"""Finite training-only robustness and one optional consensus weight hypothesis.

This is model selection, NOT a certificate of alpha. It does not receive outer
or tail indices for selection. Price outcomes outside train_idx are untouched.
"""
import hashlib
import json
import numpy as np
from lab_approval import reference_choice
from lab_backtest import evaluate_batch
from lab_io import DataError


def score_training(g,j,indices,weights,cfg):
    idx=np.asarray(indices,dtype=int)
    if len(idx)<cfg.min_train_months:raise DataError('Too few purged training months')
    if len(np.unique(idx))!=len(idx) or np.any(np.diff(idx)<=0):
        raise DataError('Training indices must be unique chronological observations')
    net=np.asarray(g['net'])[idx,j];ref=np.asarray(g['benchmark'])[idx]
    floors=np.asarray(g['floor'])[idx,j];fills=np.asarray(g['fills'])[idx,j]
    paired=np.isfinite(net)&np.isfinite(ref)
    last=min(cfg.recent_train_months,len(idx));recent=paired[-last:]
    if paired.mean()<cfg.min_fold_coverage or recent.mean()<cfg.min_fold_coverage:return None
    x=floors-ref-cfg.extra_roundtrip_stress*fills
    blocks=np.array_split(np.arange(len(idx)),3)
    b=np.array([np.nanmean(x[k]) if np.isfinite(x[k]).any() else np.nan for k in blocks])
    if not np.isfinite(b).all():return None
    full=float(np.nanmean(x));recent_mean=float(np.nanmean(x[-last:]))
    complexity=len([v for v in weights.values() if v>0])
    objective=min(full,recent_mean)-cfg.uncertainty_penalty*float(b.std())-cfg.complexity_penalty*max(0,complexity-1)
    failed=[]
    if full<=0:failed.append('NONPOSITIVE_FULL_TRAINING_COST_STRESS')
    if recent_mean<=0:failed.append('NONPOSITIVE_RECENT_TRAINING_COST_STRESS')
    if int((b>0).sum())<cfg.min_positive_inner_blocks:failed.append('TRAINING_BLOCK_DIRECTIONS_UNSTABLE')
    if objective<=0:failed.append('NONPOSITIVE_ROBUST_OBJECTIVE')
    return {'objective':float(objective),'train_excess':float(np.nanmean(net-ref)),
            'missing_slot_loss_floor_excess':float(np.nanmean(floors-ref)),
            'train_stress_excess':full,'recent_train_stress_excess':recent_mean,
            'recent_train_months':last,'inner_block_means':b.tolist(),
            'inner_blocks_include_extra_cost':True,'coverage':float(paired.mean()),
            'recent_coverage':float(recent.mean()),'train_dates':len(idx),
            'selection_eligible':not failed,'training_blockers':failed}


def _return_correlation(a,b):
    mask=np.isfinite(a)&np.isfinite(b)
    if mask.sum()<6:return 1.0
    x=a[mask];y=b[mask]
    if x.std()<1e-12 or y.std()<1e-12:return 1.0
    return float(np.corrcoef(x,y)[0,1])


def choose_robust(grids,candidates,h,train_idx,cfg,cases=None):
    idx=np.asarray(train_idx,dtype=int)
    if len(idx)<cfg.min_train_months:raise DataError('Too few purged training months')
    board=[]
    for (pool,hh),g in sorted(grids.items()):
        if hh!=h:continue
        for j,c in enumerate(candidates):
            s=score_training(g,j,idx,c['weights'],cfg)
            if s is None:continue
            # Hash only training outcomes. Equivalence is descriptive; two
            # histories can have equal returns without equal future holdings.
            digest=hashlib.sha256()
            for name in ('net','floor','fills'):
                z=np.asarray(g[name])[idx,j].astype('<f8').copy()
                z[np.isnan(z)]=np.nan
                digest.update(z.tobytes())
            board.append(dict(s,pool=pool,candidate_index=j,candidate_id=c['id'],
                name=c['name'],weights=c['weights'],complexity=c['complexity'],
                training_outcome_equivalence=digest.hexdigest()[:16],
                selection_method='REGISTERED_STATIC_CANDIDATE'))
    board.sort(key=lambda r:(-r['objective'],r['complexity'],r['pool'],r['candidate_id']))
    eligible=[r for r in board if r['selection_eligible']]
    if not eligible:
        ref=reference_choice('NO_STABLE_POSITIVE_TRAINING_CANDIDATE')
        ref['train_dates']=len(idx)
        return ref,board
    best=eligible[0]['objective']
    near=[r for r in eligible if r['objective']>=best-cfg.near_best_tolerance]
    near.sort(key=lambda r:(r['complexity'],-r['objective'],r['pool'],r['candidate_id']))
    winner=dict(near[0]);winner['consensus_tested']=False
    if cases is None or cfg.consensus_members<2:return winner,board
    pool=winner['pool'];g=grids[(pool,h)]
    peers=[r for r in eligible if r['pool']==pool and r['objective']>=winner['objective']-cfg.near_best_tolerance]
    # Anchor on the chosen static model and add up to two distinct training
    # histories. The agreement hypothesis is evaluated as an actual new RANK,
    # never by averaging old portfolio returns and calling that a new ranking.
    selected=[winner];vectors=[g['net'][idx,winner['candidate_index']]-g['benchmark'][idx]]
    for row in peers:
        if row['candidate_id']==winner['candidate_id']:continue
        vector=g['net'][idx,row['candidate_index']]-g['benchmark'][idx]
        if any(_return_correlation(vector,v)>=cfg.consensus_return_corr_limit for v in vectors):continue
        selected.append(row);vectors.append(vector)
        if len(selected)>=cfg.consensus_members:break
    if len(selected)<2:return winner,board
    weights={}
    for row in selected:
        for factor,value in row['weights'].items():weights[factor]=weights.get(factor,0.)+value/len(selected)
    keep=sorted(weights,key=lambda f:(-weights[f],f))[:cfg.max_active_factors]
    retained=sum(weights[f] for f in keep)
    weights={f:weights[f]/retained for f in sorted(keep)}
    c={'id':'consensus_'+hashlib.sha256(json.dumps(weights,sort_keys=True).encode()).hexdigest()[:12],
       'name':'training_consensus','weights':weights,'complexity':len(weights)}
    # Max one extra hypothesis per training freeze. Same indices, no tail access.
    values=[evaluate_batch(cases[(pool,h)][int(i)],[c],cfg) for i in idx]
    cg={name:np.array([r[name] for r in values]) for name in ('net','loss_floor','filled_weight')}
    cg={'net':cg['net'],'floor':cg['loss_floor'],'fills':cg['filled_weight'],'benchmark':g['benchmark'][idx]}
    s=score_training(cg,0,np.arange(len(idx)),weights,cfg)
    winner['consensus_tested']=True
    if s is not None:
        row=dict(s,pool=pool,candidate_index=-2,candidate_id=c['id'],name=c['name'],
                 weights=weights,complexity=len(weights),selection_method='TRAINING_ONLY_CONSENSUS_RERANK',
                 consensus_members=[r['candidate_id'] for r in selected],
                 consensus_removed_factor_mass=1-retained,consensus_tested=True,
                 training_outcome_equivalence=None)
        board.append(row)
        # No tolerance rescue: the recomputed consensus must score at least as
        # well as its chosen simple anchor on purged training only.
        if row['selection_eligible'] and row['objective']>=winner['objective']:
            winner=row
    board.sort(key=lambda r:(-r['objective'],r['complexity'],r['pool'],r['candidate_id']))
    return winner,board
