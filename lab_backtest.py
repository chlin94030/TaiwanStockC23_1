"""Same dates, same global factor reference, same cost assumptions for all pools."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from lab_config import FACTORS,OPTIONAL_FACTORS
from lab_io import DataError

@dataclass
class Case:
    date: pd.Timestamp
    tickers: np.ndarray
    ranks: np.ndarray
    amount20: np.ndarray
    net: np.ndarray
    filled: np.ndarray
    path_mdd: np.ndarray
    benchmark: float
    pool_net: float
    pool_missing_weight: float
    entry: pd.Timestamp
    exit: pd.Timestamp
    features: tuple


def build_cases(snap,labels,pools,cfg):
    features=tuple(FACTORS)+(tuple(OPTIONAL_FACTORS) if cfg.enable_optional_factors else ())
    by_date={d:f.set_index('ticker') for d,f in snap.groupby('date')}
    lbl={(d,int(h)):f.set_index('ticker') for (d,h),f in labels.groupby(['date','horizon'])}
    cases={}
    for (pool,d),members in pools.groupby(['pool','date'],sort=True):
        members=members.sort_values('ticker')
        syms=members.ticker.to_numpy()
        sf=by_date[d].reindex(syms)
        ranks=sf[['rank_'+c for c in features]].to_numpy(float)
        amount=members.amount20.to_numpy(float)
        for h in cfg.horizons:
            lf=lbl[(d,h)];stock=lf.reindex(syms)
            nets=stock.net.to_numpy(float)
            b=lf.loc['0050.TW'] if '0050.TW' in lf.index else None
            bench=float(b.net) if b is not None and pd.notna(b.net) else np.nan
            poolnet=float(np.mean(nets)) if np.isfinite(nets).all() else np.nan
            cases.setdefault((pool,h),[]).append(Case(d,syms,ranks,amount,nets,
                stock.filled.fillna(False).to_numpy(bool),stock.path_mdd.to_numpy(float),
                bench,poolnet,float(np.mean(~np.isfinite(nets))),
                pd.Timestamp(b.entry) if b is not None else pd.NaT,
                pd.Timestamp(b.exit) if b is not None else pd.NaT,features))
    return cases


def evaluate(case,weights,cfg):
    if not weights or not set(weights).issubset(case.features):raise DataError('Unknown/empty candidate factors')
    if any(not np.isfinite(v) or v<0 for v in weights.values()) or abs(sum(weights.values())-1)>1e-6:
        raise DataError('Candidate weights must be finite, nonnegative and sum to one')
    w=np.array([weights.get(k,0.) for k in case.features])
    known=np.isfinite(case.ranks)
    fully_known=(known*w).sum(axis=1)>=(w.sum()-1e-9)
    knowledge=float(np.mean(fully_known))
    score=np.nan_to_num(case.ranks,nan=0.)@w
    capacity=np.isfinite(case.amount20)&(case.amount20*cfg.max_amount_participation>=cfg.capital/cfg.top_k)
    eligible=fully_known&capacity
    scores=np.where(eligible,score,-np.inf)
    order=np.lexsort((case.tickers,-scores))
    chosen=order[np.isfinite(scores[order])][:cfg.top_k]
    # Missing input coverage is a data failure, not an opportunistic cash policy.
    if knowledge<cfg.min_feature_coverage:
        return {'net':np.nan,'loss_floor':-1.,'filled_weight':0.,'feature_coverage':knowledge,
                'selected':[],'missing_slots':cfg.top_k,'cash_slots':0,'state':'FEATURE_COVERAGE_BLOCKED','position_mdd':np.nan}
    vals=case.net[chosen]
    missing=int(np.sum(~np.isfinite(vals)))
    complete=missing==0
    net=float(vals.sum()/cfg.top_k) if complete else np.nan
    # Explicit sensitivity ONLY, never reported as an observed performance result.
    floor=float(np.nansum(vals)/cfg.top_k-missing/cfg.top_k)
    pmdd=case.path_mdd[chosen]
    return {'net':net,'loss_floor':floor,'filled_weight':float(case.filled[chosen].sum()/cfg.top_k),
        'feature_coverage':knowledge,'selected':case.tickers[chosen].tolist(),'missing_slots':missing,
        'cash_slots':int(cfg.top_k-len(chosen)),'state':'COMPLETE' if complete else 'SELECTED_EXIT_UNKNOWN',
        'position_mdd':float(pmdd.sum()/cfg.top_k) if np.isfinite(pmdd).all() else np.nan}


def evaluate_grid(cases,candidates,cfg,progress,check):
    grids={}
    total=len(cases);done=0
    for key,cc in sorted(cases.items()):
        result=np.full((len(cc),len(candidates)),np.nan)
        floors=result.copy();fills=np.zeros_like(result);coverage=np.zeros_like(result)
        for i,case in enumerate(cc):
            check()
            batch=evaluate_batch(case,candidates,cfg)
            result[i]=batch['net'];floors[i]=batch['loss_floor'];fills[i]=batch['filled_weight'];coverage[i]=batch['feature_coverage']
            if i%6==0:progress(phase='GRID',done=done,total=total,message=f'{key[0]} / {key[1]}d / month {i+1}/{len(cc)}')
        grids[key]={'net':result,'floor':floors,'fills':fills,'feature_coverage':coverage,
                    'benchmark':np.array([c.benchmark for c in cc]),'pool':np.array([c.pool_net for c in cc]),
                    'dates':pd.DatetimeIndex([c.date for c in cc]),'exits':pd.DatetimeIndex([c.exit for c in cc])}
        done+=1
    return grids


def evaluate_batch(case,candidates,cfg):
    """Numerical parity with evaluate(), without per-model dictionary/list work."""
    w=np.array([[c['weights'].get(k,0.) for c in candidates] for k in case.features])
    known=np.isfinite(case.ranks)
    valid=known.astype(float)@w >= (w.sum(axis=0)[None,:]-1e-9)
    knowledge=valid.mean(axis=0)
    capacity=np.isfinite(case.amount20)&(case.amount20*cfg.max_amount_participation>=cfg.capital/cfg.top_k)
    score=np.nan_to_num(case.ranks,nan=0.)@w
    score=np.where(valid&capacity[:,None],score,-np.inf)
    # Cases are ordered by ticker. Stable score sorting preserves scalar tie rules.
    order=np.argsort(-score,axis=0,kind='stable')[:cfg.top_k,:]
    col=np.arange(len(candidates))[None,:]
    selected=np.isfinite(score[order,col])
    vals=np.where(selected,case.net[order],0.)
    missing=(selected&(~np.isfinite(vals))).sum(axis=0)
    net=np.where(missing==0,np.sum(vals,axis=0)/cfg.top_k,np.nan)
    floors=np.nansum(vals,axis=0)/cfg.top_k-missing/cfg.top_k
    fills=(case.filled[order]&selected).sum(axis=0)/cfg.top_k
    blocked=knowledge<cfg.min_feature_coverage
    net[blocked]=np.nan;floors[blocked]=-1.;fills[blocked]=0.
    return {'net':net,'loss_floor':floors,'filled_weight':fills,'feature_coverage':knowledge}
