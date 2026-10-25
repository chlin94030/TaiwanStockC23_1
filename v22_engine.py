"""V22.3 stage-first stock-selection engine.

Market stage decides whether/how much to invest; stock factors decide what to buy.
V22.3 adds horizon-specific priors and optional walk-forward IC/GBM challengers,
while preserving V22's fixed-pool, stage gate and point-in-time discipline.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from lab_backtest import Case
from lab_io import DataError, atomic_json, read_json, sha
from studio_pool import digest
from studio_cache import algorithm_hash
from v223_adaptive import MODEL_TYPES, DEFAULT_ADAPTIVE, weights_for, adaptive_weights, gbm_score

HORIZONS=(5,20,60)
STAGES={'A1','A2','A3','B1','B2','B3','UNKNOWN'}

@dataclass
class V22Case:
    base: Case
    sectors: np.ndarray
    rsi: np.ndarray
    bias: np.ndarray
    double_top_rank: np.ndarray
    capitulation_rank: np.ndarray
    double_bottom_rank: np.ndarray
    stage: str
    confidence: float
    action: str
    horizon: int = 0


def _clean_stage_weights(sw, available):
    if not isinstance(sw,dict): raise DataError('Stage weights must be an object')
    clean={}
    for stage,w in sw.items():
        if stage not in STAGES: raise DataError('Unknown stage in weights')
        if not isinstance(w,dict) or not 1<=len(w)<=4 or not set(w)<=set(available):
            raise DataError('Each active stage uses 1..4 available factors')
        q={k:float(v) for k,v in sorted(w.items())}
        if any(not np.isfinite(v) or v<=0 for v in q.values()) or abs(sum(q.values())-1)>1e-8:
            raise DataError('Positive stage weights must sum to 1')
        clean[stage]=q
    return clean


def validate_profiles(obj,available):
    """Accept V22 schema-2 profiles and new V22.3 schema-3 profiles."""
    if not isinstance(obj,dict) or obj.get('schema') not in (2,3):
        raise DataError('V22.3 expects profiles schema 2 or 3')
    rows=obj.get('candidates',[])
    if not 1<=len(rows)<=32: raise DataError('Use 1..32 explicit V22 candidates')
    out=[];seen=set();names=set()
    for c in rows:
        allowed={'name','stage_weights','horizon_stage_weights','stage_exposure','discipline','model_type',
                 'adaptive','selection_eligible','gbm'}
        if set(c)-allowed: raise DataError('Unknown V22 strategy setting: '+str(sorted(set(c)-allowed)))
        name=c.get('name')
        if not isinstance(name,str) or not name or name in names: raise DataError('Candidate names must be unique')
        se=c.get('stage_exposure');disc=bool(c.get('discipline',True));model_type=str(c.get('model_type','manual'))
        if model_type not in MODEL_TYPES: raise DataError('Unknown V22.3 model_type '+model_type)
        if not isinstance(se,dict) or set(se)!=STAGES: raise DataError('Stage exposure must define A1/A2/A3/B1/B2/B3/UNKNOWN')
        exp={k:float(v) for k,v in se.items()}
        if any(not np.isfinite(v) or not 0<=v<=1 for v in exp.values()): raise DataError('Stage exposures must be within [0,1]')

        if c.get('horizon_stage_weights') is not None:
            hsw=c['horizon_stage_weights']
            if not isinstance(hsw,dict): raise DataError('horizon_stage_weights must be an object')
            weights_by_horizon={}
            for h in HORIZONS:
                src=hsw.get(str(h),hsw.get(h))
                if src is None: raise DataError(f'Horizon {h} weights are required')
                weights_by_horizon[h]=_clean_stage_weights(src,available)
        else:
            clean=_clean_stage_weights(c.get('stage_weights'),available)
            weights_by_horizon={h:clean for h in HORIZONS}

        for h,sw in weights_by_horizon.items():
            for s,v in exp.items():
                if v>0 and s!='UNKNOWN' and s not in sw:
                    raise DataError(f'Positive exposure requires {h}d weights for stage {s}')

        adaptive={**DEFAULT_ADAPTIVE,**(c.get('adaptive') or {})}
        for k in ('blend','stage_mix','ic_floor','ic_cap','prior_floor'):
            if not np.isfinite(float(adaptive[k])): raise DataError('Invalid adaptive setting '+k)
        if not 0<=float(adaptive['blend'])<=1 or not 0<=float(adaptive['stage_mix'])<=1 or not 0<=float(adaptive['prior_floor'])<=1:
            raise DataError('Adaptive blend/stage_mix/prior_floor must be within [0,1]')
        if not 6<=int(adaptive['lookback_cases'])<=120 or not 4<=int(adaptive['min_history_cases'])<=60:
            raise DataError('Invalid adaptive history length')
        gbm=dict(c.get('gbm') or {})
        selection_eligible=bool(c.get('selection_eligible', model_type!='gbm_challenger'))
        serial={'weights_by_horizon':weights_by_horizon,'stage_exposure':exp,'discipline':disc,'model_type':model_type,
                'adaptive':adaptive if model_type in ('hybrid_ic','ic_adaptive') else {},'gbm':gbm if model_type=='gbm_challenger' else {},
                'selection_eligible':selection_eligible}
        key=digest(serial)[:20]
        if key in seen: raise DataError('Duplicate V22 strategy hypothesis')
        seen.add(key);names.add(name)
        # stage_weights remains as a compatibility fallback for old helper/tests.
        fallback=weights_by_horizon[20]
        out.append({'id':key,'name':name,'weights_by_horizon':weights_by_horizon,'stage_weights':fallback,
            'stage_exposure':exp,'discipline':disc,'model_type':model_type,'adaptive':adaptive,'gbm':gbm,
            'selection_eligible':selection_eligible,
            'complexity':sum(len(w) for h in weights_by_horizon.values() for w in h.values())})
    return out


def make_cases(snap,labels,pools,context,pool,cfg,factors):
    lookup={d:x.set_index('ticker') for d,x in snap.groupby('date')}
    outcomes={(d,int(h)):x.set_index('ticker') for (d,h),x in labels.groupby(['date','horizon'])}
    contexts={(x['pool'],pd.Timestamp(x['date'])):x for x in context.to_dict('records')}
    rows=pools[pools.pool==pool]
    if rows.empty: raise DataError('V22 pool absent: '+pool)
    cases={}
    for h in cfg.horizons:
        cc=[]
        for date,m in rows.groupby('date',sort=True):
            m=m.sort_values('ticker');t=m.ticker.to_numpy();s=lookup[date].reindex(t)
            if (date,h) not in outcomes: raise DataError('Matured label case missing')
            l=outcomes[(date,h)];ll=l.reindex(t);b=l.loc['0050.TW'];net=ll.net.to_numpy(float)
            ranks=s[['rank_'+k for k in factors]].to_numpy(float)
            base=Case(pd.Timestamp(date),t,ranks,m.amount20.to_numpy(float),net,ll.filled.fillna(False).to_numpy(bool),
                ll.path_mdd.to_numpy(float),float(b.net),float(net.mean()) if np.isfinite(net).all() else np.nan,
                float((~np.isfinite(net)).mean()),pd.Timestamp(b.entry),pd.Timestamp(b.exit),tuple(factors))
            cx=contexts[(pool,pd.Timestamp(date))]
            def col(name,default=np.nan): return s[name].to_numpy(float) if name in s else np.full(len(t),default)
            cc.append(V22Case(base,s.sector.fillna('UNKNOWN').astype(str).to_numpy() if 'sector' in s else np.repeat('UNKNOWN',len(t)),
                col('rsi14'),col('bias20'),col('rank_ks_double_top_risk'),col('rank_ks_capitulation_reversal'),col('rank_ks_double_bottom'),
                str(cx['stage']),float(cx['stage_confidence']),str(cx['stage_action']),int(h)))
        cases[h]=cc
    return cases


def _base_record(sc,candidate,exposure):
    c=sc.base
    return {'date':c.date,'entry':c.entry,'exit':c.exit,'benchmark':c.benchmark,'pool_net':c.pool_net,'pool_missing_weight':c.pool_missing_weight,
          'stage':sc.stage,'stage_confidence':sc.confidence,'stage_action':sc.action,'candidate_id':candidate['id'],'candidate_name':candidate['name'],
          'model_type':candidate.get('model_type','manual'),'selection_eligible':candidate.get('selection_eligible',True),
          'stage_exposure':exposure,'market_stage_dominant':True,'decision_uses_future_outcomes':False,'is_reference':False,
          'horizon':int(getattr(sc,'horizon',0) or 0)}


def _cash_record(base,c,cfg,state,exposure=0.,diag=None):
    out=dict(base,net=0.,loss_floor=0.,filled_weight=0.,stock_filled_weight=0.,feature_coverage=1.,selected=[],core_selected=[],
        missing_slots=cfg.top_k,cash_slots=cfg.top_k,position_mdd=0.,state=state,
        excess_0050=-c.benchmark if np.isfinite(c.benchmark) else np.nan,
        excess_pool=-c.pool_net if np.isfinite(c.pool_net) else np.nan,exposure_matched_excess=0.,effective_weights={})
    if diag: out.update(adaptive_history_cases=diag.get('history_cases',0),adaptive_fallback=diag.get('fallback'))
    return out


def _evaluate_scored(sc,candidate,cfg,spec,score,coverage,effective_weights=None,diag=None):
    c=sc.base;stage=sc.stage;exposure=float(candidate['stage_exposure'].get(stage,0.0));base=_base_record(sc,candidate,exposure)
    if stage=='UNKNOWN' or sc.confidence<spec['stage_confidence_floor']:
        return _cash_record(base,c,cfg,'STAGE_UNCERTAIN_CASH',0.,diag)
    if exposure<=0:
        return _cash_record(base,c,cfg,'BOOK_STAGE_CASH_OR_SELL',0.,diag)
    if coverage<cfg.min_feature_coverage:
        out=dict(base,net=np.nan,loss_floor=-exposure,filled_weight=0.,stock_filled_weight=0.,feature_coverage=float(coverage),selected=[],core_selected=[],
            missing_slots=cfg.top_k,cash_slots=0,position_mdd=np.nan,state='INPUT_COVERAGE_BLOCKED',excess_0050=np.nan,excess_pool=np.nan,
            exposure_matched_excess=np.nan,effective_weights=effective_weights or {})
        if diag: out.update(adaptive_history_cases=diag.get('history_cases',0),adaptive_fallback=diag.get('fallback'))
        return out
    capacity=np.isfinite(c.amount20)&(c.amount20*cfg.max_amount_participation>=cfg.capital/cfg.top_k)
    score=np.where(capacity & np.isfinite(score),score,-np.inf)
    order=np.lexsort((c.tickers,-score));ranked=order[np.isfinite(score[order])]
    core=ranked[:cfg.top_k]
    # V22.4.2: discipline filters are applied BEFORE the funded list is finalized.
    # Previously we filtered only the raw top_k and left rejected slots as cash,
    # which confounded stock-model quality with an avoidable implementation underfill.
    # Scan a bounded deeper list (4x top_k) and refill until top_k valid names are found.
    scan_limit=min(len(ranked),max(cfg.top_k,cfg.top_k*4))
    scan=ranked[:scan_limit]
    chosen=[];counts={};cap=max(1,int(np.floor(cfg.top_k*spec['sector_cap'])));rank_depth_used=0
    for rank_pos,idx in enumerate(scan, start=1):
        rank_depth_used=rank_pos
        sec=sc.sectors[idx]
        if candidate['discipline']:
            if not np.isfinite(sc.rsi[idx]) or not np.isfinite(sc.bias[idx]) or sc.rsi[idx]>spec['max_rsi'] or sc.bias[idx]>spec['max_bias20']: continue
            if np.isfinite(sc.double_top_rank[idx]) and sc.double_top_rank[idx]>.90: continue
            if counts.get(sec,0)>=cap: continue
            if stage=='B3':
                confirm=max(sc.capitulation_rank[idx] if np.isfinite(sc.capitulation_rank[idx]) else 0,
                            sc.double_bottom_rank[idx] if np.isfinite(sc.double_bottom_rank[idx]) else 0)
                if confirm<spec['b3_confirmation_rank']: continue
        chosen.append(idx);counts[sec]=counts.get(sec,0)+1
        if len(chosen)>=cfg.top_k: break
    vals=c.net[chosen];missing=int((~np.isfinite(vals)).sum());net=float(vals.sum()/cfg.top_k*exposure) if missing==0 else np.nan
    filled=float(c.filled[chosen].sum()/cfg.top_k*exposure);mdd=c.path_mdd[chosen]
    loss_floor=float(np.nansum(vals)/cfg.top_k*exposure-missing/cfg.top_k*exposure)
    pmdd=float(np.nanmean(mdd)) if len(mdd) and np.isfinite(mdd).any() else (0. if not chosen else np.nan)
    state='SELECTED_EXIT_UNKNOWN' if missing else ('SELECTED' if chosen else 'NO_CONFIRMED_SETUP')
    out=dict(base,net=net,loss_floor=loss_floor,filled_weight=filled,stock_filled_weight=filled,feature_coverage=float(coverage),
        selected=c.tickers[chosen].tolist(),core_selected=c.tickers[core].tolist(),selected_scores=[float(score[i]) for i in chosen],
        missing_slots=cfg.top_k-len(chosen),cash_slots=cfg.top_k-len(chosen),position_mdd=pmdd,state=state,
        refill_used=bool(len(chosen)>len(set(chosen).intersection(set(core.tolist())))),
        rank_depth_used=int(rank_depth_used),screened_candidates=int(min(scan_limit,rank_depth_used)),
        excess_0050=net-c.benchmark if np.isfinite(net) and np.isfinite(c.benchmark) else np.nan,
        excess_pool=net-c.pool_net if np.isfinite(net) and np.isfinite(c.pool_net) else np.nan,
        exposure_matched_excess=net-c.benchmark*exposure if np.isfinite(net) and np.isfinite(c.benchmark) else np.nan,
        effective_weights=effective_weights or {})
    if diag:
        out.update(adaptive_history_cases=diag.get('history_cases',0),stage_history_cases=diag.get('stage_history_cases',0),
                   adaptive_fallback=diag.get('fallback'),adaptive_blend=diag.get('blend'),adaptive_factor_ic=diag.get('factor_ic'))
    return out


def _score_from_weights(sc,weights):
    c=sc.base
    w=np.array([weights.get(k,0.) for k in c.features],float)
    known=np.isfinite(c.ranks);fully_known=(known*w).sum(1)>=1-1e-9
    coverage=float(fully_known.mean())
    score=np.nan_to_num(c.ranks,nan=0.)@w
    score=np.where(fully_known,score,np.nan)
    return score,coverage


def evaluate(sc,candidate,cfg,spec):
    """Backward-compatible scalar evaluator. Adaptive types fall back to their prior."""
    h=int(getattr(sc,'horizon',0) or 20);weights=weights_for(candidate,h,sc.stage)
    exposure=float(candidate['stage_exposure'].get(sc.stage,0.0));base=_base_record(sc,candidate,exposure)
    if sc.stage=='UNKNOWN' or sc.confidence<spec['stage_confidence_floor']:
        return _cash_record(base,sc.base,cfg,'STAGE_UNCERTAIN_CASH')
    if exposure<=0 or not weights:
        return _cash_record(base,sc.base,cfg,'BOOK_STAGE_CASH_OR_SELL')
    score,coverage=_score_from_weights(sc,weights)
    return _evaluate_scored(sc,candidate,cfg,spec,score,coverage,weights,{'fallback':'STATIC_PRIOR_NO_HISTORY_CONTEXT','history_cases':0})


def evaluate_at(cases,i,candidate,cfg,spec):
    sc=cases[int(i)];stage=sc.stage;exposure=float(candidate['stage_exposure'].get(stage,0.0));base=_base_record(sc,candidate,exposure)
    if stage=='UNKNOWN' or sc.confidence<spec['stage_confidence_floor']:
        return _cash_record(base,sc.base,cfg,'STAGE_UNCERTAIN_CASH')
    if exposure<=0:
        return _cash_record(base,sc.base,cfg,'BOOK_STAGE_CASH_OR_SELL')
    model_type=candidate.get('model_type','manual')
    if model_type=='gbm_challenger':
        score,diag=gbm_score(cases,i,candidate)
        if score is None:
            weights=weights_for(candidate,sc.horizon,stage)
            if not weights: return _cash_record(base,sc.base,cfg,'NO_MODEL_WEIGHTS',diag=diag)
            score,coverage=_score_from_weights(sc,weights)
            diag={**diag,'fallback':diag.get('fallback') or 'GBM_TO_PRIOR'}
            return _evaluate_scored(sc,candidate,cfg,spec,score,coverage,weights,diag)
        coverage=float(np.isfinite(score).mean())
        return _evaluate_scored(sc,candidate,cfg,spec,score,coverage,{},diag)
    weights,diag=adaptive_weights(cases,i,candidate)
    if not weights:
        return _cash_record(base,sc.base,cfg,'NO_MODEL_WEIGHTS',diag=diag)
    score,coverage=_score_from_weights(sc,weights)
    return _evaluate_scored(sc,candidate,cfg,spec,score,coverage,weights,diag)


def _restore(rows):
    for r in rows:
        for k in ('date','entry','exit'):
            if k in r:r[k]=pd.Timestamp(r[k])
    return rows


def cached_rows(root,cases,candidate,cfg,spec,identity,indices,check=lambda:None):
    key=digest({'data':identity,'candidate':candidate['id'],'pool':spec['pool_rule'],
        'settings':{k:spec[k] for k in ('top_k','capital','stage_confidence_floor','sector_cap','max_rsi','max_bias20','b3_confirmation_rank')},
        'dates':[str(cases[int(i)].base.date) for i in indices],
        'code':algorithm_hash(('v22_engine.py','v223_adaptive.py'))})[:32]
    folder=Path(root)/'v22'/'strategy_cache';folder.mkdir(parents=True,exist_ok=True);path=folder/(key+'.json');meta=folder/(key+'.sha.json')
    if path.is_file() and meta.is_file():
        if sha(path)!=read_json(meta).get('sha256'):raise DataError('V22 strategy cache checksum mismatch')
        return _restore(read_json(path,[])),True
    rows=[]
    for i in indices:
        check();rows.append(evaluate_at(cases,int(i),candidate,cfg,spec))
    atomic_json(path,rows);atomic_json(meta,{'sha256':sha(path)})
    return rows,False
