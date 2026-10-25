"""Fixed-pool strategy evaluator. All choices depend on date-T inputs only."""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
from lab_io import DataError, atomic_json, read_json, sha
from lab_backtest import Case
from studio_pool import digest
from studio_cache import algorithm_hash

POLICIES=('none','discipline','risk_half','recovery_only')
@dataclass
class StudioCase:
    base: Case
    sectors: np.ndarray
    rsi: np.ndarray
    bias: np.ndarray
    recovery: np.ndarray
    regime: str


def validate_profiles(obj,available):
    if not isinstance(obj,dict) or obj.get('schema')!=1:raise DataError('Expected profiles schema 1')
    rows=obj.get('candidates',[])
    if not 1<=len(rows)<=128:raise DataError('Use 1..128 explicit candidates')
    out=[];seen=set();names=set()
    for c in rows:
        if set(c)-{'name','weights','policy'}:raise DataError('Unknown strategy setting')
        name=c.get('name');w=c.get('weights');policy=c.get('policy','none')
        if not isinstance(name,str) or not name or len(name)>80 or name in names:raise DataError('Candidate names must be unique, nonempty and <=80 chars')
        if policy not in POLICIES:raise DataError('Unknown policy')
        if not isinstance(w,dict) or not 1<=len(w)<=4 or not set(w)<=set(available):raise DataError('Use 1..4 available factors')
        try:w={k:float(v) for k,v in sorted(w.items())}
        except (ValueError,TypeError):raise DataError('Weights must be numeric')
        if any(not np.isfinite(v) or v<=0 for v in w.values()) or abs(sum(w.values())-1)>1e-8:raise DataError('Positive weights must sum to 1; no silent normalization')
        key=digest({'weights':w,'policy':policy})[:20]
        if key in seen:raise DataError('Duplicate weight/policy hypothesis')
        seen.add(key);names.add(name);out.append({'id':key,'name':name,'weights':w,'policy':policy,'complexity':len(w)})
    return out


def make_cases(snap,labels,pools,context,lock,cfg,factors):
    lookup={d:x.set_index('ticker') for d,x in snap.groupby('date')}
    outcomes={(d,int(h)):x.set_index('ticker') for (d,h),x in labels.groupby(['date','horizon'])}
    contexts={(x['pool'],pd.Timestamp(x['date'])):x for x in context.to_dict('records')}
    cases={}
    for h in cfg.horizons:
        p=lock['mapping'][str(h)];cc=[]
        for date,m in pools[pools.pool==p].groupby('date',sort=True):
            m=m.sort_values('ticker');t=m.ticker.to_numpy();s=lookup[date].reindex(t)
            if (date,h) not in outcomes:raise DataError('Matured label case missing')
            l=outcomes[(date,h)];ll=l.reindex(t);b=l.loc['0050.TW']
            net=ll.net.to_numpy(float)
            base=Case(pd.Timestamp(date),t,s[['rank_'+k for k in factors]].to_numpy(float),m.amount20.to_numpy(float),net,
                      ll.filled.fillna(False).to_numpy(bool),ll.path_mdd.to_numpy(float),float(b.net),
                      float(net.mean()) if np.isfinite(net).all() else np.nan,float((~np.isfinite(net)).mean()),
                      pd.Timestamp(b.entry),pd.Timestamp(b.exit),tuple(factors))
            cc.append(StudioCase(base,s.sector.fillna('UNKNOWN').astype(str).to_numpy() if 'sector' in s else np.repeat('UNKNOWN',len(t)),
                    s.rsi14.to_numpy(float),s.bias20.to_numpy(float),s.recovery_ready.to_numpy(float),
                    str(contexts[(p,pd.Timestamp(date))]['regime'])))
        cases[h]=cc
    return cases


def evaluate(sc,candidate,cfg):
    c=sc.base;policy=candidate['policy'];w=np.array([candidate['weights'].get(k,0.) for k in c.features])
    known=np.isfinite(c.ranks);full=(known*w).sum(1)>=1-1e-9
    coverage=float(full.mean());base={'date':c.date,'entry':c.entry,'exit':c.exit,'benchmark':c.benchmark,
        'pool_net':c.pool_net,'pool_missing_weight':c.pool_missing_weight,'regime':sc.regime,
        'candidate_id':candidate['id'],'candidate_name':candidate['name'],'policy':policy,
        'weights':candidate['weights'],'horizon':None,'is_reference':False,'exposure':1.,
        'macro_money_supply_available':False,'decision_uses_future_outcomes':False}
    blocked=(coverage<cfg.min_feature_coverage)
    if policy!='none':
        gates_known=np.isfinite(sc.rsi)&np.isfinite(sc.bias)
        if policy=='recovery_only':gates_known &= np.isfinite(sc.recovery)
        blocked |= float(gates_known.mean())<cfg.min_feature_coverage
    if policy=='risk_half' and sc.regime=='UNKNOWN':blocked=True
    if blocked:
        return dict(base,net=np.nan,loss_floor=-1.,filled_weight=0.,stock_filled_weight=0.,feature_coverage=coverage,
                    selected=[],missing_slots=cfg.top_k,cash_slots=0,position_mdd=np.nan,state='INPUT_COVERAGE_BLOCKED',
                    excess_0050=np.nan,excess_pool=np.nan,exposure_matched_excess=np.nan)
    capacity=np.isfinite(c.amount20)&(c.amount20*cfg.max_amount_participation>=cfg.capital/cfg.top_k)
    score=np.nan_to_num(c.ranks,nan=0.)@w
    score=np.where(full&capacity,score,-np.inf)
    order=np.lexsort((c.tickers,-score));core=order[np.isfinite(score[order])][:cfg.top_k]
    # Lock top-K BEFORE filters; rejected slots remain cash, never backfilled.
    chosen=[];counts={}
    for i in core:
        sec=sc.sectors[i]
        if policy!='none':
            if not np.isfinite(sc.rsi[i]) or not np.isfinite(sc.bias[i]) or sc.rsi[i]>82 or sc.bias[i]>.12:continue
            if policy=='recovery_only' and sc.recovery[i]!=1:continue
            if counts.get(sec,0)>=max(1,int(np.floor(cfg.top_k*.30))):continue
        chosen.append(i);counts[sec]=counts.get(sec,0)+1
    exposure=.5 if policy=='risk_half' and sc.regime in ('PANIC_UNCONFIRMED','EUPHORIA_DIVERGENCE') else 1.
    vals=c.net[chosen];missing=int((~np.isfinite(vals)).sum())
    net=float(vals.sum()/cfg.top_k*exposure) if missing==0 else np.nan
    filled=float(c.filled[chosen].sum()/cfg.top_k*exposure)
    mdd=c.path_mdd[chosen]
    # Position average is explicitly not portfolio daily NAV drawdown.
    result=dict(base,net=net,loss_floor=float(np.nansum(vals)/cfg.top_k*exposure-missing/cfg.top_k*exposure),
       filled_weight=filled,stock_filled_weight=filled,feature_coverage=coverage,
       selected=c.tickers[chosen].tolist(),core_selected=c.tickers[core].tolist(),
       missing_slots=missing,cash_slots=cfg.top_k-len(chosen),exposure=exposure,
       position_mdd=float(mdd.sum()/cfg.top_k*exposure) if np.isfinite(mdd).all() else np.nan,
       state='COMPLETE' if missing==0 else 'SELECTED_EXIT_UNKNOWN',
       selected_slot_weight=exposure/cfg.top_k,sector_counts=counts,
       excess_0050=net-c.benchmark,excess_pool=net-c.pool_net,
       exposure_matched_excess=net-exposure*c.benchmark)
    return result


def restore_rows(rows):
    for r in rows:
        for k in ('date','entry','exit'):r[k]=pd.Timestamp(r[k])
        for k in ('net','loss_floor','benchmark','pool_net','position_mdd','excess_0050','excess_pool','exposure_matched_excess'):
            if r.get(k) is None:r[k]=np.nan
    return rows


def cached_rows(root,scases,candidate,cfg,identity,indices,check=lambda:None):
    """One candidate x one horizon x selected time segment; changing another candidate reuses it."""
    key=digest({'data':identity,'candidate':candidate['id'],
        'evaluation':{k:getattr(cfg,k) for k in ('top_k','capital','max_amount_participation','min_feature_coverage')},
        'dates':[str(scases[int(i)].base.date) for i in indices],
        'code':algorithm_hash(('studio_engine.py',))})[:32]
    folder=Path(root)/'studio'/'strategy_cache';folder.mkdir(parents=True,exist_ok=True)
    path=folder/(key+'.json');meta=folder/(key+'.sha.json')
    if path.is_file() and meta.is_file():
        if sha(path)!=read_json(meta).get('sha256'):raise DataError('Strategy cache checksum mismatch')
        return restore_rows(read_json(path,[])),True
    rows=[]
    for i in indices:
        check();rows.append(evaluate(scases[int(i)],candidate,cfg))
    atomic_json(path,rows);atomic_json(meta,{'sha256':sha(path)})
    return rows,False
