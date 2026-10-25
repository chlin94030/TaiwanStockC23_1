"""V22 one-time local feature/pool/stage cache. No network calls."""
from pathlib import Path
import pandas as pd
import numpy as np
from lab_config import FACTORS as BASE_FACTORS, OPTIONAL_FACTORS
from lab_features import raw_adjusted_frame
from lab_price_quality import analytic_hlc_from_audit
from lab_io import DataError, atomic_json, read_json, sha, frame_write
from studio_pool import digest, load_dataset
from studio_cache import algorithm_hash, verified_cache, store_for_dataset
from studio_behavior import FACTORS as BOOK_FACTORS, behavior_features, benchmark_context
from v22_pool import build_v22_pools
from v22_regime import classify_stage


def prepare_v22(root,progress=lambda **kw:None,check=lambda:None):
    folder,meta=load_dataset(root)
    key=digest({'data':meta['dataset_id'],
        'algorithm':algorithm_hash(('v22_cache.py','v22_pool.py','v22_regime.py','studio_behavior.py','lab_features.py','lab_price_quality.py'))})[:24]
    out=Path(root)/'v22'/'feature_cache'/key
    files=('features.csv.gz','pools.csv.gz','market_stage.csv.gz','INFO.json')
    if verified_cache(out,files):return out,read_json(out/'INFO.json'),True
    out.mkdir(parents=True,exist_ok=True);check()
    store=store_for_dataset(root,meta);cfg=meta['source_config'];cal=pd.DatetimeIndex(read_json(folder/'calendar.json'))
    monthly=pd.read_csv(folder/'monthly_pools.csv.gz',parse_dates=['date'])
    if 'ADV500' not in set(monthly.pool):raise DataError('V22 requires an ADV500 prepared control pool')
    v22p=build_v22_pools(monthly)
    union=v22p[['date','ticker']].drop_duplicates();tickers=sorted(union.ticker.unique())
    def daily(t):
        raw=store.get('raw',t,cfg['start'],cfg['end']);adj=store.get('adj',t,cfg['start'],cfg['end'])
        if raw.empty or adj.empty:raise DataError('Saved daily cache missing for V22 symbol '+t)
        ana,_=analytic_hlc_from_audit(raw,store.quality_details(t));return raw_adjusted_frame(raw,adj,cal,ana)
    benchmark=daily('0050.TW');bc=benchmark_context(benchmark)
    # V21.6 already computed the 18 base price/volume factors on the exact ADV500
    # monthly union. Reuse those values; V22 adds only the new book-derived factors.
    parent=pd.read_csv(folder/'features.csv.gz',parse_dates=['date'])
    parent=union.merge(parent,on=['date','ticker'],how='left',validate='one_to_one')
    chunks=[]
    for i,t in enumerate(tickers):
        check();f=daily(t);ds=pd.DatetimeIndex(sorted(union[union.ticker==t].date.unique()))
        z=behavior_features(f).reindex(ds);z['date']=z.index;z['ticker']=t;chunks.append(z.reset_index(drop=True))
        if i%10==0:progress(phase='V22_FEATURES',done=i+1,total=len(tickers),message='V22 adds book factors only; V21.6 base factors reused, no HTTP')
    ext=pd.concat(chunks,ignore_index=True)
    replace_cols=[c for c in ext if c in parent and c not in ('date','ticker')]
    parent=parent.drop(columns=replace_cols)
    snap=parent.merge(ext,on=['date','ticker'],how='left',validate='one_to_one')
    available=list(BASE_FACTORS)+list(BOOK_FACTORS)
    if cfg.get('enable_optional_factors'):available+=list(OPTIONAL_FACTORS)
    for c in available:
        if c not in snap:raise DataError('V22 factor absent: '+c)
        snap['rank_'+c]=snap.groupby('date')[c].rank(pct=True,method='average')
    contexts=[]
    for pool,pg in v22p.groupby('pool',sort=True):
        previous=None
        for date,g in pg.groupby('date',sort=True):
            sf=snap[(snap.date==date)&snap.ticker.isin(g.ticker)]
            def frac(col,fn):
                s=sf[col];known=float(s.notna().mean())
                return (float(fn(s.dropna()).mean()) if known>=.95 and len(s.dropna()) else np.nan,known)
            breadth,bk=frac('above_ma60',lambda s:s.gt(.5))
            b20,b20k=frac('momentum20',lambda s:s.gt(0))
            b60,b60k=frac('momentum60',lambda s:s.gt(0))
            row=bc.loc[date].to_dict();row.update(date=date,pool=pool,breadth=breadth,breadth_known=bk,
                breadth_mom20=b20,breadth_mom20_known=b20k,breadth_mom60=b60,breadth_mom60_known=b60k,
                median_demand_balance=float(sf.ks_demand_balance.median(skipna=True)),
                median_amount_acceleration=float(sf.amount_acceleration.median(skipna=True)),
                proxy_holder_counts=True,news_reaction_observed=False,macro_money_supply_observed=False)
            stage=classify_stage(row,previous);previous=stage['stage'] if stage['stage']!='UNKNOWN' else previous
            row.update(stage=stage['stage'],stage_confidence=stage['confidence'],stage_action=stage['action'],
                stage_default_exposure=stage['exposure'],stage_scores=str(stage['scores']),stage_transition_guarded=stage['transition_guarded'])
            contexts.append(row)
    info={'schema':1,'version':'22.0','feature_id':key,'dataset_id':meta['dataset_id'],'factors':available,
          'pools':sorted(v22p.pool.unique()),'daily_symbols_read':len(tickers)+1,'http_requests':0,'reused_v216_base_factors':True,'pool_search_http_requests':0,
          'rank_reference':'ALL_V22_POOL_UNION_SAME_DATE','holder_counts_are_proxy':True,'news_reaction_tested':False,
          'macro_money_supply_tested':False,'source_book_formulas_claimed':False}
    frame_write(out/'features.csv.gz',snap);frame_write(out/'pools.csv.gz',v22p);frame_write(out/'market_stage.csv.gz',pd.DataFrame(contexts))
    atomic_json(out/'INFO.json',info);atomic_json(out/'CACHE.json',{n:sha(out/n) for n in files})
    return out,info,False
