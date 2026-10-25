"""Independent fixed-pool features and cost-specific label caches, no HTTP."""
from pathlib import Path
import hashlib
import json
import pandas as pd
from lab_config import FACTORS as BASE_FACTORS, OPTIONAL_FACTORS
from lab_features import raw_adjusted_frame, features, label_records
from lab_price_quality import analytic_hlc_from_audit
from lab_io import Store, DataError, atomic_json, read_json, sha, frame_write
from studio_pool import digest, load_dataset, load_pool, source_signature
from studio_behavior import FACTORS as BOOK_FACTORS, behavior_features, benchmark_context, regime


def algorithm_hash(names):
    base=Path(__file__).parent
    return digest({n:sha(base/n) for n in names})


def verified_cache(folder,names):
    m=read_json(Path(folder)/'CACHE.json')
    if not m:return False
    if any(not (Path(folder)/n).is_file() or sha(Path(folder)/n)!=m.get(n) for n in names):
        raise DataError('Cached artifact integrity failure: '+str(folder))
    return True


def store_for_dataset(root,meta):
    root=Path(root).resolve();path=(root/meta['daily_cache']).resolve()
    if not path.is_relative_to(root) or not (path/'market.sqlite').exists():
        raise DataError('New factor/changed-cost preparation needs the saved daily SQLite cache. Restore V21_DATABASE.zip; report ZIP is insufficient.')
    store=Store(path)
    if source_signature(store)!=meta.get('daily_cache_signature'):
        raise DataError('Daily cache changed since pinning. Explicitly prepare/adopt a fresh dataset before recomputing features or labels.')
    return store


def prepare_fixed_features(root,progress=lambda **kw:None,check=lambda:None):
    folder,meta=load_dataset(root);lock=load_pool(root)
    key=digest({'data':meta['dataset_id'],'mapping':lock['mapping'],
        'algorithm':algorithm_hash(('studio_behavior.py','studio_cache.py','lab_features.py','lab_price_quality.py'))})[:24]
    out=Path(root)/'studio'/'feature_cache'/key
    files=('features.csv.gz','context.csv.gz','pools.csv.gz','INFO.json')
    if verified_cache(out,files):
        return out,read_json(out/'INFO.json'),True
    out.mkdir(parents=True,exist_ok=True)
    check();store=store_for_dataset(root,meta)
    cfg=meta['source_config'];cal=pd.DatetimeIndex(read_json(folder/'calendar.json'))
    parent=pd.read_csv(folder/'features.csv.gz',parse_dates=['date'])
    pools=pd.read_csv(folder/'monthly_pools.csv.gz',parse_dates=['date'])
    pools=pools[pools.pool.isin(set(lock['mapping'].values()))].copy()
    union=pools[['date','ticker']].drop_duplicates()
    parent=union.merge(parent,on=['date','ticker'],how='left',validate='one_to_one')
    if parent.date.isna().any():raise DataError('Pinned feature dates missing')
    start,end=cfg['start'],cfg['end']
    def daily(t):
        raw=store.get('raw',t,start,end);adj=store.get('adj',t,start,end)
        if raw.empty or adj.empty:raise DataError('Saved daily input missing: '+t)
        ana,_=analytic_hlc_from_audit(raw,store.quality_details(t))
        return raw_adjusted_frame(raw,adj,cal,ana)
    benchmark=daily('0050.TW');bc=benchmark_context(benchmark)
    chunks=[];tickers=sorted(union.ticker.unique())
    # Recompute the active base formulas once from the pinned daily inputs.
    # Otherwise a formula-code change could silently reuse older base values.
    rebuild_base=True
    for i,t in enumerate(tickers):
        check();f=daily(t);ds=pd.DatetimeIndex(union[union.ticker==t].date)
        z=behavior_features(f).reindex(ds)
        if rebuild_base:
            z=z.join(features(f,benchmark).reindex(ds),how='left')
        z['date']=z.index;z['ticker']=t;chunks.append(z.reset_index(drop=True))
        if i%10==0:progress(phase='PREPARE_FIXED_FACTORS',done=i+1,total=len(tickers),message='One-time local factor preparation; no universe search or network')
    ext=pd.concat(chunks,ignore_index=True)
    replace_cols=[c for c in ext if c in parent and c not in ('date','ticker')]
    parent=parent.drop(columns=replace_cols)
    snap=parent.merge(ext,on=['date','ticker'],how='left',validate='one_to_one')
    available=list(BASE_FACTORS)+list(BOOK_FACTORS)
    if cfg.get('enable_optional_factors'):available+=list(OPTIONAL_FACTORS)
    for c in available:
        if c not in snap:raise DataError('Factor absent: '+c)
        # Stable across strategies: same fixed monthly union, never candidate-dependent.
        snap['rank_'+c]=snap.groupby('date')[c].rank(pct=True,method='average')
    context=[]
    for (pool,date),group in pools.groupby(['pool','date']):
        sf=snap[(snap.date==date)&snap.ticker.isin(group.ticker)]
        known=sf.above_ma60.notna().mean()
        breadth=float(sf.above_ma60.mean()) if known>=.95 else float('nan')
        row=bc.loc[date].to_dict();row.update(date=date,pool=pool,breadth=breadth,breadth_known_fraction=float(known),
              macro_money_supply_available=False,holder_psychology_observed=False)
        row['regime']=regime(row,breadth);context.append(row)
    info={'feature_id':key,'dataset_id':meta['dataset_id'],'pool_lock':lock['lock_id'],
          'factors':available,'daily_symbols_read':len(tickers)+1,'http_requests':0,'pool_searches':0,
          'rank_reference':'UNION_OF_FROZEN_MONTHLY_POOLS_NOT_TODAYS_SURVIVORS',
          'book_features_are_our_proxies_not_original_formulas':True,'macro_model':'NOT_AVAILABLE',
          'recomputed_legacy_base_features':rebuild_base}
    frame_write(out/'features.csv.gz',snap);frame_write(out/'context.csv.gz',pd.DataFrame(context));frame_write(out/'pools.csv.gz',pools)
    atomic_json(out/'INFO.json',info);atomic_json(out/'CACHE.json',{n:sha(out/n) for n in files})
    return out,info,False


LABEL_CFG=('fee_buy','fee_sell','stock_tax','benchmark_tax','slippage','max_gap')

def load_labels(root,feature_dir,cfg,progress=lambda **kw:None,check=lambda:None):
    folder,meta=load_dataset(root)
    if all(float(meta['source_config'][k])==float(getattr(cfg,k)) for k in LABEL_CFG):
        return pd.read_csv(folder/'labels.csv.gz',parse_dates=['date','entry','exit']),{'hit':True,'label_id':meta['files']['labels.csv.gz'],'source':'ORIGINAL_IMMUTABLE_LABELS'}
    key=digest({'dataset':meta['dataset_id'],'pool':sha(Path(feature_dir)/'pools.csv.gz'),
                'costs':{k:getattr(cfg,k) for k in LABEL_CFG},'h':list(cfg.horizons),
                'algorithm':algorithm_hash(('lab_features.py','lab_price_quality.py','studio_cache.py'))})[:24]
    out=Path(root)/'studio'/'label_cache'/key
    if verified_cache(out,('labels.csv.gz',)):
        return pd.read_csv(out/'labels.csv.gz',parse_dates=['date','entry','exit']),{'hit':True,'label_id':key,'source':'COST_REPLAY_CACHE'}
    store=store_for_dataset(root,meta);cal=pd.DatetimeIndex(read_json(folder/'calendar.json'))
    pools=pd.read_csv(Path(feature_dir)/'pools.csv.gz',parse_dates=['date']);pairs=pools[['date','ticker']].drop_duplicates()
    records=[];tickers=sorted(pairs.ticker.unique())+['0050.TW']
    for i,t in enumerate(tickers):
        check();raw=store.get('raw',t,cfg.start,cfg.end);adj=store.get('adj',t,cfg.start,cfg.end)
        ana,_=analytic_hlc_from_audit(raw,store.quality_details(t));f=raw_adjusted_frame(raw,adj,cal,ana)
        ds=pd.DatetimeIndex(sorted(pools.date.unique())) if t=='0050.TW' else pd.DatetimeIndex(sorted(pairs[pairs.ticker==t].date))
        for h in cfg.horizons:
            for row in label_records(f,ds,h,cfg,t=='0050.TW'):row['ticker']=t;records.append(row)
        if i%20==0:progress(phase='COST_REPLAY',done=i+1,total=len(tickers),message='Changed costs: local label replay only')
    labels=pd.DataFrame(records);out.mkdir(parents=True,exist_ok=True);frame_write(out/'labels.csv.gz',labels)
    atomic_json(out/'CACHE.json',{'labels.csv.gz':sha(out/'labels.csv.gz')})
    return labels,{'hit':False,'label_id':key,'source':'LOCAL_DAILY_CACHE_NO_NETWORK'}
