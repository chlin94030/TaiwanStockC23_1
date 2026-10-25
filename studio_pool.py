"""Immutable prepared-data references and explicit universe-rule locks. No network.
A lock fixes an ADV rule, NOT today's stock names throughout past history.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re
import shutil
import pandas as pd
from lab_io import DataError, atomic_json, read_json, sha, Store, JobLock

PREPARED = ('features.csv.gz','labels.csv.gz','monthly_pools.csv.gz','calendar.json','source_scope.json')
PRICE_CONFIG = ('start','end','horizons','fee_buy','fee_sell','stock_tax','benchmark_tax','slippage','max_gap')
POOL_CONFIG = ('lookback','recent_amount_floor','active_ratio60','requested_scope','include_market_cap_pools')


def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,default=str,separators=(',',':')).encode()).hexdigest()


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def verify_prepared(run):
    run=Path(run);m=read_json(run/'prepared.json')
    if not m or any(not (run/n).is_file() or sha(run/n)!=m.get(n) for n in PREPARED):
        raise DataError('Prepared snapshot missing or checksum mismatch; restore your V21_DATABASE.zip, not LAB_RESULTS.zip')
    p=read_json(run/'protocol.json')
    if not p.get('config'):raise DataError('Prepared snapshot has no price/source configuration')
    return m,p


def discover_prepared(root):
    rows=[]
    for p in (Path(root)/'runs').glob('*/prepared.json'):
        run=p.parent;proto=read_json(run/'protocol.json')
        if not all((run/n).is_file() for n in PREPARED):continue
        cfg=proto.get('config',{})
        rows.append({'run':str(run),'name':run.name,'version':proto.get('version'),
                     'start':cfg.get('start'),'end':cfg.get('end'),'mtime':p.stat().st_mtime})
    return sorted(rows,key=lambda r:(r['end'] or '',r['mtime']),reverse=True)


def source_signature(store):
    records=store.manifest()
    return digest({'downloads':[{k:r.get(k) for k in ('key','start','end','sha','source','rows')}
                                for r in records], 'imported_sha':store.meta('imported_sha')})


def adopt_dataset(root,source_run):
    root=Path(root).resolve();run=Path(source_run).resolve()
    if not run.is_relative_to(root/'runs'):raise DataError('Source must be a local prepared run under this workspace')
    with JobLock(root/'worker.lock'):
        hashes,protocol=verify_prepared(run)
        cfg=protocol['config'];meta={'files':{n:hashes[n] for n in PREPARED},
              'source_config':cfg,'source_run':run.name,'source_version':protocol.get('version')}
        key=digest({'files':meta['files'],'data_config':{k:cfg.get(k) for k in PRICE_CONFIG+POOL_CONFIG}})[:24]
        dest=root/'studio'/'datasets'/key;dest.mkdir(parents=True,exist_ok=True)
        cache=run/'import_cache' if cfg.get('source_mode')=='import' else root/'cache'
        db=cache/'market.sqlite'
        signature=None
        if db.is_file():
            store=Store(cache)
            recorded=read_json(run/'data_manifest.json').get('downloads',[])
            current={r['key']:r for r in store.manifest()}
            if recorded:
                for r in recorded:
                    if any(current.get(r['key'],{}).get(k)!=r.get(k) for k in ('sha','start','end')):
                        raise DataError('Daily cache has changed since prepared data. Prepare a new snapshot explicitly; do not mix vintages.')
            signature=source_signature(store)
        for name in PREPARED:
            out=dest/name
            if not out.exists():shutil.copy2(run/name,out)
            if sha(out)!=hashes[name]:raise DataError('Pinned snapshot content changed')
        meta.update(schema=1,dataset_id=key,pinned_at=timestamp(),daily_cache=str(cache.relative_to(root)),
                    daily_cache_signature=signature,has_daily_cache=bool(signature),
                    scope=read_json(run/'source_scope.json'),not_new_holdout=True)
        old=read_json(dest/'dataset.json')
        if old:meta=old
        else:atomic_json(dest/'dataset.json',meta)
        atomic_json(root/'studio'/'active_dataset.json',{'dataset_id':key})
        return meta


def load_dataset(root,key=None):
    root=Path(root);key=key or read_json(root/'studio'/'active_dataset.json').get('dataset_id')
    if not key or not re.fullmatch('[0-9a-f]{24}',key):raise DataError('No prepared dataset selected; adopt an existing prepared run first')
    folder=root/'studio'/'datasets'/key;meta=read_json(folder/'dataset.json')
    if meta.get('dataset_id')!=key:raise DataError('Dataset identity mismatch')
    expected=digest({'files':meta.get('files',{}),'data_config':{k:meta.get('source_config',{}).get(k) for k in PRICE_CONFIG+POOL_CONFIG}})[:24]
    if expected!=key:raise DataError('Pinned data/cost definition modified; adopt an explicit new snapshot')
    if any(not (folder/n).is_file() or sha(folder/n)!=meta.get('files',{}).get(n) for n in PREPARED):
        raise DataError('Pinned dataset checksum failed')
    return folder,meta


def lock_pool(root,mapping,reason='USER_FIXED_RESEARCH_CHOICE',evidence=None):
    """Explicit registration; choosing a pool is not proof that it is optimal."""
    root=Path(root)
    with JobLock(root/'worker.lock'):
        folder,meta=load_dataset(root)
        pools=pd.read_csv(folder/'monthly_pools.csv.gz',usecols=['date','pool','ticker','declared_size'])
        available=set(pools.pool)
        mapping={str(int(h)):str(p) for h,p in mapping.items()}
        if set(mapping)!=set(str(h) for h in meta['source_config']['horizons']):
            raise DataError('Register one fixed pool for every prepared horizon')
        if not set(mapping.values()).issubset(available):raise DataError('Requested pool not present in pinned snapshot')
        for p in set(mapping.values()):
            sizes=pools[pools.pool==p].groupby('date').agg(n=('ticker','nunique'),expected=('declared_size','first'))
            if not sizes.n.eq(sizes.expected).all():raise DataError('Historical monthly pool has silently shrunk')
        obj={'schema':1,'dataset_id':meta['dataset_id'],'mapping':mapping,
             'rules':{k:meta['source_config'].get(k) for k in POOL_CONFIG},
             'monthly_membership_sha':meta['files']['monthly_pools.csv.gz'],
             'reason':reason,'evidence':evidence or {},'optimality_proven':False,
             'members_change_monthly_using_T_only_history':True,
             'manual_or_development_lock_not_independent_outer_test':True}
        key=digest(obj)[:24];obj.update(lock_id=key,registered_at=timestamp())
        p=root/'studio'/'pool_locks'/(key+'.json')
        if not p.exists():atomic_json(p,obj)
        else:obj=read_json(p)
        atomic_json(root/'studio'/'active_pool.json',{'lock_id':key})
        return obj


def load_pool(root):
    root=Path(root);key=read_json(root/'studio'/'active_pool.json').get('lock_id')
    if not key or not re.fullmatch('[0-9a-f]{24}',key):raise DataError('Freeze a universe rule once before strategy testing')
    obj=read_json(root/'studio'/'pool_locks'/(key+'.json'))
    payload={k:v for k,v in obj.items() if k not in ('lock_id','registered_at')}
    if obj.get('lock_id')!=key or digest(payload)[:24]!=key:raise DataError('Universe lock modified; create an explicit new revision')
    _,meta=load_dataset(root)
    if obj.get('dataset_id')!=meta['dataset_id']:raise DataError('New dataset selected: explicitly rebind the same pool rule before testing')
    return obj


def report_pool_suggestions(root):
    """Only reads training-selected pool fields, never picks the best final tail."""
    folder,meta=load_dataset(root)
    source=Path(root)/'runs'/meta['source_run']
    results=read_json(source/'results.json');suggestions={}
    for h,r in results.get('models',{}).items():
        choice=r.get('freeze',{}).get('choice',{})
        p=choice.get('pool')
        if p:
            suggestions[h]={'pool':p,'source':'TRAINING_SELECTED_NOT_PROVEN_OPTIMAL',
               'training_end':r.get('freeze',{}).get('train_end'),
               'equivalent_pools':r.get('equivalent_training_pool_outcomes',[])}
    return suggestions
