"""Portable data-only checkpoints; SQLite online backup + SHA256 + ZIP validation.

Never deserialize pickle or extract paths supplied by an archive. Import merges
canonical cache rows; existing unmatched symbols are preserved. Secrets and
tracebacks are intentionally excluded. This is data integrity, not authenticity:
only import snapshots from your own trusted repository.
"""
from pathlib import Path, PurePosixPath
import hashlib
import json
import os
import shutil
import sqlite3
import stat
import tempfile
import zipfile
from datetime import datetime, timezone
from lab_io import Store, JobLock, DataError, read_json, atomic_json, sha

SCHEMA='v21-portable-data-v1'
MAX_UNPACKED=2_000_000_000
MAX_FILES=4000
RUN_FILES={'state.json','protocol.json','candidates.json','roster.json','calendar.json',
    'source_scope.json','prepared.json','features.csv.gz','labels.csv.gz',
    'monthly_pools.csv.gz','pool_coverage.csv','feature_coverage.csv',
    'results.json','selected_models.json','REPORT.txt','REPORT.md','LAB_RESULTS.zip',
    'download_failures.json','price_quality_summary.json','data_manifest.json',
    'selection_log.json','job_report.json','PRETAIL_GATES.json','DATA_READY.json',
    'feature_coverage_by_date.csv','collection_reuse.json','price_quality_audit.csv','pool_control_comparison.csv',
    'outer_oos_records.csv','final_tail_records.csv',
    'final_freeze_5.json','final_freeze_20.json','final_freeze_60.json',
    'development_candidates_5.csv','development_candidates_20.csv','development_candidates_60.csv'}
TABLE_COLS={
 'raw':['ticker','date','open','high','low','close','volume','amount'],
 'adj':['ticker','date','open','high','low','close'],
 'downloads':['key','start','end','sha','source','rows','at'],
 'meta':['key','value'],
 'quality_issues':['ticker','date','kind','detail','at'],
}


STUDIO_ROOT_FILES={'active_dataset.json','active_pool.json','latest.json','experiment_ledger.json','confirmation_ledger.json','editor.json'}
STUDIO_FILES={
 'datasets':{'dataset.json','features.csv.gz','labels.csv.gz','monthly_pools.csv.gz','calendar.json','source_scope.json'},
 'feature_cache':{'features.csv.gz','context.csv.gz','pools.csv.gz','INFO.json','CACHE.json'},
 'label_cache':{'labels.csv.gz','CACHE.json'},
 'experiments':{'experiment.json','FROZEN_STRATEGIES.json','REPORT.txt','results.json','state.json',
    'EXPORT_SHA256.json','STRATEGY_RESULTS.zip','development_walkforward.csv','final_records.csv',
    'development_ranking_5.csv','development_ranking_20.csv','development_ranking_60.csv',
    'profile_comparison_5.csv','profile_comparison_20.csv','profile_comparison_60.csv','development_regimes.csv'},
}


V22_ROOT_FILES={'latest.json','latest_pool_study.json','editor.json','confirmation_ledger.json'}
V22_FILES={
 'feature_cache':{'features.csv.gz','pools.csv.gz','market_stage.csv.gz','INFO.json','CACHE.json'},
 'experiments':{'profiles.json','settings.json','results.json','REPORT.txt','REPORT.md','state.json','EXPORT_SHA256.json','V22_TEST_REPORT.zip',
    'candidate_ranking.csv','strategy_by_stage.csv','monthly_decisions.csv','factor_sensitivity.csv','failure_analysis.csv','market_stage.csv','regime_transitions.csv','data_quality.csv','final_confirmation_records.csv'},
 'pool_study':{'POOL_STUDY.csv','POOL_SUMMARY.csv','POOL_REPORT.txt','V22_POOL_REPORT.zip'},
}

V23_REPORT_FILES={'state.json','results.json','run_spec.json','monthly_decisions.csv','daily_nav.csv',
    'candidate_comparison.csv','chronological_folds.csv','data_quality.csv','V23_1_VALIDATION.xlsx','V23_1_VALIDATION.zip','README_RESULTS.txt'}

def v23_allowed(parts):
    import re
    if len(parts)==2:return parts[1] in ('latest.json','active_strategy_release.json','mops_job_state.json')
    if len(parts)==3 and parts[1]=='evidence':return parts[2] in ('revenue.csv.gz','financial.csv.gz','flow.csv.gz')
    if len(parts)==3 and parts[1]=='mops_proxy_cache':
        return parts[2]=='status.json' or bool(re.fullmatch(r'\d{4}-(0[1-9]|1[0-2])-(sii|otc)\.csv\.gz',parts[2]))
    return (len(parts)==4 and parts[1]=='runs' and bool(re.fullmatch('[0-9a-f]{24}',parts[2]))
            and parts[3] in V23_REPORT_FILES)

def v22_allowed(parts):
    import re
    if len(parts)==2:return parts[1] in V22_ROOT_FILES
    if len(parts)==3 and parts[1]=='strategy_cache':return bool(re.fullmatch(r'[0-9a-f]{32}(\.sha)?\.json',parts[2]))
    return (len(parts)==4 and parts[1] in V22_FILES and bool(re.fullmatch('[0-9a-f]{24}',parts[2])) and parts[3] in V22_FILES[parts[1]])

def studio_allowed(parts):
    import re
    if len(parts)==2:return parts[1] in STUDIO_ROOT_FILES
    if len(parts)==3 and parts[1]=='pool_locks':return bool(re.fullmatch(r'[0-9a-f]{24}\.json',parts[2]))
    if len(parts)==3 and parts[1]=='strategy_cache':return bool(re.fullmatch(r'[0-9a-f]{32}(\.sha)?\.json',parts[2]))
    return (len(parts)==4 and parts[1] in STUDIO_FILES and bool(re.fullmatch('[0-9a-f]{24}',parts[2]))
            and parts[3] in STUDIO_FILES[parts[1]])


def allowed(name):
    pp=PurePosixPath(name)
    if pp.is_absolute() or '\\' in name or '..' in pp.parts or ':' in name:return False
    if name in ('cache/market.sqlite','latest.json','remote_context.json','SNAPSHOT.json'):return True
    if pp.parts and pp.parts[0]=='studio':return studio_allowed(pp.parts)
    if pp.parts and pp.parts[0]=='v22':return v22_allowed(pp.parts)
    if pp.parts and pp.parts[0]=='v23':return v23_allowed(pp.parts)
    return (len(pp.parts)==3 and pp.parts[0]=='runs' and
            len(pp.parts[1])==20 and all(c in '0123456789abcdef' for c in pp.parts[1])
            and pp.parts[2] in RUN_FILES)


def validate_db(path):
    with sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True) as c:
        c.execute('PRAGMA trusted_schema=OFF')
        objects=c.execute("SELECT type,name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'").fetchall()
        if any(kind!='table' or name not in TABLE_COLS for kind,name in objects):
            raise DataError('Snapshot contains unexpected SQLite objects')
        names={name for _,name in objects}
        if not {'raw','adj','downloads','meta'}.issubset(names):raise DataError('Incomplete checkpoint schema')
        for name in names:
            cols=[r[1] for r in c.execute('PRAGMA table_info('+name+')')]
            if cols!=TABLE_COLS[name]:raise DataError('Checkpoint column mismatch: '+name)
        if c.execute('PRAGMA quick_check').fetchone()[0]!='ok':raise DataError('SQLite integrity check failed')


def export_snapshot(root, output):
    root=Path(root);output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    if not (root/'cache'/'market.sqlite').is_file():raise DataError('No auto-mode cache to export yet')
    with JobLock(root/'worker.lock'):
        with tempfile.TemporaryDirectory(dir=output.parent) as td:
            stage=Path(td);(stage/'cache').mkdir()
            src=sqlite3.connect((root/'cache'/'market.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
            dst=sqlite3.connect(stage/'cache'/'market.sqlite')
            try:
                src.backup(dst,pages=2048,sleep=.01)
                dst.execute('PRAGMA journal_mode=DELETE')
            finally:src.close();dst.close()
            validate_db(stage/'cache'/'market.sqlite')
            for name in ('latest.json','remote_context.json'):
                if (root/name).is_file():shutil.copy2(root/name,stage/name)
            latest=read_json(root/'latest.json')
            # Only one active research run; archived originals remain on host.
            rid=latest.get('job_id','')
            for name in RUN_FILES:
                rel=f'runs/{rid}/{name}'
                if allowed(rel) and (root/rel).is_file():
                    (stage/rel).parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(root/rel,stage/rel)
            # Carry fixed-pool registrations, released-tail ledger and safe cache
            # artifacts across hosts. Never export credentials or arbitrary files.
            if (root/'studio').exists():
                for f in (root/'studio').rglob('*'):
                    if not f.is_file() or f.is_symlink():continue
                    rel=f.relative_to(root).as_posix()
                    if allowed(rel):
                        target=stage/rel;target.parent.mkdir(parents=True,exist_ok=True)
                        shutil.copy2(f,target)
            if (root/'v22').exists():
                for f in (root/'v22').rglob('*'):
                    if not f.is_file() or f.is_symlink():continue
                    rel=f.relative_to(root).as_posix()
                    if allowed(rel):
                        target=stage/rel;target.parent.mkdir(parents=True,exist_ok=True)
                        shutil.copy2(f,target)
            if (root/'v23').exists():
                for f in (root/'v23').rglob('*'):
                    if not f.is_file() or f.is_symlink():continue
                    rel=f.relative_to(root).as_posix()
                    if allowed(rel):
                        dst=stage/rel;dst.parent.mkdir(parents=True,exist_ok=True)
                        shutil.copy2(f,dst)
            if sum(1 for f in stage.rglob('*') if f.is_file())>=MAX_FILES:
                raise DataError('Snapshot cache file budget exceeded; archive old studio experiments outside the workspace')
            hashes={x.relative_to(stage).as_posix():sha(x) for x in stage.rglob('*') if x.is_file()}
            manifest={'schema':SCHEMA,'created_at':datetime.now(timezone.utc).isoformat(),
                      'files':hashes,'secrets_exported':False,'data_only':True}
            atomic_json(stage/'SNAPSHOT.json',manifest)
            temp=output.with_name(output.name+'.tmp')
            with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as z:
                for f in sorted(stage.rglob('*')):
                    if f.is_file():z.write(f,f.relative_to(stage).as_posix())
            with zipfile.ZipFile(temp) as z:
                if z.testzip() is not None:raise DataError('Snapshot CRC failure')
            os.replace(temp,output)
            return {'file':str(output),'bytes':output.stat().st_size,'sha256':sha(output),
                    'schema':SCHEMA,'files':len(hashes)+1}


def _stage_snapshot(path,dest):
    dest=Path(dest)
    with zipfile.ZipFile(path) as z:
        infos=z.infolist();names=[i.filename for i in infos]
        if len(infos)>MAX_FILES or len(set(names))!=len(names):raise DataError('Too many or duplicate checkpoint members')
        if sum(i.file_size for i in infos)>MAX_UNPACKED:raise DataError('Checkpoint too large')
        for i in infos:
            if not allowed(i.filename) or stat.S_ISLNK(i.external_attr>>16):raise DataError('Unsafe checkpoint member: '+i.filename)
        if 'SNAPSHOT.json' not in names or z.getinfo('SNAPSHOT.json').file_size>2_000_000:
            raise DataError('Checkpoint manifest absent or oversized')
        meta=json.loads(z.read('SNAPSHOT.json'))
        if meta.get('schema')!=SCHEMA:raise DataError('Not a V21 portable checkpoint')
        if set(meta.get('files',{}))!=set(names)-{'SNAPSHOT.json'}:raise DataError('Checkpoint manifest member mismatch')
        if 'cache/market.sqlite' not in names:raise DataError('Database missing')
        for i in infos:
            dst=dest/i.filename;dst.parent.mkdir(parents=True,exist_ok=True)
            # Copy data only after allowlist validation; never extractall().
            with z.open(i) as src, open(dst,'wb') as target:shutil.copyfileobj(src,target,1024*1024)
            if i.filename!='SNAPSHOT.json' and sha(dst)!=meta['files'][i.filename]:
                raise DataError('Checkpoint checksum mismatch: '+i.filename)
    validate_db(dest/'cache'/'market.sqlite')
    return meta


def import_snapshot(path,root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with JobLock(root/'worker.lock'), tempfile.TemporaryDirectory(dir=root) as td:
        stage=Path(td);meta=_stage_snapshot(path,stage)
        # Validate configuration and localize paths before touching user data.
        latest=read_json(stage/'latest.json')
        rid=latest.get('job_id','')
        if latest:
            if not allowed(f'runs/{rid}/state.json'):raise DataError('Invalid checkpoint job identity')
            from lab_config import Config
            cfg=Config(**latest.get('config',{})).resolved()
            if cfg.source_mode!='auto_current_roster':raise DataError('This checkpoint format supports the auto-mode cache only')
        store=Store(root/'cache')
        src=sqlite3.connect((stage/'cache'/'market.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
        merged=0
        try:
            imported={r[0]:r for r in src.execute('SELECT * FROM downloads')}
            with store.connect() as target:
                existing={r[0]:r for r in target.execute('SELECT * FROM downloads')}
                for key,row in imported.items():
                    table,ticker=key.split(':',1)
                    if table not in ('raw','adj'):raise DataError('Unknown data table in checkpoint')
                    old=existing.get(key)
                    # Prefer the newer per-symbol source query; preserve other symbols.
                    newer=not old or float(row[6])>=float(old[6])
                    mode='REPLACE' if newer else 'IGNORE'
                    cols=TABLE_COLS[table];marks=','.join('?' for _ in cols)
                    cur=src.execute('SELECT * FROM '+table+' WHERE ticker=?',(ticker,))
                    while True:
                        batch=cur.fetchmany(10000)
                        if not batch:break
                        target.executemany('INSERT OR '+mode+' INTO '+table+' VALUES('+marks+')',batch)
                        merged+=len(batch)
                    if newer:
                        # Imported metadata describes its declared interval only.
                        # Never merge disconnected time ranges into claimed coverage.
                        target.execute('INSERT OR REPLACE INTO downloads VALUES(?,?,?,?,?,?,?)',row)
                if src.execute("SELECT name FROM sqlite_master WHERE name='quality_issues'").fetchone():
                    for row in src.execute('SELECT * FROM quality_issues'):
                        target.execute('INSERT OR REPLACE INTO quality_issues VALUES(?,?,?,?,?)',row)
                # Only provider quota/provenance keys, never workspace paths or secrets.
                for key in ('finmind_requests','raw_download_failures','finmind_retry_until'):
                    r=src.execute('SELECT value FROM meta WHERE key=?',(key,)).fetchone()
                    if r and key=='finmind_requests':
                        vals=json.loads(r[0]);oldrow=target.execute('SELECT value FROM meta WHERE key=?',(key,)).fetchone();old=json.loads(oldrow[0]) if oldrow else []
                        target.execute('INSERT OR REPLACE INTO meta VALUES(?,?)',(key,json.dumps(sorted(set(old+vals)))))
                    elif r and key=='finmind_retry_until':
                        value=float(json.loads(r[0]))
                        if not 0<=value<1e12:raise DataError('Invalid quota resume timestamp')
                        target.execute('INSERT OR REPLACE INTO meta VALUES(?,?)',(key,json.dumps(value)))
                    elif r:
                        failures=json.loads(r[0])
                        failures={k:v for k,v in failures.items() if not target.execute('SELECT 1 FROM downloads WHERE key=?',('raw:'+k,)).fetchone()}
                        target.execute('INSERT OR REPLACE INTO meta VALUES(?,?)',(key,json.dumps(failures)))
        finally:src.close()
        for folder in (stage/'runs').glob('*') if (stage/'runs').exists() else []:
            dest=root/'runs'/folder.name;dest.mkdir(parents=True,exist_ok=True)
            for f in folder.iterdir():shutil.copy2(f,dest/f.name)
            state=read_json(dest/'state.json')
            if state:
                state['run']=str(dest)
                if state.get('status') in ('RUNNING','QUEUED','RETRY_WAIT'):
                    state['status']='PAUSED';state['message']='Imported checkpoint; continue explicitly on this host'
                atomic_json(dest/'state.json',state)
        if latest:
            latest['run']=str(root/'runs'/rid);latest['input_dir']=None
            atomic_json(root/'latest.json',latest)
        if (stage/'studio').exists():
            # Immutable identities may be imported; never overwrite divergent
            # confirmation decisions from this host with an older snapshot.
            for f in (stage/'studio').rglob('*'):
                if not f.is_file():continue
                rel=f.relative_to(stage);target=root/rel;target.parent.mkdir(parents=True,exist_ok=True)
                if rel.as_posix()=='studio/confirmation_ledger.json':
                    old=read_json(target);incoming=read_json(f)
                    for era,record in incoming.items():
                        if era not in old:old[era]=record
                    atomic_json(target,old)
                elif rel.as_posix()=='studio/experiment_ledger.json':
                    rows={r['experiment_id']:r for r in read_json(f,[])}
                    rows.update({r['experiment_id']:r for r in read_json(target,[])})
                    atomic_json(target,list(rows.values()))
                elif not target.exists():shutil.copy2(f,target)
            sl=read_json(root/'studio'/'latest.json')
            if sl.get('experiment_id'):
                eid=sl['experiment_id']
                if not allowed(f'studio/experiments/{eid}/state.json'):raise DataError('Invalid studio experiment identity')
                dest=root/'studio'/'experiments'/eid;sl['run']=str(dest);atomic_json(root/'studio'/'latest.json',sl)
                ss=read_json(dest/'state.json')
                if ss.get('status') in ('RUNNING','QUEUED'):
                    ss['status']='PAUSED';ss['message']='Restored studio checkpoint; continue explicitly'
                    atomic_json(dest/'state.json',ss)
        if (stage/'v22').exists():
            for f in (stage/'v22').rglob('*'):
                if not f.is_file():continue
                rel=f.relative_to(stage);target=root/rel;target.parent.mkdir(parents=True,exist_ok=True)
                if rel.as_posix()=='v22/confirmation_ledger.json':
                    old=read_json(target);incoming=read_json(f)
                    for era,record in incoming.items():
                        if era not in old:old[era]=record
                    atomic_json(target,old)
                elif not target.exists():shutil.copy2(f,target)
            vl=read_json(root/'v22'/'latest.json')
            if vl.get('experiment_id'):
                eid=vl['experiment_id']
                if not allowed(f'v22/experiments/{eid}/state.json'):raise DataError('Invalid V22 experiment identity')
                dest=root/'v22'/'experiments'/eid;vl['run']=str(dest);atomic_json(root/'v22'/'latest.json',vl)
                ss=read_json(dest/'state.json')
                if ss.get('status') in ('RUNNING','QUEUED'):
                    ss['status']='PAUSED';ss['message']='Restored V22 checkpoint; continue explicitly';atomic_json(dest/'state.json',ss)
        if (stage/'v23').exists():
            from v231_data import persist_events
            import pandas as pd
            for kind in ('revenue','financial','flow'):
                p=stage/'v23'/'evidence'/(kind+'.csv.gz')
                if p.is_file():
                    persist_events(root,pd.read_csv(p),kind)
            for f in (stage/'v23').rglob('*'):
                if not f.is_file():continue
                rel=f.relative_to(stage)
                if rel.parts[1]=='evidence' or rel.as_posix()=='v23/latest.json':continue
                target_file=root/rel;target_file.parent.mkdir(parents=True,exist_ok=True)
                if not target_file.exists():shutil.copy2(f,target_file)
            vl=read_json(stage/'v23'/'latest.json')
            v23_rid=vl.get('run_id')
            if v23_rid:
                if not allowed(f'v23/runs/{v23_rid}/state.json'):raise DataError('V23 checkpoint identity malformed')
                if not (root/'v23'/'latest.json').is_file():
                    atomic_json(root/'v23'/'latest.json',{'run':str(root/'v23'/'runs'/v23_rid),'run_id':v23_rid})
            # A restored RUNNING task is not alive; user must explicitly resume.
            for f in (root/'v23'/'runs').glob('*/state.json'):
                q=read_json(f)
                if q.get('status')=='RUNNING':
                    q['status']='PAUSED';q['message']='Snapshot restored: press Start/Resume';atomic_json(f,q)
        if (stage/'remote_context.json').exists():shutil.copy2(stage/'remote_context.json',root/'remote_context.json')
        return {'imported_rows':merged,'job_id':rid,'source_created_at':meta['created_at'],
                'existing_unmatched_symbols_preserved':True}
