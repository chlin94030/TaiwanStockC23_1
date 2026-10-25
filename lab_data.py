"""Two-pass data pipeline: whole-roster actual amount -> monthly union -> price factors."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import zipfile
import numpy as np
import pandas as pd
from lab_io import Store,DataError,atomic_json,read_json,frame_write,sha
from lab_sources import Client
from lab_bulk import FinLabBulk, finlab_ready
from lab_yahoo_bulk import YahooFastBulk
from lab_official_bulk import OfficialDailyBulk
from lab_universe import signal_calendar,turnover_snapshots,make_pools
from lab_features import raw_adjusted_frame,features,attach_evidence,percentile_reference,return_record,label_records

BENCHMARK='0050.TW'
ALLOWED_IMPORT={'prices.csv','prices.csv.gz','calendar.csv','catalog.json','manifest.json',
                'membership.csv','market_caps.csv','evidence.csv'}

def extract_dataset(path,destination):
    dest=Path(destination);dest.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path) as z:
        infos=z.infolist()
        if sum(i.file_size for i in infos)>1_000_000_000:raise DataError('Dataset expands beyond 1GB')
        for i in infos:
            if i.filename not in ALLOWED_IMPORT or i.is_dir():raise DataError('Unexpected input archive member: '+i.filename)
            if (i.external_attr>>16)&0o170000==0o120000:raise DataError('Symlinks forbidden')
        if len({i.filename for i in infos})!=len(infos):raise DataError('Duplicate archive names')
        if z.testzip():raise DataError('Input ZIP CRC failed')
        for i in infos:
            with z.open(i) as src,open(dest/i.filename,'wb') as dst:shutil.copyfileobj(src,dst)
    return dest


def import_prices(directory,store,cfg,progress,check):
    root=Path(directory)
    manifest=read_json(root/'manifest.json')
    cat=read_json(root/'catalog.json',[])
    if not manifest or not cat:raise DataError('Import requires manifest.json and catalog.json')
    if manifest.get('amount_unit')!='TWD' or manifest.get('volume_unit')!='SHARES':raise DataError('Declare actual TWD amount and share volume units')
    if manifest.get('price_basis')!='ADJUSTED_PROVIDER':raise DataError('Comparable adjusted OHLC required')
    if manifest.get('amount_kind')!='ACTUAL_TRADED_VALUE':raise DataError('Estimated close x volume is not annual actual amount')
    if not manifest.get('source'):raise DataError('Input source required')
    tickers=[r['ticker'] for r in cat]
    if len(set(tickers))!=len(tickers):raise DataError('Duplicate catalog')
    if any(not re.fullmatch(r'[1-9][0-9]{3}\.(TW|TWO)',s) for s in tickers):raise DataError('Catalog must contain common-equity symbols, not ETFs/warrants')
    synthetic=manifest.get('data_kind')=='SYNTHETIC'
    if not synthetic and len(cat)<cfg.min_current_roster:raise DataError('Partial catalog cannot represent the requested all-market universe')
    if cfg.requested_scope=='HISTORICAL_MEMBERSHIP':
        if not (root/'membership.csv').exists():raise DataError('Historical membership.csv absent')
        if manifest.get('membership_scope')!='HISTORICAL_MEMBERSHIP':raise DataError('Historical scope not declared')
    c=pd.read_csv(root/'calendar.csv')
    calendar=sorted(pd.to_datetime(c['date']).dt.strftime('%Y-%m-%d').unique())
    calendar=[d for d in calendar if cfg.start<=d<=cfg.end]
    path=root/'prices.csv.gz'
    if not path.exists():path=root/'prices.csv'
    if not path.exists():raise DataError('prices.csv(.gz) absent; a v19_results report is not market-price input')
    fingerprint=sha(path)
    if store.meta('imported_sha')!=fingerprint:
        if store.meta('import_in_progress_sha')!=fingerprint:
            with store.connect() as conn:
                for t in ('raw','adj','downloads'):conn.execute('DELETE FROM '+t)
            store.set_meta('import_in_progress_sha',fingerprint)
            store.set_meta('import_rows_done',0)
        done=int(store.meta('import_rows_done',0))
        rows=0
        required=['ticker','date']+['raw_'+c for c in ('open','high','low','close')]+['volume','amount']+['adj_'+c for c in ('open','high','low','close')]
        for chunk in pd.read_csv(path,chunksize=100000,dtype={'ticker':str}):
            check()
            rows+=len(chunk)
            if rows<=done:continue
            if not set(required).issubset(chunk):raise DataError('Normalized input fields missing: '+str(set(required)-set(chunk)))
            if not chunk.ticker.isin(tickers+[BENCHMARK]).all():raise DataError('Input symbol absent from catalog')
            chunk['date']=pd.to_datetime(chunk.date,errors='raise').dt.strftime('%Y-%m-%d')
            if chunk[['ticker','date']].duplicated().any():raise DataError('Duplicate input (ticker,date)')
            for col in required[2:]:chunk[col]=pd.to_numeric(chunk[col],errors='raise')
            if (chunk[['volume','amount']].dropna()<0).any().any():raise DataError('Negative volume/amount')
            raw_cols=['ticker','date','raw_open','raw_high','raw_low','raw_close','volume','amount']
            adj_cols=['ticker','date','adj_open','adj_high','adj_low','adj_close']
            # One transaction per 100k-row chunk, not two connections per stock.
            # Primary keys detect duplicates across chunks. The resume marker is
            # committed atomically with both raw and adjusted records.
            raw_vals=[tuple(None if pd.isna(v) else v for v in r) for r in chunk[raw_cols].itertuples(index=False,name=None)]
            adj_vals=[tuple(None if pd.isna(v) else v for v in r) for r in chunk[adj_cols].itertuples(index=False,name=None)]
            try:
                with store.connect() as conn:
                    conn.executemany('INSERT INTO raw VALUES(?,?,?,?,?,?,?,?)',raw_vals)
                    conn.executemany('INSERT INTO adj VALUES(?,?,?,?,?,?)',adj_vals)
                    conn.execute('INSERT OR REPLACE INTO meta VALUES(?,?)',('import_rows_done',str(rows)))
            except Exception as e:raise DataError('Input transaction failed; possible duplicate (ticker,date). Completed chunks are retained.') from e
            progress(message='Importing canonical rows',done=rows,total=None)
        store.set_meta('imported_sha',fingerprint)
    scope='SYNTHETIC' if synthetic else ('USER_SUPPLIED_HISTORICAL_MEMBERSHIP' if cfg.requested_scope=='HISTORICAL_MEMBERSHIP' else 'CURRENT_SNAPSHOT_DIAGNOSTIC')
    return cat,calendar,{'scope':scope,'data_kind':manifest.get('data_kind','PROVIDER_HISTORY'),
            'manifest':manifest,'import_file_sha':fingerprint,'source_independently_audited':False}


def build_dataset(store,cfg,run,progress,check,token=None,input_dir=None):
    run=Path(run);run.mkdir(parents=True,exist_ok=True)
    client=bulk=None;turnover=None
    if cfg.source_mode=='import':
        if not input_dir:raise DataError('An input directory is required')
        cat,calendar,source=import_prices(input_dir,store,cfg,progress,check)
    else:
        client=Client(store,cfg,progress,check,token)
        cat=read_json(run/'roster.json',[])
        if not cat:
            progress(phase='ROSTER',message='讀取臺灣證交所與櫃買中心現行公司名冊',done=0,total=2)
            cat=client.catalog();atomic_json(run/'roster.json',cat)
        ready=finlab_ready()
        if cfg.history_provider=='FINLAB_BULK':
            if not ready:raise DataError('FINLAB_BULK requested but package/three server credentials are not available')
            bulk=FinLabBulk(cfg,progress,check);provider='FINLAB_BULK'
        elif cfg.history_provider=='YAHOO_FAST':
            bulk=YahooFastBulk(cfg,progress,check);provider='YAHOO_FAST'
        elif cfg.history_provider=='OFFICIAL_BULK':
            bulk=OfficialDailyBulk(cfg,progress,check).bind_cache(store.root);provider='OFFICIAL_BULK'
        elif cfg.history_provider in ('AUTO','AUTO_FAST'):
            # Fast-development default: licensed exact whole-market FinLab first;
            # otherwise use Yahoo multi-ticker OHLCV proxy instead of ~2,000
            # serial FinMind requests.  The proxy is disclosed in source_scope.
            bulk=FinLabBulk(cfg,progress,check) if ready else YahooFastBulk(cfg,progress,check)
            provider='FINLAB_BULK' if ready else 'YAHOO_FAST'
        else:
            provider='FINMIND'
        progress(phase='CALENDAR',message=f'名冊 {len(cat)} 檔，確認交易日與起迄期間｜provider={provider}',done=0,total=1,roster_count=len(cat))
        calendar=read_json(run/'calendar.json',[]);bulk_fallback_reason=None
        if not calendar:
            if bulk:
                try:calendar=bulk.calendar()
                except DataError as exc:
                    if cfg.history_provider not in ('AUTO','AUTO_FAST'):raise
                    if provider=='FINLAB_BULK':
                        bulk_fallback_reason='FinLab -> Yahoo fast: '+str(exc)[:200]
                        bulk=YahooFastBulk(cfg,progress,check);provider='YAHOO_FAST';calendar=bulk.calendar()
                    else:
                        raise DataError('Yahoo快速批次來源不可用；AUTO_FAST不會自動退回7小時FinMind逐檔模式。可重試Yahoo或明確選FINMIND。') from exc
            else:calendar=client.calendar()
            calendar=sorted({str(x)[:10] for x in calendar if cfg.start<=str(x)[:10]<=cfg.end})
            atomic_json(run/'calendar.json',calendar)
        source={'scope':'CURRENT_SNAPSHOT_DIAGNOSTIC','data_kind':'PROVIDER_HISTORY',
                'whole_current_roster_count':len(cat),'historical_membership_verified':False,
                'history_provider':provider,'bulk_fallback_reason':bulk_fallback_reason,
                'amount_source':('FinLab price:成交金額 whole-market table' if provider=='FINLAB_BULK' else
                                 ('TWSE/TPEx official whole-market daily traded amount' if provider=='OFFICIAL_BULK' else
                                  ('Yahoo proxy ((High+Low+Close)/3)*Volume' if provider=='YAHOO_FAST' else 'FinMind TaiwanStockPrice Trading_money'))),
                'amount_is_exchange_reported':provider in ('FINLAB_BULK','OFFICIAL_BULK','FINMIND'),
                'development_turnover_proxy':provider=='YAHOO_FAST',
                'adjusted_price_source':('FinLab etl:adj_close ratio OHLC' if provider=='FINLAB_BULK' else 'Yahoo via yfinance'),
                'source_independently_audited':False}
    source['frozen_collection_context']=read_json(run/'collection_reuse.json')
    atomic_json(run/'source_scope.json',source);atomic_json(run/'roster.json',cat);atomic_json(run/'calendar.json',calendar)
    cal=pd.DatetimeIndex(calendar)
    signals=signal_calendar(cal,cfg)
    minimum=cfg.min_train_months+cfg.min_outer_folds*cfg.outer_test_months+cfg.holdout_months+4

    if cfg.source_mode!='import' and bulk:
        # Whole-market turnover ranks are computed directly from one wide actual-
        # amount table. Do not first write ~4 million full-market daily rows.
        try:
            turnover,bulk_meta=bulk.turnover(signals,cat,store.root)
            source['bulk_turnover']=bulk_meta
            progress(phase='POOL',done=len(cat),total=len(cat),message='Whole-market turnover checkpoint ready; no per-symbol amount requests used')
        except DataError as exc:
            if cfg.history_provider not in ('AUTO','AUTO_FAST'):raise
            if provider=='FINLAB_BULK':
                source['bulk_fallback_reason']='FinLab -> Yahoo fast: '+str(exc)[:200]
                bulk=YahooFastBulk(cfg,progress,check);provider='YAHOO_FAST';source['history_provider']='YAHOO_FAST'
                source['amount_source']='Yahoo proxy ((High+Low+Close)/3)*Volume';source['amount_is_exchange_reported']=False;source['development_turnover_proxy']=True;source['adjusted_price_source']='Yahoo via yfinance'
                try:
                    turnover,bulk_meta=bulk.turnover(signals,cat,store.root);source['bulk_turnover']=bulk_meta
                    exc=None
                except DataError as yahoo_exc:
                    exc=yahoo_exc
            if provider=='YAHOO_FAST' and turnover is None:
                # V22.2: a non-empty Yahoo bulk result can still be historically
                # sparse (rate-limit/partial-download behaviour).  AUTO_FAST must
                # not fail after a four-minute bootstrap and must never fall back
                # to the 7+ hour serial FinMind path.  Switch to resumable, exact
                # TWSE/TPEx whole-market-by-date data instead.
                source['bulk_fallback_reason']='Yahoo fast -> official bulk: '+str(exc)[:240]
                progress(phase='OFFICIAL_FALLBACK',done=0,total=1,
                         message='Yahoo歷史完整度不足；自動改用TWSE/TPEx整市場精確回補，不會改走7小時FinMind逐檔模式')
                bulk=OfficialDailyBulk(cfg,progress,check).bind_cache(store.root);provider='OFFICIAL_BULK'
                source['history_provider']='OFFICIAL_BULK'
                source['amount_source']='TWSE/TPEx official whole-market daily traded amount'
                source['amount_is_exchange_reported']=True
                source['development_turnover_proxy']=False
                source['adjusted_price_source']='Yahoo via yfinance (adjusted OHLC only)'
                turnover,bulk_meta=bulk.turnover(signals,cat,store.root);source['bulk_turnover']=bulk_meta
    if turnover is None:
        if cfg.source_mode!='import':
            requested=cat+[{'ticker':BENCHMARK}]
            for i,row in enumerate(requested):
                check();t=row['ticker']
                progress(phase='AMOUNT',done=i,total=len(requested),current_symbol=t,
                         message=f'全市場歷史成交金額：正在檢查／取得 {t} （{i}/{len(requested)}）',roster_count=len(cat),
                         finmind_safe_limit=client.finmind_limit() if client and getattr(client,'token','') and hasattr(client,'finmind_limit') else 260)
                cached=store.cached('raw:'+t,cfg.start,cfg.end)
                if not cached:
                    covered=store.covered_range('raw:'+t);start_date=cfg.start
                    if covered and covered[0]<=cfg.start and covered[1]<cfg.end:
                        start_date=max(cfg.start,str((pd.Timestamp(covered[1])-pd.Timedelta(days=7)).date()))
                    try:
                        f=client.raw(t,start=start_date)
                        if f.empty:raise DataError('No actual amount rows for requested symbol '+t)
                        store.put('raw',t,f,cfg.start,cfg.end,'FinMind_TaiwanStockPrice_ACTUAL_AMOUNT')
                        failures=store.meta('raw_download_failures',{});failures.pop(t,None);store.set_meta('raw_download_failures',failures)
                    except DataError as exc:
                        failures=store.meta('raw_download_failures',{})
                        failures[t]={'error_type':type(exc).__name__,'message':str(exc)[:240]};store.set_meta('raw_download_failures',failures)
                        progress(phase='AMOUNT',done=i+1,total=len(requested),current_symbol=t,quarantined_symbols=len(failures),
                                 message='Amount data invalid for '+t+'; recorded, continuing collection. Research will block until repaired.')
                        continue
                failure_count=len(store.meta('raw_download_failures',{}));committed_count=i+1-failure_count
                progress(phase='AMOUNT',done=i+1,total=len(requested),current_symbol=t,last_committed_symbol=t,
                         roster_count=len(cat),committed_symbols=committed_count,
                         message=f'已處理 {i+1}/{len(requested)}；成功儲存 {committed_count} 檔歷史量值（{t}；'+('已使用快取）' if cached else '新取得資料）'))
            failures=store.meta('raw_download_failures',{});unresolved={t:e for t,e in failures.items() if t in {r['ticker'] for r in requested}}
            atomic_json(run/'download_failures.json',unresolved);atomic_json(run/'price_quality_summary.json',store.quality_summary())
            if unresolved:raise DataError('Amount collection completed with unresolved symbols: '+','.join(sorted(unresolved))+'; cached good symbols retained, no full-market research claim')
        amt=[];empty=[]
        for i,row in enumerate(cat):
            check();t=row['ticker'];raw=store.get('raw',t,cfg.start,cfg.end)
            if raw.empty:empty.append(t);continue
            ss=turnover_snapshots(raw,cal,signals,cfg);ss['ticker']=t;amt.append(ss)
            if i%30==0:progress(phase='POOL',done=i+1,total=len(cat),message='Computing monthly trailing-252 turnover ranks')
        if empty:raise DataError('No raw history for requested symbols: '+','.join(empty[:20])+'; full-market ranking is blocked')
        turnover=pd.concat(amt,ignore_index=True)

    if len(signals)<minimum:raise DataError(f'Only {len(signals)} matured monthly dates; protocol needs at least {minimum}. Extend actual history, do not recycle dates.')

    members=caps=evidence=None
    if input_dir:
        root=Path(input_dir)
        if (root/'membership.csv').exists():members=pd.read_csv(root/'membership.csv')
        if (root/'market_caps.csv').exists():caps=pd.read_csv(root/'market_caps.csv')
        if cfg.enable_optional_factors:
            if not (root/'evidence.csv').exists():raise DataError('PIT evidence.csv missing')
            evidence=pd.read_csv(root/'evidence.csv')
    pools,pool_diag=make_pools(turnover,signals,cfg,members,caps)
    frame_write(run/'monthly_pools.csv.gz',pools);frame_write(run/'pool_coverage.csv',pool_diag)
    union=sorted(set(pools.ticker))

    if client:
        required=union+[BENCHMARK]
        missing=[t for t in required if not (store.cached('raw:'+t,cfg.start,cfg.end) and store.cached('adj:'+t,cfg.start,cfg.end))]
        if bulk and missing:
            try:source['bulk_selected_prices']=bulk.populate_selected_prices(store,missing);missing=[]
            except DataError as exc:
                if cfg.history_provider not in ('AUTO','AUTO_FAST'):raise
                if provider=='YAHOO_FAST':
                    raise DataError('Yahoo固定母池批次下載失敗；AUTO_FAST不會改走逐檔FinMind。已完成的Yahoo快取會保留。') from exc
                source['bulk_selected_price_fallback_reason']=str(exc)[:240]
        # FinLab selected-union fallback may still use per-symbol sources.  Yahoo
        # fast mode never silently converts into the ~2,000-symbol FinMind path.
        for i,t in enumerate(missing):
            check();progress(phase='SELECTED_FALLBACK',done=i,total=len(missing),current_symbol=t,
                message=f'固定母池聯集 fallback：{i}/{len(missing)}（{t}）')
            if not store.cached('raw:'+t,cfg.start,cfg.end):
                f=client.raw(t,start=cfg.start);store.put('raw',t,f,cfg.start,cfg.end,'FinMind_SELECTED_UNION_ACTUAL_AMOUNT')
            if not store.cached('adj:'+t,cfg.start,cfg.end):
                a=client.adjusted(t);store.put('adj',t,a,cfg.start,cfg.end,'YAHOO_ADJUSTED_PROVIDER');client.wait(.5,'Adjusted price pacing')
            progress(phase='SELECTED_FALLBACK',done=i+1,total=len(missing),last_committed_symbol=t,
                     message=f'固定母池聯集已儲存 {i+1}/{len(missing)} 檔')

    from lab_price_quality import analytic_hlc_from_audit
    br=store.get('raw',BENCHMARK,cfg.start,cfg.end);ba=store.get('adj',BENCHMARK,cfg.start,cfg.end)
    if br.empty or ba.empty:raise DataError('Benchmark daily history is incomplete after selected-union preparation')
    bra,bstat=analytic_hlc_from_audit(br,store.quality_details(BENCHMARK));benchmark_frame=raw_adjusted_frame(br,ba,cal,bra)
    quality_rows=[dict(bstat,ticker=BENCHMARK)];snapshots=[];labels=[]
    for i,t in enumerate(union+[BENCHMARK]):
        check();r=store.get('raw',t,cfg.start,cfg.end);a=store.get('adj',t,cfg.start,cfg.end)
        analytical,stats=analytic_hlc_from_audit(r,store.quality_details(t));f=raw_adjusted_frame(r,a,cal,analytical)
        quality_rows.append(dict(stats,ticker=t,valid_adjusted_close=int(f.adj_close.notna().sum()),valid_execution_open=int(f.raw_open.notna().sum())))
        if t!=BENCHMARK:
            z=features(f,benchmark_frame).loc[signals].copy();z['date']=z.index;z['ticker']=t
            valid_dates=set(pools.loc[pools.ticker==t,'date']);snapshots.append(z[z.date.isin(valid_dates)].reset_index(drop=True))
        used=signals if t==BENCHMARK else pd.DatetimeIndex([d for d in signals if d in valid_dates])
        for h in cfg.horizons:
            batch=label_records(f,used,h,cfg,t==BENCHMARK)
            for q in batch:q['ticker']=t
            labels.extend(batch)
        if i%10==0:progress(phase='FEATURES',done=i+1,total=len(union)+1,message='Computing features and immutable outcome labels once')
    snap=pd.concat(snapshots,ignore_index=True);snap=snap.merge(turnover[['date','ticker','amount20']],on=['date','ticker'],how='left')
    names={r['ticker']:r.get('name',r['ticker']) for r in cat};sectors={r['ticker']:r.get('sector','UNKNOWN') for r in cat}
    snap['name']=snap.ticker.map(names);snap['sector']=snap.ticker.map(sectors)
    if evidence is not None:snap=attach_evidence(snap,evidence)
    snap=percentile_reference(snap,cfg)
    progress(phase='SAVE',message='Saving compact monthly feature checkpoint');frame_write(run/'features.csv.gz',snap)
    progress(phase='SAVE',message='Saving shared outcome checkpoint');lab=pd.DataFrame(labels);frame_write(run/'labels.csv.gz',lab)
    frame_write(run/'price_quality_audit.csv',pd.DataFrame(quality_rows))
    source['analytical_hlc_recovery_is_not_execution_repair']=True;source['monthly_union_count']=len(union)
    source['feature_reference']='SAME_DATE_UNION_OF_ALL_PREREGISTERED_POOLS';source['price_revision_vintages_verified']=False
    atomic_json(run/'source_scope.json',source)
    progress(phase='HASH',message='Hashing consistent input database')
    atomic_json(run/'data_manifest.json',{'canonical_data_hash':store.data_hash(),'downloads':store.manifest(),
                    'source':source,'input_rows_are_not_trade_evidence':True})
    return snap,lab,pools,cal,source

