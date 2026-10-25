"""Offline, fixed-universe development loop and one explicitly released confirmation.
Fast experiments never evaluate or display the final tail. Conditional development
folds are NOT fresh OOS evidence after a pool has been chosen from past results.
"""
from pathlib import Path
from dataclasses import fields, replace
import json
import time
import threading
import traceback
import zipfile
import numpy as np
import pandas as pd
from lab_config import Config
from lab_io import DataError, PauseRequested, JobLock, atomic_json, read_json, frame_write, sha
from lab_search import matured_training_indices, summarize_rows, verdict
from lab_robust import score_training
from lab_approval import reference_choice, is_reference
from studio_pool import load_dataset, load_pool, digest, timestamp
from studio_cache import prepare_fixed_features, load_labels, algorithm_hash
from studio_engine import make_cases, validate_profiles, cached_rows

STUDIO_VERSION='21.5-fixed-pool-studio'
DEFAULT_SPEC={'schema':1,'top_k':10,'capital':1000000.0,'fast_folds':3,
              'fast_bootstrap_samples':200,'final_bootstrap_samples':2000,'cost_overrides':{}}
_ALLOWED=set(DEFAULT_SPEC)
_THREADS={};_LOCK=threading.Lock()


def configure(meta,spec):
    if set(spec)-_ALLOWED:raise DataError('Unknown studio setting')
    s={**DEFAULT_SPEC,**spec}
    if s['schema']!=1 or not isinstance(s['top_k'],int) or not 10<=s['top_k']<=30:
        raise DataError('Use 10..30 funded slots; this studio caps planned single-name weight at 10%')
    if not np.isfinite(s['capital']) or s['capital']<=0:raise DataError('Invalid capital')
    if not 1<=s['fast_folds']<=12:raise DataError('Fast development uses 1..12 folds')
    if not 100<=s['fast_bootstrap_samples']<=2000 or not 200<=s['final_bootstrap_samples']<=10000:raise DataError('Invalid bootstrap budget')
    if set(s['cost_overrides'])-{'fee_buy','fee_sell','stock_tax','benchmark_tax','slippage','max_gap','extra_roundtrip_stress'}:
        raise DataError('Unsupported cost change; no hidden outcome transformation')
    allowed={f.name for f in fields(Config)}
    base={k:v for k,v in meta['source_config'].items() if k in allowed}
    
    for key,value in s['cost_overrides'].items():
        if not isinstance(value,(int,float)) or not np.isfinite(value) or not 0<=value<=1:
            raise DataError('Cost rates must be finite fractions in [0,1]')
    usable_pools=tuple(n for n in base.get('pools',(100,200,300,500)) if n>=s['top_k'])
    if not usable_pools:raise DataError('Requested funded slots exceed every prepared pool')
    cfg=replace(Config(**base),pools=usable_pools,top_k=s['top_k'],capital=float(s['capital']),**s['cost_overrides']).resolved()
    if cfg.max_gap<0 or cfg.max_gap>1:raise DataError('Invalid maximum entry gap')
    return cfg,s


def _grid(rows,n):
    g={key:np.full((n,1),np.nan) for key in ('net','floor','fills','feature_coverage')}
    g['benchmark']=np.full(n,np.nan)
    for i,r in rows.items():
        for out,field in [('net','net'),('floor','loss_floor'),('fills','filled_weight'),('feature_coverage','feature_coverage')]:g[out][i,0]=r[field]
        g['benchmark'][i]=r['benchmark']
    return g


def reference_row(sc,h):
    c=sc.base;known=np.isfinite(c.benchmark)
    return {'date':c.date,'entry':c.entry,'exit':c.exit,'horizon':h,'pool':None,'benchmark':c.benchmark,
            'pool_net':c.pool_net,'pool_missing_weight':c.pool_missing_weight,'regime':sc.regime,
            'candidate_id':'REFERENCE_0050','candidate_name':'0050_REFERENCE_NOT_ALPHA','weights':{},'policy':'reference',
            'net':c.benchmark,'loss_floor':c.benchmark,'filled_weight':float(known),'stock_filled_weight':0.,
            'feature_coverage':1.,'is_reference':True,'excess_0050':0. if known else np.nan,'excess_pool':np.nan,
            'selected':['0050.TW'] if known else [],'state':'REFERENCE' if known else 'BENCHMARK_UNKNOWN',
            'position_mdd':np.nan,'missing_slots':0 if known else 1,'cash_slots':0,'decision_uses_future_outcomes':False}


def export_report(run,result):
    run=Path(run);atomic_json(run/'results.json',result)
    lines=['Alpha Radar '+STUDIO_VERSION, '\u56fa\u5b9a\u6bcd\u6c60\u7b56\u7565\u5de5\u4f5c\u53f0',
           'Mode: '+result['mode'],'Universe lock: '+result['pool_lock']['lock_id'],
           'Universe search calls: 0 | Network calls: 0',
           'Same historical data and repeated development are NOT an independent holdout.',
           'Book-inspired signals are hypotheses, not original book formulas or proven alpha.',
           'Market turnover is NOT macro money supply; ownership psychology is NOT directly observed.',
           'All returns are holding-window arithmetic returns, not continuous portfolio NAV.','']
    for h,r in result['models'].items():
        lines+=['=== '+h+' sessions ===','Fixed pool: '+r['pool'],
                'Training choice: '+r['choice']['name'], 'Weights: '+json.dumps(r['choice']['weights'],ensure_ascii=False),
                'Verdict: '+r['verdict']]
        label='final_confirmation' if result['mode']=='confirm' else 'development_validation'
        sm=r.get(label,{})
        for name in ('versus_0050','versus_same_pool'):
            x=sm.get(name,{})
            fmt=lambda v:'--' if v is None else f'{100*v:+.3f}%'
            lines += [name+': '+fmt(x.get('mean'))+' | block95 low '+fmt(x.get('block95_low'))+
                ' | known '+str(x.get('complete'))+'/'+str(x.get('planned'))]
        lines += ['Final tail inspected: '+str(result['mode']=='confirm'),
                  'Candidate selected by training only; NO automatic trades.','']
    lines+=['CACHE / TIMINGS',json.dumps(result['cache'],ensure_ascii=False),
            'LIMITATIONS: current-snapshot membership bias; repeated historical exposure; no measured macro liquidity; '
            'no intraday fills, continuous NAV, forced-selling model, or automatic order authorization.']
    (run/'REPORT.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8-sig')
    files=[p for p in run.iterdir() if p.is_file() and p.suffix in ('.txt','.csv','.json') and p.name not in ('state.json','error.txt','EXPORT_SHA256.json')]
    atomic_json(run/'EXPORT_SHA256.json',{p.name:sha(p) for p in files})
    path=run/'STRATEGY_RESULTS.zip';tmp=run/'STRATEGY_RESULTS.tmp.zip'
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files+[run/'EXPORT_SHA256.json']:z.write(p,p.name)
    with zipfile.ZipFile(tmp) as z:
        if z.testzip():raise DataError('Report ZIP CRC failed')
    tmp.replace(path)


def assert_confirmation_unused(ledger,eid,era,windows):
    """Changing end date/pool must not turn overlapping outcomes into a fresh test."""
    for old_era,record in ledger.items():
        if record.get('experiment_id')==eid:continue
        old=record.get('holding_windows')
        if old_era==era or not old:
            raise DataError('This tail was already released under another experiment; no new holdout claim is permitted.')
        if any(a<=d and c<=b for a,b in windows for c,d in old):
            raise DataError('An overlapping holding-period tail was already released. Use development mode or a genuinely later preregistered period.')


def run_experiment(root,profiles,spec=None,mode='fast',progress=None,check=None,resume=False):
    root=Path(root);spec=spec or DEFAULT_SPEC;start=time.perf_counter()
    if mode not in ('fast','confirm'):raise DataError('Mode must be fast or confirm')
    _,meta=load_dataset(root);lock=load_pool(root);cfg,spec=configure(meta,spec)
    import re
    if any(int(re.search(r'[0-9]+$',p).group())<cfg.top_k for p in lock['mapping'].values()):
        raise DataError('Funded slots exceed the explicitly frozen pool; change the slot count, not the locked universe')
    identity={'version':STUDIO_VERSION,'dataset_id':meta['dataset_id'],'pool_lock':lock['lock_id'],
              'profiles':profiles,'settings':spec,'mode':mode,
              'code':algorithm_hash(('studio_runner.py','studio_engine.py','studio_behavior.py','studio_cache.py','lab_robust.py','lab_stats.py','lab_search.py','lab_approval.py','lab_config.py'))}
    eid=digest(identity)[:24];run=root/'studio'/'experiments'/eid;run.mkdir(parents=True,exist_ok=True)
    def emit(**kw):
        state=read_json(run/'state.json');state.update(kw,updated_at=timestamp());atomic_json(run/'state.json',state)
        if progress:progress(**kw)
    def check_stop():
        if (run/'PAUSE').exists():raise PauseRequested('Paused, per-candidate results are retained')
        if check:check()
    with JobLock(root/'worker.lock'):
        if resume:(run/'PAUSE').unlink(missing_ok=True)
        atomic_json(root/'studio'/'latest.json',{'experiment_id':eid,'run':str(run)})
        if read_json(run/'state.json').get('status')=='COMPLETE':return run
        emit(status='RUNNING',phase='VERIFY',message='Using a fixed dataset and pool; no network or pool search',started_at=timestamp())
        atomic_json(run/'experiment.json',{**identity,'registered_at':timestamp(),'no_auto_orders':True,
                     'pool_size_not_reoptimized':True,'parameters_are_research_hypotheses':True})
        registry=read_json(root/'studio'/'experiment_ledger.json',[])
        if not any(x['experiment_id']==eid for x in registry):
            registry.append({'experiment_id':eid,'registered_at':timestamp(),'mode':mode,
                             'dataset_id':meta['dataset_id'],'pool_lock':lock['lock_id'],
                             'profiles_hash':digest(profiles),'repeated_development_not_fresh_significance':True})
            atomic_json(root/'studio'/'experiment_ledger.json',registry)
        try:
            fdir,info,feature_hit=prepare_fixed_features(root,emit,check_stop)
            snap=pd.read_csv(fdir/'features.csv.gz',parse_dates=['date']);pools=pd.read_csv(fdir/'pools.csv.gz',parse_dates=['date'])
            context=pd.read_csv(fdir/'context.csv.gz',parse_dates=['date'])
            candidates=validate_profiles(profiles,info['factors'])
            labels,label_info=load_labels(root,fdir,cfg,emit,check_stop)
            calendar=pd.DatetimeIndex(read_json(root/'studio'/'datasets'/meta['dataset_id']/'calendar.json'))
            cases=make_cases(snap,labels,pools,context,lock,cfg,info['factors'])
            ftime=time.perf_counter()-start
            results={};hit=miss=0;outer_export=[];tail_export=[];frozen={};candidate_rows={};regime_export=[]
            for h,cc in cases.items():
                dates=pd.DatetimeIndex([c.base.date for c in cc]);exits=pd.DatetimeIndex([c.base.exit for c in cc])
                if len(cc)<cfg.holdout_months+cfg.min_train_months+6:raise DataError('Not enough monthly history for the locked training/tail design')
                boundary=dates[len(cc)-cfg.holdout_months]
                dev=matured_training_indices(dates,exits,boundary,calendar,cfg.gap_sessions)
                if len(dev)<cfg.min_train_months:raise DataError('Purged development history too short')
                grids={};records={}
                for j,cand in enumerate(candidates):
                    check_stop();emit(phase='FAST_STRATEGY',done=j+1,total=len(candidates),message=f'{h}d: {cand["name"]} on {lock["mapping"][str(h)]}')
                    rr,cached=cached_rows(root,cc,cand,cfg,{'feature':info['feature_id'],'labels':label_info['label_id'],'h':h},dev,check_stop)
                    hit+=cached;miss+=not cached
                    for r in rr:r.update(horizon=h,pool=lock['mapping'][str(h)],candidate_name=cand['name'])
                    records[cand['id']]={int(i):r for i,r in zip(dev,rr)}
                    grids[cand['id']]=_grid(records[cand['id']],len(cc))
                def choose(indices):
                    board=[]
                    for cand in candidates:
                        q=score_training(grids[cand['id']],0,indices,cand['weights'],cfg)
                        if q is not None:board.append({**q,'candidate_id':cand['id'],'name':cand['name'],'weights':cand['weights'],'policy':cand['policy']})
                    board.sort(key=lambda x:(-x['objective'],len(x['weights']),x['candidate_id']))
                    eligible=[q for q in board if q.get('selection_eligible')]
                    return (eligible[0] if eligible else reference_choice('NO_SUPPORTED_TRAINING_CANDIDATE')),board
                folds=[];allfolds=[]
                for t in range(cfg.min_train_months,len(dev),cfg.outer_test_months):
                    test=dev[t:t+cfg.outer_test_months]
                    if len(test)<3:continue
                    train=matured_training_indices(dates,exits,dates[test[0]],calendar,cfg.gap_sessions)
                    train=np.intersect1d(train,dev)
                    if len(train)>=cfg.min_train_months:allfolds.append((train,test))
                allfolds=allfolds[-spec['fast_folds']:] if mode=='fast' else allfolds
                outer=[]
                for train,test in allfolds:
                    choice,_=choose(train)
                    rr=[reference_row(cc[int(i)],h) if is_reference(choice) else dict(records[choice['candidate_id']][int(i)]) for i in test]
                    for r in rr:r['fold']=len(folds)+1
                    outer+=rr;folds.append({'train_max_exit':str(exits[train].max()),'test_start':str(dates[test[0]]),
                        'test_end':str(dates[test[-1]]),'choice':choice,'not_independent_after_pool_selection':True})
                # Same development-validation dates for every registered profile.
                # These comparisons are intentionally exposed tuning diagnostics.
                test_ids=sorted(set(int(i) for _,ii in allfolds for i in ii))
                comparison=[]
                from lab_stats import finite_mean
                for candidate in candidates:
                    rr=[records[candidate['id']][i] for i in test_ids]
                    if not rr:continue
                    comparison.append({'horizon':h,'pool':lock['mapping'][str(h)],
                        'name':candidate['name'],'policy':candidate['policy'],
                        'weights':json.dumps(candidate['weights'],sort_keys=True),
                        'planned':len(rr),'known':sum(np.isfinite(r['excess_0050']) for r in rr),
                        'mean_net':finite_mean([r['net'] for r in rr]),
                        'excess_0050':finite_mean([r['excess_0050'] for r in rr]),
                        'excess_same_pool':finite_mean([r['excess_pool'] for r in rr]),
                        'extra_cost_stress':finite_mean([r['excess_0050']-cfg.extra_roundtrip_stress*r['stock_filled_weight'] for r in rr]),
                        'stock_capital_fraction':finite_mean([r['stock_filled_weight'] for r in rr]),
                        'scope':'EXPOSED_DEVELOPMENT_DIAGNOSTIC_NOT_FRESH_OOS'})
                    for state in sorted(set(r['regime'] for r in rr)):
                        sub=[r for r in rr if r['regime']==state]
                        regime_export.append({'horizon':h,'name':candidate['name'],'policy':candidate['policy'],
                            'regime':state,'planned':len(sub),'known':sum(np.isfinite(r['excess_0050']) for r in sub),
                            'excess_0050':finite_mean([r['excess_0050'] for r in sub]),
                            'macro_money_supply_observed':False,'not_causal':True})
                frame_write(run/f'profile_comparison_{h}.csv',pd.DataFrame(comparison))
                choice,board=choose(dev)
                frozen[str(h)]={'choice':choice,'pool':lock['mapping'][str(h)],'train_max_exit':str(exits[dev].max()),
                                'tail_start':str(boundary),'tail_end':str(dates[-1]),'tail_used_for_selection':False}
                sm=summarize_rows(outer,replace(cfg,block_bootstrap_samples=spec['fast_bootstrap_samples']))
                results[str(h)]={'pool':lock['mapping'][str(h)],'choice':choice,'best_active_candidate':board[0] if board else None,
                   'profile_comparison':comparison,'development_validation':sm,'folds':folds,'verdict':'DEVELOPMENT_ONLY_NOT_ACCEPTANCE',
                   'final_tail_inspected':False,'global_optimum_proven':False,'macro_liquidity_tested':False}
                frame_write(run/f'development_ranking_{h}.csv',pd.DataFrame(board))
                candidate_rows[h]=(cc,records,dev,candidates)
                outer_export+=outer
            atomic_json(run/'FROZEN_STRATEGIES.json',frozen)
            if mode=='confirm':
                # One release per tail calendar. Changing weights or universe does not
                # buy a fresh tail. An interrupted identical experiment may resume.
                era=digest({str(h):[str(c.base.date) for c in cc[-cfg.holdout_months:]] for h,cc in cases.items()})[:24]
                ledgerfile=root/'studio'/'confirmation_ledger.json';ledger=read_json(ledgerfile)
                existing=ledger.get(era)
                windows=sorted(set((str(c.base.date.date()),str(c.base.exit.date()))
                    for cc in cases.values() for c in cc[-cfg.holdout_months:] if pd.notna(c.base.exit)))
                assert_confirmation_unused(ledger,eid,era,windows)
                ledger[era]={'experiment_id':eid,'reserved_at':existing.get('reserved_at') if existing else timestamp(),
                             'status':'RESERVED','freeze_hash':sha(run/'FROZEN_STRATEGIES.json'),
                             'holding_windows':windows,'same_or_overlapping_release_disallowed':True}
                atomic_json(ledgerfile,ledger)
                for h,(cc,records,dev,cands) in candidate_rows.items():
                    choice=results[str(h)]['choice'];idx=np.arange(len(cc)-cfg.holdout_months,len(cc))
                    if is_reference(choice):tail=[reference_row(cc[int(i)],h) for i in idx]
                    else:
                        cand=next(c for c in cands if c['id']==choice['candidate_id'])
                        tail,cached=cached_rows(root,cc,cand,cfg,{'feature':info['feature_id'],'labels':label_info['label_id'],'h':h},idx,check_stop)
                        hit+=cached;miss+=not cached
                    for r in tail:r.update(horizon=h,fold='FINAL_RELEASED_ONCE',pool=lock['mapping'][str(h)],candidate_name=choice['name'])
                    summary=summarize_rows(tail,replace(cfg,block_bootstrap_samples=spec['final_bootstrap_samples']))
                    v='NO_STOCK_ALPHA_SELECTED' if is_reference(choice) else verdict(summary,cfg)
                    # Historical support never confers forward certification.
                    results[str(h)].update(final_confirmation=summary,final_tail_inspected=True,verdict=v)
                    tail_export+=tail
                ledger[era]['status']='COMPLETE';atomic_json(ledgerfile,ledger)
            frame_write(run/'development_walkforward.csv',pd.DataFrame(outer_export))
            frame_write(run/'development_regimes.csv',pd.DataFrame(regime_export))
            if tail_export:frame_write(run/'final_records.csv',pd.DataFrame(tail_export))
            data_kind=meta['scope'].get('data_kind')
            if data_kind=='SYNTHETIC':
                for r in results.values():r['verdict']='SYNTHETIC_ENGINE_TEST_NOT_PERFORMANCE'
            result={'version':STUDIO_VERSION,'experiment_id':eid,'mode':mode,'pool_lock':lock,
                    'dataset_id':meta['dataset_id'],'data_scope':meta['scope'],'models':results,
                    'total_registered_experiments':len(registry),'no_automatic_orders':True,
                    'no_pool_search':True,'independent_future_holdout':False,
                    'tail_previously_exposed_by_prior_versions':True,
                    'cache':{'feature_hit':feature_hit,'label_hit':label_info['hit'],'strategy_hits':int(hit),'strategy_misses':int(miss),
                             'http_requests':0,'pool_searches':0,'feature_stage_seconds':round(ftime,3),
                             'total_seconds':round(time.perf_counter()-start,3)}}
            export_report(run,result);emit(status='COMPLETE',phase='DONE',message='Fixed-pool experiment complete; no automatic next experiment')
            return run
        except PauseRequested as exc:
            emit(status='PAUSED',phase='PAUSED',message=str(exc));return run
        except Exception as exc:
            (run/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
            emit(status='BLOCKED' if isinstance(exc,DataError) else 'ERROR',message=str(exc)[:500])
            raise


def start_studio(root,profiles,spec,mode='fast'):
    root=Path(root);key=str(root.resolve())
    with _LOCK:
        if _THREADS.get(key) and _THREADS[key].is_alive():raise DataError('A studio task is already running')
        def target():
            try:run_experiment(root,profiles,spec,mode,resume=True)
            except Exception as exc:
                atomic_json(root/'studio'/'launch_error.json',{'type':type(exc).__name__,'message':str(exc)[:500],'at':timestamp()})
        (root/'studio').mkdir(parents=True,exist_ok=True)
        atomic_json(root/'studio'/'editor.json',{'profiles':profiles,'settings':spec,'mode':mode})
        (root/'studio'/'launch_error.json').unlink(missing_ok=True)
        latest=read_json(root/'studio'/'latest.json')
        if latest.get('run'):(Path(latest['run'])/'PAUSE').unlink(missing_ok=True)
        t=threading.Thread(target=target,daemon=True,name='fixed-pool-studio');_THREADS[key]=t;t.start()


def studio_alive(root):
    t=_THREADS.get(str(Path(root).resolve()));return bool(t and t.is_alive())
