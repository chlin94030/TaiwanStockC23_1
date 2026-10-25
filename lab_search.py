"""Chronological, label-maturity-purged inner search / outer tests / one held-out tail.
Pool size is selected INSIDE the inner search, not using the outer/holdout scores.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from lab_io import DataError,atomic_json,frame_write
from lab_backtest import evaluate
from lab_stats import uncertainty,finite_mean,disjoint_indices
from lab_approval import reference_choice, is_reference, pre_tail_gate, equivalent_training_pools


def matured_training_indices(dates,exits,boundary,calendar,gap):
    cal=pd.DatetimeIndex(calendar);b=cal.searchsorted(pd.Timestamp(boundary))-gap
    if b<=0:return np.array([],int)
    cutoff=cal[b]
    return np.flatnonzero((pd.DatetimeIndex(dates)<boundary)&(pd.DatetimeIndex(exits)<cutoff))


def choose_inner(grids,candidates,h,train_idx,cfg,cases=None):
    from lab_robust import choose_robust
    return choose_robust(grids,candidates,h,train_idx,cfg,cases)


def record_choice(choice,keycases,idx,cfg):
    rows=[]
    for i in idx:
        c=keycases[i]
        if is_reference(choice):
            # The benchmark has its own precomputed entry, costs and price
            # availability. A reference is not a stock-picking result.
            q={'net':c.benchmark, 'loss_floor':c.benchmark,
               'filled_weight':1.0 if np.isfinite(c.benchmark) else 0.0,
               'feature_coverage':1.0, 'selected':['0050.TW'] if np.isfinite(c.benchmark) else [],
               'missing_slots':0 if np.isfinite(c.benchmark) else 1,
               'cash_slots':0, 'state':'BENCHMARK_REFERENCE' if np.isfinite(c.benchmark) else 'BENCHMARK_UNKNOWN',
               'position_mdd':np.nan}
        else:
            q=evaluate(c,choice['weights'],cfg)
        q['stock_filled_weight']=0.0 if is_reference(choice) else q['filled_weight']
        q['is_reference']=is_reference(choice)
        q['selected_slot_weight']=0.0 if is_reference(choice) else 1.0/cfg.top_k
        rows.append(dict(q,date=c.date,exit=c.exit,entry=c.entry,pool=choice['pool'],
                    benchmark=c.benchmark,pool_net=c.pool_net,pool_missing_weight=c.pool_missing_weight,
                    excess_0050=q['net']-c.benchmark if np.isfinite(q['net']) and np.isfinite(c.benchmark) else np.nan,
                    excess_pool=(q['net']-c.pool_net if not is_reference(choice) else np.nan),
                    candidate_id=choice['candidate_id'],weights=choice['weights'],horizon=None))
    return rows


def summarize_rows(rows,cfg):
    if not rows:return {'status':'NO_EVALUATION_ROWS'}
    dates=[r['date'] for r in rows];exits=[r['exit'] for r in rows]

    # V22.4 compatibility guard: older cached JSON may contain ``null`` for
    # unavailable numeric evidence. JSON null becomes Python None; direct
    # arithmetic such as ``None - cost`` raises TypeError.  Treat unavailable
    # evidence as NaN so the existing coverage/insufficient-evidence guards can
    # handle it without forcing users to delete caches or rebuild market data.
    def num(value):
        try:
            if value is None:return np.nan
            x=float(value)
            return x if np.isfinite(x) else np.nan
        except (TypeError,ValueError):
            return np.nan

    stress=[]
    for r in rows:
        excess=num(r.get('excess_0050'))
        filled=num(r.get('stock_filled_weight',r.get('filled_weight')))
        stress.append(excess-cfg.extra_roundtrip_stress*filled if np.isfinite(excess) and np.isfinite(filled) else np.nan)

    out={'versus_0050':uncertainty([num(r.get('excess_0050')) for r in rows],dates,exits,cfg),
         'versus_same_pool':uncertainty([num(r.get('excess_pool')) for r in rows],dates,exits,cfg),
         'net_mean':finite_mean([num(r.get('net')) for r in rows]),
         'benchmark_mean':finite_mean([num(r.get('benchmark')) for r in rows]),
         'extra_cost_stress_excess':finite_mean(stress),
         'filled_capital_fraction':finite_mean([num(r.get('filled_weight')) for r in rows]),
         'stock_capital_fraction':finite_mean([num(r.get('stock_filled_weight',r.get('filled_weight'))) for r in rows]),
         'reference_months':sum(bool(r.get('is_reference')) for r in rows),
         'pool_missing_weight_mean':finite_mean([num(r.get('pool_missing_weight')) for r in rows])}
    ids=disjoint_indices(dates,exits)
    closed=[num(rows[i].get('net')) for i in ids]
    if closed and all(np.isfinite(closed)):
        equity=np.r_[1,np.cumprod(1+np.array(closed))]
        out['nonoverlap_closed_trade_return']=float(equity[-1]-1)
        out['closed_trade_curve_drawdown_NOT_DAILY_NAV']=float(np.min(equity/np.maximum.accumulate(equity)-1))
    else:out['nonoverlap_closed_trade_return']=None
    return out


def audit_factors(cases,choice,idx,cfg):
    if is_reference(choice):
        return {'ablations':[], 'factor_ic':[], 'reason':'REFERENCE_HAS_NO_STOCK_FACTORS'}
    base=[evaluate(cases[i],choice['weights'],cfg)['net'] for i in idx]
    ablations=[]
    if len(choice['weights'])>1:
        for factor in choice['weights']:
            w={k:v for k,v in choice['weights'].items() if k!=factor}
            total=sum(w.values());w={k:v/total for k,v in w.items()}
            alt=np.array([evaluate(cases[i],w,cfg)['net'] for i in idx])
            delta=np.array(base)-alt
            ablations.append({'removed':factor,'pairs':int(np.isfinite(delta).sum()),'full_minus_removed':finite_mean(delta),
                              'scope':'PRE_HOLDOUT_TRAINING_DIAGNOSTIC_NOT_CAUSAL'})
    ic=[]
    for fi,name in enumerate(cases[0].features):
        vals=[]
        for i in idx:
            c=cases[i];mask=np.isfinite(c.ranks[:,fi])&np.isfinite(c.net)
            if mask.sum()<20:continue
            x=pd.Series(c.ranks[mask,fi]).rank();y=pd.Series(c.net[mask]).rank()
            if x.std()>0 and y.std()>0:vals.append(float(x.corr(y)))
        ic.append({'factor':name,'ic_months':len(vals),'mean_rank_ic':finite_mean(vals),
                   'scope':'TRAINING_ONLY_OBSERVED_OUTCOMES','not_return_or_win_probability':True})
    return {'ablations':ablations,'factor_ic':ic}


def verdict(summary,cfg):
    if 'versus_0050' not in summary:return 'INSUFFICIENT_EVIDENCE'
    for key in ('versus_0050','versus_same_pool'):
        s=summary[key]
        if s['coverage']<cfg.min_fold_coverage or s.get('disjoint_complete',0)<cfg.min_disjoint_cases:
            return 'INSUFFICIENT_EVIDENCE'
    positive=all(summary[k]['mean']>=cfg.economic_excess_floor and summary[k].get('block95_low',-1)>0
                 and (summary[k].get('disjoint95_low') or -1)>0 for k in ('versus_0050','versus_same_pool'))
    return 'HISTORICAL_SUPPORT_ONLY' if positive and (summary['extra_cost_stress_excess'] or -1)>0 else 'NOT_SUPPORTED'


def nested_study(grids,cases,candidates,calendar,cfg,run,source,progress,check):
    run=Path(run);results={};all_outer=[];all_tail=[];pool_tables=[]
    calendar=pd.DatetimeIndex(calendar)
    for h in cfg.horizons:
        key=next(k for k in grids if k[1]==h);g=grids[key];dates=g['dates'];exits=g['exits']
        # All universes must use the same predeclared signal calendar.
        if any(not v['dates'].equals(dates) for k,v in grids.items() if k[1]==h):raise DataError('Mismatched comparison calendars')
        hold_start=len(dates)-cfg.holdout_months
        boundary=dates[hold_start]
        dev=matured_training_indices(dates,exits,boundary,calendar,cfg.gap_sessions)
        folds=[];outer_rows=[]
        for start in range(cfg.min_train_months,len(dates),cfg.outer_test_months):
            test=np.array([i for i in range(start,min(start+cfg.outer_test_months,len(dates))) if i in set(dev)],int)
            if len(test)<3:continue
            train=matured_training_indices(dates,exits,dates[test[0]],calendar,cfg.gap_sessions)
            if len(train)<cfg.min_train_months:continue
            check();progress(phase='NESTED',message=f'{h}d chronological fold {len(folds)+1}',done=len(folds)+1,total=None)
            choice,board=choose_inner(grids,candidates,h,train,cfg,cases)
            if choice is None:
                folds.append({'fold':len(folds)+1,'status':'NO_DATA_QUALIFIED_CANDIDATE','test_start':dates[test[0]],'test_end':dates[test[-1]]});continue
            # Freeze BEFORE producing test records; no winner selection on test returns.
            fold={'fold':len(folds)+1,'train_start':dates[train[0]],'train_end':dates[train[-1]],
                  'max_train_label_exit':exits[train].max(),'test_start':dates[test[0]],'test_end':dates[test[-1]],'choice':choice}
            atomic_json(run/f'fold_{h}_{len(folds)+1}_freeze.json',fold)
            sample_pool=choice['pool'] if not is_reference(choice) else sorted(k[0] for k in cases if k[1]==h)[0]
            rows=record_choice(choice,cases[(sample_pool,h)],test,cfg)
            for r in rows:r['horizon']=h;r['fold']=len(folds)+1
            fold['test']=summarize_rows(rows,cfg);fold['status']='EVALUATED';folds.append(fold);outer_rows+=rows
        if sum(f.get('status')=='EVALUATED' for f in folds)<cfg.min_outer_folds:
            raise DataError(f'{h}d: insufficient valid outer folds. No forced winner; inspect input coverage or history length.')
        winner,leaderboard=choose_inner(grids,candidates,h,dev,cfg,cases)
        if winner is None:raise DataError('No data-qualified model for final freeze')
        freeze={'horizon':h,'choice':winner,
                'best_active_training_challenger':leaderboard[0] if leaderboard else None,'train_start':dates[dev[0]],'train_end':dates[dev[-1]],
                'max_train_label_exit':exits[dev].max(),'holdout_start':boundary,'holdout_end':dates[-1],
                'holdout_used_for_selection':False,'selector':cfg.selector,'funded_slots':cfg.top_k,
                'max_extra_training_consensus_candidates':1,
                'candidate_count':len(candidates),'pool_count':len([k for k in grids if k[1]==h])}
        atomic_json(run/f'final_freeze_{h}.json',freeze)
        # Only this frozen model is reported on the final tail. Alternatives are not
        # ranked by their tail outcomes, and a failed tail never triggers refitting.
        tail_idx=np.arange(hold_start,len(dates))
        tail_pool=winner['pool'] if not is_reference(winner) else sorted(k[0] for k in cases if k[1]==h)[0]
        tail=record_choice(winner,cases[(tail_pool,h)],tail_idx,cfg)
        for r in tail:r['horizon']=h;r['fold']='FINAL_TAIL'
        tail_summary=summarize_rows(tail,cfg);outer_summary=summarize_rows(outer_rows,cfg)
        pre_tail_guard=pre_tail_gate(winner,outer_summary,folds,cfg)
        equivalent=equivalent_training_pools({k:v for k,v in grids.items() if k[1]==h},winner,dev,cases=cases,cfg=cfg)
        pre_tail_guard['evidence_scope']='OUTER_RECORDS_VALIDATE_THE_SELECTOR_NOT_THIS_FINAL_WEIGHT_VECTOR'
        data_scope=source['scope']
        exposed=bool(cfg.exposed_through and boundary<=pd.Timestamp(cfg.exposed_through))
        def concentration(rows):
            counts={}
            for row in rows:
                if row.get('is_reference'):continue
                for ticker in row.get('selected',[]):counts[ticker]=counts.get(ticker,0)+1
            active=sum(not r.get('is_reference',False) for r in rows)
            return {'active_months':active,'unique_stocks':len(counts),
                    'funded_slots':cfg.top_k,'maximum_planned_single_name_weight':1/cfg.top_k,
                    'selection_frequency':[{'ticker':t,'months':n,'fraction_of_active_months':n/active if active else None}
                        for t,n in sorted(counts.items(),key=lambda x:(-x[1],x[0]))[:20]],
                    'not_evidence_of_sector_diversification':True}
        result={'horizon':h,'freeze':freeze,'outer_folds':folds,'outer_oos':outer_summary,
                'final_tail':tail_summary,'verdict':verdict(tail_summary,cfg),
                'scope':data_scope,'historical_tail_previously_exposed':exposed,
                'independent_future_holdout':False,'automatic_orders_authorized':False,
                'global_optimum_proven':False,'best_definition':'TRAINING_ROBUST_OBJECTIVE_WITH_ONE_BOUNDED_CONSENSUS_NOT_GLOBAL_OPTIMUM',
                'diagnostics':audit_factors(cases[(tail_pool,h)],winner,dev,cfg),
                'pre_tail_activation_gate':pre_tail_guard,
                'concentration':concentration(tail),
                'equivalent_training_pool_outcomes':equivalent,
                'pool_identification':'OBSERVATIONALLY_EQUIVALENT_ON_TRAIN' if len(equivalent)>1 else 'DISTINGUISHABLE_ON_TRAIN',
                'reference_is_zero_excess_by_definition':is_reference(winner),
                'active_stock_selection_authorized':False,
                'active_stock_selection_research_gate':pre_tail_guard['status']}
        if is_reference(winner):
            result['verdict']='NO_STOCK_ALPHA_SELECTED'
        if result['verdict']=='HISTORICAL_SUPPORT_ONLY':
            ov=outer_summary.get('versus_0050',{})
            if ov.get('coverage',0)<cfg.min_fold_coverage:
                result['verdict']='INSUFFICIENT_EVIDENCE'
            elif (ov.get('mean') or -1)<=0 or (outer_summary.get('extra_cost_stress_excess') or -1)<=0:
                result['verdict']='NOT_SUPPORTED'
            elif pre_tail_guard['status'] != 'PRETAIL_RESEARCH_GATE_PASSED':
                # A favorable exposed tail must not rescue a candidate already
                # rejected by the frozen pre-tail promotion checks.
                result['verdict']='NOT_SUPPORTED'
        if source.get('data_kind')=='SYNTHETIC':result['verdict']='SYNTHETIC_ENGINE_TEST_NOT_PERFORMANCE'
        results[str(h)]=result;all_outer+=outer_rows;all_tail+=tail
        # Factor weights table is training-only. Final-tail scores never enter it.
        frame_write(run/f'development_candidates_{h}.csv',pd.DataFrame([{**r,'weights':str(r['weights']),'inner_block_means':str(r['inner_block_means'])} for r in leaderboard]))
        for pool in sorted({k[0] for k in grids if k[1]==h}):
            gg=grids[(pool,h)];c0=next(i for i,c in enumerate(candidates) if c['name']=='v20_control_80_20')
            pool_tables.append({'horizon':h,'pool':pool,'scope':'PURGED_PRE_HOLDOUT_ONLY',
                               'pool_equal_weight_vs_0050':finite_mean(gg['pool'][dev]-gg['benchmark'][dev]),
                               'fixed_80_20_vs_0050':finite_mean(gg['net'][dev,c0]-gg['benchmark'][dev]),
                               'fixed_80_20_vs_pool':finite_mean(gg['net'][dev,c0]-gg['pool'][dev]),
                               'dates':len(dev),'pool_complete':int(np.isfinite(gg['pool'][dev]).sum())})
    frame_write(run/'outer_oos_records.csv',pd.DataFrame(all_outer));frame_write(run/'final_tail_records.csv',pd.DataFrame(all_tail))
    frame_write(run/'pool_control_comparison.csv',pd.DataFrame(pool_tables))
    from lab_config import VERSION
    return {'version':VERSION,'data_scope':source,'config':cfg.to_dict(),'models':results,
            'finished_finite_search':True,'auto_retest_on_failure':False,'automatic_orders_authorized':False,
            'limitations':['Current snapshot mode is survivor-biased, not complete historical membership.',
              'Adjusted histories have no historical revision vintage ledger.',
              'Monthly overlapping holding-period arithmetic means are not continuous portfolio NAV.',
              'Unknown outcomes are not zero; same-pool coverage can be insufficient.',
              'This finite search is not a proof of a global optimum or a future profit guarantee.',
              'Intraday paths, official limit-price executions and portfolio-wide drawdown are not validated.',
              'Reported bootstrap intervals are descriptive, not familywise multiple-testing-adjusted significance.',
              'A benchmark reference has exactly zero excess by construction; this is not earned alpha.',
              'V21.4 factors, ten-slot portfolio and robust selector were designed after exposed V21.0/V21.3 evidence; this is new development, not fresh blind confirmation.',
              'Outer OOS aggregates different training-selected strategies and evaluates the selection procedure, not a fixed final weight vector.',
              'A residual-momentum feature is not a fully beta-neutral investment portfolio; it only removes an estimated prior market exposure in that feature.',
              'Feature-only HLC recovery preserves recorded values and does not repair executable raw opens or exits.',
              'Ten equal funded slots cap a selected name at 10% only when top_k=10; cash slots and missing fills remain explicit.']}

