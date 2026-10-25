"""Readable small exports first. Market inputs are not embedded in the report ZIP."""
from pathlib import Path
import json
import zipfile
from lab_io import atomic_json,sha
from lab_config import VERSION


def percent(x): return '--' if x is None else f'{x*100:+.2f}%'

def write_report(result,run):
    run=Path(run)
    atomic_json(run/'results.json',result)
    lines=[f'Alpha Radar {VERSION} | Robust-Consensus Universe Research',
           '\u6bcd\u6c60\uff0b\u56e0\u5b50\u6b0a\u91cd\u7814\u7a76\u5831\u544a',
           '', 'This is not proof of an optimal live stock-picking strategy.',
           'Scope: '+str(result['data_scope'].get('scope')),
           'Candidates are selected using PURGED TRAINING data only.',
           'Selector: '+str(result['config'].get('selector')),
           'Funded equal-weight stock slots: '+str(result['config'].get('top_k')),
           'Maximum 64 default static candidates plus one train-only consensus per freeze.',
           'Outer tests and the final tail do not select their own winner.',
           'All reported mean returns are per sampled holding window, NOT annualized.',
           'No automatic orders. No automatic extra tests after an unfavorable result.','']
    champions={}
    for h,r in result['models'].items():
        c=r['freeze']['choice'];m=r['final_tail'];o=r['outer_oos']
        guard=r.get('pre_tail_activation_gate',{})
        equivalent=r.get('equivalent_training_pool_outcomes',[])
        report_pool=c['pool'] or 'REFERENCE_NOT_A_STOCK_POOL'
        active=c.get('candidate_id')!='REFERENCE_0050'
        lines += [f'=== {h} trading sessions ===',
                  'Verdict: '+r['verdict'],
                  'Research-selected pool: '+report_pool,
                  'Pre-tail activation: '+guard.get('status','NOT_CHECKED'),
                  'Pre-tail blockers: '+(', '.join(guard.get('failed_checks',[])) or 'None'),
                  'Training-equivalent pool results: '+(', '.join(equivalent) or 'None'),
                  'Research candidate selected stocks: '+str(active),
                  'Reference=0050 is not an alpha result; its excess is zero by definition.',
                  'Model: '+c['name'],
                  'Weight construction: '+c.get('selection_method','REFERENCE'),
                  'Recent training stress excess: '+percent(c.get('recent_train_stress_excess')),
                  'Weights: '+json.dumps(c['weights'],ensure_ascii=False),
                  'Outer OOS vs 0050: '+percent(o.get('versus_0050',{}).get('mean')),
                  'Final-tail vs 0050: '+percent(m.get('versus_0050',{}).get('mean')),
                  'Final-tail vs same pool: '+percent(m.get('versus_same_pool',{}).get('mean')),
                  'Final-tail block 95% lower: '+percent(m.get('versus_0050',{}).get('block95_low')),
                  'Final-tail known cases: '+str(m.get('versus_0050',{}).get('complete'))+'/'+str(m.get('versus_0050',{}).get('planned')),
                  'Final-tail disjoint known cases: '+str(m.get('versus_0050',{}).get('disjoint_complete')),
                  'Final-tail extra STOCK-cost stress: '+percent(m.get('extra_cost_stress_excess')),
                  'Reference extra stock cost is ZERO by construction.',
                  'Final-tail stock capital fraction: '+str(m.get('stock_capital_fraction')),
                  'Selector OOS is NOT confirmation of the exact final weight vector.',
                  'Tail overlaps previously exposed history: '+str(r['historical_tail_previously_exposed']),
                  'Historical/current-roster bias still applies: '+r['scope'],'']
        champions[h]={'pool':c['pool'],'weights':c['weights'],'verdict':r['verdict'],
                      'research_only_pool':c['pool'],
                      'research_only_weights':c['weights'],
                      'actual_trade_model_approved':False,
                      'reference_default':guard.get('fallback')=='REFERENCE_0050',
                      'reference_default_is_alpha':False,
                      'pre_tail_activation_gate':guard,
                      'pool_identification':r.get('pool_identification'),
                      'equivalent_training_pool_outcomes':equivalent,
                      'benchmark_default_if_unproven':guard.get('fallback')=='REFERENCE_0050',
                      'research_challenger':r['freeze'].get('best_active_training_challenger'),
                      'funded_slots':result['config'].get('top_k'),
                      'concentration':r.get('concentration'),
                      'weight_construction':c.get('selection_method'),
                      'registered_train_cutoff':str(r['freeze']['train_end']),
                      'holdout_used_for_selection':False,'use':'PAPER_RESEARCH_ONLY',
                      'automatic_orders_authorized':False,'production_approved':False}
    lines+=['LIMITATIONS']+result['limitations']
    text='\n'.join(lines)+'\n'
    (run/'REPORT.txt').write_text(text,encoding='utf-8-sig')
    atomic_json(run/'selected_models.json',champions)
    atomic_json(run/'PRETAIL_GATES.json',{h:r.get('pre_tail_activation_gate',{}) for h,r in result['models'].items()})
    # Small outputs avoid embedding many MB of price inputs in a download button.
    names=['REPORT.txt','results.json','selected_models.json','PRETAIL_GATES.json','protocol.json','candidates.json',
           'source_scope.json','pool_control_comparison.csv','outer_oos_records.csv','final_tail_records.csv',
           'collection_reuse.json','pool_coverage.csv','feature_coverage.csv','feature_coverage_by_date.csv','price_quality_audit.csv']
    names += [p.name for p in run.glob('final_freeze_*.json')]
    names += [p.name for p in run.glob('development_candidates_*.csv')]
    names=[n for n in names if (run/n).is_file()]
    manifest={n:sha(run/n) for n in names};atomic_json(run/'EXPORT_SHA256.json',manifest)
    path=run/'LAB_RESULTS.zip';tmp=run/'LAB_RESULTS.tmp.zip'
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
        for n in names+['EXPORT_SHA256.json']:z.write(run/n,n)
    with zipfile.ZipFile(tmp) as z:
        if z.testzip():raise ValueError('Report ZIP CRC failed')
    tmp.replace(path)
    return path
