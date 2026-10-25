"""Bounded external worker. No browser and no Streamlit runtime required.

Each normal time slice commits a data-only checkpoint. Next Actions job restores
it. An abrupt machine death can lose work since the LAST uploaded snapshot,
not the entire earlier database. This is NOT a performance-optimization run.
"""
import argparse
from pathlib import Path
import json
import os
from lab_config import load_config,Config
from lab_io import read_json,atomic_json,code_hash
from lab_jobs import identifier,run_until_stop
from lab_snapshot import export_snapshot
from lab_upgrade import run_config


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='data/v21_universe_lab')
    p.add_argument('--config',default='config.json')
    p.add_argument('--minutes',type=float,default=30)
    p.add_argument('--output',default='background_output')
    a=p.parse_args(argv)
    if not 0<a.minutes<=300:raise ValueError('minutes must be >0 and <=300')
    root=Path(a.root);root.mkdir(parents=True,exist_ok=True)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    latest=read_json(root/'latest.json')
    # Preserve the original end date even if downloads cross midnight or a reboot.
    prior=read_json(Path(latest['run'])/'protocol.json') if latest.get('run') else {}
    cfg=run_config(load_config(a.config),latest,prior)
    if cfg.source_mode!='auto_current_roster':raise ValueError('External background workflow uses the auto-mode cache')
    run=root/'runs'/identifier(cfg)
    state=read_json(run/'state.json')
    if state.get('status')=='COMPLETE':
        print('Already complete: no download, no refit, no new study.')
        atomic_json(out/'outcome.json',{'status':'COMPLETE','already_complete':True})
        return 0
    (run/'PAUSE').unlink(missing_ok=True)
    run=run_until_stop(root,cfg,token=os.getenv('FINMIND_TOKEN',''),max_seconds=a.minutes*60)
    state=read_json(run/'state.json')
    atomic_json(root/'remote_context.json',{'config':cfg.to_dict(),'code_hash':code_hash(),
                'run_id':os.getenv('GITHUB_RUN_ID','local'),'state':state.get('status')})
    snapshot=export_snapshot(root,out/'V21_DATABASE.zip') if (root/'cache'/'market.sqlite').is_file() else None
    outcome={'status':state.get('status'),'phase':state.get('phase'),
             'done':state.get('done'),'total':state.get('total'),
             'message':state.get('message'),'snapshot':snapshot}
    atomic_json(out/'outcome.json',outcome)
    (out/'STATUS.txt').write_text(json.dumps(outcome,ensure_ascii=False,indent=2),encoding='utf-8')
    for name in ('REPORT.txt','results.json','selected_models.json','LAB_RESULTS.zip'):
        f=run/name
        if f.is_file():
            import shutil
            shutil.copy2(f,out/name)
    summary=os.getenv('GITHUB_STEP_SUMMARY')
    if summary:
        with open(summary,'a',encoding='utf-8') as f:
            f.write('## V21 background checkpoint\n\n```json\n'+json.dumps(outcome,ensure_ascii=False,indent=2)+'\n```\n')
    ghout=os.getenv('GITHUB_OUTPUT')
    if ghout:
        with open(ghout,'a') as f:
            f.write('status='+str(state.get('status','unknown')).lower()+'\n')
            f.write('code_hash='+code_hash()[:12]+'\n')
    print(json.dumps(outcome,ensure_ascii=False))
    return 0 if state.get('status') in ('COMPLETE','PAUSED','RETRY_WAIT') else 2

if __name__=='__main__':raise SystemExit(main())
