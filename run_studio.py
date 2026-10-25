"""Offline CLI. Never makes network requests or re-optimizes a locked universe."""
import argparse,json
from pathlib import Path
from studio_pool import discover_prepared,adopt_dataset,lock_pool,load_pool
from studio_runner import run_experiment,DEFAULT_SPEC
from lab_io import read_json

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--root',default='data/v21_universe_lab')
    p.add_argument('--action',choices=['adopt','lock','fast','confirm'],default='fast')
    p.add_argument('--source-run');p.add_argument('--pool',default='ADV200')
    p.add_argument('--profiles',default='studio_profiles.json');p.add_argument('--settings',default='studio_settings.json')
    a=p.parse_args(argv)
    if a.action=='adopt':
        choices=discover_prepared(a.root)
        source=a.source_run or (choices[0]['run'] if choices else None)
        if not source:raise ValueError('Restore a prepared V21_DATABASE.zip or run explicit data preparation first')
        out=adopt_dataset(a.root,source)
    elif a.action=='lock':
        from studio_pool import load_dataset
        _,m=load_dataset(a.root);out=lock_pool(a.root,{str(h):a.pool for h in m['source_config']['horizons']})
    else:
        run=run_experiment(a.root,read_json(a.profiles),read_json(a.settings,DEFAULT_SPEC),a.action,resume=True)
        out={'run':str(run),'state':read_json(run/'state.json')}
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
