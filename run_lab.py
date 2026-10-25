"""CLI: python run_lab.py --config config.json [--input-dir dataset]"""
import argparse
from pathlib import Path
from lab_config import load_config
from lab_jobs import run_pipeline
from lab_io import read_json

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='config.json');p.add_argument('--root',default='data/v21_universe_lab')
    p.add_argument('--input-dir',default=None)
    a=p.parse_args();cfg=load_config(a.config)
    run=run_pipeline(a.root,cfg,input_dir=a.input_dir)
    state=read_json(Path(run)/'state.json');print(state);print('Output:',run)
    raise SystemExit(0 if state.get('status')=='COMPLETE' else 2)
