"""Generate synthetic engineering fixtures only. Never called by the real-data UI."""
from pathlib import Path
from dataclasses import replace
import argparse
import numpy as np
import pandas as pd
from lab_config import Config
from lab_io import atomic_json,frame_write


def create(directory,stocks=60,sessions=1450):
    root=Path(directory);root.mkdir(parents=True,exist_ok=True)
    cal=pd.bdate_range('2018-01-01',periods=sessions)
    rng=np.random.default_rng(421031)
    market=rng.normal(.00025,.01,sessions)
    frames=[];catalog=[]
    for i in range(stocks+1):
        ticker='0050.TW' if i==stocks else str(1000+i)+('.TW' if i%2==0 else '.TWO')
        if i!=stocks:catalog.append({'ticker':ticker,'name':'SYNTH_'+str(i),'board':'TWSE' if i%2==0 else 'TPEX','sector':'SYNTHETIC'})
        ret=market if i==stocks else .7*market+rng.normal(.00005,.012,sessions)
        close=(30+i)*np.cumprod(1+ret)
        op=np.r_[close[0],close[:-1]]*(1+rng.normal(0,.002,sessions))
        hi=np.maximum(op,close)*1.005;lo=np.minimum(op,close)*.995
        vol=rng.integers(2_000_000,4_000_000,sessions).astype(float)
        f=pd.DataFrame({'ticker':ticker,'date':cal,'raw_open':op,'raw_high':hi,'raw_low':lo,'raw_close':close,
                        'volume':vol,'amount':vol*(op+close)/2,'adj_open':op,'adj_high':hi,'adj_low':lo,'adj_close':close})
        frames.append(f)
    frame_write(root/'prices.csv.gz',pd.concat(frames,ignore_index=True))
    frame_write(root/'calendar.csv',pd.DataFrame({'date':cal}))
    atomic_json(root/'catalog.json',catalog)
    atomic_json(root/'manifest.json',{'data_kind':'SYNTHETIC','amount_kind':'ACTUAL_TRADED_VALUE',
            'amount_unit':'TWD','volume_unit':'SHARES','price_basis':'ADJUSTED_PROVIDER','source':'SYNTHETIC_ENGINE_FIXTURE_NOT_MARKET_DATA',
            'membership_scope':'CURRENT_SNAPSHOT'})
    cfg=Config(start=str(cal[0].date()),end=str(cal[-1].date()),pools=(10,20,30,40),
               candidate_budget=20,min_train_months=12,outer_test_months=3,min_outer_folds=2,
               holdout_months=6,block_bootstrap_samples=100,source_mode='import',capital=100000.)
    cfg.validate();atomic_json(root/'demo_config.json',cfg.to_dict())
    return cfg

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='demo_input');p.add_argument('--stocks',type=int,default=60)
    p.add_argument('--sessions',type=int,default=1450);a=p.parse_args()
    create(a.out,a.stocks,a.sessions)
    print('SYNTHETIC ONLY. Example: python run_lab.py --config '+a.out+'/demo_config.json --input-dir '+a.out+' --root demo_run')
