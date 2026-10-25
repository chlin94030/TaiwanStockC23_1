"""Entire V23.1 report workflow against a SYNTHETIC pinned-data fixture.
Does NOT measure Taiwanese stock returns.
"""
from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).parents[1]))
import v231_runner as vr

class FakeStore:
    def __init__(self,cal):self.cal=cal
    def get(self,table,ticker,start=None,end=None):
        cal=self.cal; rate=.0003 if ticker=='0050.TW' else .0006
        cl=30*np.exp(rate*np.arange(len(cal)))
        op=np.r_[cl[0],cl[:-1]]
        return pd.DataFrame({'date':cal,'open':op,'close':cl})


def make_base():
    cal=pd.bdate_range('2018-01-01',periods=2050)
    dates=pd.DatetimeIndex(pd.Series(cal,index=cal).resample('ME').last().dropna().tolist())
    ticker=[f'{2000+i}.TW' for i in range(250)]
    source={'start':str(cal[0].date()),'end':str(cal[-1].date())}
    meta={'dataset_id':'synthetic-v231-qa','source_config':source,'scope':{'scope':'CURRENT_SNAPSHOT_DIAGNOSTIC'}}
    pool=[];fs=[];stages=[];labels=[]
    for j,d in enumerate(dates):
        stage={'date':d,'pool':'KOSTO250','stage':'A2','stage_confidence':.7}
        stages.append(stage)
        for i,t in enumerate(ticker):
            pool.append({'date':d,'ticker':t,'pool':'KOSTO250','amount20':100_000_000.})
            fs.append({'date':d,'ticker':t,'sector':f'SECT{i%10}',
                       'rsi14':60,'bias20':.03,'momentum20':.012+i*.00005,
                       'momentum120_skip20':.04+i*.0003,'high252':.65+i*.0007,
                       'residual_lowvol60':-.05+i*.00004})
        ci=cal.get_indexer([d])[0]
        for h in (5,20,60):
            if ci+h>=len(cal):continue
            entry=cal[ci+1];exit=cal[ci+h]
            for t in ticker+['0050.TW']:
                lab={'date':d,'ticker':t,'horizon':h,'net':.001*h,
                     'entry':entry,'exit':exit,'filled':True,'state':'ASSUMED_OPEN_FILL'}
                labels.append(lab)
    store=FakeStore(cal)
    return {'meta':meta,'features':pd.DataFrame(fs),'pools':pd.DataFrame(pool),
            'stages':pd.DataFrame(stages),'labels':pd.DataFrame(labels),
            'cal':cal,'store':store,'bench':store.get('adj','0050.TW'),
            'cache_hit':True,'info':{'feature_id':'SYNTH_FIXTURE'},'labels_info':{'source':'SYNTH_FIXTURE'}}


def test_full_integrated_export(tmp_path,monkeypatch):
    base=make_base()
    monkeypatch.setattr(vr,'load_dataset',lambda root:(tmp_path,base['meta']))
    monkeypatch.setattr(vr,'_read_base',lambda root,cfg,progress:base)
    monkeypatch.setattr(vr,'evidence_signature',lambda root:{'revenue':None,'financial':None,'flow':None})
    monkeypatch.setattr(vr,'load_events',lambda root:{k:pd.DataFrame() for k in ('revenue','financial','flow')})
    out=vr.run_research(tmp_path)
    data=json.loads((out/'results.json').read_text(encoding='utf-8'))
    assert len(data['models'])==12
    assert data['holdout_protocol']['final_tail_inspected'] is False
    assert data['genuine_unseen_acceptance'] is False
    assert (out/'V23_1_VALIDATION.xlsx').is_file()
    assert (out/'V23_1_VALIDATION.zip').is_file()
    assert any(x.get('cagr') is not None for x in data['models'] if x['model']=='PRICE_RS_CONTROL')
    import zipfile
    with zipfile.ZipFile(out/'V23_1_VALIDATION.zip') as z:assert z.testzip() is None
    # Identity cache: re-run should not recompute any historical prices.
    assert vr.run_research(tmp_path)==out
