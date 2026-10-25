"""Deterministic offline tests. Synthetic prices only, no claimed market returns."""
from __future__ import annotations
import io
import sys
from pathlib import Path
import tempfile
import pandas as pd
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]))
from lab_io import DataError
from v231_data import validate_events,persist_events,load_events
from v231_factors import build_asof_fundamentals, benchmark_rs120
from v231_engine import V23Config,select_asof,simulate_nav,choose_cohorts
from v231_mops import parse_html


def monthly_rows(last='2024-07'):
    periods=pd.period_range(end=last,periods=20,freq='M')
    rows=[]
    for i,p in enumerate(periods):
        date=p.to_timestamp(how='end').normalize()+pd.Timedelta(days=10)
        rows.append(dict(ticker='2330.TW',period=str(p),revenue_ntd=float(1000+40*i),
                         available_at=date.strftime('%Y-%m-%d')+'T18:00:00+08:00',source='fixture-filing',quality='VERIFIED'))
    return pd.DataFrame(rows)


def financial_rows():
    rows=[]
    for i,p in enumerate(pd.period_range(end='2024Q2',periods=6,freq='Q')):
        date=p.to_timestamp(how='end').normalize()+pd.Timedelta(days=40)
        rows.append(dict(ticker='2330.TW',fiscal_quarter=str(p),eps=1+i*.3,gross_margin=.4+i*.01,
                         available_at=date.strftime('%Y-%m-%d')+'T18:00:00+08:00',source='fixture-filing',quality='VERIFIED'))
    return pd.DataFrame(rows)


def test_pit_15_oclock_no_after_close_knowledge():
    revenue=validate_events(monthly_rows(),'revenue')
    x=build_asof_fundamentals({'revenue':revenue,'financial':pd.DataFrame(),'flow':pd.DataFrame()},'2024-07-10')
    assert not x.empty
    assert x.iloc[0].revenue_published_at < pd.Timestamp('2024-07-10T15:00:00+08:00').tz_convert('UTC')
    after=build_asof_fundamentals({'revenue':revenue,'financial':pd.DataFrame(),'flow':pd.DataFrame()},'2024-07-11')
    assert after.iloc[0].revenue_published_at >= pd.Timestamp('2024-07-10T18:00:00+08:00').tz_convert('UTC')


def test_verified_requires_timezone():
    data=monthly_rows();data.loc[0,'available_at']='2023-01-10 18:00:00'
    with pytest.raises(DataError,match='時區'):validate_events(data,'revenue')


def test_revision_preservation_duplicate_reject(tmp_path):
    f=validate_events(monthly_rows(),'revenue')
    persist_events(tmp_path,f,'revenue')
    assert len(load_events(tmp_path)['revenue'])==20
    persist_events(tmp_path,f,'revenue')
    assert len(load_events(tmp_path)['revenue'])==20
    fake=f.copy();fake.loc[0,'revenue_ntd']+=999
    with pytest.raises(DataError,match='不同'):persist_events(tmp_path,fake,'revenue')
    amendment=f.iloc[[0]].copy();amendment['available_at']='2024-08-20T18:00:00+08:00'
    amendment['revenue_ntd']+=1000
    persist_events(tmp_path,amendment,'revenue')
    assert len(load_events(tmp_path)['revenue'])==21


def test_financial_consecutive_and_quality():
    x=validate_events(financial_rows(),'financial')
    q=build_asof_fundamentals({'revenue':pd.DataFrame(),'financial':x,'flow':pd.DataFrame()},'2024-06-01')
    assert (not q.empty) and np.isfinite(q.iloc[0].eps_growth_yoy)
    assert q.iloc[0].financial_quality=='VERIFIED'


def make_panel(n=80):
    df=pd.DataFrame({'ticker':[f'{2000+i}.TW' for i in range(n)], 'sector':[f'S{i%8}' for i in range(n)],
        'amount20':[100_000_000.]*n,'rsi14':[60.]*n,'bias20':[.04]*n,
        'rs_0050_120':np.linspace(-.15,.25,n),'momentum120_skip20':np.linspace(-.1,.15,n),
        'high252':np.linspace(.5,.99,n),'residual_lowvol60':np.linspace(-.06,-.01,n)})
    return df


def test_missing_fundamentals_never_silent_alpha():
    d=make_panel();cfg=V23Config()
    price=select_asof(d,'PRICE_RS_CONTROL','A2',cfg)
    assert price['state']=='PICKS_READY' and len(price['selected'])==10 and price['target_stock_fraction']==.65
    growth=select_asof(d,'GROWTH_RS_40_35_15_10','A2',cfg)
    assert growth['state']=='FUNDAMENTAL_COVERAGE_BLOCKED' and growth['target_stock_fraction']==0


def test_growth_factor_weights_and_sector_cap():
    x=make_panel()
    for k in ('revenue_yoy_3m','revenue_acceleration','eps_growth_yoy','gross_margin_delta','institutional_flow5'):
        x[k]=np.linspace(-.3,.5,len(x))
    r=select_asof(x,'GROWTH_RS_40_35_15_10','A2',V23Config())
    assert r['state']=='PICKS_READY' and len(r['selected'])==10
    assert max([sum(p['sector']==s for p in r['selected']) for s in set(x.sector)])<=3
    assert abs(sum(r['factors'].values())-1)<1e-9


def test_no_overlapping_capital():
    d=pd.DataFrame({'date':pd.to_datetime(['2024-01-02','2024-01-31','2024-02-29']),
       'entry':pd.to_datetime(['2024-01-03','2024-02-01','2024-03-01']),
       'exit':pd.to_datetime(['2024-03-28','2024-04-20','2024-05-01']),
       'selected':[[],[],[]],'target_stock_fraction':[.5,.5,.5]})
    a=choose_cohorts(d,60)
    assert list(a.active_cohort)==[True,False,False]


def test_benchmark_only_exactly_tracks_etf():
    cal=pd.bdate_range('2024-01-01',periods=30)
    bench=pd.DataFrame({'date':cal,'open':100*np.exp(.001*np.arange(30)),
                       'close':100*np.exp(.001*np.arange(30))})
    daily,mt=simulate_nav(cal,bench,{},pd.DataFrame(),V23Config())
    assert abs(mt['cagr']-mt['benchmark_cagr'])<1e-12
    assert abs(mt['mdd']-mt['benchmark_mdd'])<1e-12
    assert np.max(np.abs(daily.strategy_nav-daily.benchmark_nav))<1e-12


def test_stock_gain_and_cost_stress_decrease_nav():
    cal=pd.bdate_range('2024-01-01',periods=70)
    flat=pd.DataFrame({'date':cal,'open':np.full(len(cal),100.),'close':np.full(len(cal),100.)})
    close=30*(1.004)**np.arange(len(cal));opening=np.r_[close[0],close[:-1]]
    stock=pd.DataFrame({'date':cal,'open':opening,'close':close})
    start=cal[10];end=cal[29]
    picks=[{'ticker':'2000.TW','score':.8,'sector':'X'}]
    cohort=pd.DataFrame([{'date':cal[9],'entry':start,'exit':end,
                          'selected':picks,'target_stock_fraction':.50,'active_cohort':True}])
    normal,metrics=simulate_nav(cal,flat,{'2000.TW':stock},cohort,V23Config())
    stressed,ms=simulate_nav(cal,flat,{'2000.TW':stock},cohort,V23Config(),stress=True)
    assert normal.strategy_nav.iloc[-1]>1
    assert stressed.strategy_nav.iloc[-1]<normal.strategy_nav.iloc[-1]
    assert normal.positions.max()==1
    assert normal.positions.iloc[-1]==0


def test_missing_held_price_fail_closed():
    cal=pd.bdate_range('2024-01-01',periods=40)
    flat=pd.DataFrame({'date':cal,'open':[100.]*40,'close':[100.]*40})
    broken=flat.copy();broken.loc[19,'close']=np.nan
    cohort=pd.DataFrame([{'date':cal[9],'entry':cal[10],'exit':cal[30],
                          'selected':[{'ticker':'2000.TW'}],'target_stock_fraction':.5,'active_cohort':True}])
    with pytest.raises(DataError,match='價格缺漏'):
        simulate_nav(cal,flat,{'2000.TW':broken},cohort,V23Config())


def test_mops_proxy_not_per_company_disclosure():
    html='<table><tr><th>公司代號</th><th>當月營收</th></tr>'+\
         ''.join(f'<tr><td>{2000+i}</td><td>1,000</td></tr>' for i in range(100))+'</table>'
    out=parse_html(html,2024,2,'sii','https://doc.twse.com.tw/example')
    assert len(out)==100 and set(out.quality)=={'PROXY'}
    assert out.iloc[0].available_at=='2024-03-20T15:30:00+08:00'


def test_benchmark_rs_from_existing_features():
    cal=pd.bdate_range('2023-01-01',periods=170)
    close=100*np.exp(.0003*np.arange(len(cal)))
    b=pd.DataFrame({'date':cal,'close':close})
    t=cal[130]
    stock20=.03;stockskip=.12
    s=pd.DataFrame({'ticker':['2330.TW'],'date':[t],'momentum20':[stock20],
                    'momentum120_skip20':[stockskip]})
    q=benchmark_rs120(s,b,[t])
    expected=np.log((1+stock20)*(1+stockskip))-np.log(close[130]/close[10])
    assert abs(q.iloc[0].rs_0050_120-expected)<1e-12


def test_v23_json_release_is_versioned_and_cache_invalidated(tmp_path):
    import json
    from v231_release import bundled,parse,persist,load
    baseline=bundled()
    assert parse(json.dumps(baseline).encode('utf-8'))==baseline
    invalid=baseline.copy();invalid['requires_engine']='22.4.1-guided'
    with pytest.raises(DataError,match='版本不符'):parse(json.dumps(invalid))
    modified=json.loads(json.dumps(baseline));modified['release_id']='R231-CANDIDATE-A'
    modified['models']['PRICE_RS_CONTROL']['rs_0050_120']=.46
    modified['models']['PRICE_RS_CONTROL']['high252']=.19
    modified.pop('release_sha',None)
    stored=persist(tmp_path,modified)
    assert stored['release_sha']!=baseline['release_sha']
    assert load(tmp_path)['release_id']=='R231-CANDIDATE-A'
    corrupted=dict(stored);corrupted['release_sha']='0000000000000000'
    with pytest.raises(DataError,match='SHA'):parse(json.dumps(corrupted))


def test_v23_checkpoint_allowlist():
    from lab_snapshot import allowed
    assert allowed('v23/evidence/revenue.csv.gz')
    assert allowed('v23/evidence/financial.csv.gz')
    assert allowed('v23/active_strategy_release.json')
    assert allowed('v23/mops_job_state.json')
    assert allowed('v23/runs/0123456789abcdef01234567/V23_1_VALIDATION.zip')
    assert not allowed('v23/../../.streamlit/secrets.toml')
    assert not allowed('v23/evidence/anything.py')
    assert not allowed('v23/runs/bad/evil.pickle')


def test_v23_engine_selfcheck_offline(tmp_path):
    from v231_health import selfcheck
    r=selfcheck(tmp_path,Path(__file__).parents[1])
    assert r['result']=='PASS'
    assert r['etf_only_vs_0050_excess_pp']==0.0
