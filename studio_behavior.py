"""Kostolany-inspired hypotheses, NOT quoted book formulas or proven signals.
Only contemporaneous price/volume proxies are used. Turnover != money supply.
"""
import numpy as np
import pandas as pd

BEHAVIOR_VERSION='ks-proxies-v2-v22-stage-first'
FACTORS=('ks_demand_balance','ks_absorption','ks_confirmed_recovery',
         'ks_heat_safety','ks_quiet_strength','ks_patience',
         'ks_double_bottom','ks_double_top_risk','ks_capitulation_reversal','ks_cold_resilience')


def behavior_features(f):
    c=f.adj_close;hi=f.adj_high;lo=f.adj_low;amount=f.amount
    r=c.pct_change(fill_method=None)
    ma5=c.rolling(5,min_periods=5).mean();ma20=c.rolling(20,min_periods=20).mean()
    ma60=c.rolling(60,min_periods=60).mean()
    a5=amount.rolling(5,min_periods=5).mean();a60=amount.rolling(60,min_periods=60).mean()
    previous_peak=c.rolling(60,min_periods=60).max();dd=c/previous_peak-1
    low20=c.rolling(20,min_periods=20).min();r5=c/c.shift(5)-1
    complete=r.notna()&amount.notna()&c.notna()&hi.notna()&lo.notna()
    upamount=amount.where(r.gt(0),0).where(complete)
    downamount=amount.where(r.lt(0),0).where(complete)
    u=upamount.rolling(20,min_periods=20).sum();d=downamount.rolling(20,min_periods=20).sum()
    out=pd.DataFrame(index=f.index)
    out['ks_demand_balance']=np.log((u+1)/(d+1))
    width=(hi-lo);clv=((2*c-hi-lo)/width.where(width>0)).mask(width.eq(0),0)
    # Turnover on down days with closes well off their lows: absorption proxy,
    # not a claim about actual holder identities or net capital inflow.
    pressure=(amount*clv.where(r.lt(0),0)).where(complete)
    out['ks_absorption']=pressure.rolling(20,min_periods=20).sum()/amount.rolling(20,min_periods=20).sum().where(lambda x:x>0)
    dd_window=dd.rolling(20,min_periods=20).min()
    readiness=dd_window.lt(-.10)&r5.gt(0)&c.gt(ma5)
    recovery=(c/low20-1).clip(0,.25)
    out['ks_confirmed_recovery']=recovery.where(readiness,0).where(low20.notna()&dd_window.notna()&ma5.notna()&r5.notna())
    bias=c/ma20-1;impulse=a5/a60.where(a60>0)
    out['ks_heat_safety']=-(bias.clip(lower=0)*impulse.clip(lower=1))
    out['ks_quiet_strength']=((c/c.shift(20)-1).clip(lower=0)/(impulse.clip(lower=.25))).where(ma60.notna())
    direction=r.gt(0).astype(float).where(r.notna()).rolling(60,min_periods=60).mean()
    out['ks_patience']=direction*(c/c.shift(60)-1).clip(lower=0)
    out['recovery_ready']=readiness.astype(float).where(dd_window.notna()&r5.notna()&ma5.notna())
    out['above_ma60']=c.gt(ma60).astype(float).where(c.notna()&ma60.notna())

    # V22 chart-structure proxies. These are deterministic price/turnover patterns,
    # not literal evidence of hidden support groups or investor identities.
    old_low=c.shift(20).rolling(40,min_periods=40).min()
    new_low=c.rolling(20,min_periods=20).min()
    mid_high=c.shift(5).rolling(35,min_periods=35).max()
    similarity=(1-(new_low/old_low-1).abs()/.08).clip(0,1)
    rebound=((mid_high/np.minimum(old_low,new_low)-1)/.15).clip(0,1)
    recovery=((c/new_low-1)/.12).clip(0,1)
    out['ks_double_bottom']=(similarity*rebound*recovery).where(old_low.notna()&new_low.notna()&mid_high.notna())
    old_high=c.shift(20).rolling(40,min_periods=40).max()
    new_high=c.rolling(20,min_periods=20).max()
    mid_low=c.shift(5).rolling(35,min_periods=35).min()
    top_similarity=(1-(new_high/old_high-1).abs()/.08).clip(0,1)
    drop=((np.maximum(old_high,new_high)/mid_low-1)/.15).clip(0,1)
    rollover=((new_high/c-1)/.10).clip(0,1)
    out['ks_double_top_risk']=(top_similarity*drop*rollover).where(old_high.notna()&new_high.notna()&mid_low.notna())
    a20=amount.rolling(20,min_periods=20).mean();a120=amount.rolling(120,min_periods=120).mean()
    spike=(a20/a120.where(a120>0)).clip(.25,4)
    prior_dd=(c/c.rolling(60,min_periods=60).max()-1)
    capitulation=(spike-1).clip(0,3)/3 * (-prior_dd/.20).clip(0,1) * (r5/.06).clip(0,1)
    out['ks_capitulation_reversal']=capitulation.where(a120.notna()&prior_dd.notna()&r5.notna())
    residual_strength=(c/c.shift(20)-1).clip(lower=0)
    out['ks_cold_resilience']=(residual_strength/(spike.clip(lower=.35))).where(a120.notna())
    return out.replace([np.inf,-np.inf],np.nan)


def benchmark_context(f):
    c=f.adj_close;ret=c.pct_change(fill_method=None)
    ma20=c.rolling(20,min_periods=20).mean();ma200=c.rolling(200,min_periods=200).mean()
    dd=c/c.rolling(60,min_periods=60).max()-1
    vol=ret.rolling(20,min_periods=20).std()
    base=vol.shift(1).rolling(120,min_periods=100).median()
    amount=f.amount
    a5=amount.rolling(5,min_periods=5).mean();a60=amount.rolling(60,min_periods=60).mean();a20=amount.rolling(20,min_periods=20).mean();a120=amount.rolling(120,min_periods=120).mean()
    return pd.DataFrame({'market_drawdown60':dd,'market_bias20':c/ma20-1,
            'market_above200':c.gt(ma200).astype(float).where(ma200.notna()&c.notna()),
            'market_ret5':c/c.shift(5)-1,'market_ret20':c/c.shift(20)-1,'market_ret60':c/c.shift(60)-1,
            'market_amount_ratio5_60':a5/a60.where(a60>0),'market_amount_ratio20_120':a20/a120.where(a120>0),
            'market_vol_ratio':vol/base.where(base>0)},index=f.index)


def regime(row,breadth):
    vals=[row.get(k,np.nan) for k in ('market_drawdown60','market_bias20','market_above200','market_ret5','market_ret20','market_vol_ratio')]
    if not all(np.isfinite(v) for v in vals) or not np.isfinite(breadth):return 'UNKNOWN'
    dd,bias,above,r5,r20,vr=vals
    if dd<-.15 and r5<0 and vr>1.5:return 'PANIC_UNCONFIRMED'
    if dd<-.10 and r5>0 and bias>-.02:return 'RECOVERY_PROXY'
    if bias>.10 and breadth<.55:return 'EUPHORIA_DIVERGENCE'
    if above==1 and r20>0 and breadth>=.55:return 'EXPANSION_PROXY'
    return 'NEUTRAL_PROXY'
