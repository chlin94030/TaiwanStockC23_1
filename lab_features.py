"""Date-T features with explicit units, no label-dependent universe selection."""
import numpy as np
import pandas as pd
from lab_config import FACTORS, OPTIONAL_FACTORS
from lab_io import DataError


def raw_adjusted_frame(raw,adj,calendar,analytical_raw=None):
    from lab_price_quality import field_masks
    idx=pd.DatetimeIndex(calendar)
    r=raw.set_index('date').reindex(idx)
    a=adj.set_index('date').reindex(idx)
    analytical=r if analytical_raw is None else analytical_raw.set_index('date').reindex(idx)
    rm=field_masks(r);am=field_masks(analytical);ym=field_masks(a)
    f=pd.DataFrame(index=idx)
    for col in ('open','high','low','close'):
        # Raw execution fields are NEVER restored from quarantined ledger rows.
        f['raw_'+col]=pd.to_numeric(r[col],errors='coerce').where(rm[col])
        allowed=rm[col] if col=='open' else am[col]
        f['adj_'+col]=pd.to_numeric(a[col],errors='coerce').where(allowed&ym[col])
    f['volume']=pd.to_numeric(r.volume,errors='coerce')
    f['amount']=pd.to_numeric(r.amount,errors='coerce')
    return f


def features(f,benchmark=None):
    a=f.adj_close;hi=f.adj_high;lo=f.adj_low
    ret=a.pct_change(fill_method=None)
    ma20=a.rolling(20,min_periods=20).mean();ma60=a.rolling(60,min_periods=60).mean()
    d=a.diff();up=d.clip(lower=0).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    down=(-d.clip(upper=0)).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    denom=up+down
    rsi=100*up/denom.where(denom>0);rsi=rsi.mask(denom.eq(0),50)
    ema12=a.ewm(span=12,adjust=False,min_periods=12).mean();ema26=a.ewm(span=26,adjust=False,min_periods=26).mean()
    macd=ema12-ema26;hist=macd-macd.ewm(span=9,adjust=False,min_periods=9).mean()
    tr=pd.concat([hi-lo,(hi-a.shift()).abs(),(lo-a.shift()).abs()],axis=1).max(axis=1)
    atr=tr.rolling(14,min_periods=14).mean()
    out=pd.DataFrame(index=f.index)
    out['liquidity20']=np.log1p(f.amount.rolling(20,min_periods=20).mean())
    out['high252']=a/hi.rolling(252,min_periods=252).max()
    out['momentum20']=a/a.shift(20)-1
    out['momentum60']=a/a.shift(60)-1
    out['momentum120_skip20']=a.shift(20)/a.shift(120)-1
    movement=ret.abs().rolling(60,min_periods=60).sum()
    out['trend_eff60']=out.momentum60/movement.where(movement>0)
    out['trend_eff60']=out.trend_eff60.mask(movement.eq(0),0)
    out['downside60']=-np.sqrt(ret.clip(upper=0).pow(2).rolling(60,min_periods=60).mean())
    out['reversal5']=-(a/a.shift(5)-1)
    v5=f.volume.rolling(5,min_periods=5).mean();v60=f.volume.rolling(60,min_periods=60).mean()
    out['volume_impulse']=np.log(v5/v60.where(v60>0))
    span=hi-lo
    clv=((2*a-hi-lo)/span.where(span>0)).mask(span.eq(0),0)
    v20=f.volume.rolling(20,min_periods=20).sum()
    out['money_flow20']=(clv*f.volume).rolling(20,min_periods=20).sum()/v20.where(v20>0)
    out['breakout20']=(a-hi.shift(1).rolling(20,min_periods=20).max())/atr.where(atr>0)
    out['pullback20']=-(a-ma20).abs()/atr.where(atr>0)-(ma20<ma60).astype(float)
    out['rsi_recovery']=rsi.diff(3)*(rsi.shift(3)<50).astype(float)
    out['macd_acceleration']=hist.diff(3)/a
    out['rsi14']=rsi;out['bias20']=a/ma20-1
    # Estimate each stock's exposure using only prior days. A common subtraction
    # would leave cross-sectional ranks unchanged; stock-specific beta does not.
    for name in ('residual_momentum20','residual_momentum60','residual_lowvol60'):
        out[name]=np.nan
    out['benchmark_beta120']=np.nan
    if benchmark is not None:
        b=benchmark.adj_close.reindex(f.index)
        stock_lr=np.log(a.where(a>0)).diff()
        market_lr=np.log(b.where(b>0)).diff()
        paired=stock_lr.notna()&market_lr.notna()
        sx=stock_lr.where(paired);mx=market_lr.where(paired)
        variance=mx.rolling(120,min_periods=108).var()
        beta=sx.rolling(120,min_periods=108).cov(mx)/variance.where(variance>1e-12)
        # At time d, the beta fit ends at d-1. No future prices enter the fit.
        prior_beta=beta.shift(1)
        residual=sx-prior_beta*mx
        out['benchmark_beta120']=prior_beta
        out['residual_momentum20']=residual.rolling(20,min_periods=20).sum()
        out['residual_momentum60']=residual.rolling(60,min_periods=60).sum()
        out['residual_lowvol60']=-residual.rolling(60,min_periods=60).std()
    a20=f.amount.rolling(20,min_periods=20).mean()
    a120=f.amount.rolling(120,min_periods=120).mean()
    out['amount_acceleration']=np.log(a20.where(a20>0)/a120.where(a120>0))
    return out.replace([np.inf,-np.inf],np.nan)


def attach_evidence(frame,evidence):
    """Revisions use their own available_at timestamp. No forward-period matching."""
    e=evidence.copy()
    for c in ('ticker','available_at','factor','value','source'):
        if c not in e:raise DataError('PIT evidence lacks '+c)
    if not set(e.factor).issubset(OPTIONAL_FACTORS):raise DataError('Unsupported optional factor')
    if e.source.fillna('').str.strip().eq('').any():raise DataError('Evidence provenance absent')
    e['available_at']=pd.to_datetime(e.available_at,utc=True)
    e['value']=pd.to_numeric(e.value,errors='coerce')
    if e[['ticker','factor','available_at']].duplicated().any():raise DataError('Ambiguous evidence revision')
    for factor in OPTIONAL_FACTORS:
        frame[factor]=np.nan
        for ticker,group in frame.groupby('ticker'):
            ee=e[(e.ticker==ticker)&(e.factor==factor)].sort_values('available_at')
            if ee.empty:continue
            left=pd.DataFrame({'row':group.index,'time':pd.to_datetime(group.date).dt.tz_localize('Asia/Taipei').dt.tz_convert('UTC')+pd.Timedelta(hours=23,minutes=59)})
            joined=pd.merge_asof(left.sort_values('time'),ee[['available_at','value']],left_on='time',right_on='available_at',direction='backward',tolerance=pd.Timedelta(days=120))
            frame.loc[joined.row,factor]=joined.value.to_numpy()
    return frame


def percentile_reference(frame,cfg):
    """One same-date reference union shared by ALL pools. Not re-ranked inside N."""
    cols=list(FACTORS)+(list(OPTIONAL_FACTORS) if cfg.enable_optional_factors else [])
    out=frame.copy()
    for c in cols:
        # No imputation to zero; candidate-specific known-weight coverage is enforced.
        out['rank_'+c]=out.groupby('date')[c].rank(method='average',pct=True)
    return out


def return_record(f,signal,h,cfg,benchmark=False):
    """T-close decision, T+1 adjusted open entry, T+h adjusted close exit.
    Known no-fill is cash. Missing data remains UNKNOWN. No path/order inference.
    """
    cal=f.index;t=cal.get_indexer([pd.Timestamp(signal)])[0]
    base={'date':pd.Timestamp(signal),'horizon':h,'net':np.nan,'state':'UNKNOWN','filled':False,'path_mdd':np.nan}
    if t<0 or t+h>=len(cal):return dict(base,state='NOT_MATURED')
    e=f.iloc[t+1];x=f.iloc[t+h];prev=f.iloc[t]
    base.update(entry=cal[t+1],exit=cal[t+h])
    if not pd.notna(e.volume):return dict(base,state='ENTRY_VOLUME_UNKNOWN')
    if pd.notna(e.volume) and e.volume==0:
        return dict(base,net=np.nan if benchmark else 0.,state='BENCHMARK_NOT_TRADABLE' if benchmark else 'NO_TRADE_VOLUME',path_mdd=np.nan if benchmark else 0.)
    if not np.isfinite(e.raw_open) or e.raw_open<=0 or not np.isfinite(e.adj_open) or e.adj_open<=0:
        return dict(base,state='ENTRY_PRICE_UNKNOWN')
    if not np.isfinite(prev.adj_close) or prev.adj_close<=0:return dict(base,state='SIGNAL_PRICE_UNKNOWN')
    gap=e.adj_open/prev.adj_close-1
    if not benchmark and gap>cfg.max_gap:
        return dict(base,net=0.,state='PREDECLARED_GAP_NO_FILL',path_mdd=0.)
    base['filled']=True
    if not np.isfinite(x.raw_close) or x.raw_close<=0 or not np.isfinite(x.adj_close) or not pd.notna(x.volume) or x.volume<=0:
        return dict(base,state='EXIT_PRICE_UNKNOWN')
    tax=cfg.benchmark_tax if benchmark else cfg.stock_tax
    gross=x.adj_close/e.adj_open
    net=gross*(1-cfg.fee_sell-tax-cfg.slippage)/(1+cfg.fee_buy+cfg.slippage)-1
    path=f.adj_close.iloc[t+1:t+h+1].to_numpy()
    mdd=np.nan
    if np.isfinite(path).all():
        p=np.r_[e.adj_open,path];mdd=float(np.min(p/np.maximum.accumulate(p)-1))
    return dict(base,net=float(net),state='ASSUMED_OPEN_FILL',path_mdd=mdd)


def label_records(f,signals,h,cfg,benchmark=False):
    """Vectorized counterpart of return_record; all labels share one calendar.
    Prices/features are computed once per stock, not once per model or cutoff.
    """
    cal=f.index;positions=cal.get_indexer(pd.DatetimeIndex(signals))
    rows=[]
    valid=(positions>=0)&(positions+h<len(cal))
    for d in pd.DatetimeIndex(signals)[~valid]:
        rows.append({'date':d,'horizon':h,'net':np.nan,'state':'NOT_MATURED','filled':False,
                     'entry':pd.NaT,'exit':pd.NaT,'path_mdd':np.nan})
    t=positions[valid];dates=pd.DatetimeIndex(signals)[valid]
    if not len(t):return rows
    raw_open=f.raw_open.to_numpy(float);raw_close=f.raw_close.to_numpy(float)
    ao=f.adj_open.to_numpy(float);ac=f.adj_close.to_numpy(float);vol=f.volume.to_numpy(float)
    n=len(t);state=np.full(n,'ASSUMED_OPEN_FILL',dtype=object)
    net=np.full(n,np.nan);filled=np.ones(n,bool);path_mdd=np.full(n,np.nan)
    eo=ao[t+1];xc=ac[t+h];prev=ac[t]
    volume_missing=~np.isfinite(vol[t+1])
    zero=np.isfinite(vol[t+1])&(vol[t+1]==0)
    entry_missing=(~np.isfinite(raw_open[t+1]))|(raw_open[t+1]<=0)|(~np.isfinite(eo))|(eo<=0)
    signal_missing=(~np.isfinite(prev))|(prev<=0)
    with np.errstate(divide='ignore',invalid='ignore'):
        gap=eo/prev-1
    nofill=(gap>cfg.max_gap)&(not benchmark)
    exit_missing=(~np.isfinite(raw_close[t+h]))|(raw_close[t+h]<=0)|(~np.isfinite(xc))|(~np.isfinite(vol[t+h]))|(vol[t+h]<=0)
    tax=cfg.benchmark_tax if benchmark else cfg.stock_tax
    with np.errstate(divide='ignore',invalid='ignore'):
        net=xc/eo*(1-cfg.fee_sell-tax-cfg.slippage)/(1+cfg.fee_buy+cfg.slippage)-1
    state[exit_missing]='EXIT_PRICE_UNKNOWN';net[exit_missing]=np.nan
    state[nofill]='PREDECLARED_GAP_NO_FILL';net[nofill]=0.;filled[nofill]=False
    state[signal_missing]='SIGNAL_PRICE_UNKNOWN';net[signal_missing]=np.nan;filled[signal_missing]=False
    state[entry_missing]='ENTRY_PRICE_UNKNOWN';net[entry_missing]=np.nan;filled[entry_missing]=False
    state[zero]='BENCHMARK_NOT_TRADABLE' if benchmark else 'NO_TRADE_VOLUME'
    net[zero]=np.nan if benchmark else 0.;filled[zero]=False
    state[volume_missing]='ENTRY_VOLUME_UNKNOWN';net[volume_missing]=np.nan;filled[volume_missing]=False
    paths=ac[t[:,None]+np.arange(1,h+1)[None,:]]
    good=(state=='ASSUMED_OPEN_FILL')&np.isfinite(paths).all(axis=1)
    pp=np.column_stack([eo[good],paths[good]])
    if len(pp):path_mdd[good]=np.min(pp/np.maximum.accumulate(pp,axis=1)-1,axis=1)
    path_mdd[np.isfinite(net)&(~filled)]=0.
    for i,d in enumerate(dates):
        rows.append({'date':d,'horizon':h,'net':float(net[i]),'state':state[i],'filled':bool(filled[i]),
                     'entry':cal[t[i]+1],'exit':cal[t[i]+h],'path_mdd':float(path_mdd[i])})
    return rows
