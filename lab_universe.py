"""Monthly date-T universes, actual rolling turnover, no future-selected membership."""
import numpy as np
import pandas as pd
from lab_io import DataError


def signal_calendar(calendar,cfg):
    cal=pd.DatetimeIndex(calendar)
    # Last observed session of each FULL month. Do not treat a mid-month as month end.
    rows=pd.Series(cal,index=cal).groupby(cal.to_period('M')).last()
    last_period=pd.Timestamp(cfg.end).to_period('M')
    rows=rows[rows.index < last_period]
    values=[pd.Timestamp(d) for d in rows if cal.get_loc(d)>=cfg.lookback-1 and cal.get_loc(d)+max(cfg.horizons)<len(cal)]
    return pd.DatetimeIndex(values)


def turnover_snapshots(raw,calendar,signals,cfg):
    """One stock at a time. Missing amount is not zero; real zero turnover is included."""
    f=raw.set_index('date').reindex(pd.DatetimeIndex(calendar))
    amount=pd.to_numeric(f.amount,errors='coerce')
    volume=pd.to_numeric(f.volume,errors='coerce')
    annual=amount.rolling(252,min_periods=252).mean()
    a20=amount.rolling(20,min_periods=20).mean()
    a60=amount.rolling(60,min_periods=60).mean()
    act=volume.gt(0).where(volume.notna()).rolling(60,min_periods=60).mean()
    out=pd.DataFrame({'date':f.index,'amount252':annual.values,'amount20':a20.values,
                      'amount60':a60.values,'active60':act.values})
    out=out[out.date.isin(signals)].copy()
    out['eligible']=out.amount252.gt(0)&out.amount20.ge(cfg.recent_amount_floor)&out.active60.ge(cfg.active_ratio60)
    return out


def validate_membership(members):
    needed={'date','ticker','available_at','source'}
    if not needed.issubset(members):raise DataError('membership.csv requires date,ticker,available_at,source')
    f=members.copy();f['date']=pd.to_datetime(f.date);f['available_at']=pd.to_datetime(f.available_at,utc=True)
    cutoff=(f.date+pd.Timedelta(hours=23,minutes=59)).dt.tz_localize('Asia/Taipei').dt.tz_convert('UTC')
    if (f.available_at>cutoff).any():raise DataError('Membership was not available by its snapshot date')
    if f[['date','ticker']].duplicated().any():raise DataError('Duplicate membership rows')
    if f.source.fillna('').str.strip().eq('').any():raise DataError('Membership source absent')
    return f


def make_pools(turnover,signals,cfg,membership=None,caps=None):
    """Pool membership is fixed BEFORE fetching future adjusted outcomes.
    Every requested pool must contain its declared N. No silent shrink to 150.
    Optional market cap is dated supplied data, never current shares x past prices.
    """
    m=validate_membership(membership) if membership is not None else None
    if cfg.requested_scope=='HISTORICAL_MEMBERSHIP' and m is None:raise DataError('Historical membership required')
    if cfg.include_market_cap_pools and caps is None:raise DataError('market_caps.csv required; paid-in capital is not market cap')
    if caps is not None:
        caps=caps.copy()
        for c in ('date','ticker','market_cap','available_at','source'):
            if c not in caps:raise DataError('Market-cap metadata absent: '+c)
        caps['date']=pd.to_datetime(caps.date);caps['available_at']=pd.to_datetime(caps.available_at,utc=True)
        if caps[['date','ticker']].duplicated().any():raise DataError('Duplicate cap snapshot')
    pools=[];diagnostics=[]
    for d in signals:
        f=turnover[(turnover.date==d)&turnover.eligible].copy()
        expected=None
        if m is not None:
            mm=m[m.date==d]
            if mm.empty:raise DataError('No exact monthly historical membership for '+str(d.date()))
            expected=len(mm);f=f[f.ticker.isin(mm.ticker)]
        f=f.sort_values(['amount252','ticker'],ascending=[False,True])
        if len(f)<max(cfg.pools):
            raise DataError(f'{d.date()}: only {len(f)} eligible turnover histories, cannot claim Top-{max(cfg.pools)}. No pool shrink performed.')
        f['amount_rank']=np.arange(1,len(f)+1)
        for n in cfg.pools:
            a=f.head(n)[['date','ticker','amount252','amount20','amount_rank']].copy()
            a['pool']='ADV'+str(n);a['declared_size']=n;pools.append(a)
        if cfg.include_market_cap_pools:
            cc=caps[caps.date==d].copy()
            at=(pd.Timestamp(d)+pd.Timedelta(hours=23,minutes=59)).tz_localize('Asia/Taipei').tz_convert('UTC')
            if (cc.available_at>at).any():raise DataError('Future market-cap availability')
            merged=f.merge(cc[['ticker','market_cap']],on='ticker',how='left')
            # Unobserved caps could reorder the top list: do not ignore them.
            if not merged.market_cap.gt(0).all():raise DataError('Incomplete monthly market caps; cap pools blocked')
            merged=merged.sort_values(['market_cap','ticker'],ascending=[False,True])
            for n in cfg.pools:
                a=merged.head(n)[['date','ticker','amount252','amount20','amount_rank','market_cap']].copy()
                a['pool']='MCAP'+str(n);a['declared_size']=n;pools.append(a)
        diagnostics.append({'date':d,'rankable_turnover_stocks':len(f),'historical_members':expected,
                            'method':'TRAILING_252_ACTUAL_AMOUNT_MONTHLY_NEXT_OPEN'})
    if not pools:raise DataError('No monthly pools')
    return pd.concat(pools,ignore_index=True),pd.DataFrame(diagnostics)
