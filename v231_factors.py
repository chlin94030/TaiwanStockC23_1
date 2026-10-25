"""Point-in-time fundamental factor builder for V23.1.

All as-of features are reconstructed from dated, revision-preserving disclosures.
Signal is after the Taiwan exchange close, 15:00 Asia/Taipei. A release at
18:00 on signal date cannot be used for next-open orders from that signal.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from lab_io import DataError

QUARTER_MAX_AGE_DAYS=200
REVENUE_MAX_AGE_DAYS=80
FLOW_MAX_AGE_DAYS=28


def _cutoff(date):
    return pd.Timestamp(date).normalize().tz_localize('Asia/Taipei')+pd.Timedelta(hours=15)


def _known_latest(events,cutoff,keys):
    if events is None or events.empty:return events.iloc[0:0].copy()
    known=events.loc[events.available_at<=cutoff.tz_convert('UTC')].sort_values('available_at')
    return known.drop_duplicates(keys,keep='last')


def _consecutive(index, freq, n):
    p=pd.PeriodIndex(index,freq=freq)
    return len(p)==n and p.is_unique and all(p[i].ordinal-p[i-1].ordinal==1 for i in range(1,n))


def monthly_revenue_factors(events, signal_date):
    columns=['ticker','revenue_yoy_3m','revenue_acceleration','revenue_published_at','revenue_quality']
    if events is None or events.empty:return pd.DataFrame(columns=columns)
    cutoff=_cutoff(signal_date)
    x=_known_latest(events,cutoff,['ticker','period'])
    vals=[]
    for ticker,g in x.groupby('ticker',sort=False):
        g=g.sort_values('period')
        current=g.iloc[-1]
        if cutoff.tz_convert('UTC')-current.available_at>pd.Timedelta(days=REVENUE_MAX_AGE_DAYS): continue
        if len(g)<18:continue
        z=g.iloc[-18:].copy()
        if not _consecutive(z.period,'M',18):continue
        r=z.revenue_ntd.to_numpy(dtype=float)
        last3=float(r[-3:].sum());prev_year=float(r[-15:-12].sum())
        older3=float(r[-6:-3].sum());older_year=float(r[-18:-15].sum())
        if prev_year<=0 or older_year<=0:continue
        yoy=last3/prev_year-1
        prev=older3/older_year-1
        if not np.isfinite(yoy) or not np.isfinite(prev):continue
        quality='VERIFIED' if z.quality.eq('VERIFIED').all() else 'NOT_VERIFIED'
        vals.append({'ticker':ticker,'revenue_yoy_3m':yoy,'revenue_acceleration':yoy-prev,
                     'revenue_published_at':current.available_at,'revenue_quality':quality})
    return pd.DataFrame(vals,columns=columns)


def quarterly_financial_factors(events,signal_date):
    columns=['ticker','eps_growth_yoy','gross_margin_delta','earnings_published_at','financial_quality']
    if events is None or events.empty:return pd.DataFrame(columns=columns)
    cutoff=_cutoff(signal_date)
    x=_known_latest(events,cutoff,['ticker','fiscal_quarter'])
    rows=[]
    for ticker,g in x.groupby('ticker',sort=False):
        g=g.sort_values('fiscal_quarter')
        if len(g)<5:continue
        last=g.iloc[-5:]
        if not _consecutive(last.fiscal_quarter,'Q',5):continue
        current=last.iloc[-1]
        if cutoff.tz_convert('UTC')-current.available_at>pd.Timedelta(days=QUARTER_MAX_AGE_DAYS):continue
        p=last.iloc[0]
        # EPS zero/negative base is economically ambiguous; block rather than
        # fabricating extreme apparent growth.
        if p.eps<=0:continue
        eps_growth=float(current.eps/p.eps-1)
        margin_delta=float(current.gross_margin-last.iloc[-2].gross_margin)
        if not np.isfinite(eps_growth+margin_delta):continue
        quality='VERIFIED' if last.quality.eq('VERIFIED').all() else 'NOT_VERIFIED'
        rows.append({'ticker':ticker,'eps_growth_yoy':eps_growth,'gross_margin_delta':margin_delta,
                     'earnings_published_at':current.available_at,'financial_quality':quality})
    return pd.DataFrame(rows,columns=columns)


def institutional_flow_factors(events,signal_date):
    columns=['ticker','institutional_flow5','flow_published_at','flow_quality']
    if events is None or events.empty:return pd.DataFrame(columns=columns)
    cutoff=_cutoff(signal_date)
    x=_known_latest(events,cutoff,['ticker'])
    x=x[(cutoff.tz_convert('UTC')-x.available_at)<=pd.Timedelta(days=FLOW_MAX_AGE_DAYS)]
    if x.empty:return pd.DataFrame(columns=columns)
    x=x.rename(columns={'available_at':'flow_published_at','quality':'flow_quality'})
    return x[columns]


def build_asof_fundamentals(events,date):
    """Strictly backward join, one row/ticker; absent values stay NaN."""
    items=[monthly_revenue_factors(events.get('revenue'),date),
           quarterly_financial_factors(events.get('financial'),date),
           institutional_flow_factors(events.get('flow'),date)]
    full=items[0]
    for q in items[1:]:full=full.merge(q,on='ticker',how='outer',validate='one_to_one')
    return full


def benchmark_rs120(frame,bench,dates):
    """Exact stock 120-session momentum reconstructed from existing month-T 20
    and 120skip20 factors; compare with benchmark's 120-session adjusted close.
    All values end at the signal date. Do not use a future adjusted vintage
    without separate price revision auditing.
    """
    b=bench[['date','close']].dropna().copy()
    b['date']=pd.to_datetime(b.date)
    b=b.sort_values('date').drop_duplicates('date').set_index('date')
    growth=b.close/b.close.shift(120)-1
    q=frame.copy()
    q['stock_growth120']=(1+pd.to_numeric(q.momentum20,errors='coerce'))*(1+pd.to_numeric(q.momentum120_skip20,errors='coerce'))-1
    bg=q.date.map(growth)
    safe=(q.stock_growth120>-0.999)&(bg>-0.999)
    q['rs_0050_120']=np.log1p(q.stock_growth120.where(safe))-np.log1p(bg.where(safe))
    return q
