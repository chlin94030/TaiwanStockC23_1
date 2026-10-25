"""One-time V22 pool candidates built from already-prepared monthly ADV data.
No HTTP, no future returns. The Kostolany pool deliberately avoids using only
short-term hot turnover, because the source material warns about over-heated crowds.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from lab_io import DataError

POOL_RULES=('ADV200','ADV300','KOSTO250','KOSTO300')


def _base500(monthly):
    b=monthly[monthly.pool=='ADV500'].copy()
    if b.empty:raise DataError('V22 requires prepared ADV500 monthly membership for one-time pool construction')
    needed={'date','ticker','amount252','amount20'}
    if not needed.issubset(b):raise DataError('Monthly pool lacks '+str(sorted(needed-set(b))))
    return b


def _kosto_one(g,n):
    if len(g)<500:raise DataError('ADV500 month silently shrank')
    g=g.sort_values(['amount252','ticker'],ascending=[False,True]).copy()
    liq=g.amount252.rank(method='average',pct=True)
    ratio=(g.amount20/g.amount252.where(g.amount252>0)).clip(.05,20)
    # Lower recent/annual turnover is useful only after long-run liquidity is secured;
    # it is a crowding/temperature proxy, not literal strong/weak-hand ownership.
    cold=(-np.log(ratio)).rank(method='average',pct=True)
    stable=(-np.log(ratio).abs()).rank(method='average',pct=True)
    core_n=int(round(n*.70));sat_n=n-core_n
    core=g.assign(score=.75*liq+.25*stable).sort_values(['score','amount252','ticker'],ascending=[False,False,True]).head(core_n)
    remaining=g[~g.ticker.isin(core.ticker)].copy()
    rliq=remaining.amount252.rank(method='average',pct=True)
    rratio=(remaining.amount20/remaining.amount252.where(remaining.amount252>0)).clip(.05,20)
    rcold=(-np.log(rratio)).rank(method='average',pct=True)
    # Satellite retains meaningful long-run capacity but prefers names not already in a hot-turnover extreme.
    sat=remaining.assign(score=.65*rliq+.35*rcold).sort_values(['score','amount252','ticker'],ascending=[False,False,True]).head(sat_n)
    out=pd.concat([core,sat],ignore_index=True)
    if out.ticker.nunique()!=n:raise DataError('Kostolany pool construction failed declared size')
    out=out[['date','ticker','amount252','amount20']].copy();out['pool']='KOSTO'+str(n);out['declared_size']=n
    out['pool_rule']='70% stable liquid core + 30% liquid non-hot satellite; date-T turnover only'
    return out


def build_v22_pools(monthly):
    b=_base500(monthly);rows=[]
    for d,g in b.groupby('date',sort=True):
        for n in (200,300):
            q=g.sort_values(['amount252','ticker'],ascending=[False,True]).head(n)[['date','ticker','amount252','amount20']].copy()
            q['pool']='ADV'+str(n);q['declared_size']=n;q['pool_rule']='trailing 252-session actual turnover control';rows.append(q)
        rows.extend([_kosto_one(g,250),_kosto_one(g,300)])
    out=pd.concat(rows,ignore_index=True)
    counts=out.groupby(['date','pool']).ticker.nunique()
    for (_,p),n in counts.items():
        target=int(''.join(x for x in p if x.isdigit()))
        if n!=target:raise DataError(f'{p} declared {target} but has {n}')
    return out.sort_values(['date','pool','ticker']).reset_index(drop=True)
