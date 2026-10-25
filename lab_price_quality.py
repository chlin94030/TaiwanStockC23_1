"""Field-level analytical validity without inventing prices or executable fills.

Older caches quarantined an entire row for an invalid open. When original H/L/C
are still present in the quality ledger and mutually consistent, they may be
used as ANALYTICAL evidence only. The trading raw row remains unchanged.
"""
import numpy as np
import pandas as pd


def analytic_hlc_from_audit(raw, issues):
    out=raw.copy()
    stats={'ledger_rows':len(issues), 'analytical_hlc_restored':0,
           'execution_raw_modified':False, 'prices_interpolated':0}
    if raw.empty:return out,stats
    where={pd.Timestamp(d).normalize():i for i,d in out.date.items()}
    seen=set()
    for issue in issues:
        if issue.get('kind') not in ('OHLC_RANGE_MISMATCH','NO_VALID_OHLC'):continue
        try:
            day=pd.Timestamp(issue['date']).normalize();v=issue['observed']
            high,low,close=[float(v[k]) for k in ('high','low','close')]
        except (TypeError,ValueError,KeyError):continue
        if day not in where or day in seen:continue
        if not np.isfinite([high,low,close]).all() or low<=0 or not low<=close<=high:continue
        i=where[day]
        old=out.loc[i,['high','low','close']].to_numpy(float)
        new=np.array([high,low,close])
        known=np.isfinite(old)
        # A ledger must never overwrite an inconsistent newer provider value.
        if np.any(known & ~np.isclose(old,new,rtol=1e-9,atol=1e-8)):continue
        if not known.all():
            out.loc[i,['high','low','close']]=np.where(known,old,new)
            stats['analytical_hlc_restored']+=1
        seen.add(day)
    return out,stats


def field_masks(raw):
    p={k:pd.to_numeric(raw[k],errors='coerce') for k in ('open','high','low','close')}
    pos={k:np.isfinite(x)&(x>0) for k,x in p.items()}
    hl=pos['high']&pos['low']&p['high'].ge(p['low'])
    close_ok=pos['close'] & (~(pos['high']&p['close'].gt(p['high']+1e-6))) & (~(pos['low']&p['close'].lt(p['low']-1e-6)))
    hlc=hl&close_ok&p['close'].ge(p['low']-1e-6)&p['close'].le(p['high']+1e-6)
    open_ok=hlc&pos['open']&p['open'].ge(p['low']-1e-6)&p['open'].le(p['high']+1e-6)
    return {'close':close_ok,'high':hlc,'low':hlc,'open':open_ok}
