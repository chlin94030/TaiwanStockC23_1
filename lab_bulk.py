"""Whole-market transport helpers for V21.6.

The research definition is unchanged. This module only replaces thousands of
per-symbol HTTP requests with a whole-market table when a licensed FinLab
account is explicitly configured. No credentials are written to disk.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import importlib.util
import os
import numpy as np
import pandas as pd

from lab_io import DataError, frame_write, read_json, atomic_json, sha

FINLAB_ENV = ('FINLAB_REFRESH_TOKEN','FINLAB_SESSION_ID','FINLAB_API_KEY')


def finlab_credentials_present(env=None):
    env = os.environ if env is None else env
    return all(str(env.get(k,'')).strip() for k in FINLAB_ENV)


def finlab_package_present():
    return importlib.util.find_spec('finlab') is not None


def finlab_ready():
    return finlab_credentials_present() and finlab_package_present()


def _slice(df,start,end,columns=None):
    if df is None or getattr(df,'empty',True):
        raise DataError('FinLab returned an empty whole-market table')
    out=pd.DataFrame(df).copy()
    out.index=pd.to_datetime(out.index,errors='coerce')
    out=out[~out.index.isna()].sort_index()
    out=out.loc[(out.index>=pd.Timestamp(start))&(out.index<=pd.Timestamp(end))]
    if columns is not None:
        wanted=[c for c in columns if c in out.columns]
        out=out[wanted]
    return out


def _code(ticker):
    return str(ticker).split('.')[0]


def require_coverage(df,start,end,label='FinLab table',tolerance_days=10):
    idx=pd.DatetimeIndex(df.index)
    if idx.empty:raise DataError(label+' has no dated rows')
    first,last=idx.min(),idx.max();s=pd.Timestamp(start);e=pd.Timestamp(end)
    if first>s+pd.Timedelta(days=tolerance_days):
        raise DataError(f'{label} begins {first.date()}, later than requested start {s.date()}')
    if last<e-pd.Timedelta(days=tolerance_days):
        raise DataError(f'{label} ends {last.date()}, earlier than requested end {e.date()}; current FinLab entitlement may be stale')
    return df


def vectorized_turnover(amount,signals,cfg,roster_codes):
    """Create the same monthly turnover statistics as the per-stock path.

    Activity is inferred from actual traded amount > 0. For valid exchange
    records this is equivalent to traded volume > 0 and avoids downloading a
    second full-market matrix merely to decide whether a stock traded.
    Missing amount stays missing; it is never converted to zero.
    """
    amount=_slice(amount,cfg.start,cfg.end,list(roster_codes))
    amount=amount.apply(pd.to_numeric,errors='coerce')
    annual=amount.rolling(252,min_periods=252).mean()
    a20=amount.rolling(20,min_periods=20).mean()
    a60=amount.rolling(60,min_periods=60).mean()
    active=amount.gt(0).where(amount.notna()).rolling(60,min_periods=60).mean()
    sig=pd.DatetimeIndex(signals)
    frames=[]
    for name,wide in [('amount252',annual),('amount20',a20),('amount60',a60),('active60',active)]:
        x=wide.reindex(sig)
        x.index.name='date'
        long=x.reset_index().melt('date',var_name='code',value_name=name)
        frames.append(long)
    out=frames[0]
    for f in frames[1:]:out=out.merge(f,on=['date','code'],how='outer',validate='one_to_one')
    out['ticker']=out.code.map(roster_codes)
    out=out[out.ticker.notna()].drop(columns='code')
    out['eligible']=out.amount252.gt(0)&out.amount20.ge(cfg.recent_amount_floor)&out.active60.ge(cfg.active_ratio60)
    return out.sort_values(['date','ticker']).reset_index(drop=True)


@dataclass
class FinLabBulk:
    cfg: object
    progress: object
    check_pause: object

    def _data(self):
        if not finlab_package_present():
            raise DataError('FinLab bulk selected but finlab package is not installed')
        if not finlab_credentials_present():
            raise DataError('FinLab bulk selected but FINLAB_REFRESH_TOKEN / FINLAB_SESSION_ID / FINLAB_API_KEY are not all set')
        from finlab import data
        data.set_market('tw')
        try:data.reset_universe()
        except Exception:pass
        return data

    def _get(self,key):
        self.check_pause()
        data=self._data()
        try:
            return data.get(key)
        except Exception as exc:
            raise DataError('FinLab whole-market table unavailable: '+key+' | '+type(exc).__name__) from exc

    def amount_table(self):
        cached=getattr(self,'_amount_cache',None)
        if cached is not None:return cached
        self.progress(phase='BULK_AMOUNT',message='FinLab whole-market traded-value table: one bulk download, not ~2,000 symbol requests')
        self._amount_cache=require_coverage(_slice(self._get('price:成交金額'),self.cfg.start,self.cfg.end),self.cfg.start,self.cfg.end,'FinLab price:成交金額')
        return self._amount_cache

    def turnover(self,signals,cat,cache_dir):
        cache_dir=Path(cache_dir);cache_dir.mkdir(parents=True,exist_ok=True)
        path=cache_dir/f'finlab_turnover_{self.cfg.start}_{self.cfg.end}.csv.gz'
        meta=cache_dir/f'finlab_turnover_{self.cfg.start}_{self.cfg.end}.json'
        codes={_code(r['ticker']):r['ticker'] for r in cat}
        old=read_json(meta)
        if path.exists() and old.get('roster_count')==len(codes) and old.get('start')==self.cfg.start and old.get('end')==self.cfg.end:
            self.progress(phase='BULK_AMOUNT',message='Reusing cached whole-market turnover checkpoint')
            return pd.read_csv(path,parse_dates=['date']),old
        amount=self.amount_table()
        out=vectorized_turnover(amount,signals,self.cfg,codes)
        frame_write(path,out)
        info={'provider':'FINLAB_BULK','start':self.cfg.start,'end':self.cfg.end,'roster_count':len(codes),
              'rows':len(out),'actual_amount':True,'estimated_amount':False}
        atomic_json(meta,info);info['checkpoint_sha']=sha(path);atomic_json(meta,info)
        self._amount_cache=None
        try:
            from finlab import data
            data.free_memory()
        except Exception:pass
        return out,info

    def calendar(self):
        amount=self.amount_table()
        return [d.strftime('%Y-%m-%d') for d in pd.DatetimeIndex(amount.index)]

    def populate_selected_prices(self,store,tickers):
        """Fetch raw whole-market tables once, slice to the fixed union, then store.

        Adjusted OHLC uses FinLab adjusted close / raw close as the daily ratio.
        Actual amount/volume stay unadjusted exchange quantities.
        """
        tickers=sorted(set(tickers));codes=[_code(t) for t in tickers]
        ticker_by_code={_code(t):t for t in tickers}
        fields={'open':'price:開盤價','high':'price:最高價','low':'price:最低價','close':'price:收盤價',
                'volume':'price:成交股數','amount':'price:成交金額','adj_close':'etl:adj_close'}
        parts={}
        for i,(name,key) in enumerate(fields.items(),1):
            self.progress(phase='BULK_PRICES',done=i-1,total=len(fields),message=f'FinLab bulk table {i}/{len(fields)}: {name}')
            parts[name]=require_coverage(_slice(self._get(key),self.cfg.start,self.cfg.end,codes),self.cfg.start,self.cfg.end,'FinLab '+key)
        idx=pd.DatetimeIndex([])
        for x in parts.values():idx=idx.union(pd.DatetimeIndex(x.index))
        for i,code in enumerate(codes):
            self.check_pause();t=ticker_by_code[code]
            f=pd.DataFrame(index=idx)
            for name,x in parts.items():f[name]=pd.to_numeric(x[code],errors='coerce').reindex(idx) if code in x else np.nan
            f.index.name='date';f=f.reset_index()
            raw=f[['date','open','high','low','close','volume','amount']].copy()
            ratio=f.adj_close/f.close.where(f.close>0)
            ratio=ratio.where(np.isfinite(ratio)&(ratio>0))
            adj=pd.DataFrame({'date':f.date})
            for c in ('open','high','low','close'):adj[c]=f[c]*ratio
            # Match the FinMind quarantine policy: preserve actual amount/volume,
            # never invent OHLC.  The audit ledger lets analytical H/L/C be
            # recovered later where independently consistent; execution opens stay unknown.
            positive=raw[['open','high','low','close']].gt(0).all(axis=1)
            high_bad=positive & (raw.high < raw[['open','close','low']].max(axis=1)-1e-6)
            low_bad=positive & (raw.low > raw[['open','close','high']].min(axis=1)+1e-6)
            nonfinite=~np.isfinite(raw[['open','high','low','close']]).all(axis=1)
            bad=high_bad|low_bad|nonfinite|(~positive)
            issues=[]
            for ix,row in raw.loc[bad].iterrows():
                issues.append({'ticker':t,'date':pd.Timestamp(row.date).strftime('%Y-%m-%d'),
                    'kind':'OHLC_RANGE_MISMATCH' if bool(high_bad.loc[ix] or low_bad.loc[ix]) else 'NO_VALID_OHLC',
                    'observed':{k:(float(row[k]) if pd.notna(row[k]) and np.isfinite(row[k]) else None)
                                for k in ('open','high','low','close','volume','amount')}})
            if bad.any():raw.loc[bad,['open','high','low','close']]=np.nan
            raw.attrs['quality_issues']=issues
            store.put('raw',t,raw,self.cfg.start,self.cfg.end,'FINLAB_BULK_ACTUAL_AMOUNT_RAW_OHLC')
            store.put('adj',t,adj,self.cfg.start,self.cfg.end,'FINLAB_ETL_ADJ_CLOSE_RATIO_OHLC')
            if i%25==0 or i+1==len(codes):
                self.progress(phase='BULK_PRICES',done=i+1,total=len(codes),last_committed_symbol=t,
                              message=f'FinLab bulk selected-union prices committed: {i+1}/{len(codes)}')
        try:
            from finlab import data
            data.free_memory()
        except Exception:pass
        return {'provider':'FINLAB_BULK','symbols':len(codes),'tables':list(fields.values())}
