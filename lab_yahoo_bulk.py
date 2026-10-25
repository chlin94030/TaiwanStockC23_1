"""Fast no-key historical bootstrap using Yahoo Finance multi-ticker downloads.

V22.2 hardens the V22.1 fast path against *partial* yfinance bulk results.
A yfinance call may return a non-empty DataFrame while many tickers are rate-limited
or have only a sparse slice of history.  V22.1 treated the presence of a column as
success, which could later produce a misleading 1,985/1,985 progress display while
fewer than 500 stocks actually had 252-session turnover history.

This module now:
  1. uses smaller bulk chunks;
  2. writes a versioned cache so old sparse checkpoints are not silently reused;
  3. audits rankable 252-session history at every monthly signal date;
  4. retries only the highest-priority sparse tickers in smaller chunks; and
  5. raises a precise coverage error so AUTO_FAST can fall back to official
     whole-market daily data rather than to 7+ hours of serial FinMind requests.

Yahoo turnover remains a DEVELOPMENT proxy:
    typical_price * volume = ((high + low + close) / 3) * volume
It is never presented as exchange-reported Trading_money.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import time
import numpy as np
import pandas as pd

from lab_io import DataError, atomic_json, sha
from lab_bulk import vectorized_turnover

BENCHMARK = '0050.TW'
CACHE_SCHEMA = 2


def _clean_index(index):
    idx=pd.DatetimeIndex(index)
    if idx.tz is not None:
        try: idx=idx.tz_convert('Asia/Taipei').tz_localize(None)
        except Exception: idx=idx.tz_localize(None)
    return idx.normalize()


def _ticker_frame(downloaded,ticker):
    """Extract one ticker from either yfinance MultiIndex orientation."""
    if downloaded is None or getattr(downloaded,'empty',True):
        return pd.DataFrame()
    d=pd.DataFrame(downloaded)
    if isinstance(d.columns,pd.MultiIndex):
        l0=[str(x) for x in d.columns.get_level_values(0)]
        l1=[str(x) for x in d.columns.get_level_values(1)]
        if ticker in l0:
            try: out=d.xs(ticker,axis=1,level=0,drop_level=True)
            except Exception:return pd.DataFrame()
        elif ticker in l1:
            try: out=d.xs(ticker,axis=1,level=1,drop_level=True)
            except Exception:return pd.DataFrame()
        else:return pd.DataFrame()
    else:
        out=d.copy()
    out.index=_clean_index(out.index)
    return out[~out.index.duplicated(keep='last')].sort_index()


def _raw_adj(frame,ticker):
    if frame is None or frame.empty:return None,None
    aliases={str(c).lower().replace('_',' ').strip():c for c in frame.columns}
    def col(*names):
        for n in names:
            key=n.lower().replace('_',' ').strip()
            if key in aliases:return pd.to_numeric(frame[aliases[key]],errors='coerce')
        return pd.Series(np.nan,index=frame.index,dtype=float)
    op,hi,lo,cl,vol=col('Open'),col('High'),col('Low'),col('Close'),col('Volume')
    ac=col('Adj Close','AdjClose')
    typical=(hi+lo+cl)/3.0
    # Yahoo Finance historical pages expose Taiwan historical Volume in shares.
    # Do not multiply by 1,000 here; the Taiwan quote UI separately displays lots.
    amount=(typical*vol).where((typical>0)&(vol>=0))
    raw=pd.DataFrame({'date':frame.index,'open':op.to_numpy(),'high':hi.to_numpy(),'low':lo.to_numpy(),'close':cl.to_numpy(),'volume':vol.to_numpy(),'amount':amount.to_numpy()}).reset_index(drop=True)
    positive=raw[['open','high','low','close']].gt(0).all(axis=1)
    high_bad=positive & (raw.high < raw[['open','close','low']].max(axis=1)-1e-6)
    low_bad=positive & (raw.low > raw[['open','close','high']].min(axis=1)+1e-6)
    nonfinite=~np.isfinite(raw[['open','high','low','close']]).all(axis=1)
    bad=high_bad|low_bad|nonfinite|(~positive)
    issues=[]
    for ix,row in raw.loc[bad].iterrows():
        issues.append({'ticker':ticker,'date':pd.Timestamp(row.date).strftime('%Y-%m-%d'),
            'kind':'YAHOO_OHLC_RANGE_MISMATCH' if bool(high_bad.loc[ix] or low_bad.loc[ix]) else 'YAHOO_NO_VALID_OHLC',
            'observed':{k:(float(row[k]) if pd.notna(row[k]) and np.isfinite(row[k]) else None)
                        for k in ('open','high','low','close','volume','amount')}})
    if bad.any():raw.loc[bad,['open','high','low','close']]=np.nan
    raw.attrs['quality_issues']=issues
    ratio=(ac/cl.where(cl>0)).where(np.isfinite(ac)&np.isfinite(cl)&(ac>0)&(cl>0))
    adj=pd.DataFrame({'date':frame.index})
    for name,series in [('open',op),('high',hi),('low',lo),('close',cl)]:adj[name]=(series*ratio).to_numpy()
    adj=adj.reset_index(drop=True)
    return raw,adj


def _chunk_key(tickers,start,end):
    payload=('v'+str(CACHE_SCHEMA)+'|'+'|'.join(tickers)+'|'+str(start)+'|'+str(end)).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _coverage_stats(amount,signals,cfg):
    """Audit rankability independently from turnover thresholds.

    A stock is history-complete at a signal only when the trailing 252 market rows
    contain 252 non-missing Yahoo amount observations.  This intentionally matches
    the original V21/V22 pool definition rather than silently relaxing it.
    """
    if amount is None or amount.empty:
        return {'min_history_complete':0,'min_eligible':0,'bad_signals':[], 'eligible_counts':{}}
    x=amount.copy()
    mapping={str(c).split('.')[0]:str(c) for c in x.columns}
    x.columns=[str(c).split('.')[0] for c in x.columns]
    out=vectorized_turnover(x,signals,cfg,mapping)
    eligible=out.groupby('date').eligible.sum().astype(int)
    aligned=amount.sort_index()
    counts=aligned.notna().rolling(252,min_periods=1).sum().reindex(pd.DatetimeIndex(signals))
    complete=counts.ge(252).sum(axis=1).astype(int)
    target=max(cfg.pools)
    bad=[]
    for d in pd.DatetimeIndex(signals):
        ec=int(eligible.get(d,0));hc=int(complete.get(d,0))
        if ec<target:bad.append({'date':d.strftime('%Y-%m-%d'),'eligible':ec,'history_complete':hc})
    return {'min_history_complete':int(complete.min()) if len(complete) else 0,
            'min_eligible':int(eligible.min()) if len(eligible) else 0,
            'bad_signals':bad,
            'eligible_counts':{pd.Timestamp(k).strftime('%Y-%m-%d'):int(v) for k,v in eligible.items()}}


def _replace_if_better(amount,ticker,new_series):
    """Replace cached ticker history only if the retry has more usable rows."""
    if new_series is None or new_series.empty:return False
    s=pd.to_numeric(new_series,errors='coerce').sort_index()
    old=amount[ticker] if ticker in amount.columns else pd.Series(dtype=float)
    if int(s.notna().sum()) <= int(pd.to_numeric(old,errors='coerce').notna().sum()):return False
    idx=amount.index.union(s.index).sort_values()
    amount=amount.reindex(idx)
    amount[ticker]=s.reindex(idx)
    return amount


@dataclass
class YahooFastBulk:
    cfg: object
    progress: object
    check_pause: object
    # 80-ticker chunks were too aggressive for yfinance 0.2.66 on Cloud and
    # could return a non-empty but sparse result.  Smaller chunks trade a little
    # wall-clock time for dramatically better completeness.
    chunk_size: int = 20
    repair_chunk_size: int = 5
    batch_pause: float = 0.20

    provider_name='YAHOO_FAST'
    actual_amount=False

    def _yf(self):
        try:import yfinance as yf
        except Exception as exc:raise DataError('Yahoo fast mode requires yfinance') from exc
        return yf

    def _download(self,tickers,threads=None):
        yf=self._yf();tickers=list(tickers)
        end=(pd.Timestamp(self.cfg.end)+pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        last=None
        for attempt in range(3):
            self.check_pause()
            try:
                nthreads=max(1,min(int(threads or 8),len(tickers)))
                d=yf.download(tickers=tickers,start=self.cfg.start,end=end,auto_adjust=False,
                    actions=False,progress=False,threads=nthreads,group_by='ticker',
                    timeout=self.cfg.request_timeout,keepna=False)
                if d is not None and not d.empty:return d
                last=DataError('Yahoo returned empty batch')
            except Exception as exc:last=exc
            time.sleep(2*(attempt+1))
        raise DataError('Yahoo batch download failed: '+type(last).__name__)

    def calendar(self):
        cached=getattr(self,'_calendar_cache',None)
        if cached is not None:return list(cached)
        try:d=self._download([BENCHMARK],threads=1)
        except TypeError:d=self._download([BENCHMARK])
        f=_ticker_frame(d,BENCHMARK)
        if f.empty:raise DataError('Yahoo benchmark calendar is empty')
        idx=_clean_index(f.index)
        idx=idx[(idx>=pd.Timestamp(self.cfg.start))&(idx<=pd.Timestamp(self.cfg.end))]
        if len(idx)<300:raise DataError('Yahoo benchmark calendar incomplete')
        self._calendar_cache=[x.strftime('%Y-%m-%d') for x in idx]
        return list(self._calendar_cache)

    def _batch_amount(self,chunk,threads=None):
        # Preserve monkeypatch/backward compatibility with one-argument test/download callables.
        d=self._download(chunk) if threads is None else self._download(chunk,threads=threads)
        series={};failed=[]
        for t in chunk:
            f=_ticker_frame(d,t)
            if f.empty:failed.append(t);continue
            raw,_=_raw_adj(f,t)
            if raw is None or raw.empty:failed.append(t);continue
            s=pd.to_numeric(raw.set_index('date')['amount'],errors='coerce');s.name=t
            series[t]=s
        return series,failed

    def _repair_sparse(self,amount,signals,target):
        """Retry sparse histories in small chunks, prioritising liquid names.

        This is bounded.  If it cannot restore enough 252-session histories,
        AUTO_FAST will switch to OFFICIAL_BULK instead of silently looping.
        """
        stats=_coverage_stats(amount,signals,self.cfg)
        if stats['min_eligible']>=target:return amount,stats,0,[]
        sig=pd.DatetimeIndex(signals)
        counts=amount.notna().rolling(252,min_periods=1).sum().reindex(sig)
        deficient=set()
        for d in sig:
            row=counts.loc[d]
            deficient.update(str(t) for t,v in row.items() if pd.isna(v) or float(v)<252)
        # Prioritise names with the highest recent proxy turnover.  They are the
        # most likely to matter for a Top-500 pool, so a bounded repair can stop
        # early once every signal has >= target eligible names.
        recent=amount.tail(min(60,len(amount))).mean(axis=0,skipna=True).fillna(-1)
        ordered=sorted(deficient,key=lambda t:(-float(recent.get(t,-1)),t))
        repairs=0;failed=[]
        chunks=[ordered[i:i+self.repair_chunk_size] for i in range(0,len(ordered),self.repair_chunk_size)]
        # Keep repair bounded so Yahoo rate-limits cannot turn this into another
        # multi-hour path.  1,000 symbols is already enough to reconstruct a
        # Top-500 universe in normal conditions; otherwise use official bulk.
        max_symbols=min(len(ordered),1000)
        seen=0
        for ci,chunk in enumerate(chunks,1):
            if seen>=max_symbols:break
            chunk=chunk[:max_symbols-seen];seen+=len(chunk)
            self.check_pause()
            self.progress(phase='YAHOO_REPAIR',done=seen-len(chunk),total=max_symbols,
                          message=f'Yahoo稀疏歷史修復 {seen-len(chunk)}/{max_symbols}；只重抓缺漏ticker')
            try:series,bad=self._batch_amount(chunk,threads=min(4,len(chunk)))
            except DataError:
                failed.extend(chunk);continue
            failed.extend(bad)
            for t,s in series.items():
                better=_replace_if_better(amount,t,s)
                if isinstance(better,pd.DataFrame):
                    amount=better;repairs+=1
            if ci%10==0 or seen>=max_symbols:
                stats=_coverage_stats(amount,signals,self.cfg)
                self.progress(phase='YAHOO_REPAIR',done=seen,total=max_symbols,
                              message=f'Yahoo修復後最少可排名 {stats["min_eligible"]}/{target}；252日完整史 {stats["min_history_complete"]}')
                if stats['min_eligible']>=target:break
            time.sleep(max(0,float(self.batch_pause)))
        stats=_coverage_stats(amount,signals,self.cfg)
        return amount,stats,repairs,sorted(set(failed))

    def turnover(self,signals,cat,cache_dir):
        # Schema-specific directory means V22.1 sparse Yahoo checkpoints are not
        # reused after upgrading. Other SQLite/raw caches remain untouched.
        root=Path(cache_dir)/f'yahoo_fast_v{CACHE_SCHEMA}';root.mkdir(parents=True,exist_ok=True)
        tickers=sorted({r['ticker'] for r in cat})
        manifest={'provider':'YAHOO_FAST','cache_schema':CACHE_SCHEMA,'start':self.cfg.start,'end':self.cfg.end,
                  'roster_count':len(tickers),'actual_amount':False,'estimated_amount':True,
                  'proxy_formula':'((High+Low+Close)/3)*Volume','development_only':True,
                  'yahoo_chunk_size':self.chunk_size}
        wide_parts=[];available=set();failed=[]
        chunks=[tickers[i:i+self.chunk_size] for i in range(0,len(tickers),self.chunk_size)]
        for ci,chunk in enumerate(chunks,1):
            self.check_pause();key=_chunk_key(chunk,self.cfg.start,self.cfg.end);path=root/f'turnover_{ci:03d}_{key}.csv.gz'
            if path.exists():
                w=pd.read_csv(path,index_col=0,parse_dates=True);w.index=pd.to_datetime(w.index)
                self.progress(phase='YAHOO_FAST',done=ci,total=len(chunks),message=f'Yahoo批次快取 {ci}/{len(chunks)}')
            else:
                self.progress(phase='YAHOO_FAST',done=ci-1,total=len(chunks),message=f'Yahoo批次下載 {ci}/{len(chunks)}（每批最多{self.chunk_size}檔）')
                series,bad=self._batch_amount(chunk);failed.extend(bad)
                if not series:raise DataError('Yahoo batch returned no usable symbols')
                w=pd.concat(series.values(),axis=1).sort_index();w.to_csv(path,index_label='date',compression={'method':'gzip','compresslevel':1})
                self.progress(phase='YAHOO_FAST',done=ci,total=len(chunks),message=f'Yahoo批次已儲存 {ci}/{len(chunks)}')
                time.sleep(max(0,float(self.batch_pause)))
            available.update(str(c) for c in w.columns);wide_parts.append(w)
        amount=pd.concat(wide_parts,axis=1).sort_index();amount=amount.loc[:,~amount.columns.duplicated()]
        # Use the union of returned Taiwan trading dates for this bulk checkpoint.
        # This avoids another Yahoo request merely to obtain a calendar.  With a
        # broad roster the union should contain every market session; the 252-row
        # completeness audit below will catch sparse per-symbol histories.
        calendar=pd.DatetimeIndex(amount.index).sort_values().unique()
        amount=amount.reindex(calendar)
        stats=_coverage_stats(amount,signals,self.cfg)
        repairs=0;repair_failed=[];target=min(max(self.cfg.pools),len(tickers))
        if stats['min_eligible']<target:
            amount,stats,repairs,repair_failed=self._repair_sparse(amount,signals,target)
        # Check broad recent coverage *after* bounded repair. A partially rate-limited
        # first pass is exactly what the repair phase is intended to fix.
        recent=amount.tail(min(60,len(amount))).notna().sum(axis=0).ge(50)
        recent_fraction=float(recent.sum()/max(1,len(tickers)))
        if recent_fraction<0.85:
            raise DataError(f'Yahoo fast recent 60-session coverage only {recent_fraction:.1%} after repair; bulk response is too sparse')
        if stats['min_eligible']<target:
            first=stats['bad_signals'][0] if stats['bad_signals'] else {}
            raise DataError('YAHOO_HISTORY_COVERAGE_INSUFFICIENT: '
                f'{first.get("date","unknown")}: eligible={first.get("eligible",0)}, '
                f'252-session-complete={first.get("history_complete",0)}, required={target}. '
                'Yahoo bulk returned partial histories after bounded repair; use OFFICIAL_BULK fallback.')
        codes={str(r['ticker']).split('.')[0]:r['ticker'] for r in cat}
        amount.columns=[str(c).split('.')[0] for c in amount.columns]
        out=vectorized_turnover(amount,signals,self.cfg,codes)
        manifest.update({'rows':len(out),'recent_symbol_coverage_50_of_60':recent_fraction,
                         'failed_symbols':sorted(set(failed))[:200],
                         'repair_failed_symbols':repair_failed[:200],
                         'repair_replaced_histories':repairs,
                         'min_eligible_at_signal':stats['min_eligible'],
                         'min_252_session_complete_at_signal':stats['min_history_complete'],
                         'bad_signals_after_repair':stats['bad_signals'][:20]})
        meta=root/f'turnover_{self.cfg.start}_{self.cfg.end}.json';atomic_json(meta,manifest);manifest['checkpoint_sha']=sha(meta);atomic_json(meta,manifest)
        return out,manifest

    def populate_adjusted_only(self,store,tickers):
        tickers=sorted(set(tickers));chunks=[tickers[i:i+self.chunk_size] for i in range(0,len(tickers),self.chunk_size)]
        saved=0;missing=[]
        for ci,chunk in enumerate(chunks,1):
            self.check_pause();self.progress(phase='YAHOO_ADJUSTED',done=ci-1,total=len(chunks),message=f'Yahoo還原價批次 {ci}/{len(chunks)}')
            d=self._download(chunk)
            for t in chunk:
                f=_ticker_frame(d,t);_,adj=_raw_adj(f,t)
                if adj is None or adj.empty:missing.append(t);continue
                store.put('adj',t,adj,self.cfg.start,self.cfg.end,'YAHOO_ADJ_CLOSE_RATIO_OHLC');saved+=1
            self.progress(phase='YAHOO_ADJUSTED',done=ci,total=len(chunks),message=f'Yahoo還原價已儲存 {saved}/{len(tickers)}')
            time.sleep(max(0,float(self.batch_pause)))
        if missing:raise DataError('Yahoo adjusted missing symbols: '+','.join(missing[:30]))
        return {'provider':'YAHOO_ADJUSTED_BATCH','symbols':saved}

    def populate_selected_prices(self,store,tickers):
        tickers=sorted(set(tickers));chunks=[tickers[i:i+self.chunk_size] for i in range(0,len(tickers),self.chunk_size)]
        saved=0;missing=[]
        for ci,chunk in enumerate(chunks,1):
            self.check_pause();self.progress(phase='YAHOO_SELECTED',done=ci-1,total=len(chunks),message=f'Yahoo固定母池批次 {ci}/{len(chunks)}')
            d=self._download(chunk)
            for t in chunk:
                f=_ticker_frame(d,t)
                raw,adj=_raw_adj(f,t)
                if raw is None or raw.empty or adj is None or adj.empty:
                    missing.append(t);continue
                store.put('raw',t,raw,self.cfg.start,self.cfg.end,'YAHOO_FAST_PROXY_AMOUNT_RAW_OHLC')
                store.put('adj',t,adj,self.cfg.start,self.cfg.end,'YAHOO_ADJ_CLOSE_RATIO_OHLC')
                saved+=1
            self.progress(phase='YAHOO_SELECTED',done=ci,total=len(chunks),message=f'Yahoo固定母池已儲存 {saved}/{len(tickers)}')
            time.sleep(max(0,float(self.batch_pause)))
        if missing:
            raise DataError('Yahoo selected-union missing symbols: '+','.join(missing[:30]))
        return {'provider':'YAHOO_FAST','symbols':saved,'actual_amount':False,'estimated_amount':True,
                'proxy_formula':'((High+Low+Close)/3)*Volume','development_only':True}
