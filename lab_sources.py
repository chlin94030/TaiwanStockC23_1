"""Current official roster + actual per-stock turnover; adjusted prices kept separate.
No sponsor-only datasets, bypasses, raw-price return fallback, or fake 150-stock market.
"""
import os
import re
import time
from datetime import timedelta
import numpy as np
import pandas as pd
import requests
from lab_io import DataError, SourceBlocked, SourceDeferred

FINMIND='https://api.finmindtrade.com/api/v4/data'
TWSE='https://openapi.twse.com.tw/v1/opendata/t187ap03_L'
TPEX='https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O'

def number(s):
    if s is None:return np.nan
    return pd.to_numeric(str(s).replace(',','').strip(),errors='coerce')

def parse_roster(rows,board):
    if not isinstance(rows,list):raise DataError('Official roster is not a list')
    out=[]
    for row in rows:
        code=str(row.get('\u516c\u53f8\u4ee3\u865f',row.get('SecuritiesCompanyCode',''))).strip()
        if not re.fullmatch(r'[1-9][0-9]{3}',code):continue
        name=str(row.get('\u516c\u53f8\u7c21\u7a31',row.get('\u516c\u53f8\u540d\u7a31',code)))
        sector=str(row.get('\u7522\u696d\u5225','UNKNOWN'))
        out.append({'ticker':code+('.TW' if board=='TWSE' else '.TWO'),
                    'name':name,'sector':sector,'board':board})
    if len({x['ticker'] for x in out})!=len(out):raise DataError('Duplicate official ticker')
    return out

def parse_raw(payload,ticker,price_policy="strict"):
    if price_policy not in ("strict", "quarantine"):
        raise ValueError("Unknown price policy")
    if not isinstance(payload,dict) or int(payload.get('status',0))!=200:raise DataError('Invalid raw price payload')
    f=pd.DataFrame(payload.get('data') or [])
    cols=['date','open','high','low','close','volume','amount']
    if f.empty:return pd.DataFrame(columns=cols)
    expected={'date','stock_id','open','max','min','close','Trading_Volume','Trading_money'}
    if not expected.issubset(f):raise DataError('Actual Trading_money is required; price times volume is not substituted')
    if not f.stock_id.astype(str).eq(ticker.split('.')[0]).all():raise DataError('Raw symbol mismatch')
    f=f.rename(columns={'max':'high','min':'low','Trading_Volume':'volume','Trading_money':'amount'})[cols]
    f['date']=pd.to_datetime(f.date,errors='raise')
    for c in cols[1:]:f[c]=f[c].map(number)
    if (f[['volume','amount']].dropna()<0).any().any():raise DataError('Negative actual amount or shares')
    if price_policy=='quarantine' and not np.isfinite(f[['volume','amount']]).all().all():
        raise DataError('Missing or nonfinite actual amount/shares; full-market ranking must wait')
    # Official no-price rows remain zero/NaN; never forward fill them.
    if f.date.duplicated().any():raise DataError('Duplicate raw dates')
    positive=f[['open','high','low','close']].gt(0).all(axis=1)
    high_bad=positive & (f.high < f[['open','close','low']].max(axis=1)-1e-6)
    low_bad=positive & (f.low > f[['open','close','high']].min(axis=1)+1e-6)
    if price_policy == 'strict':
        if high_bad.any():raise DataError('OHLC high mismatch')
        if low_bad.any():raise DataError('OHLC low mismatch')
    else:
        # Historical emerging-market open may be yesterday's average, NOT an
        # executable opening price. Never widen H/L or invent an open to fit.
        # Keep verified actual amount/shares; quarantine the whole OHLC row.
        nonfinite=~np.isfinite(f[['open','high','low','close']]).all(axis=1)
        no_price=~positive
        bad=high_bad | low_bad | nonfinite | no_price
        issues=[]
        for ix,row in f.loc[bad].iterrows():
            kind='OHLC_RANGE_MISMATCH' if high_bad.loc[ix] or low_bad.loc[ix] else 'NO_VALID_OHLC'
            issues.append({'ticker':ticker,'date':row.date.strftime('%Y-%m-%d'),
                'kind':kind, 'observed':{k:(float(row[k]) if pd.notna(row[k]) and np.isfinite(row[k]) else None)
                for k in ('open','high','low','close','volume','amount')}})
        if bad.any():f.loc[bad,['open','high','low','close']]=np.nan
        f.attrs['quality_issues']=issues
        f.attrs['price_policy']='quarantine_invalid_ohlc_preserve_actual_amount'

    return f.sort_values('date')

def parse_yahoo_history(f):
    if f is None or f.empty:raise DataError('No adjusted price history')
    required={'Open','High','Low','Close','Adj Close'}
    if not required.issubset(f):raise DataError('Adjusted close absent; raw returns are not substituted')
    out=pd.DataFrame(index=f.index)
    ratio=pd.to_numeric(f['Adj Close'],errors='coerce')/pd.to_numeric(f['Close'],errors='coerce')
    ratio=ratio.where(np.isfinite(ratio)&(ratio>0))
    for a,b in [('open','Open'),('high','High'),('low','Low'),('close','Close')]:
        out[a]=pd.to_numeric(f[b],errors='coerce')*ratio
    idx=pd.DatetimeIndex(f.index)
    if idx.tz is not None:idx=idx.tz_convert('Asia/Taipei').tz_localize(None)
    out['date']=idx.normalize();out=out.reset_index(drop=True)
    if out.date.duplicated().any():raise DataError('Duplicate adjusted dates')
    return out[['date','open','high','low','close']]

class Client:
    def __init__(self,store,cfg,progress,check_pause,token=None):
        self.store,self.cfg=store,cfg
        self.progress,self.check_pause=progress,check_pause
        self.token=(token or os.getenv('FINMIND_TOKEN','')).strip()
        self.session=requests.Session()
        self.session.headers.update({'User-Agent':'AlphaRadarUniverseLab/21.2 (personal research)','Accept':'application/json'})
    def wait(self,seconds,reason):
        end=time.time()+max(0,seconds)
        last=0
        while time.time()<end:
            self.check_pause()
            if time.time()-last>5:
                self.progress(message=reason,wait_seconds=int(end-time.time()));last=time.time()
            time.sleep(min(.5,max(0,end-time.time())))
    def finmind_limit(self,force=False):
        # FinMind exposes the account quota.  Use it instead of treating every
        # authenticated plan as 520 requests/hour.  Failure to inspect quota is
        # non-fatal: fall back conservatively and never expose the token.
        if not self.token:return 260
        cached=self.store.meta('finmind_quota_info',{}) or {}
        now=time.time()
        if not force and cached.get('checked_at',0)>now-1800 and int(cached.get('safe_limit',0))>0:
            return int(cached['safe_limit'])
        safe=520;reported=None;used=None
        try:
            r=self.session.get('https://api.web.finmindtrade.com/v2/user_info',
                headers={'Authorization':'Bearer '+self.token},timeout=(5,min(15,self.cfg.request_timeout)))
            body=r.json() if r.ok else {}
            reported=int(body.get('api_request_limit') or 0);used=int(body.get('user_count') or 0)
            if reported>0:
                safe=max(60,int(reported*float(getattr(self.cfg,'finmind_quota_utilization',.90))))
        except Exception:
            pass
        self.store.set_meta('finmind_quota_info',{'checked_at':now,'reported_limit':reported,
            'reported_used':used,'safe_limit':safe,'credential_present':True})
        return safe
    def throttle(self):
        history=self.store.meta('finmind_requests',[])
        now=time.time();history=[x for x in history if now-x<3600]
        limit=self.finmind_limit()
        spacing=3600/limit+.02
        delay=max(0,(history[-1]+spacing-now) if history else 0,
                  (history[0]+3602-now) if len(history)>=limit else 0,
                  float(self.store.meta('finmind_retry_until',0))-now)
        if delay:self.wait(delay,f'FinMind quota pacing ({limit}/h safe rate); cached work retained')
        history=[x for x in history if time.time()-x<3600]+[time.time()]
        self.store.set_meta('finmind_requests',history)
    def json_get(self,url,params=None,finmind=False):
        for attempt in range(3):
            self.check_pause()
            if finmind:self.throttle()
            headers={'Authorization':'Bearer '+self.token} if finmind and self.token else {}
            try:
                r=self.session.get(url,params=params,headers=headers,timeout=(5,self.cfg.request_timeout))
                try: body=r.json()
                except ValueError:body={}
                status=int(body.get('status',r.status_code)) if isinstance(body,dict) else r.status_code
                if r.status_code in (401,403) or status in (401,403):
                    raise SourceBlocked('Source permission/HTTP 401 or 403; no paid endpoint fallback')
                if r.status_code in (402,429) or status in (402,429):
                    msg=str(body.get('msg','')) if isinstance(body,dict) else ''
                    if re.search(r'sponsor|backer|permission|premium|\u6b0a\u9650|\u6703\u54e1',msg,re.I):
                        raise SourceBlocked('Requested data unavailable at this entitlement. Import licensed data instead.')
                    if attempt==2:raise SourceBlocked('Repeated rate limit; job stopped safely after bounded cooldowns')
                    try:delay=max(1,min(3600,int(r.headers.get('Retry-After','3600'))))
                    except ValueError:delay=3600
                    # Long silent cooldowns look like deadlocks and can be killed
                    # by cloud host sleep. Stop clearly instead; resume later
                    # will keep all downloaded SQLite rows.
                    if delay>90:
                        self.store.set_meta('finmind_retry_until',time.time()+delay)
                        raise SourceDeferred(delay, f'FinMind配額暫時不足；來源建議約 {delay} 秒後重試。已下載的資料保留。')
                    self.wait(delay,'FinMind 配額暫停，正倒數等待');continue
                r.raise_for_status()
                if status!=200:raise DataError('Source status '+str(status))
                return body
            except requests.RequestException as e:
                if attempt==2:raise SourceBlocked(type(e).__name__+': source request failed; cache retained') from e
                self.wait(3*(attempt+1),'Retrying a transient network failure')
        raise SourceBlocked('Bounded source retries exhausted')
    def catalog(self):
        rows=parse_roster(self.json_get(TWSE),'TWSE')+parse_roster(self.json_get(TPEX),'TPEX')
        if len(rows)<self.cfg.min_current_roster:raise DataError('Official whole-roster sanity check failed; no random fallback')
        if len({x['ticker'].split('.')[0] for x in rows})!=len(rows):raise DataError('Conflicting official board codes')
        return sorted(rows,key=lambda r:r['ticker'])
    def calendar(self):
        p=self.json_get(FINMIND,{'dataset':'TaiwanStockTradingDate'},True)
        values=sorted({str(x['date'])[:10] for x in p.get('data',[])})
        values=[d for d in values if self.cfg.start<=d<=self.cfg.end]
        if len(values)<300:raise DataError('Trading calendar incomplete')
        return values
    def raw(self,ticker,start=None):
        p=self.json_get(FINMIND,{'dataset':'TaiwanStockPrice','data_id':ticker.split('.')[0],
                                  'start_date':start or self.cfg.start,'end_date':self.cfg.end},True)
        return parse_raw(p,ticker,price_policy="quarantine")
    def adjusted(self,ticker):
        # Lazy import keeps offline analysis/tests independent from network packages.
        import yfinance as yf
        end=(pd.Timestamp(self.cfg.end)+timedelta(days=1)).strftime('%Y-%m-%d')
        for attempt in range(3):
            self.check_pause()
            try:
                f=yf.Ticker(ticker).history(start=self.cfg.start,end=end,auto_adjust=False,
                        back_adjust=False,actions=False,repair=False,raise_errors=True,
                        timeout=self.cfg.request_timeout)
                return parse_yahoo_history(f)
            except Exception as e:
                if attempt==2:raise SourceBlocked('Adjusted price unavailable for '+ticker+'; raw fallback forbidden') from e
                self.wait(5*(attempt+1),'Adjusted price retry '+ticker)
