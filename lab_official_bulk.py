"""Free exact historical bootstrap from TWSE/TPEx whole-market daily pages.

One request per market/date replaces one request per symbol.  The provider is
resumable and caches parsed daily files.  It uses exchange-reported turnover
amounts and raw OHLC; adjusted OHLC is fetched in Yahoo multi-ticker batches.

This is a courtesy backfill, not a high-frequency scraper.  The default pacing
runs TWSE and TPEx in parallel for one date, then waits between dates.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import re
import time
import numpy as np
import pandas as pd
import requests

from lab_io import DataError, SourceBlocked, atomic_json, read_json, sha
from lab_bulk import vectorized_turnover
from lab_yahoo_bulk import YahooFastBulk

TWSE_URL='https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX'
TPEX_URL='https://www.tpex.org.tw/www/zh-tw/afterTrading/dailyQuotes'
TPEX_LEGACY='https://www.tpex.org.tw/web/stock/aftertrading/otc_quotes_no1430/stk_wn1430_result.php'


def _num(x):
    if x is None:return np.nan
    s=re.sub(r'<[^>]+>','',str(x)).replace(',','').replace('--','').strip()
    if s in ('','---','-'):return np.nan
    return pd.to_numeric(s,errors='coerce')


def _norm(x):
    return re.sub(r'[\s\(\)（）\[\]\-_/]','',str(x)).replace('（元）','').replace('(元)','')


def _field_index(fields,candidates):
    norm=[_norm(x) for x in fields]
    for cand in candidates:
        c=_norm(cand)
        for i,x in enumerate(norm):
            if x==c:return i
    for cand in candidates:
        c=_norm(cand)
        for i,x in enumerate(norm):
            if c in x:return i
    return None


def _rows_from_table(fields,data,board,date):
    idx={
        'code':_field_index(fields,['證券代號','代號']),
        'open':_field_index(fields,['開盤價','開盤']),
        'high':_field_index(fields,['最高價','最高']),
        'low':_field_index(fields,['最低價','最低']),
        'close':_field_index(fields,['收盤價','收盤']),
        'volume':_field_index(fields,['成交股數','成交量']),
        'amount':_field_index(fields,['成交金額']),
    }
    if any(v is None for v in idx.values()):return pd.DataFrame()
    out=[]
    for row in data or []:
        try:code=re.sub(r'\s+','',str(row[idx['code']]))
        except Exception:continue
        if not re.fullmatch(r'\d{4}',code):continue
        vals={k:_num(row[i]) for k,i in idx.items() if k!='code'}
        out.append({'date':pd.Timestamp(date),'ticker':code+('.TW' if board=='TWSE' else '.TWO'),'board':board,**vals})
    return pd.DataFrame(out)


def parse_twse(payload,date):
    if not isinstance(payload,dict):raise DataError('TWSE payload is not JSON object')
    if payload.get('stat')!='OK':return pd.DataFrame()
    for t in payload.get('tables',[]):
        f=t.get('fields') or [];d=t.get('data') or []
        if _field_index(f,['證券代號']) is not None and _field_index(f,['成交金額']) is not None and _field_index(f,['收盤價']) is not None:
            out=_rows_from_table(f,d,'TWSE',date)
            if not out.empty:return out
    # Old response shape.
    for n in range(1,20):
        f=payload.get(f'fields{n}');d=payload.get(f'data{n}')
        if f and d and _field_index(f,['證券代號']) is not None and _field_index(f,['成交金額']) is not None:
            out=_rows_from_table(f,d,'TWSE',date)
            if not out.empty:return out
    raise DataError('TWSE whole-market OHLCV table not found')


def parse_tpex(payload,date):
    if not isinstance(payload,dict):raise DataError('TPEx payload is not JSON object')
    # New website JSON may expose tables.
    for t in payload.get('tables',[]):
        f=t.get('fields') or [];d=t.get('data') or []
        if _field_index(f,['代號','證券代號']) is not None and _field_index(f,['成交金額']) is not None and _field_index(f,['收盤','收盤價']) is not None:
            out=_rows_from_table(f,d,'TPEX',date)
            if not out.empty:return out
    # Some endpoints return a list of dict rows.
    rows=payload.get('data')
    if isinstance(rows,list) and rows and isinstance(rows[0],dict):
        fields=list(rows[0]);data=[[r.get(k) for k in fields] for r in rows]
        out=_rows_from_table(fields,data,'TPEX',date)
        if not out.empty:return out
    # Legacy aaData order: code,name,close,chg,open,high,low,volume,amount,...
    aa=payload.get('aaData')
    if isinstance(aa,list):
        out=[]
        for row in aa:
            if not isinstance(row,(list,tuple)) or len(row)<9:continue
            code=re.sub(r'\s+','',str(row[0]))
            if not re.fullmatch(r'\d{4}',code):continue
            out.append({'date':pd.Timestamp(date),'ticker':code+'.TWO','board':'TPEX','close':_num(row[2]),
                        'open':_num(row[4]),'high':_num(row[5]),'low':_num(row[6]),'volume':_num(row[7]),'amount':_num(row[8])})
        return pd.DataFrame(out)
    stat=str(payload.get('stat',payload.get('msg','')))
    if stat and stat not in ('OK','ok',''):return pd.DataFrame()
    return pd.DataFrame()


def _roc_date(date):
    d=pd.Timestamp(date);return f'{d.year-1911}/{d.month:02d}/{d.day:02d}'


@dataclass
class OfficialDailyBulk:
    cfg: object
    progress: object
    check_pause: object
    interval: float = 0.75

    provider_name='OFFICIAL_BULK'
    actual_amount=True

    def __post_init__(self):
        self.twse_session=requests.Session();self.tpex_session=requests.Session()
        for sess in (self.twse_session,self.tpex_session):sess.headers.update({'User-Agent':'AlphaRadar/22.1 research backfill','Accept':'application/json'})
        self._cache_root=None

    def _get_json(self,url,params,session=None):
        last=None
        for attempt in range(3):
            self.check_pause()
            try:
                r=(session or self.twse_session).get(url,params=params,timeout=(5,self.cfg.request_timeout))
                if r.status_code in (401,403):raise SourceBlocked(f'Official source HTTP {r.status_code}; cloud IP may be blocked')
                if r.status_code==429:
                    time.sleep(4*(attempt+1));continue
                r.raise_for_status();return r.json()
            except SourceBlocked:raise
            except Exception as exc:last=exc;time.sleep(2*(attempt+1))
        raise DataError('Official bulk request failed: '+type(last).__name__)

    def _fetch_twse(self,date):
        p=self._get_json(TWSE_URL,{'date':pd.Timestamp(date).strftime('%Y%m%d'),'type':'ALLBUT0999','response':'json'},self.twse_session)
        return parse_twse(p,date)

    def _fetch_tpex(self,date):
        d=pd.Timestamp(date).strftime('%Y/%m/%d')
        try:
            p=self._get_json(TPEX_URL,{'date':d,'response':'json'},self.tpex_session)
            out=parse_tpex(p,date)
            if not out.empty:return out
            stat=str(p.get('stat',p.get('msg',''))).strip() if isinstance(p,dict) else ''
            total=p.get('total',p.get('iTotalRecords')) if isinstance(p,dict) else None
            if (stat and stat.lower() not in ('ok','success')) or total==0:return pd.DataFrame()
        except (DataError,SourceBlocked):
            # Permission errors on the new endpoint should still try the legacy
            # public historical endpoint before giving up.
            pass
        p=self._get_json(TPEX_LEGACY,{'l':'zh-tw','d':_roc_date(date),'se':'EW'},self.tpex_session)
        return parse_tpex(p,date)

    def _paths(self,root,date):
        d=pd.Timestamp(date);folder=Path(root)/'official_bulk'/f'{d.year:04d}'/f'{d.month:02d}';folder.mkdir(parents=True,exist_ok=True)
        stem=d.strftime('%Y-%m-%d');return folder/(stem+'.csv.gz'),folder/(stem+'.empty')

    def _ensure(self,cache_dir):
        self._cache_root=Path(cache_dir);days=pd.bdate_range(self.cfg.start,self.cfg.end)
        done=0;trading=[]
        for di,d in enumerate(days,1):
            self.check_pause();path,empty=self._paths(cache_dir,d)
            if path.exists():trading.append(d.strftime('%Y-%m-%d'));done+=1;continue
            if empty.exists():done+=1;continue
            self.progress(phase='OFFICIAL_BULK',done=done,total=len(days),message=f'官方整批歷史 {d.date()}｜TWSE+TPEx 同日各1次')
            with ThreadPoolExecutor(max_workers=2) as ex:
                a=ex.submit(self._fetch_twse,d);b=ex.submit(self._fetch_tpex,d)
                tw=a.result();otc=b.result()
            if tw.empty and otc.empty:
                empty.write_text('non-trading-day',encoding='ascii')
            elif tw.empty != otc.empty:
                raise DataError(f'Official board mismatch on {d.date()}; refusing partial-market cache')
            else:
                out=pd.concat([tw,otc],ignore_index=True)
                # Exact actual amount and shares must be present; raw zero-price rows may exist.
                if out.ticker.duplicated().any():raise DataError('Duplicate official ticker on '+str(d.date()))
                if (out[['volume','amount']].dropna()<0).any().any():raise DataError('Negative official volume/amount')
                out.to_csv(path,index=False,compression={'method':'gzip','compresslevel':1});trading.append(d.strftime('%Y-%m-%d'))
            done+=1;self.progress(phase='OFFICIAL_BULK',done=done,total=len(days),message=f'官方整批已快取 {done}/{len(days)} 個工作日')
            time.sleep(max(0,float(self.interval)))
        return trading

    def calendar(self):
        if self._cache_root is None:
            # build_dataset calls calendar before turnover; use the store cache
            # root injected later via turnover is unavailable, so a temporary
            # attribute is set by bind_cache in lab_data.
            root=getattr(self,'cache_dir',None)
            if root is None:raise DataError('Official bulk cache directory not bound')
        else:root=self._cache_root
        return self._ensure(root)

    def bind_cache(self,path):self.cache_dir=Path(path);return self

    def _read_all(self,cache_dir,cols=None):
        self._ensure(cache_dir);frames=[]
        for p in sorted((Path(cache_dir)/'official_bulk').glob('*/*/*.csv.gz')):
            q=pd.read_csv(p,usecols=cols,parse_dates=['date'] if cols is None or 'date' in cols else None)
            frames.append(q)
        if not frames:raise DataError('Official bulk cache has no trading data')
        return pd.concat(frames,ignore_index=True)

    def turnover(self,signals,cat,cache_dir):
        long=self._read_all(cache_dir,['date','ticker','amount'])
        keep={r['ticker'] for r in cat};long=long[long.ticker.isin(keep)]
        wide=long.pivot(index='date',columns='ticker',values='amount').sort_index()
        codes={str(r['ticker']).split('.')[0]:r['ticker'] for r in cat};wide.columns=[str(x).split('.')[0] for x in wide.columns]
        out=vectorized_turnover(wide,signals,self.cfg,codes)
        meta={'provider':'OFFICIAL_BULK','start':self.cfg.start,'end':self.cfg.end,'roster_count':len(cat),'rows':len(out),
              'actual_amount':True,'estimated_amount':False,'source':'TWSE MI_INDEX + TPEx dailyQuotes historical whole-market'}
        return out,meta

    def populate_selected_prices(self,store,tickers):
        wanted=set(tickers);long=self._read_all(self._cache_root or self.cache_dir,['date','ticker','open','high','low','close','volume','amount'])
        long=long[long.ticker.isin(wanted)].copy();missing=[]
        for i,t in enumerate(sorted(wanted),1):
            f=long[long.ticker==t].drop(columns='ticker').sort_values('date')
            if f.empty:missing.append(t);continue
            positive=f[['open','high','low','close']].gt(0).all(axis=1)
            high_bad=positive&(f.high < f[['open','close','low']].max(axis=1)-1e-6)
            low_bad=positive&(f.low > f[['open','close','high']].min(axis=1)+1e-6)
            bad=high_bad|low_bad|(~np.isfinite(f[['open','high','low','close']]).all(axis=1))|(~positive)
            issues=[]
            for ix,row in f.loc[bad].iterrows():issues.append({'ticker':t,'date':pd.Timestamp(row.date).strftime('%Y-%m-%d'),'kind':'OFFICIAL_NO_VALID_OHLC','observed':{}})
            if bad.any():f.loc[bad,['open','high','low','close']]=np.nan
            f.attrs['quality_issues']=issues;store.put('raw',t,f,self.cfg.start,self.cfg.end,'TWSE_TPEX_OFFICIAL_BULK_ACTUAL_AMOUNT')
            self.progress(phase='OFFICIAL_SELECTED',done=i,total=len(wanted),last_committed_symbol=t,message=f'官方固定母池raw {i}/{len(wanted)}')
        if missing:raise DataError('Official bulk selected-union missing: '+','.join(missing[:30]))
        # Only adjusted OHLC comes from Yahoo; raw amount/volume remains official.
        y=YahooFastBulk(self.cfg,self.progress,self.check_pause)
        y.populate_adjusted_only(store,sorted(wanted))
        return {'provider':'OFFICIAL_BULK','symbols':len(wanted),'actual_amount':True,'estimated_amount':False,
                'adjusted_source':'Yahoo Adj Close ratio'}
