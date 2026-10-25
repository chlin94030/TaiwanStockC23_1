"""V23.1 supplementary data ingestion and conservative time-of-knowledge policy.

Do not infer issuer disclosure timestamps from an accounting period or a table's
print date. Official TWSE/TPEx current OpenAPI snapshots are archived at FETCH time.
Historical MOPS summary pages are marked PROXY and are NOT eligible for strong
point-in-time/independent acceptance. No network calls during normal UI reruns.
"""
from __future__ import annotations
import hashlib
import io
import json
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import requests
from lab_io import DataError, atomic_json, read_json, sha

TZ = 'Asia/Taipei'
QUALITY = {'VERIFIED', 'USER_ATTESTED', 'PROXY', 'OBSERVED_ONLY'}
CONTRACTS = {
    'revenue': ['ticker', 'period', 'revenue_ntd', 'available_at', 'source', 'quality'],
    'financial': ['ticker', 'fiscal_quarter', 'eps', 'gross_margin', 'available_at', 'source', 'quality'],
    'flow': ['ticker', 'institutional_flow5', 'available_at', 'source', 'quality'],
}
SOURCES = {
    'TWSE': 'https://openapi.twse.com.tw/v1/opendata/t187ap05_L',
    'TPEX': 'https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap05_O',
}


def normalize_ticker(s):
    s = str(s).strip().upper()
    if s.endswith(('.TW', '.TWO')): return s
    if s.isdigit() and len(s) == 4: return s + '.TW'  # ambiguous board: audit must resolve
    raise DataError('股票代號需為 2330.TW 或 6488.TWO：' + str(s))


def _timestamp_column(s, quality):
    x = pd.Series(s, copy=True).astype('string')
    # VERIFIED inputs require an unambiguous timezone offset in each timestamp.
    good = x.str.contains(r'(?:Z|[+-]\d\d:\d\d)$', case=False, regex=True, na=False)
    q = pd.Series(quality).astype('string').str.upper()
    if ((q == 'VERIFIED') & ~good).any():
        raise DataError('VERIFIED 資料 available_at 必須帶時區，例如 2025-05-10T18:00:00+08:00')
    # For unverified timestamps, missing timezone is taken as Taiwan local time;
    # this does NOT upgrade their quality to VERIFIED.
    x = x.where(good, x + '+08:00')
    out = pd.to_datetime(x, utc=True, errors='coerce')
    if out.isna().any(): raise DataError('available_at 有無法解析的公告日期')
    return out


def _validate_period(col, fmt):
    vals = pd.Series(col).astype(str).str.strip()
    if fmt == 'monthly':
        if not vals.str.fullmatch(r'\d{4}-(0[1-9]|1[0-2])').all(): raise DataError('period 格式必須是 YYYY-MM')
        pd.PeriodIndex(vals, freq='M')
    else:
        if not vals.str.fullmatch(r'\d{4}Q[1-4]').all(): raise DataError('fiscal_quarter 格式必須是 YYYYQ1~Q4')
        pd.PeriodIndex(vals, freq='Q')
    return vals


def validate_events(frame:pd.DataFrame, kind:str) -> pd.DataFrame:
    if kind not in CONTRACTS: raise DataError('未知的補充資料類型')
    need = CONTRACTS[kind]
    if not isinstance(frame, pd.DataFrame) or not set(need).issubset(frame):
        raise DataError(f'{kind} 缺少欄位：' + ', '.join(sorted(set(need)-set(frame.columns))))
    x = frame[need].copy()
    if len(x)>600000: raise DataError('單次事件檔案超過600000筆上限')
    x['ticker'] = x.ticker.map(normalize_ticker)
    if x['source'].isna().any() or x['source'].astype(str).str.strip().eq('').any(): raise DataError('source 來源欄不得空白')
    x['quality'] = x.quality.astype(str).str.strip().str.upper()
    if not x.quality.isin(QUALITY).all(): raise DataError('quality 限 VERIFIED / USER_ATTESTED / PROXY / OBSERVED_ONLY')
    x['available_at'] = _timestamp_column(x.available_at, x.quality)
    if kind == 'revenue':
        x['period'] = _validate_period(x.period, 'monthly')
        x['revenue_ntd'] = pd.to_numeric(x.revenue_ntd, errors='coerce')
        if (~np.isfinite(x.revenue_ntd) | (x.revenue_ntd < 0)).any(): raise DataError('revenue_ntd 必須是有限且非負數值（新台幣元）')
        p = pd.PeriodIndex(x.period, freq='M').to_timestamp(how='start').tz_localize(TZ).tz_convert('UTC')
        if (x.available_at < p).any(): raise DataError('公告時間早於營收資料月份')
        subset=['ticker','period','available_at']
    elif kind == 'financial':
        x['fiscal_quarter'] = _validate_period(x.fiscal_quarter, 'quarterly')
        for k in ('eps','gross_margin'):
            x[k] = pd.to_numeric(x[k], errors='coerce')
            if (~np.isfinite(x[k])).any(): raise DataError(k+' 必須為有限數字')
        if (x.gross_margin.abs()>1.0).any(): raise DataError('gross_margin 請用小數如0.52，不是52%')
        p=pd.PeriodIndex(x.fiscal_quarter,freq='Q').to_timestamp(how='start').tz_localize(TZ).tz_convert('UTC')
        if (x.available_at<p).any(): raise DataError('財報公告時間早於所屬季度')
        subset=['ticker','fiscal_quarter','available_at']
    else:
        x['institutional_flow5']=pd.to_numeric(x.institutional_flow5,errors='coerce')
        if (~np.isfinite(x.institutional_flow5)).any(): raise DataError('institutional_flow5 必須為有限值')
        subset=['ticker','available_at']
    if x.duplicated(subset).any(): raise DataError('同股票、期別、時間存在重複事件；請保留修訂版本的不同公告時間')
    # VERIFIED must mean exact dated source; still independently audited externally.
    return x.sort_values(subset).reset_index(drop=True)


def persist_events(root,frame,kind):
    """Append-only revisions, atomic CSV replace, provenance checksum; no secret access."""
    directory=Path(root)/'v23'/'evidence';directory.mkdir(parents=True,exist_ok=True)
    x=validate_events(frame,kind)
    path=directory/(kind+'.csv.gz')
    if path.exists():
        old=validate_events(pd.read_csv(path),kind)
        unique=['ticker','available_at']+(['period'] if kind=='revenue' else ['fiscal_quarter'] if kind=='financial' else [])
        oldkeys=old[unique].astype(str).agg('|'.join,axis=1)
        newkeys=x[unique].astype(str).agg('|'.join,axis=1)
        collide=set(oldkeys)&set(newkeys)
        if collide:
            # identical resubmissions allowed; changing historical vintage silently is not.
            oldmap=old.set_index(unique);newmap=x.set_index(unique)
            for key in set(oldmap.index)&set(newmap.index):
                a=oldmap.loc[key].astype(str).to_dict();b=newmap.loc[key].astype(str).to_dict()
                if a!=b: raise DataError('舊公告歷史紀錄內容不同；須另用新公告時間寫入修訂版本')
        x=pd.concat([old,x],ignore_index=True).drop_duplicates(unique)
    tmp=path.with_suffix('.tmp')
    x.to_csv(tmp,index=False,compression={'method':'gzip','compresslevel':2})
    tmp.replace(path)
    return {'kind':kind,'rows':len(x),'checksum':sha(path),'quality_counts':x.quality.value_counts().to_dict()}


def load_events(root):
    base=Path(root)/'v23'/'evidence';out={}
    for kind in CONTRACTS:
        p=base/(kind+'.csv.gz')
        out[kind]=validate_events(pd.read_csv(p),kind) if p.exists() else pd.DataFrame(columns=CONTRACTS[kind])
    return out


def evidence_signature(root):
    base=Path(root)/'v23'/'evidence'
    return {k:sha(base/(k+'.csv.gz')) if (base/(k+'.csv.gz')).exists() else None for k in CONTRACTS}


def _roc_month(s):
    t=str(s).strip().replace('/','').replace('-','')
    if not t.isdigit() or len(t)<5:raise DataError('TWSE 月營收年月格式不明：'+t)
    year=int(t[:-2]);month=int(t[-2:]); year=year+1911 if year<1911 else year
    if not 2000<=year<=2100 or not 1<=month<=12:raise DataError('TWSE 月營收年月超出範圍：'+t)
    return f'{year:04d}-{month:02d}'


def fetch_latest_official(root,timeout=25):
    """Fetch an observed snapshot; never backdate its available_at to report month."""
    moment=datetime.now(timezone.utc).isoformat()
    rows=[];warnings=[]
    for board,url in SOURCES.items():
        try:
            response=requests.get(url,timeout=timeout,headers={'User-Agent':'Mozilla/5.0 AlphaRadarResearch/23.1'})
            response.raise_for_status(); payload=response.json()
            if not isinstance(payload,list) or len(payload)<50:raise DataError('官方端點回傳不足')
            added=0
            for v in payload:
                try:
                    ticker=str(v.get('公司代號','')).strip()
                    if not (ticker.isdigit() and len(ticker)==4):continue
                    period=_roc_month(v.get('資料年月',''))
                    amount=str(v.get('營業收入-當月營收','')).replace(',','').strip()
                    if not amount or amount=='-':continue
                    value=float(amount)*1000
                    if not np.isfinite(value) or value<0:continue
                    rows.append({'ticker':ticker+('.TW' if board=='TWSE' else '.TWO'),
                                 'period':period,'revenue_ntd':value,'available_at':moment,'source':url,
                                 'quality':'OBSERVED_ONLY'})
                    added+=1
                except (ValueError,TypeError,DataError):continue
            if added<50:raise DataError('官方資料欄位不相容，解析不足50筆')
        except Exception as exc:
            warnings.append(board+': '+str(exc))
    if not rows:raise DataError('官方月營收端點無可用資料；'+ '; '.join(warnings))
    result=persist_events(root,pd.DataFrame(rows),'revenue')
    result['warnings']=warnings;result['asof_observation_utc']=moment
    return result


def export_templates(root):
    """Return a small ZIP with explicit schema examples (no fabricated numeric financial data)."""
    import zipfile
    io_obj=io.BytesIO()
    with zipfile.ZipFile(io_obj,'w',zipfile.ZIP_DEFLATED) as z:
        for name,cols in CONTRACTS.items():
            z.writestr(name+'_template.csv',','.join(cols)+'\n')
        z.writestr('READ_ME.txt','公告時間須為 ISO8601+08:00，VERIFIED 需有明確可查的公告來源。不可用月份日期代替公告時間。\n')
    return io_obj.getvalue()
