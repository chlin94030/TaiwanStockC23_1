"""Optional public historical MOPS revenue collector (strictly research-only).

MOPS static monthly aggregates do NOT disclose issuer-level publication times.
Every collected row is PROXY with intentionally late usable date (next-month
20th after 15:00 TW). This does not qualify for independent/PIT acceptance.
Each market-month is atomically saved; one failed request never overwrites data.
"""
from __future__ import annotations
import io
import re
from pathlib import Path
import pandas as pd
import requests
from lab_io import DataError, atomic_json, read_json
from v231_data import persist_events

HOSTS=('https://mopsov.twse.com.tw','https://doc.twse.com.tw')


def parse_html(html,year,month,board,url):
    try:
        tables=pd.read_html(io.StringIO(html),flavor='lxml')
    except (ValueError,ImportError) as exc:
        raise DataError('MOPS 網頁無法解析表格：'+str(exc))
    rows=[]
    for table in tables:
        t=table.copy()
        if isinstance(t.columns,pd.MultiIndex):
            t.columns=[''.join(str(z) for z in col if not str(z).startswith('Unnamed')).replace(' ','') for col in t.columns]
        else:t.columns=[str(col).replace(' ','') for col in t.columns]
        tickcols=[col for col in t.columns if '公司代號' in str(col)]
        revcols=[col for col in t.columns if '當月營收' in str(col) and '去年' not in str(col) and '上月' not in str(col)]
        if (not tickcols or not revcols) and len(t)>2:
            # Static HTML can put the column header inside body row 0 or 1.
            for k in range(min(5,len(t))):
                v=t.iloc[k].astype(str).str.replace(' ','',regex=False).tolist()
                ix=[i for i,x in enumerate(v) if '公司代號' in x]
                ry=[i for i,x in enumerate(v) if '當月營收' in x and not any(q in x for q in ('去年','上月'))]
                if len(ix)==1 and len(ry)==1:
                    t.columns=[str(x) for x in v]
                    t=t.iloc[k+1:].copy()
                    tickcols=[t.columns[ix[0]]];revcols=[t.columns[ry[0]]]
                    break
        if len(tickcols)!=1 or len(revcols)!=1:continue
        for x,y in zip(t[tickcols[0]],t[revcols[0]]):
            tx=str(x).strip()
            if not re.fullmatch(r'\d{4}',tx):continue
            raw=str(y).strip().replace(',','')
            try: v=float(raw)*1000. # MOPS unit: NTD thousand
            except ValueError:continue
            if not (v>=0 and v<1e17):continue
            # Date is a conservative PROXY, never presented as issuer timestamp.
            py=year+(month==12);pm=month%12+1
            available=f'{py:04d}-{pm:02d}-20T15:30:00+08:00'
            rows.append({'ticker':tx+('.TW' if board=='sii' else '.TWO'),
                         'period':f'{year:04d}-{month:02d}', 'revenue_ntd':v,
                         'available_at':available,'quality':'PROXY','source':url})
    frame=pd.DataFrame(rows)
    if not frame.empty:frame=frame.drop_duplicates(['ticker','period'])
    return frame


def collect_month(root,year,month,timeout=18):
    if not (2013<=year<=2100 and 1<=month<=12):raise DataError('月營收年月超出允許範圍')
    cache=Path(root)/'v23'/'mops_proxy_cache';cache.mkdir(parents=True,exist_ok=True)
    status=read_json(cache/'status.json')
    allrows=[];errors=[]
    for board in ('sii','otc'):
        key=f'{year:04d}-{month:02d}-{board}'
        path=cache/(key+'.csv.gz')
        if status.get(key)=='DONE' and path.exists():
            allrows.append(pd.read_csv(path))
            continue
        roc=year-1911
        success=False
        for host in HOSTS:
            for extra in ('0','1'):
                url=f'{host}/nas/t21/{board}/t21sc03_{roc}_{month}_{extra}.html'
                try:
                    r=requests.get(url,timeout=timeout,headers={'User-Agent':'Mozilla/5.0','Accept-Language':'zh-TW,zh;q=0.9'})
                    r.raise_for_status()
                    r.encoding=r.apparent_encoding or 'big5'
                    p=parse_html(r.text,year,month,board,url)
                    if len(p)<50:raise DataError('來源解析不足50檔')
                    allrows.append(p);success=True
                    break
                except Exception as exc:errors.append(url+': '+str(exc)[:160])
            if success:break
        if success:
            # Keep a per-board recovery file; don't pretend success for the
            # missing board. The stored event table is merged below atomically.
            d=allrows[-1]
            d.to_csv(path,index=False,compression='gzip')
            status[key]='DONE';atomic_json(cache/'status.json',status)
    if not allrows:raise DataError('MOPS歷史月營收未能取得：'+'; '.join(errors[-3:]))
    df=pd.concat(allrows,ignore_index=True)
    result=persist_events(root,df,'revenue')
    result.update({'month':f'{year:04d}-{month:02d}','rows_this_month':len(df),'errors':errors[-5:],
                   'official_company_timestamps_verified':False,'historical_summary_proxy_only':True})
    return result
