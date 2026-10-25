"""Resumable incremental MOPS historical snapshot/proxy collector."""
from __future__ import annotations
import threading
import traceback
from pathlib import Path
import pandas as pd
from lab_io import DataError,read_json,atomic_json
from v231_mops import collect_month

_LOCK=threading.Lock();_THREADS={}

def mops_job_state(root):return read_json(Path(root)/'v23'/'mops_job_state.json')

def mops_busy(root):
    with _LOCK:
        t=_THREADS.get(str(Path(root).resolve()))
        return bool(t and t.is_alive())

def start_mops(root,year_start,year_end):
    root=Path(root).resolve()
    if not (2013<=year_start<=year_end<=2100):raise DataError('研究年分不合法')
    # End at the last fully reported month; never request a future accounting period.
    now=pd.Timestamp.now(tz='Asia/Taipei')
    months=pd.period_range(f'{year_start}-01',f'{year_end}-12',freq='M')
    months=[m for m in months if m <= (now.to_period('M')-1)]
    p=root/'v23'/'mops_job_state.json';p.parent.mkdir(parents=True,exist_ok=True)
    with _LOCK:
        old=_THREADS.get(str(root))
        if old and old.is_alive():raise DataError('已有歷史營收下載任務')
        def worker():
            completed=0;errors=[]
            try:
                for m in months:
                    atomic_json(p,{'status':'RUNNING','completed':completed,'total':len(months),'message':str(m)})
                    try:
                        collect_month(root,int(m.year),int(m.month));completed+=1
                    except Exception as exc:
                        errors.append(str(m)+':'+str(exc)[:200])
                        # A transient failure should not destroy previously
                        # downloaded market-months; retry it on next run.
                atomic_json(p,{'status':'COMPLETE_WITH_GAPS' if errors else 'COMPLETE','completed':completed,
                               'total':len(months),'message':'; '.join(errors[-5:]),'errors':errors})
            except Exception as exc:
                atomic_json(p,{'status':'ERROR','completed':completed,'total':len(months),
                               'message':str(exc),'trace':traceback.format_exc()[-5000:]})
        t=threading.Thread(target=worker,daemon=True,name='V231-MOPS');_THREADS[str(root)]=t;t.start()
    return True
