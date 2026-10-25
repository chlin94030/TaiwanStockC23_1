"""Atomic, pickle-free persistence. Never log tokens or load executable archives."""
from pathlib import Path
from contextlib import contextmanager
import hashlib
import json
import os
import sqlite3
import threading
import time
import numpy as np
import pandas as pd

class DataError(RuntimeError): pass
class PauseRequested(RuntimeError): pass
class SourceBlocked(RuntimeError): pass
class SourceDeferred(SourceBlocked):
    def __init__(self, retry_after, message):
        super().__init__(message)
        self.retry_after = max(1, min(7200, int(retry_after)))

def plain(x):
    if isinstance(x,dict): return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple,np.ndarray)): return [plain(v) for v in x]
    if isinstance(x,(np.bool_,)): return bool(x)
    if isinstance(x,(np.integer,)): return int(x)
    if isinstance(x,(float,np.floating)): return float(x) if np.isfinite(x) else None
    if isinstance(x,(pd.Timestamp,)): return x.isoformat()
    if isinstance(x,Path): return str(x)
    return x

def atomic_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.{threading.get_ident()}.tmp')
    temp.write_text(json.dumps(plain(obj),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    os.replace(temp,path)

def read_json(path,default=None):
    try: return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError,ValueError): return {} if default is None else default

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def code_hash():
    root=Path(__file__).parent
    h=hashlib.sha256()
    for p in sorted(root.glob('*.py')):
        if p.name.startswith('test_'): continue
        h.update(p.name.encode());h.update(p.read_bytes())
    return h.hexdigest()

def frame_write(path,df):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+f'.{threading.get_ident()}.tmp')
    df.to_csv(tmp,index=False,compression={'method':'gzip','compresslevel':1} if p.name.endswith('.gz') else None)
    os.replace(tmp,p)

class Store:
    def __init__(self,root):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.db=self.root/'market.sqlite'
        with self.connect() as c:
            c.executescript('''
            CREATE TABLE IF NOT EXISTS raw(ticker TEXT,date TEXT,open REAL,high REAL,low REAL,close REAL,volume REAL,amount REAL,PRIMARY KEY(ticker,date));
            CREATE TABLE IF NOT EXISTS adj(ticker TEXT,date TEXT,open REAL,high REAL,low REAL,close REAL,PRIMARY KEY(ticker,date));
            CREATE TABLE IF NOT EXISTS downloads(key TEXT PRIMARY KEY, start TEXT,end TEXT,sha TEXT,source TEXT,rows INTEGER,at REAL);
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT);
            CREATE TABLE IF NOT EXISTS quality_issues(
                ticker TEXT,date TEXT,kind TEXT,detail TEXT,at REAL,
                PRIMARY KEY(ticker,date,kind));
            ''')
    @contextmanager
    def connect(self):
        c=sqlite3.connect(self.db,timeout=30)
        try:
            c.execute('PRAGMA journal_mode=WAL');c.execute('PRAGMA busy_timeout=30000')
            c.execute('PRAGMA cache_size=-4096')
            with c:
                yield c
        finally:
            c.close()
    def meta(self,key,default=None):
        with self.connect() as c: r=c.execute('SELECT value FROM meta WHERE key=?',(key,)).fetchone()
        return json.loads(r[0]) if r else default
    def set_meta(self,key,val):
        with self.connect() as c: c.execute('INSERT OR REPLACE INTO meta VALUES(?,?)',(key,json.dumps(plain(val),allow_nan=False)))
    def covered_range(self,key):
        with self.connect() as c:r=c.execute('SELECT start,end FROM downloads WHERE key=?',(key,)).fetchone()
        return tuple(r) if r else None
    def cached(self,key,start,end):
        with self.connect() as c: r=c.execute('SELECT start,end FROM downloads WHERE key=?',(key,)).fetchone()
        return bool(r and r[0] <= start and r[1] >= end)
    def put(self,table,ticker,frame,start,end,source):
        if table not in ('raw','adj'): raise DataError('Invalid table')
        cols=['date','open','high','low','close']+(['volume','amount'] if table=='raw' else [])
        f=frame[cols].copy();f['date']=pd.to_datetime(f.date).dt.strftime('%Y-%m-%d')
        if f.date.duplicated().any(): raise DataError('Duplicate source dates')
        payload=f.to_csv(index=False).encode(); digest=hashlib.sha256(payload).hexdigest()
        vals=[]
        for row in f.itertuples(index=False,name=None):
            vals.append((ticker,)+tuple(None if pd.isna(v) else v for v in row))
        placeholders=','.join('?' for _ in range(len(cols)+1))
        with self.connect() as c:
            if table == 'raw':
                
                if not f.empty:
                    c.execute('DELETE FROM quality_issues WHERE ticker=? AND date>=? AND date<=?', (ticker,str(f.date.min()),str(f.date.max())))
                for issue in frame.attrs.get('quality_issues',[]):
                    c.execute('INSERT OR REPLACE INTO quality_issues VALUES(?,?,?,?,?)',
                              (ticker,issue['date'],issue['kind'],json.dumps(plain(issue),allow_nan=False),time.time()))
            c.executemany(f'INSERT OR REPLACE INTO {table} VALUES({placeholders})',vals)
            c.execute('INSERT OR REPLACE INTO downloads VALUES(?,?,?,?,?,?,?)',(table+':'+ticker,start,end,digest,source,len(f),time.time()))
    def get(self,table,ticker,start=None,end=None):
        if table not in ('raw','adj'): raise DataError('Invalid table')
        args=[ticker];where='ticker=?'
        if start: where+=' AND date>=?';args.append(start)
        if end: where+=' AND date<=?';args.append(end)
        with self.connect() as c: f=pd.read_sql_query(f'SELECT * FROM {table} WHERE {where} ORDER BY date',c,params=args)
        f['date']=pd.to_datetime(f.date)
        return f
    def quality_details(self,ticker):
        with self.connect() as c:
            rows=c.execute('SELECT detail FROM quality_issues WHERE ticker=? ORDER BY date,at DESC',(ticker,)).fetchall()
        return [json.loads(row[0]) for row in rows]
    def quality_summary(self):
        with self.connect() as c:
            return pd.read_sql_query('SELECT ticker,kind,COUNT(*) AS rows FROM quality_issues GROUP BY ticker,kind',c).to_dict('records')
    def manifest(self):
        with self.connect() as c:
            return pd.read_sql_query('SELECT * FROM downloads ORDER BY key',c).to_dict('records')
    def data_hash(self):
        # Single-writer workspace lock is held by the caller. Checkpoint WAL before
        # hashing the actual database bytes (including provenance/cache metadata).
        # This is an artifact identity, not a semantic row-order-independent hash.
        with self.connect() as c:
            status=c.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchone()
            if status[0]!=0:raise DataError('Unable to checkpoint database for a consistent hash')
        return sha(self.db)

class JobLock:
    """OS advisory lock releases on crash; supported on Linux/macOS and Windows."""
    def __init__(self,path): self.path=Path(path);self.file=None
    def __enter__(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.file=open(self.path,'a+b')
        try:
            if os.name=='nt':
                import msvcrt
                self.file.seek(0);self.file.write(b'0');self.file.flush();self.file.seek(0)
                msvcrt.locking(self.file.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(self.file,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except (OSError,IOError):
            self.file.close();raise DataError('A job already owns this workspace.')
        return self
    def __exit__(self,*args):
        if self.file: self.file.close()
