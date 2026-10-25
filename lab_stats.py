"""Descriptive dependent-data uncertainty; not guaranteed independent market trials."""
import numpy as np
import pandas as pd

def finite_mean(x):
    a=np.asarray(x,float);a=a[np.isfinite(a)]
    return float(a.mean()) if len(a) else None

def disjoint_indices(dates,exits):
    ids=[];last=pd.Timestamp.min
    for i in np.argsort(pd.DatetimeIndex(dates).values,kind='stable'):
        d=pd.Timestamp(dates[i]);e=pd.Timestamp(exits[i])
        if d>last and pd.notna(e):ids.append(int(i));last=e
    return ids

def uncertainty(values,dates,exits,cfg):
    arr=np.asarray(values,float);valid=np.isfinite(arr);n=int(valid.sum())
    out={'planned':len(arr),'complete':n,'coverage':n/len(arr) if len(arr) else 0,
         'mean':finite_mean(arr),'positive_fraction':float(np.mean(arr[valid]>0)) if n else None,
         'missing_dates':[str(pd.Timestamp(dates[i]).date()) for i in range(len(arr)) if not valid[i]],
         'not_compounded_or_annualized':True}
    if not n:return out
    rng=np.random.default_rng(cfg.seed+len(arr));B=cfg.block_bootstrap_samples
    a=arr[valid]
    iid=a[rng.integers(0,len(a),(B,len(a)))].mean(axis=1)
    out.update(iid95_low=float(np.quantile(iid,.025)),iid95_high=float(np.quantile(iid,.975)))
    # Circular moving blocks preserve nearby observed-date dependence. The protocol
    # never interprets a large raw date count as independent replication.
    L=min(len(arr),cfg.block_months);samples=[]
    for _ in range(B):
        starts=rng.integers(0,len(arr),int(np.ceil(len(arr)/L)))
        idx=np.concatenate([(s+np.arange(L))%len(arr) for s in starts])[:len(arr)]
        x=arr[idx]
        if np.isfinite(x).sum():samples.append(float(np.nanmean(x)))
    out['block95_low']=float(np.quantile(samples,.025));out['block95_high']=float(np.quantile(samples,.975))
    ids=disjoint_indices(dates,exits)  # Select using calendar before observing missing returns.
    b=arr[ids];b=b[np.isfinite(b)]
    out['disjoint_planned']=len(ids);out['disjoint_complete']=len(b)
    out['disjoint_mean']=finite_mean(b)
    if len(b):
        boot=b[rng.integers(0,len(b),(B,len(b)))].mean(axis=1)
        out['disjoint95_low']=float(np.quantile(boot,.025));out['disjoint95_high']=float(np.quantile(boot,.975))
    else:out['disjoint95_low']=None;out['disjoint95_high']=None
    years={}
    for year in sorted(set(pd.DatetimeIndex(dates).year)):
        years[str(year)]=finite_mean(arr[pd.DatetimeIndex(dates).year==year])
    out['year_means']=years
    return out
