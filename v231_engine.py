"""V23.1 benchmark-first model, date-T selection, monthly non-overlap ledger.

Runs only on existing V22 pinned market data. No forecast/optimality promises.
The price-only control is NOT automatically promoted to production; 0050 is
always the safe uninvestigated default. All trades are simulated, never sent.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from lab_io import DataError

VERSION='23.1.0-benchmark-first'

# Precommitted research hypotheses, not optimized on the exposed history.
MODELS={
    '0050_ONLY': {},
    'PRICE_RS_CONTROL': {
        'rs_0050_120':.45,'momentum120_skip20':.25,'high252':.20,'residual_lowvol60':.10},
    'GROWTH_RS_40_35_15_10': {
        'revenue_yoy_3m':.12,'revenue_acceleration':.18,'eps_growth_yoy':.10,
        'rs_0050_120':.13,'momentum120_skip20':.12,'high252':.10,
        'gross_margin_delta':.08,'residual_lowvol60':.07,'institutional_flow5':.10},
    'GROWTH_NO_FLOW_ABLATION': {
        'revenue_yoy_3m':.16,'revenue_acceleration':.20,'eps_growth_yoy':.14,
        'rs_0050_120':.14,'momentum120_skip20':.13,'high252':.10,
        'gross_margin_delta':.07,'residual_lowvol60':.06},
}
assert all(abs(sum(v.values())-1)<1e-9 for k,v in MODELS.items() if v)

@dataclass(frozen=True)
class V23Config:
    pool_rule:str='KOSTO250'
    capital:float=1_000_000.
    top_k:int=10
    min_eligible:int=30
    min_fundamental_coverage:float=.70
    max_rsi:float=84.
    max_bias20:float=.14
    sector_cap:int=3
    amount_participation:float=.005
    stage_exposure_a1:float=.70
    stage_exposure_a2:float=.65
    stage_exposure_b3:float=.35
    final_holdout_months:int=24
    min_dev_train_months:int=24
    extra_stock_roundtrip_stress:float=.005
    benchmark_excess_cagr_min_pp:float=5.
    ir_min:float=.50
    max_extra_drawdown_pp:float=10.
    required_folds:int=3

    def validate(self):
        if self.pool_rule not in ('KOSTO250','KOSTO300','ADV200','ADV300'):raise DataError('不支援的固定母池')
        if not 5<=self.top_k<=20 or not 3<=self.sector_cap<=self.top_k:raise DataError('錯誤的持股或產業上限')
        if not 0<self.amount_participation<=.05 or self.capital<=0:raise DataError('錯誤的資金或成交量限制')
        if not 0<self.min_fundamental_coverage<=1:raise DataError('不合法的基本面覆蓋率要求')
        if not 0<=self.max_bias20<=.5 or not 30<=self.max_rsi<=100:raise DataError('不合法的追價門檻')
        if not 6<=self.final_holdout_months<=36 or not 12<=self.min_dev_train_months<=48:raise DataError('回測月份不足')
        if not all(0<=v<=1 for v in (self.stage_exposure_a1,self.stage_exposure_a2,self.stage_exposure_b3)):raise DataError('階段曝險無效')
        if not 0<=self.extra_stock_roundtrip_stress<=.03:raise DataError('成本壓力值無效')
        if not 0<=self.max_extra_drawdown_pp<=100:raise DataError('最大回撤限制無效')
        return self


def _stage_budget(stage,cfg):
    return {'A1':cfg.stage_exposure_a1,'A2':cfg.stage_exposure_a2,'B3':cfg.stage_exposure_b3}.get(stage,0.)


def select_asof(group,model,stage,cfg,models=None):
    """A deterministic signal ranking with full-data hard gates, not fitted weights."""
    cfg.validate()
    candidates=MODELS if models is None else models
    if model not in candidates:raise DataError('未知的V23候選')
    if model=='0050_ONLY':return {'state':'ETF_BASELINE','eligible':0,'coverage':1.,'selected':[],'target_stock_fraction':0.}
    ff=candidates[model]
    n=len(group)
    if n<max(cfg.top_k,cfg.min_eligible):
        return {'state':'POOL_INCOMPLETE','eligible':0,'coverage':0.,'selected':[],'target_stock_fraction':0.}
    base=group.copy()
    for f in ff:base[f]=pd.to_numeric(base[f],errors='coerce') if f in base else np.nan
    # Require all economic factors for a fully funded strategy. Never fill
    # missing fundamentals with neutral zeros or silently change weights.
    known=base[list(ff)].notna().all(axis=1)
    fundamental='revenue_yoy_3m' in ff
    coverage=float(known.mean())
    if fundamental and coverage<cfg.min_fundamental_coverage:
        return {'state':'FUNDAMENTAL_COVERAGE_BLOCKED','eligible':int(known.sum()),'coverage':coverage,
                'selected':[],'target_stock_fraction':0.}
    liquid=pd.to_numeric(base.amount20,errors='coerce')*cfg.amount_participation >= cfg.capital/cfg.top_k
    risk=pd.to_numeric(base.rsi14,errors='coerce').le(cfg.max_rsi)&pd.to_numeric(base.bias20,errors='coerce').le(cfg.max_bias20)
    valid=base[known & liquid & risk].copy()
    if len(valid)<cfg.min_eligible:
        return {'state':'ELIGIBLE_COUNT_BLOCKED','eligible':len(valid),'coverage':coverage,
                'selected':[],'target_stock_fraction':0.}
    valid['score']=0.
    for f,w in ff.items():
        valid['score']+=float(w)*valid[f].rank(pct=True,method='average')
    valid=valid.sort_values(['score','ticker'],ascending=[False,True],kind='mergesort')
    chosen=[];caps={}
    for r in valid.itertuples(index=False):
        sector=str(r.sector)
        if caps.get(sector,0)>=cfg.sector_cap:continue
        chosen.append({'ticker':str(r.ticker),'sector':sector,'score':float(r.score)})
        caps[sector]=caps.get(sector,0)+1
        if len(chosen)==cfg.top_k:break
    target=_stage_budget(stage,cfg)*len(chosen)/cfg.top_k
    if not chosen:target=0.
    return {'state':'PICKS_READY' if chosen else 'NO_PICKS','eligible':len(valid),
            'coverage':coverage,'selected':chosen,'target_stock_fraction':float(target),
            'model':model,'factors':ff.copy()}


def choose_cohorts(decisions,horizon):
    """No overlapping capital: a new cohort cannot open until previous exits.

    Dataframe rows: date, entry, exit, candidate, selected, target_stock_fraction.
    Signal T is fixed month-end. T+1 adjusted open starts stock ownership.
    """
    d=decisions.sort_values('date').reset_index(drop=True)
    rows=[];last_exit=pd.Timestamp.min
    for rec in d.to_dict('records'):
        entry=pd.Timestamp(rec['entry']);end=pd.Timestamp(rec['exit'])
        if pd.isna(entry) or pd.isna(end):continue
        rec['active_cohort']=bool(entry>last_exit)
        if rec['active_cohort']:last_exit=end
        rows.append(rec)
    return pd.DataFrame(rows)


def _metric(daily,periods=252):
    if daily.empty:return {'sessions':0,'cagr':None,'mdd':None,'ir':None}
    ret=daily.strategy_nav.pct_change().fillna(0).to_numpy(float)
    ref=daily.benchmark_nav.pct_change().fillna(0).to_numpy(float)
    n=max(1,len(ret)-1);yw=n/periods
    s=float(daily.strategy_nav.iloc[-1]/daily.strategy_nav.iloc[0]);b=float(daily.benchmark_nav.iloc[-1]/daily.benchmark_nav.iloc[0])
    cagr=s**(1/yw)-1;bcagr=b**(1/yw)-1
    nav=daily.strategy_nav.to_numpy(float)
    mdd=float(np.min(nav/np.maximum.accumulate(nav)-1))
    bnav=daily.benchmark_nav.to_numpy(float)
    benchmark_mdd=float(np.min(bnav/np.maximum.accumulate(bnav)-1))
    active=ret-ref
    te=float(np.std(active,ddof=1)*np.sqrt(periods)) if len(active)>2 else 0.
    ir=float(np.mean(active)*periods/te) if te>1e-12 else None
    return {'sessions':len(daily),'cagr':float(cagr),'benchmark_cagr':float(bcagr),
            'excess_cagr_pp':float(100*(cagr-bcagr)),'mdd':mdd,'benchmark_mdd':benchmark_mdd,'ir':ir,
            'active_hit_ratio':float((active[1:]>0).mean()) if len(active)>1 else None}


def simulate_nav(calendar,benchmark,prices,cohorts,cfg,stress=False):
    """Daily NAV ledger with ETF overnight gap, open-entry, close-exit, drag/cost.

    Adj prices are a single vendor's current adjusted-vintage history and must
    pass a separate revision-vintage audit before true final acceptance.
    ETF-only cash substitute is represented as 0050 shares, NOT zero return.
    """
    cfg.validate()
    if not isinstance(cohorts,pd.DataFrame):cohorts=pd.DataFrame(cohorts)
    cal=pd.DatetimeIndex(calendar)
    if cal.has_duplicates or not cal.is_monotonic_increasing:raise DataError('Calendar invalid')
    if len(cal)<2:raise DataError('Insufficient daily trading history')
    bench=benchmark.set_index('date').reindex(cal)
    if not {'open','close'}.issubset(bench):raise DataError('ETF價格需含開盤與收盤')
    if (bench['close']<=0).any() or bench[['open','close']].isna().any().any():raise DataError('0050基準資料有缺漏，不能以零報酬補齊')
    # Convert adjusted close/open into per-stock aligned series; do not forward-fill
    # missing/suspended stock prices.
    close={};opening={}
    entries={};exits={};excluded=0
    for r in cohorts.to_dict('records') if len(cohorts) else []:
        if not r.get('active_cohort',True):continue
        fraction=float(r.get('target_stock_fraction',0))
        if fraction<=0:continue
        valid=list(r['selected'])
        if not valid:continue
        when=pd.Timestamp(r['entry']);end=pd.Timestamp(r['exit'])
        if when not in cal or end not in cal or end<when:raise DataError('持股日期不屬交易日曆')
        if when in entries or end in exits:raise DataError('持股期間重疊，不可假裝投入兩倍資金')
        for pick in valid:
            ticker=pick['ticker'] if isinstance(pick,dict) else pick
            if ticker not in close:
                if ticker not in prices:raise DataError('缺少歷史股價：'+ticker)
                p=prices[ticker].set_index('date').reindex(cal)
                close[ticker]=pd.to_numeric(p.close,errors='coerce')
                opening[ticker]=pd.to_numeric(p.open,errors='coerce')
            seg=close[ticker].loc[when:end]
            if seg.isna().any() or (seg<=0).any() or not np.isfinite(opening[ticker].loc[when]) or opening[ticker].loc[when]<=0:
                raise DataError('持股期間價格缺漏（禁止隱性補值）：'+ticker+' '+str(when.date()))
        row={**r,'selected':valid}
        entries[when]=row;exits[end]=row
    nav=1.;bench_nav=1.;etf=1.;stock_vals={};prevclose=None
    buy_stock_cost=.001425+.001
    sell_stock_cost=.001425+.003+.001
    buy_etf_cost=.001425+.001
    sell_etf_cost=.001425+.001+.001
    if stress:
        sell_stock_cost+=cfg.extra_stock_roundtrip_stress/2
        buy_stock_cost+=cfg.extra_stock_roundtrip_stress/2
    rows=[]
    for i,dt in enumerate(cal):
        bop=float(bench.loc[dt,'open']);bcc=float(bench.loc[dt,'close'])
        if i==0:
            # Initial ETF holdings start from initial adjusted close; first day is zero growth.
            prevclose=bcc
            rows.append({'date':dt,'strategy_nav':nav,'benchmark_nav':bench_nav,'alpha_weight':0.,'positions':0,'turnover_cost':0.})
            continue
        daily_cost=0.
        if dt in entries:
            if stock_vals:raise DataError('Overlap at entry; no silent capital reuse')
            row=entries[dt];fraction=float(row['target_stock_fraction'])
            # Asset held as ETF from previous close until the buy-at-open transition.
            etf*=bop/prevclose
            total=etf
            desired=total*fraction
            picks=row['selected']
            # Fixed-fee approximation; all charges taken from NAV at entry.
            daily_cost=desired*(sell_etf_cost+buy_stock_cost)
            etf=total-desired-daily_cost
            if etf<0:raise DataError('Cost-adjusted negative ETF balance')
            stock_vals={ (p['ticker'] if isinstance(p,dict) else p): desired/len(picks) for p in picks}
            for ticker in stock_vals:
                stock_vals[ticker]*=float(close[ticker].loc[dt]/opening[ticker].loc[dt])
            etf*=bcc/bop
        else:
            ratio=bcc/prevclose
            etf*=ratio
            for ticker in stock_vals:
                old_close=float(close[ticker].loc[cal[i-1]])
                new_close=float(close[ticker].loc[dt])
                if old_close<=0:raise DataError('Zero previous close')
                stock_vals[ticker]*=new_close/old_close
        if dt in exits and stock_vals:
            totalstocks=sum(stock_vals.values())
            daily_cost+=totalstocks*(sell_stock_cost+buy_etf_cost)
            etf+=totalstocks*(1-sell_stock_cost-buy_etf_cost)
            stock_vals={}
        nav=etf+sum(stock_vals.values())
        bench_nav*=bcc/prevclose
        if not (np.isfinite(nav) and nav>0):raise DataError('NAV nonfinite')
        rows.append({'date':dt,'strategy_nav':nav,'benchmark_nav':bench_nav,
                     'alpha_weight':sum(stock_vals.values())/nav if nav>0 else 0.,
                     'positions':len(stock_vals),'turnover_cost':daily_cost})
        prevclose=bcc
    df=pd.DataFrame(rows)
    return df,_metric(df)
