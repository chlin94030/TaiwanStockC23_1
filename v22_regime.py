"""V22 Kostolany stage engine.

The six stages are a quantitative research translation of the user's supplied
book excerpts, not formulas printed by Kostolany. Holder counts and news
reaction are not observed in this dataset, so breadth/turnover are disclosed
proxies rather than literal investor-identity measurements.
"""
from __future__ import annotations
import math
import numpy as np

STAGES=('A1','A2','A3','B1','B2','B3')
ACTIONS={
    'A1':'BUY_ACCUMULATE',
    'A2':'HOLD_WAIT_PROXY',
    'A3':'SELL_AVOID_NEW_BUYS',
    'B1':'SELL_AVOID_NEW_BUYS',
    'B2':'WAIT_CASH',
    'B3':'BUY_CONTRARIAN_WITH_CONFIRMATION',
    'UNKNOWN':'WAIT_CASH',
}
DEFAULT_EXPOSURE={'A1':1.0,'A2':0.85,'A3':0.0,'B1':0.0,'B2':0.0,'B3':0.60,'UNKNOWN':0.0}


def _clip(x,lo=-1.,hi=1.):
    if x is None or not np.isfinite(x): return np.nan
    return float(np.clip(x,lo,hi))

def _safe_log_ratio(x,scale):
    if x is None or not np.isfinite(x) or x<=0:return np.nan
    return _clip(math.log(x)/math.log(scale))

def _need(row,names):
    vals=[row.get(k,np.nan) for k in names]
    return vals if all(np.isfinite(v) for v in vals) else None


def stage_scores(row):
    """Return six nonnegative stage scores from date-T market/pool inputs only."""
    names=('market_drawdown60','market_bias20','market_above200','market_ret5','market_ret20','market_ret60',
           'market_amount_ratio5_60','market_vol_ratio','breadth','breadth_mom20','breadth_mom60',
           'median_demand_balance','median_amount_acceleration')
    vals=_need(row,names)
    if vals is None:return {s:np.nan for s in STAGES}
    dd,bias,above,r5,r20,r60,ar,vr,breadth,b20,b60,demand,amtacc=vals
    deep=_clip(-dd/.20,0,1); mild=_clip((-dd-.03)/.12,0,1)
    up5=_clip(r5/.04,0,1); down5=_clip(-r5/.04,0,1)
    up20=_clip(r20/.10,0,1); down20=_clip(-r20/.10,0,1)
    up60=_clip(r60/.20,0,1); down60=_clip(-r60/.20,0,1)
    hot=_clip((bias-.04)/.10,0,1)
    turn_hot=_clip((_safe_log_ratio(ar,2.0) or 0),0,1)
    turn_cold=_clip(-(_safe_log_ratio(ar,2.0) or 0),0,1)
    vol_hot=_clip((vr-1.0)/1.0,0,1)
    broad_hi=_clip((breadth-.55)/.35,0,1); broad_lo=_clip((.45-breadth)/.35,0,1)
    b20_hi=_clip((b20-.50)/.35,0,1); b20_lo=_clip((.50-b20)/.35,0,1)
    b60_hi=_clip((b60-.50)/.35,0,1); b60_lo=_clip((.50-b60)/.35,0,1)
    demand_pos=_clip(demand/.60,0,1); demand_neg=_clip(-demand/.60,0,1)
    acc_pos=_clip(amtacc/.50,0,1); acc_neg=_clip(-amtacc/.50,0,1)
    below=1-float(above); above=float(above)

    # A1: correction/turning phase. Price is still depressed, turnover is subdued,
    # but short-term price/breadth/demand begin to stabilize.
    A1=.25*mild+.18*turn_cold+.16*up5+.14*demand_pos+.12*b20_hi+.08*below+.07*acc_neg
    # A2: rising/following phase. Broad participation and persistent trend matter.
    A2=.25*up20+.18*up60+.16*above+.16*broad_hi+.10*b20_hi+.08*demand_pos+.07*(1-hot)
    # A3: bull overheat. Price extension + active turnover + broad/crowded participation.
    A3=.24*hot+.19*turn_hot+.14*vol_hot+.16*up20+.12*broad_hi+.08*acc_pos+.07*above
    # B1: first decline after a mature rise. Trend weakens while the market is not yet deeply washed out.
    B1=.24*down5+.17*down20+.16*above+.14*b20_lo+.10*mild+.10*(1-deep)+.09*turn_cold
    # B2: bear follow-through. Persistent decline + weak breadth; turnover may expand as holders retreat.
    B2=.22*down20+.18*down60+.15*below+.15*broad_lo+.10*b60_lo+.08*demand_neg+.05*acc_pos+.07*down5
    # B3: capitulation/overheated decline. Deep drawdown, broad weakness and heavy turnover;
    # the action engine still requires stock-level confirmation before a funded slot is used.
    B3=.28*deep+.22*turn_hot+.15*vol_hot+.12*broad_lo+.08*b60_lo+.06*demand_neg+.09*up5
    return {k:float(max(0,v)) for k,v in zip(STAGES,(A1,A2,A3,B1,B2,B3))}


def classify_stage(row,previous=None):
    scores=stage_scores(row)
    if not scores or not all(np.isfinite(v) for v in scores.values()):
        return {'stage':'UNKNOWN','confidence':0.0,'action':ACTIONS['UNKNOWN'],'exposure':0.0,'scores':scores,'transition_guarded':False}
    order=sorted(scores,key=lambda s:(-scores[s],STAGES.index(s)))
    top,second=order[:2];total=sum(scores.values())+1e-12
    confidence=float(np.clip((scores[top]-scores[second])/(max(scores[top],1e-12)),0,1))
    guarded=False
    if previous in STAGES and top!=previous:
        pi=STAGES.index(previous);ti=STAGES.index(top)
        allowed={previous,STAGES[(pi+1)%6],STAGES[(pi-1)%6]}
        # The book describes a cycle. Reject large jumps unless evidence is clearly stronger.
        if top not in allowed and scores[top] < scores[previous]+.18:
            top=previous;guarded=True
            confidence=float(np.clip((scores[top]-scores[second])/max(scores[top],1e-12),0,1))
    return {'stage':top,'confidence':confidence,'action':ACTIONS[top],
            'exposure':DEFAULT_EXPOSURE[top],'scores':scores,'transition_guarded':guarded,
            'proxy_holder_counts':True,'news_reaction_observed':False,'macro_money_supply_observed':False}
