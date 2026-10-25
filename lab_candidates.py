"""Bounded, seeded candidate registry, generated BEFORE looking at returns."""
import hashlib
import json
import numpy as np
from lab_config import FACTORS,OPTIONAL_FACTORS


def candidate_registry(cfg):
    names=list(FACTORS)+(list(OPTIONAL_FACTORS) if cfg.enable_optional_factors else [])
    candidates=[];seen=set()
    def add(weights,label):
        if len(candidates) >= cfg.candidate_budget:return
        weights={k:float(v) for k,v in weights.items() if v>0}
        total=sum(weights.values());weights={k:round(v/total,8) for k,v in sorted(weights.items())}
        # Final normalization removes decimal serialization drift.
        total=sum(weights.values());weights={k:v/total for k,v in weights.items()}
        key=json.dumps(weights,sort_keys=True)
        if key in seen:return
        seen.add(key)
        candidates.append({'id':hashlib.sha256(key.encode()).hexdigest()[:12],
                           'name':label,'weights':weights,'complexity':len(weights)})
    add({'liquidity20':.8,'high252':.2},'v20_control_80_20')
    for n in names:add({n:1},'single_'+n)
    add({'momentum60':.5,'high252':.3,'downside60':.2},'trend_leadership')
    add({'reversal5':.4,'money_flow20':.3,'rsi_recovery':.3},'reversal_accumulation')
    add({'breakout20':.4,'volume_impulse':.3,'trend_eff60':.3},'breakout_volume')
    add({'momentum120_skip20':.4,'downside60':.3,'high252':.3},'medium_quality')
    # Non-random new economic hypotheses; none of their weights uses the tail.
    add({'residual_momentum60':.5,'residual_lowvol60':.3,'amount_acceleration':.2},'residual_quality_flow')
    add({'residual_momentum20':.4,'high252':.3,'amount_acceleration':.3},'residual_leadership_flow')
    add({'residual_momentum60':.5,'momentum120_skip20':.3,'downside60':.2},'residual_persistence')
    add({'reversal5':.3,'residual_momentum60':.4,'money_flow20':.3},'pullback_in_residual_trend')
    rng=np.random.default_rng(cfg.seed)
    while len(candidates)<cfg.candidate_budget:
        k=int(rng.integers(2,cfg.max_active_factors+1)) if cfg.max_active_factors>=2 else 1
        selected=rng.choice(names,k,replace=False)
        if k==1:
            if len(seen)>=len(names)+5:break
            continue
        # Coarse tenths, no arbitrary 3-decimal precision search.
        cuts=np.sort(rng.choice(np.arange(1,10),k-1,replace=False))
        ws=np.diff(np.r_[0,cuts,10])/10
        if max(ws)>.7:continue
        add(dict(zip(selected,ws)),'sparse_'+str(len(candidates)))
    return candidates
