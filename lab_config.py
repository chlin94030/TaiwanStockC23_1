"""Finite universe/model selection protocol. Defaults are not performance claims."""
from dataclasses import dataclass, asdict, fields
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import hashlib
import json

VERSION = '22.2-yahoo-coverage-fix'
FACTORS = (
    'liquidity20', 'high252', 'momentum20', 'momentum60',
    'momentum120_skip20', 'trend_eff60', 'downside60', 'reversal5',
    'volume_impulse', 'money_flow20', 'breakout20', 'pullback20',
    'rsi_recovery', 'macd_acceleration',
    'residual_momentum20', 'residual_momentum60',
    'residual_lowvol60', 'amount_acceleration',
)
OPTIONAL_FACTORS = ('revenue_yoy', 'eps_growth', 'foreign_flow20', 'trust_flow20')

@dataclass(frozen=True)
class Config:
    research_action: str = 'universe_search'

    start: str = '2018-01-01'
    end: str = ''
    pools: tuple = (100, 200, 300, 500)
    horizons: tuple = (5, 20, 60)
    top_k: int = 10
    lookback: int = 252
    recent_amount_floor: float = 5_000_000.
    active_ratio60: float = .95
    candidate_budget: int = 64
    max_active_factors: int = 4
    seed: int = 210031
    min_train_months: int = 24
    outer_test_months: int = 6
    min_outer_folds: int = 3
    holdout_months: int = 24
    gap_sessions: int = 5
    min_fold_coverage: float = .95
    min_feature_coverage: float = .98
    fee_buy: float = .001425
    fee_sell: float = .001425
    stock_tax: float = .003
    benchmark_tax: float = .001
    slippage: float = .001
    extra_roundtrip_stress: float = .005
    max_gap: float = .04
    capital: float = 1_000_000.
    max_amount_participation: float = .005
    block_bootstrap_samples: int = 2000
    block_months: int = 4
    min_disjoint_cases: int = 6
    economic_excess_floor: float = .002
    near_best_tolerance: float = .001
    complexity_penalty: float = .0005
    uncertainty_penalty: float = .5
    source_mode: str = 'auto_current_roster'
    # AUTO prefers FinLab whole-market tables when authenticated, then falls back
    # to FinMind per-symbol history. This changes transport, not the research rule.
    history_provider: str = 'AUTO_FAST'
    finmind_quota_utilization: float = .90
    min_current_roster: int = 1500
    requested_scope: str = 'CURRENT_OFFICIAL_SNAPSHOT'
    # All old provider history through this date has been exposed in development.
    exposed_through: str = '2026-10-20'
    include_market_cap_pools: bool = False
    enable_optional_factors: bool = False
    request_timeout: int = 25
    # Registered BEFORE new outcomes; not selected by exposed final-tail returns.
    selector: str = 'RECENT_AND_FULL_MAXIMIN_WITH_TRAINING_CONSENSUS'
    recent_train_months: int = 24
    min_positive_inner_blocks: int = 2
    consensus_members: int = 3
    consensus_return_corr_limit: float = .90
    freeze_existing_data_end: bool = True

    def resolved(self):
        d = asdict(self)
        if not self.end:
            d['end'] = (datetime.now(ZoneInfo('Asia/Taipei')).date()-timedelta(days=1)).isoformat()
        for k in ('pools','horizons'): d[k] = tuple(d[k])
        c = Config(**d)
        c.validate()
        return c

    def validate(self):
        if self.research_action not in ('universe_search','prepare_data'):
            raise ValueError('Unknown research_action')

        if self.lookback != 252: raise ValueError('A one-year pool requires 252 market sessions.')
        if not self.pools or any(n < self.top_k for n in self.pools): raise ValueError('Invalid pool sizes')
        if len(set(self.pools)) != len(self.pools): raise ValueError('Duplicate pools')
        if not self.horizons or any(h not in (5,20,60) for h in self.horizons): raise ValueError('Use 5/20/60 horizons')
        if not 20 <= self.candidate_budget <= 128: raise ValueError('Candidate budget must be 20..128')
        if not 2 <= self.max_active_factors <= 4: raise ValueError('Sparse weights use at most four factors')
        if self.holdout_months < 6 or self.min_train_months < 12: raise ValueError('Insufficient chronological design')
        if not 0 < self.min_fold_coverage <= 1 or not 0 < self.min_feature_coverage <= 1: raise ValueError('Invalid coverage')
        if self.source_mode not in ('auto_current_roster','import'): raise ValueError('Invalid source mode')
        if self.history_provider not in ('AUTO','AUTO_FAST','FINLAB_BULK','OFFICIAL_BULK','YAHOO_FAST','FINMIND'): raise ValueError('Invalid history provider')
        if not .50 <= self.finmind_quota_utilization <= .95: raise ValueError('FinMind quota utilization must be 0.50..0.95')
        if self.requested_scope not in ('CURRENT_OFFICIAL_SNAPSHOT','HISTORICAL_MEMBERSHIP'): raise ValueError('Invalid scope')
        if self.capital <= 0 or not 0 < self.max_amount_participation < 1: raise ValueError('Invalid capacity')
        if min(self.fee_buy,self.fee_sell,self.stock_tax,self.benchmark_tax,self.slippage,self.extra_roundtrip_stress) < 0: raise ValueError('Costs cannot be negative')
        if self.selector != 'RECENT_AND_FULL_MAXIMIN_WITH_TRAINING_CONSENSUS':
            raise ValueError('Unknown registered selector')
        if self.recent_train_months < 12: raise ValueError('Recent window must be >=12 months')
        if not 1 <= self.min_positive_inner_blocks <= 3: raise ValueError('Invalid inner block requirement')
        if not 1 <= self.consensus_members <= 3: raise ValueError('Consensus budget is at most three members')
        if not 0 < self.consensus_return_corr_limit <= 1: raise ValueError('Invalid consensus correlation bound')
        if self.top_k < 1: raise ValueError('At least one funded slot is required')
        if self.start >= self.end: raise ValueError('Start must precede end')
        if self.min_outer_folds < 1 or self.outer_test_months < 3 or self.gap_sessions < 0: raise ValueError('Invalid folds')
        if self.requested_scope == 'HISTORICAL_MEMBERSHIP' and self.source_mode != 'import':
            raise ValueError('Verified historical membership requires a supplied membership file; a current roster is not a replacement.')
        if self.enable_optional_factors and self.candidate_budget < 24:raise ValueError('Optional registry needs at least 24 candidate slots')
        if self.enable_optional_factors and self.source_mode != 'import':
            raise ValueError('Optional factors require a point-in-time evidence.csv import.')

    def to_dict(self): return asdict(self)
    def digest(self):
        return hashlib.sha256(json.dumps(self.to_dict(),sort_keys=True).encode()).hexdigest()

def load_config(path=None):
    if path:
        with open(path, encoding='utf-8') as f: obj=json.load(f)
        allowed={f.name for f in fields(Config)}
        extra=set(obj)-allowed
        if extra: raise ValueError('Unknown config keys: '+str(sorted(extra)))
        return Config(**obj).resolved()
    return Config().resolved()
