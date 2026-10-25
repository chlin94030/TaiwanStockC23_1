"""V22.3 adaptive challengers inspired by the uploaded Claude engine.

The important constraint is chronology: adaptive weights/models may use only
outcomes whose holding window has already ended before the current signal date.
They never use the current/future label, and they never change the V22 market
stage gate or fixed-pool membership.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
import pandas as pd

MODEL_TYPES = ('manual', 'hybrid_ic', 'ic_adaptive', 'gbm_challenger')

DEFAULT_ADAPTIVE = {
    'blend': 0.35,
    'lookback_cases': 36,
    'min_history_cases': 12,
    'stage_mix': 0.60,
    'ic_floor': 0.0,
    'ic_cap': 0.25,
    'prior_floor': 0.10,
}


def weights_for(candidate, horizon, stage):
    """Return the frozen prior weights for one horizon/stage."""
    by_h = candidate.get('weights_by_horizon', {})
    h = int(horizon)
    if h in by_h:
        return by_h[h].get(stage)
    if str(h) in by_h:
        return by_h[str(h)].get(stage)
    # Backward-compatible internal representation.
    return candidate.get('stage_weights', {}).get(stage)


def _rank_corr(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 20:
        return np.nan
    a = pd.Series(x[m]).rank(method='average').to_numpy(float)
    b = pd.Series(y[m]).rank(method='average').to_numpy(float)
    a -= a.mean(); b -= b.mean()
    den = math.sqrt(float(np.dot(a, a) * np.dot(b, b)))
    if den <= 0:
        return np.nan
    return float(np.dot(a, b) / den)


def matured_history_indices(cases, i, lookback_cases=36, stage=None):
    """Cases are eligible only after their label exit is before the new signal."""
    now = pd.Timestamp(cases[int(i)].base.date)
    idx = []
    for j in range(int(i)):
        c = cases[j]
        if pd.isna(c.base.exit) or pd.Timestamp(c.base.exit) >= now:
            continue
        if stage is not None and c.stage != stage:
            continue
        idx.append(j)
    if lookback_cases and len(idx) > int(lookback_cases):
        idx = idx[-int(lookback_cases):]
    return idx


def _factor_ic(cases, idx, factor):
    vals = []
    for j in idx:
        c = cases[int(j)]
        try:
            fi = c.base.features.index(factor)
        except ValueError:
            continue
        ic = _rank_corr(c.base.ranks[:, fi], c.base.net)
        if np.isfinite(ic):
            vals.append(float(ic))
    if not vals:
        return {'mean_ic': np.nan, 'positive_ratio': np.nan, 'cases': 0}
    a = np.asarray(vals, float)
    return {'mean_ic': float(a.mean()), 'positive_ratio': float((a > 0).mean()), 'cases': int(len(a))}


def ic_diagnostics(cases, i, prior, stage, adaptive=None):
    a = {**DEFAULT_ADAPTIVE, **(adaptive or {})}
    global_idx = matured_history_indices(cases, i, a['lookback_cases'])
    stage_idx = matured_history_indices(cases, i, a['lookback_cases'], stage=stage)
    rows = {}
    for factor in prior:
        g = _factor_ic(cases, global_idx, factor)
        s = _factor_ic(cases, stage_idx, factor)
        use_stage = s['cases'] >= max(4, int(a['min_history_cases']) // 3)
        if use_stage and np.isfinite(s['mean_ic']):
            mix = float(a['stage_mix'])
            mean_ic = mix * s['mean_ic'] + (1 - mix) * (g['mean_ic'] if np.isfinite(g['mean_ic']) else s['mean_ic'])
            pr = mix * s['positive_ratio'] + (1 - mix) * (g['positive_ratio'] if np.isfinite(g['positive_ratio']) else s['positive_ratio'])
        else:
            mean_ic, pr = g['mean_ic'], g['positive_ratio']
        rows[factor] = {
            'mean_ic': mean_ic,
            'positive_ratio': pr,
            'global_cases': g['cases'],
            'stage_cases': s['cases'],
            'stage_component_used': bool(use_stage),
        }
    return rows, global_idx, stage_idx


def adaptive_weights(cases, i, candidate):
    sc = cases[int(i)]
    h = int(getattr(sc, 'horizon', 0) or 0)
    stage = sc.stage
    prior = weights_for(candidate, h, stage)
    if not prior:
        return None, {'history_cases': 0, 'fallback': 'NO_STAGE_PRIOR'}
    model_type = candidate.get('model_type', 'manual')
    if model_type == 'manual':
        return dict(prior), {'history_cases': 0, 'fallback': 'MANUAL_PRIOR', 'model_type': model_type}

    a = {**DEFAULT_ADAPTIVE, **candidate.get('adaptive', {})}
    diag, global_idx, stage_idx = ic_diagnostics(cases, i, prior, stage, a)
    if len(global_idx) < int(a['min_history_cases']):
        return dict(prior), {'history_cases': len(global_idx), 'stage_history_cases': len(stage_idx),
                             'fallback': 'INSUFFICIENT_MATURED_HISTORY', 'model_type': model_type,
                             'factor_ic': diag}
    raw = {}
    for factor in prior:
        ic = diag[factor]['mean_ic']
        pr = diag[factor]['positive_ratio']
        if not np.isfinite(ic) or not np.isfinite(pr):
            raw[factor] = 0.0
            continue
        ic = float(np.clip(ic, float(a['ic_floor']), float(a['ic_cap'])))
        # Reliability multiplier prevents one lucky positive month from dominating.
        reliability = float(np.clip((pr - .40) / .40, 0.0, 1.0))
        raw[factor] = max(0.0, ic) * reliability
    total = sum(raw.values())
    if total <= 0:
        return dict(prior), {'history_cases': len(global_idx), 'stage_history_cases': len(stage_idx),
                             'fallback': 'NO_POSITIVE_IC_SUPPORT', 'model_type': model_type,
                             'factor_ic': diag}
    icw = {k: v / total for k, v in raw.items()}
    blend = float(a.get('blend', .35))
    if model_type == 'ic_adaptive':
        blend = max(blend, .60)
    elif model_type == 'hybrid_ic':
        blend = min(blend, .50)
    floor = float(a.get('prior_floor', .10))
    w = {k: (1 - blend) * float(prior[k]) + blend * float(icw.get(k, 0.0)) for k in prior}
    # Preserve a small fraction of the book prior so the adaptive layer cannot
    # silently become a completely different factor model.
    for k in prior:
        w[k] = max(w[k], floor * float(prior[k]))
    z = sum(w.values())
    w = {k: float(v / z) for k, v in w.items()}
    return w, {'history_cases': len(global_idx), 'stage_history_cases': len(stage_idx),
               'fallback': None, 'model_type': model_type, 'blend': blend,
               'factor_ic': diag, 'ic_weights': icw}


def factor_ic_summary(cases, indices):
    """Exposed-development diagnostic, never a fresh holdout claim."""
    rows = []
    if not cases:
        return rows
    factors = tuple(cases[0].base.features)
    stages = ('ALL', 'A1', 'A2', 'A3', 'B1', 'B2', 'B3')
    idx = [int(i) for i in indices]
    for stage in stages:
        use = idx if stage == 'ALL' else [i for i in idx if cases[i].stage == stage]
        for factor in factors:
            q = _factor_ic(cases, use, factor)
            rows.append({'stage': stage, 'factor': factor, 'cases': q['cases'],
                         'mean_rank_ic': q['mean_ic'], 'positive_ratio': q['positive_ratio'],
                         'scope': 'EXPOSED_DEVELOPMENT_DIAGNOSTIC_NOT_CAUSAL'})
    return rows


def static_weight_suggestion(cases, indices, candidate, horizon, blend=.40):
    """Suggest, but do not automatically activate, a static next-iteration profile."""
    rows = []
    indices = [int(i) for i in indices]
    for stage in ('A1', 'A2', 'B3'):
        prior = weights_for(candidate, horizon, stage)
        if not prior:
            continue
        use = [i for i in indices if cases[i].stage == stage]
        raw = {}
        for factor in prior:
            q = _factor_ic(cases, use, factor)
            ic = q['mean_ic']; pr = q['positive_ratio']
            raw[factor] = 0.0 if not np.isfinite(ic) or not np.isfinite(pr) else max(0., ic) * max(0., pr - .4)
        if sum(raw.values()) > 0:
            q = {k: v / sum(raw.values()) for k, v in raw.items()}
            w = {k: (1-blend)*prior[k] + blend*q.get(k, 0.) for k in prior}
            z = sum(w.values()); w = {k: float(v/z) for k,v in w.items()}
        else:
            w = dict(prior)
        for factor in prior:
            rows.append({'horizon': int(horizon), 'stage': stage, 'factor': factor,
                         'prior_weight': float(prior[factor]), 'suggested_weight': float(w[factor]),
                         'auto_activated': False,
                         'scope': 'NEXT_ITERATION_SUGGESTION_FROM_EXPOSED_DEVELOPMENT'})
    return rows


def gbm_score(cases, i, candidate):
    """Optional research-only walk-forward challenger from the Claude design.

    Uses only matured past labels. The default V22.3 profile file does not enable
    this challenger, keeping normal tuning fast.
    """
    sc = cases[int(i)]
    h = int(getattr(sc, 'horizon', 0) or 0)
    prior = weights_for(candidate, h, sc.stage)
    if not prior:
        return None, {'fallback': 'NO_STAGE_PRIOR'}
    settings = {'lookback_cases': 36, 'min_history_cases': 12, 'max_iter': 80,
                'max_depth': 3, 'learning_rate': .05, 'l2_regularization': 1.0,
                **candidate.get('gbm', {})}
    idx = matured_history_indices(cases, i, settings['lookback_cases'])
    if len(idx) < int(settings['min_history_cases']):
        return None, {'fallback': 'INSUFFICIENT_MATURED_HISTORY', 'history_cases': len(idx)}
    support = [f for f in prior if f in sc.base.features]
    fi = [sc.base.features.index(f) for f in support]
    X=[];Y=[]
    for j in idx:
        c = cases[j]
        x = c.base.ranks[:, fi]
        y = c.base.net
        m = np.isfinite(y) & np.isfinite(x).all(axis=1)
        if int(m.sum()) < 20: continue
        # Cross-sectional target removes market-level return from the stock model.
        yr = pd.Series(y[m]).rank(pct=True, method='average').to_numpy(float) - .5
        X.append(x[m]); Y.append(yr)
    if not X or sum(len(x) for x in X) < 800:
        return None, {'fallback': 'INSUFFICIENT_TRAINING_ROWS', 'history_cases': len(idx)}
    from sklearn.ensemble import HistGradientBoostingRegressor
    model = HistGradientBoostingRegressor(max_iter=int(settings['max_iter']), max_depth=int(settings['max_depth']),
        learning_rate=float(settings['learning_rate']), l2_regularization=float(settings['l2_regularization']), random_state=0)
    model.fit(np.vstack(X), np.concatenate(Y))
    cur = sc.base.ranks[:, fi]
    valid = np.isfinite(cur).all(axis=1)
    score = np.full(len(cur), np.nan)
    if valid.any(): score[valid] = model.predict(cur[valid])
    return score, {'fallback': None, 'history_cases': len(idx), 'support': support,
                   'training_rows': int(sum(len(x) for x in X))}
