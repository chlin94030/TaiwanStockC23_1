"""Pre-final, evidence-based activation safeguards.

The research challenger is distinct from permission to use it as an active
allocation. Neither this gate nor its parameters observes the final tail.
The 0050 benchmark is a reference, not an alpha result or trading instruction.
"""

import math

REFERENCE_ID = 'REFERENCE_0050'


def reference_choice(reason='NO_TRAINING_ALPHA'):
    return {
        'pool': None,
        'candidate_index': -1,
        'candidate_id': REFERENCE_ID,
        'name': 'benchmark_reference_no_stock_alpha',
        'weights': {},
        'complexity': 0,
        'objective': 0.0,
        'train_excess': 0.0,
        'missing_slot_loss_floor_excess': 0.0,
        'train_stress_excess': 0.0,
        'inner_block_means': [0.0, 0.0, 0.0],
        'coverage': 1.0,
        'train_dates': 0,
        'reason': reason,
    }


def is_reference(choice):
    return choice.get('candidate_id') == REFERENCE_ID


def _number(value):
    try:
        a = float(value)
        return a if math.isfinite(a) else None
    except (ValueError, TypeError):
        return None


def pre_tail_gate(choice, outer_summary, outer_folds, cfg):
    """No holdout data enters this gate.

    Require multiple independent-looking pre-tail indicators, but do not
    mistake them for formal independence or future return guarantees.
    The conservative response to failure is research-only; it is NOT proof
    that holding the reference will produce positive absolute returns.
    """
    failed = []
    required = {
        'mean_excess_floor': cfg.economic_excess_floor,
        'coverage_floor': cfg.min_fold_coverage,
        'minimum_disjoint': cfg.min_disjoint_cases,
        'minimum_positive_fold_fraction': 0.60,
        'training_min_positive_inner_blocks': 2,
        'block_95_lower_gt_zero': 0.0,
    }
    if is_reference(choice):
        return {'status': 'NO_ACTIVE_ALPHA', 'failed_checks': ['TRAINING_CHOSE_REFERENCE'],
                'required': required, 'source': 'TRAIN_AND_OUTER_ONLY',
                'final_tail_used': False, 'fallback': REFERENCE_ID}

    objective = _number(choice.get('objective'))
    stress = _number(choice.get('train_stress_excess'))
    blocks = [v for v in (_number(x) for x in choice.get('inner_block_means', [])) if v is not None]
    if objective is None or objective <= 0: failed.append('NONPOSITIVE_TRAINING_OBJECTIVE')
    if stress is None or stress <= cfg.economic_excess_floor:
        failed.append('TRAINING_COST_STRESS_NOT_ECONOMICALLY_POSITIVE')
    if len(blocks) < 3 or sum(v > 0 for v in blocks) < 2:
        failed.append('TRAINING_TIME_BLOCKS_UNSTABLE')

    for key in ('versus_0050', 'versus_same_pool'):
        s = outer_summary.get(key, {})
        if (_number(s.get('coverage')) or 0) < cfg.min_fold_coverage:
            failed.append(key.upper() + '_INSUFFICIENT_COVERAGE')
        if int(s.get('disjoint_complete') or 0) < cfg.min_disjoint_cases:
            failed.append(key.upper() + '_INSUFFICIENT_NONOVERLAP')
        if (_number(s.get('mean')) or -math.inf) <= cfg.economic_excess_floor:
            failed.append(key.upper() + '_INSUFFICIENT_MEAN_ALPHA')
        if (_number(s.get('block95_low')) or -math.inf) <= 0:
            failed.append(key.upper() + '_BLOCK_95_NOT_POSITIVE')
        if (_number(s.get('disjoint95_low')) or -math.inf) <= 0:
            failed.append(key.upper() + '_NONOVERLAP_95_NOT_POSITIVE')

    if (_number(outer_summary.get('extra_cost_stress_excess')) or -math.inf) <= 0:
        failed.append('OUTER_EXTRA_COST_STRESS_NOT_POSITIVE')
    evaluated = [f for f in outer_folds if f.get('status') == 'EVALUATED']
    positives = sum((_number(f.get('test', {}).get('versus_0050', {}).get('mean')) or -math.inf) > 0 for f in evaluated)
    fraction = positives / len(evaluated) if evaluated else 0.0
    if fraction < required['minimum_positive_fold_fraction']:
        failed.append('OUTER_FOLD_DIRECTION_UNSTABLE')
    return {
        'status': 'RESEARCH_CANDIDATE_NOT_PROMOTED' if failed else 'PRETAIL_RESEARCH_GATE_PASSED',
        'failed_checks': sorted(set(failed)),
        'required': required,
        'outer_evaluated_folds': len(evaluated),
        'outer_positive_folds': positives,
        'outer_positive_fraction': fraction,
        'source': 'TRAIN_AND_OUTER_ONLY',
        'final_tail_used': False,
        'fallback': REFERENCE_ID if failed else None,
        'automatic_orders_authorized': False,
    }


def equivalent_training_pools(grids, choice, train_idx, cases=None, cfg=None):
    """Economic outcome equivalence, not a claim of identical stock picks."""
    if is_reference(choice):
        return []
    key = (choice['pool'], None)
    target = None
    for (pool, horizon), g in grids.items():
        if pool == choice['pool']:
            key = (pool, horizon)
            target = g
            break
    if target is None:
        return []
    j = choice['candidate_index']
    if j < 0:
        if cases is None or cfg is None:
            return []
        from lab_backtest import evaluate
        import numpy as np
        def vals(pool):
            rs=[evaluate(cases[(pool,key[1])][int(i)],choice['weights'],cfg) for i in train_idx]
            return np.array([[r[x] for x in ('net','loss_floor','filled_weight','feature_coverage')] for r in rs],float)
        target_values=vals(choice['pool'])
        return sorted(pool for (pool,h) in grids if h==key[1] and
                      np.allclose(target_values,vals(pool),equal_nan=True,atol=1e-12,rtol=0))
    arrays = ('net', 'floor', 'fills', 'feature_coverage')
    import numpy as np
    same = []
    for (pool, h), other in grids.items():
        if h != key[1]:
            continue
        if all(np.allclose(target[a][train_idx, j], other[a][train_idx, j],
                           equal_nan=True, atol=1e-12, rtol=0) for a in arrays):
            same.append(pool)
    return sorted(same)
