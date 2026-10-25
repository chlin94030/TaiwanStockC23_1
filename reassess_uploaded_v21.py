"""Read an existing V21 LAB_RESULTS.zip without retraining or using tail for selection.

Example:
    python reassess_uploaded_v21.py LAB_RESULTS.zip --out-dir review

This is a diagnostic replay of the *old* run's observed metrics. It is NOT a
new strategy backtest, fresh holdout or validation of revised factor weights.
"""
import argparse
import ast
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

from lab_approval import pre_tail_gate
from lab_config import Config


def _r(v):
    return round(v * 100, 3) if isinstance(v, (int, float)) and math.isfinite(v) else None


def _read(archive):
    files = {}
    with zipfile.ZipFile(archive) as z:
        if z.testzip():
            raise ValueError('Damaged report archive')
        for name in z.namelist():
            if name.startswith('/') or '..' in Path(name).parts:
                raise ValueError('Unsafe report member')
            if z.getinfo(name).file_size > 20_000_000:
                raise ValueError('Unexpectedly large report member')
            files[name] = z.read(name)
    manifest = json.loads(files['EXPORT_SHA256.json'])
    for name, expected in manifest.items():
        if name not in files or hashlib.sha256(files[name]).hexdigest() != expected:
            raise ValueError('Report integrity failure at ' + name)
    return files


def _csv(files, name):
    if name not in files:
        return []
    return list(csv.DictReader(io.StringIO(files[name].decode('utf-8-sig'))))


def review(archive):
    files = _read(archive)
    report = json.loads(files['results.json'])
    cfg = Config(**report['config'])
    feature = sorted(_csv(files, 'feature_coverage.csv'), key=lambda x: float(x['coverage']))
    pool = _csv(files, 'pool_control_comparison.csv')
    tails = _csv(files, 'final_tail_records.csv')
    models = {}
    for horizon, entry in sorted(report['models'].items(), key=lambda x: int(x[0])):
        choice = entry['freeze']['choice']
        gate = pre_tail_gate(choice, entry['outer_oos'], entry['outer_folds'], cfg)
        tail = entry['final_tail']
        candidates = _csv(files, f'development_candidates_{horizon}.csv')
        equivalent = []
        for r in candidates:
            if r['candidate_id'] != choice['candidate_id']:
                continue
            if (abs(float(r['objective']) - float(choice['objective'])) < 1e-12
                and abs(float(r['train_excess']) - float(choice['train_excess'])) < 1e-12
                and abs(float(r['train_stress_excess']) - float(choice['train_stress_excess'])) < 1e-12):
                equivalent.append(r['pool'])
        held = Counter()
        for rec in tails:
            if rec.get('horizon') != horizon:
                continue
            try:
                symbols = ast.literal_eval(rec['selected'])
                if isinstance(symbols, list):
                    held.update(str(s) for s in symbols)
            except (ValueError, SyntaxError, TypeError):
                pass
        x = tail['versus_0050']
        y = tail['versus_same_pool']
        models[horizon] = {
            'training_pool': choice['pool'],
            'training_candidate': choice['name'],
            'training_weights': choice['weights'],
            'training_objective_pct': _r(choice['objective']),
            'training_excess_pct': _r(choice['train_excess']),
            'outer_oos_excess_pct': _r(entry['outer_oos']['versus_0050']['mean']),
            'final_tail_excess_pct': _r(x['mean']),
            'final_tail_vs_pool_pct': _r(y['mean']),
            'final_tail_95pct_block_lower_pct': _r(x.get('block95_low')),
            'final_tail_known': x.get('complete'),
            'final_tail_planned': x.get('planned'),
            'final_tail_same_pool_known': y.get('complete'),
            'final_tail_nonoverlap_complete': x.get('disjoint_complete'),
            'pre_tail_gate_reassessment_on_old_exposed_results': gate,
            'economic_training_pool_equivalents_not_holding_proof': sorted(set(equivalent)),
            'most_frequent_holdings': held.most_common(5),
            'original_verdict': entry['verdict'],
        }
    return {
        'kind': 'READ_ONLY_REASSESSMENT_OF_OLD_EXPOSED_REPORT',
        'run_version': report.get('version'),
        'existing_export_sha256_checked': True,
        'model_weights_changed': False,
        'new_returns_computed': False,
        'new_independent_holdout': False,
        'data_scope': report['data_scope'],
        'models': models,
        'factor_coverage': [
            {'factor': x['factor'], 'coverage_pct': _r(float(x['coverage']))}
            for x in feature
        ],
        'pool_control_completeness': [
            {'horizon': x['horizon'], 'pool': x['pool'], 'comparable_pool_dates': int(x['pool_complete']),
             'preholdout_dates': int(x['dates'])}
            for x in pool
        ],
    }


def render_text(result):
    lines = [
        'Alpha Radar V21.3: read-only review of existing V21.2 evidence',
        'Integrity: source export hashes checked.',
        'NOT a new alpha validation. Original final-tail outcomes remain exposed.',
        'No new winning weights inferred from the final tail.', '',
    ]
    for h, r in result['models'].items():
        g = r['pre_tail_gate_reassessment_on_old_exposed_results']
        lines += [
            f'[{h} trading sessions]',
            'Old research candidate: ' + r['training_pool'] + ' / ' + r['training_candidate'],
            'Weights: ' + json.dumps(r['training_weights'], ensure_ascii=False),
            f"Train vs 0050: {r['training_excess_pct']}%",
            f"Outer OOS vs 0050: {r['outer_oos_excess_pct']}%",
            f"Old final tail vs 0050: {r['final_tail_excess_pct']}%",
            f"Old final tail vs own pool: {r['final_tail_vs_pool_pct']}%",
            f"Known tail vs 0050: {r['final_tail_known']}/{r['final_tail_planned']}",
            f"Known tail vs same pool: {r['final_tail_same_pool_known']}/{r['final_tail_planned']}",
            f"Gate using training + outer only: {g['status']}",
            'Gate blockers: ' + (', '.join(g['failed_checks']) or 'None'),
            'Economically indistinguishable training pool outcomes: '
            + (', '.join(r['economic_training_pool_equivalents_not_holding_proof']) or 'None'),
            'Repeated holdings: ' + str(r['most_frequent_holdings']), '',
        ]
    lines += ['[Observed factor availability]'] + [
        f"{x['factor']}: {x['coverage_pct']}%" for x in result['factor_coverage']
    ]
    lines += ['', '[Boundaries]',
              'Current-roster survivorship bias remains.',
              'Same-pool comparison is incomplete when even one member has an unknown exit.',
              'An always-0050 reference has zero excess BY DEFINITION, not earned alpha.',
              'No factor weight can be honestly claimed optimal from this aggregate export.']
    return '\n'.join(lines) + '\n'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('archive')
    p.add_argument('--out-dir', default='.')
    a = p.parse_args()
    result = review(a.archive)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'V21_3_EVIDENCE_REVIEW.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    (out / 'V21_3_EVIDENCE_REVIEW.txt').write_text(render_text(result), encoding='utf-8')
    print(render_text(result))


if __name__ == '__main__':
    main()
