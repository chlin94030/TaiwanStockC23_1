"""V22.4.2 one-click guided validation pipeline.

Normal users should not have to choose pool/model/JSON controls in the UI.
This module resumes from durable checkpoints:
  prepared dataset -> one-time V22-derived pool lock -> fast strategy comparison -> report.
The final blind confirmation is deliberately NOT part of this pipeline because the
report is intended for iterative adjustment; repeatedly opening the final tail
would destroy its value as a holdout.
"""
from __future__ import annotations
from pathlib import Path
import copy
import threading
import traceback
import pandas as pd

from lab_io import DataError, atomic_json, read_json, sha
from studio_pool import load_dataset, load_pool, timestamp, digest
from v22_cache import prepare_v22
from v22_runner import run_pool_study, run_experiment

GUIDED_VERSION = '22.4.2-guided-validation-refill'
_STATE = 'guided_state.json'
_POOL_LOCK = 'guided_pool_lock.json'
_THREADS = {}
_LOCK = threading.Lock()


def _state_path(root):
    return Path(root) / 'v22' / _STATE


def _pool_lock_path(root):
    return Path(root) / 'v22' / _POOL_LOCK


def guided_state(root):
    return read_json(_state_path(root))


def _write(root, **updates):
    root = Path(root)
    root.joinpath('v22').mkdir(parents=True, exist_ok=True)
    state = read_json(_state_path(root))
    state.update(updates, updated_at=timestamp(), guided_version=GUIDED_VERSION)
    atomic_json(_state_path(root), state)
    return state


def _guided_profiles(profiles):
    """Keep only the three normal comparison models; GBM/diagnostics stay advanced."""
    obj = copy.deepcopy(profiles)
    keep = []
    for c in obj.get('candidates', []):
        mt = c.get('model_type', 'manual')
        if mt in ('manual', 'hybrid_ic', 'ic_adaptive') and c.get('selection_eligible', True):
            c['selection_eligible'] = True
            keep.append(c)
    if not keep:
        raise DataError('V22.4 guided mode found no normal strategy candidates')
    obj['candidates'] = keep
    return obj


def _read_guided_pool_lock(root, dataset_id, prepared=None):
    """Read the V22-specific pool lock.

    V22 pools (especially KOSTO250/KOSTO300) are deterministic derivatives of the
    pinned ADV500 monthly snapshot. They are not required to be native pool names in
    studio/monthly_pools.csv.gz, so the legacy studio_pool.lock_pool validator is the
    wrong validator for these derived pools.
    """
    obj = read_json(_pool_lock_path(root))
    if not obj or obj.get('dataset_id') != dataset_id:
        return None
    lock_id = obj.get('lock_id')
    payload = {k: v for k, v in obj.items() if k not in ('lock_id', 'registered_at')}
    if not lock_id or digest(payload)[:24] != lock_id:
        return None
    vals = list((obj.get('mapping') or {}).values())
    if not vals or len(set(vals)) != 1:
        return None
    pool = vals[0]
    if prepared is not None:
        fdir, info, available = prepared
        if pool not in available:
            return None
        if obj.get('feature_id') != info.get('feature_id'):
            return None
        p = Path(fdir) / 'pools.csv.gz'
        if not p.is_file() or obj.get('derived_pool_membership_sha') != sha(p):
            return None
    return pool


def _current_pool_lock(root, dataset_id, prepared=None):
    # V22.4.1 derived lock is authoritative for the guided path.
    pool = _read_guided_pool_lock(root, dataset_id, prepared=prepared)
    if pool:
        return pool
    # Backward compatibility: older workspaces may have a legacy studio pool lock.
    try:
        obj = load_pool(root)
    except Exception:
        return None
    if obj.get('dataset_id') != dataset_id:
        return None
    vals = list((obj.get('mapping') or {}).values())
    if not vals or len(set(vals)) != 1:
        return None
    return vals[0]


def _latest_pool_suggestion(root, dataset_id=None):
    latest = read_json(Path(root) / 'v22' / 'latest_pool_study.json')
    if dataset_id and latest.get('dataset_id') and latest.get('dataset_id') != dataset_id:
        return None, None
    pool = latest.get('suggested_pool')
    if pool in ('ADV200', 'ADV300', 'KOSTO250', 'KOSTO300'):
        return pool, latest.get('run')
    return None, None


def _available_v22_pools(root):
    """Return deterministic V22 derived pools for the active pinned dataset."""
    fdir, info, _ = prepare_v22(root)
    pools = set(info.get('pools') or [])
    if not pools:
        raise DataError('V22 derived pool cache contains no usable pools')
    return fdir, info, pools


def _validate_selected_pool_membership(fdir, pool):
    """Validate the selected derived pool has not silently shrunk by month."""
    p = Path(fdir) / 'pools.csv.gz'
    if not p.is_file():
        raise DataError('V22 derived pool membership file is missing')
    frame = pd.read_csv(p, usecols=['date', 'pool', 'ticker', 'declared_size'])
    q = frame[frame.pool.astype(str) == str(pool)]
    if q.empty:
        raise DataError(f'V22 derived pool {pool} is not present in the current pinned dataset')
    sizes = q.groupby('date').agg(n=('ticker', 'nunique'), expected=('declared_size', 'first'))
    if not sizes.n.eq(sizes.expected).all():
        raise DataError(f'V22 derived pool {pool} has silently shrunk in one or more months')
    return sha(p)


def _freeze_pool(root, dataset_id, pool, evidence_run=None, prepared=None):
    """Freeze a V22-derived pool without requiring it to be native in the snapshot.

    Integrity is anchored to the pinned dataset id + V22 feature/pool algorithm cache
    + derived monthly membership SHA. This is stronger and more semantically correct
    than asking the legacy studio pool lock to recognize a derived KOSTO pool name.
    """
    root = Path(root)
    folder, meta = load_dataset(root)
    if meta.get('dataset_id') != dataset_id:
        raise DataError('Active pinned dataset changed while the guided run was starting')
    if prepared is None:
        fdir, info, available = _available_v22_pools(root)
    else:
        fdir, info, available = prepared
    if info.get('dataset_id') != dataset_id:
        raise DataError('V22 derived pool cache belongs to a different pinned dataset')
    if pool not in available:
        raise DataError('Requested V22 pool is unavailable in this pinned dataset; available: ' + ', '.join(sorted(available)))
    membership_sha = _validate_selected_pool_membership(fdir, pool)
    horizons = [str(int(h)) for h in meta.get('source_config', {}).get('horizons', [])]
    if not horizons:
        raise DataError('Pinned dataset has no configured validation horizons')
    mapping = {h: pool for h in horizons}
    evidence = {
        'source': 'V22_DEVELOPMENT_POOL_STUDY',
        'run': evidence_run,
        'guided_version': GUIDED_VERSION,
        'final_tail_used': False,
    }
    payload = {
        'schema': 2,
        'dataset_id': dataset_id,
        'mapping': mapping,
        'pool_rule': pool,
        'feature_id': info.get('feature_id'),
        'derived_pool_membership_sha': membership_sha,
        'source_monthly_pools_sha': meta.get('files', {}).get('monthly_pools.csv.gz'),
        'derivation': 'v22_pool.build_v22_pools_from_pinned_ADV500',
        'reason': 'V22_4_2_GUIDED_DEVELOPMENT_LOCK',
        'evidence': evidence,
        'optimality_proven': False,
        'members_change_monthly_using_T_only_history': True,
        'manual_or_development_lock_not_independent_outer_test': True,
    }
    lock_id = digest(payload)[:24]
    obj = dict(payload, lock_id=lock_id, registered_at=timestamp())
    root.joinpath('v22').mkdir(parents=True, exist_ok=True)
    atomic_json(_pool_lock_path(root), obj)
    archive = root / 'v22' / 'pool_locks' / (lock_id + '.json')
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        atomic_json(archive, obj)
    _write(root, dataset_id=dataset_id, pool_rule=pool, pool_lock_id=lock_id,
           pool_study_run=evidence_run, phase='POOL_LOCKED', progress=0.32,
           message=f'母池已固定為 {pool}；後續策略驗證不再重跑母池。')
    return pool


def run_guided(root, profiles, settings, force_pool_recalibration=False, force_fast=False):
    root = Path(root)
    _, meta = load_dataset(root)
    dataset_id = meta['dataset_id']
    profiles = _guided_profiles(profiles)
    settings = dict(settings)
    identity = digest({'guided_version': GUIDED_VERSION, 'dataset_id': dataset_id,
                       'profiles': profiles, 'settings_without_pool': {k:v for k,v in settings.items() if k != 'pool_rule'}})[:24]

    old = guided_state(root)
    if old.get('dataset_id') != dataset_id or old.get('identity') != identity:
        old = {}
    _write(root, status='RUNNING', phase='PRECHECK', progress=0.05,
           dataset_id=dataset_id, identity=identity, error=None,
           message='資料庫檢查完成，確認 V22 衍生母池與既有 checkpoint。')

    # Build/verify the deterministic V22 pool cache once. KOSTO pools live here,
    # not necessarily as native pool labels in the pinned studio snapshot.
    prepared = _available_v22_pools(root)
    _, _, available = prepared

    # 1) Reuse a compatible guided/legacy lock first.
    pool = None if force_pool_recalibration else _current_pool_lock(root, dataset_id, prepared=prepared)
    pool_run = None
    if pool not in available:
        pool = None

    # 2) Reuse a completed one-time study from the same dataset when possible.
    if pool is None and not force_pool_recalibration:
        pool, pool_run = _latest_pool_suggestion(root, dataset_id)
        if pool not in available:
            pool, pool_run = None, None

    # 3) Only if neither exists, run the one-time development pool study.
    if pool is None:
        _write(root, phase='POOL_CALIBRATION', progress=0.12,
               message='第一次才會做：自動比較 ADV200 / ADV300 / KOSTO250 / KOSTO300。')
        folder = run_pool_study(root, profiles, settings)
        pool_run = str(folder)
        latest = read_json(root / 'v22' / 'latest_pool_study.json')
        pool = latest.get('suggested_pool')
        if pool not in available:
            # The study and available cache should agree. If not, fail with an
            # actionable integrity message instead of reusing a stale suggestion.
            raise DataError('母池校準結果與目前 V22 衍生母池快取不一致；請保留資料庫並重新執行一次母池校準')

    pool = _freeze_pool(root, dataset_id, pool, pool_run, prepared=prepared)

    # Development-only fast validation. Never open the final holdout here.
    settings['pool_rule'] = pool
    prior_fast = old.get('fast_run') if old else None
    if prior_fast and not force_fast:
        p = Path(prior_fast)
        if read_json(p / 'state.json').get('status') == 'COMPLETE' and (p / 'V22_TEST_REPORT.zip').is_file():
            _write(root, status='COMPLETE', phase='REPORT_READY', progress=1.0,
                   fast_run=str(p), report_zip=str(p/'V22_TEST_REPORT.zip'), report_xlsx=str(p/'V22_TEST_REPORT.xlsx'),
                   message='完整快速驗證已完成；可直接下載報告供下一輪調整。')
            return p

    _write(root, phase='FAST_VALIDATION', progress=0.42,
           message='固定母池後，自動比較固定權重 / Hybrid IC / IC Adaptive，並分別驗證 5 / 20 / 60 日。')
    run = run_experiment(root, profiles, settings, mode='fast')
    if read_json(run / 'state.json').get('status') != 'COMPLETE':
        raise DataError('快速策略驗證未完成')
    if not (run / 'V22_TEST_REPORT.zip').is_file():
        raise DataError('策略驗證完成但報告 ZIP 未產生')

    _write(root, status='COMPLETE', phase='REPORT_READY', progress=1.0,
           fast_run=str(run), report_zip=str(run/'V22_TEST_REPORT.zip'),
           report_xlsx=str(run/'V22_TEST_REPORT.xlsx'),
           message='完整快速驗證已完成；下載報告後即可依結果調整，不需查看最後保留期。')
    return run


def _mark_error(root, exc):
    msg = str(exc) or type(exc).__name__
    _write(root, status='ERROR', progress=guided_state(root).get('progress', 0.0),
           error={'type': type(exc).__name__, 'message': msg[:1000], 'traceback': traceback.format_exc()[-5000:]},
           message='本次停在目前步驟；已完成的資料與母池結果會保留。修正後按「開始／繼續」即可從未完成步驟接續。')


def start_guided(root, profiles, settings, force_pool_recalibration=False, force_fast=False):
    root = Path(root)
    key = str(root.resolve())
    with _LOCK:
        if _THREADS.get(key) and _THREADS[key].is_alive():
            raise DataError('完整驗證程序已在執行')
        root.joinpath('v22').mkdir(parents=True, exist_ok=True)
        # Clear the old V22.3 launch banner. Guided state owns errors now.
        (root/'v22'/'launch_error.json').unlink(missing_ok=True)
        def target():
            try:
                run_guided(root, profiles, settings, force_pool_recalibration, force_fast)
            except Exception as exc:
                _mark_error(root, exc)
        t = threading.Thread(target=target, daemon=True, name='v2241-guided-one-click')
        _THREADS[key] = t
        t.start()


def guided_alive(root):
    t = _THREADS.get(str(Path(root).resolve()))
    return bool(t and t.is_alive())
