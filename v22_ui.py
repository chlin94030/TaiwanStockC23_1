"""Alpha Radar V22.5 Hot-Swap validation UI.

One-time foundation change: replace this file once.
Afterwards strategy/model tuning is delivered as ONE JSON file uploaded in the app.
No GitHub edits or reboot are required for ordinary weight/factor/adaptive-setting changes.
"""
from __future__ import annotations

from pathlib import Path
import copy
import json
import pandas as pd

from lab_io import read_json, atomic_json, DataError
from studio_pool import discover_prepared, adopt_dataset, load_dataset, digest
from v22_runner import start_v22, v22_alive, V22_VERSION, configure
from v22_engine import validate_profiles
from v22_cache import prepare_v22
from v224_guided import start_guided, guided_alive, guided_state, GUIDED_VERSION

HERE = Path(__file__).resolve().parent
UI_VERSION = '22.5-hot-swap-ui'
SUPPORTED_ENGINES = {'22.4.2-validation-refill'}
_ACTIVE_RELEASE = 'active_strategy_release.json'
_RELEASE_DIR = 'strategy_releases'


def _fmt_pct(v):
    try:
        if v is None or not pd.notna(v):
            return '—'
        return f'{float(v):+.2%}'
    except Exception:
        return '—'


def _release_path(root):
    return Path(root) / 'v22' / _ACTIVE_RELEASE


def _release_dir(root):
    return Path(root) / 'v22' / _RELEASE_DIR


def _bundled_release():
    profiles = read_json(HERE / 'v22_profiles.json')
    settings = read_json(HERE / 'v22_settings.json')
    obj = {
        'schema': 1,
        'release_id': 'BUNDLED-22.4.2-BASELINE',
        'requires_engine': '22.4.2-validation-refill',
        'note': 'Bundled baseline; future tuning should be uploaded as one strategy_release.json.',
        'profiles': profiles,
        'settings': settings,
    }
    return _stamp_release(obj)


def _stamp_release(obj):
    clean = copy.deepcopy(obj)
    clean.pop('release_sha', None)
    release_sha = digest(clean)[:16]
    clean['release_sha'] = release_sha
    return clean


def _parse_release_bytes(data):
    try:
        obj = json.loads(data.decode('utf-8-sig') if isinstance(data, (bytes, bytearray)) else str(data))
    except Exception as exc:
        raise DataError('策略檔不是有效 JSON：' + str(exc))
    if not isinstance(obj, dict):
        raise DataError('策略檔最外層必須是 JSON object')
    allowed = {'schema', 'release_id', 'requires_engine', 'note', 'profiles', 'settings', 'release_sha'}
    extra = set(obj) - allowed
    if extra:
        raise DataError('策略檔含未知欄位：' + ', '.join(sorted(extra)))
    if obj.get('schema') != 1:
        raise DataError('strategy_release schema 必須為 1')
    rid = str(obj.get('release_id') or '').strip()
    if not rid or len(rid) > 80:
        raise DataError('release_id 必須為 1~80 字元')
    req = str(obj.get('requires_engine') or '').strip()
    if not req:
        raise DataError('requires_engine 不可空白')
    if not isinstance(obj.get('profiles'), dict) or not isinstance(obj.get('settings'), dict):
        raise DataError('策略檔必須同時包含 profiles 與 settings')
    stamped = _stamp_release(obj)
    supplied = obj.get('release_sha')
    if supplied and str(supplied) != stamped['release_sha']:
        raise DataError('策略檔 release_sha 不符；檔案可能被部分修改或損壞')
    return stamped


def _load_active_release(root):
    p = _release_path(root)
    if p.is_file():
        try:
            return _parse_release_bytes(p.read_bytes()), 'UPLOADED'
        except Exception:
            # Do not brick the UI because of a damaged runtime override.
            pass
    return _bundled_release(), 'BUNDLED'


def _validate_release_against_dataset(root, release):
    if release['requires_engine'] != V22_VERSION:
        raise DataError(
            f"策略 {release['release_id']} 需要核心 {release['requires_engine']}，"
            f"目前實際核心是 {V22_VERSION}。為避免混版，系統已阻止執行。"
        )
    _, meta = load_dataset(root)
    configure(meta, release['settings'])
    _, info, _ = prepare_v22(root)
    validate_profiles(release['profiles'], info.get('factors') or [])
    return True


def _profiles_for_run(release):
    obj = copy.deepcopy(release['profiles'])
    # Top-level metadata is ignored by validate_profiles but is included in the
    # guided identity and exported profiles.json. Therefore changing one release
    # automatically invalidates only the strategy checkpoint, not data/pool caches.
    obj['_release'] = {
        'release_id': release['release_id'],
        'release_sha': release['release_sha'],
        'requires_engine': release['requires_engine'],
        'ui_version': UI_VERSION,
        'note': release.get('note', ''),
    }
    return obj


def _guided_profiles(default_profiles):
    q = copy.deepcopy(default_profiles)
    q['candidates'] = [
        c for c in q.get('candidates', [])
        if c.get('model_type', 'manual') in ('manual', 'hybrid_ic', 'ic_adaptive')
        and c.get('selection_eligible', True)
    ]
    return q


def _state_release(state):
    # New Hot-Swap runs put release metadata into the identity via profiles.
    # Keep an explicit copy in guided_state when we launch, which makes UI checks cheap.
    return (state or {}).get('strategy_release') or {}


def _save_release(root, release):
    root = Path(root)
    root.joinpath('v22').mkdir(parents=True, exist_ok=True)
    atomic_json(_release_path(root), release)
    arc = _release_dir(root)
    arc.mkdir(parents=True, exist_ok=True)
    atomic_json(arc / f"{release['release_id']}_{release['release_sha']}.json", release)


def _mark_state_release(root, release):
    p = Path(root) / 'v22' / 'guided_state.json'
    s = read_json(p)
    s['strategy_release'] = {
        'release_id': release['release_id'],
        'release_sha': release['release_sha'],
        'engine': V22_VERSION,
        'ui': UI_VERSION,
    }
    atomic_json(p, s)


def render(st, root):
    root = Path(root)
    release, release_source = _load_active_release(root)
    engine_ok = V22_VERSION in SUPPORTED_ENGINES and release['requires_engine'] == V22_VERSION

    st.title('Alpha Radar｜快速驗效工作台')
    st.caption('以後調模型只換 1 個 strategy_release.json；資料庫與固定母池不動。')

    # Single source of truth: show actual imported module versions, never a manually typed page version.
    c1, c2, c3 = st.columns(3)
    c1.metric('核心引擎', V22_VERSION)
    c2.metric('策略版本', release['release_id'])
    c3.metric('策略 SHA', release['release_sha'])
    st.caption(f'UI {UI_VERSION}｜Guided {GUIDED_VERSION}｜策略來源：{release_source}')

    if not engine_ok:
        st.error(
            '偵測到核心／策略不相容，已自動鎖住驗證，避免再產生混版報告。'
            f"\n\n目前核心：{V22_VERSION}｜策略要求：{release['requires_engine']}"
        )
        st.info('一般模型微調不應再改 Python；若核心真的需要升級，只做一次基礎版升級，之後全部走單一策略檔。')

    from lab_jobs import workspace_busy
    busy = guided_alive(root) or v22_alive(root) or workspace_busy(root)
    prepared = discover_prepared(root)
    dataset = None
    try:
        if (root / 'studio' / 'active_dataset.json').exists():
            _, dataset = load_dataset(root)
    except Exception as exc:
        st.warning(str(exc))

    if dataset:
        st.success('① 資料庫已就緒｜' + str(dataset['source_config'].get('start')) + ' ～ ' + str(dataset['source_config'].get('end')))
    else:
        st.subheader('① 選擇已完成的資料庫')
        if not prepared:
            st.warning('目前沒有可沿用的完整資料庫。請先切到「資料準備與備份」完成一次資料建置。')
            return
        selected = st.selectbox('已準備資料', prepared, format_func=lambda p: p['name'] + '｜' + str(p['start']) + '～' + str(p['end']))
        if st.button('採用這份資料並繼續', type='primary', disabled=busy, use_container_width=True):
            try:
                adopt_dataset(root, selected['run'])
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
        return

    # ---- Hot-swap: future tuning happens here, not in GitHub. ----
    with st.expander('🔄 套用新的策略調整（以後只用這裡）', expanded=True):
        st.write('收到我提供的新 `strategy_release.json` 後，直接上傳並按一次。**不用 GitHub、不用 Reboot、不用重跑資料／母池。**')
        upload = st.file_uploader('strategy_release.json', type=['json'], key='strategy_release_upload')
        parsed = None
        if upload is not None:
            try:
                parsed = _parse_release_bytes(upload.getvalue())
                st.success(f"已讀取：{parsed['release_id']}｜SHA {parsed['release_sha']}｜要求核心 {parsed['requires_engine']}")
                if parsed['requires_engine'] != V22_VERSION:
                    st.error(f"這個策略檔要求核心 {parsed['requires_engine']}，目前是 {V22_VERSION}，不會讓你誤跑。")
            except Exception as exc:
                st.error(str(exc))
        if st.button('套用這個策略並立即重新驗證', type='primary', disabled=busy or parsed is None or not engine_ok or (parsed is not None and parsed['requires_engine'] != V22_VERSION), use_container_width=True):
            try:
                _validate_release_against_dataset(root, parsed)
                _save_release(root, parsed)
                profiles = _guided_profiles(_profiles_for_run(parsed))
                settings = copy.deepcopy(parsed['settings'])
                _mark_state_release(root, parsed)
                start_guided(root, profiles, settings, force_fast=True)
                st.success('已套用；只重跑策略驗證。資料庫與固定母池均保留。')
                st.rerun()
            except Exception as exc:
                st.error(type(exc).__name__ + '：' + str(exc))

        st.download_button(
            '下載目前策略檔（備份／比對）',
            json.dumps(release, ensure_ascii=False, indent=2).encode('utf-8'),
            file_name=f"strategy_release_{release['release_id']}.json",
            mime='application/json',
            use_container_width=True,
            on_click='ignore',
        )

    # Validate the active release once against the active dataset. If something is
    # incompatible, the main run button is disabled automatically.
    release_valid = engine_ok
    release_error = None
    if release_valid:
        try:
            _validate_release_against_dataset(root, release)
        except Exception as exc:
            release_valid = False
            release_error = str(exc)
    if release_error:
        st.error('目前策略檔驗證失敗：' + release_error)

    state = guided_state(root)
    state_release = _state_release(state)
    state_matches_release = (
        state_release.get('release_sha') == release['release_sha']
        and state_release.get('engine') == V22_VERSION
    )

    pool = state.get('pool_rule')
    if pool:
        st.success('② 母池已固定｜' + str(pool) + '｜策略調整不重跑母池')
    else:
        latest_pool = read_json(root / 'v22' / 'latest_pool_study.json')
        if latest_pool.get('suggested_pool'):
            st.success('② 已找到先前母池校準｜將沿用 ' + str(latest_pool['suggested_pool']))
        else:
            st.info('② 尚未做母池校準｜第一次才會自動比較四種母池；之後固定。')

    st.subheader('③ 驗證目前策略')
    st.write('固定母池後自動跑：**Manual → Hybrid IC → IC Adaptive → 5/20/60日 → Excel＋ZIP**。')
    st.caption('不查看最後保留期；策略版本／SHA 會自動綁定 checkpoint。')

    current_complete = state_matches_release and state.get('status') == 'COMPLETE'
    if current_complete:
        label = f"✓ {release['release_id']} 已完成"
    elif state and not state_matches_release:
        label = f"▶ 驗證新策略 {release['release_id']}（保留資料／母池）"
    else:
        label = f"▶ 開始／繼續 {release['release_id']}"

    if st.button(label, type='primary', disabled=busy or current_complete or not release_valid, use_container_width=True):
        try:
            profiles = _guided_profiles(_profiles_for_run(release))
            settings = copy.deepcopy(release['settings'])
            _mark_state_release(root, release)
            start_guided(root, profiles, settings, force_fast=not state_matches_release)
            st.success('已啟動。策略 SHA 已綁定本次驗證；不會沿用不同策略的舊報告。')
        except Exception as exc:
            st.error(type(exc).__name__ + '：' + str(exc))

    @st.fragment(run_every='3s')
    def monitor():
        s = guided_state(root)
        if not s:
            return
        sr = _state_release(s)
        st.caption('本次任務策略：' + str(sr.get('release_id', '舊版未標記')) + '｜SHA ' + str(sr.get('release_sha', '—')))
        progress = float(s.get('progress') or 0.0)
        st.progress(max(0.0, min(1.0, progress)), text=str(s.get('message') or s.get('phase') or '準備中'))
        st.caption('目前步驟：' + str(s.get('phase', '—')) + '｜狀態：' + str(s.get('status', '—')))

        if s.get('status') == 'ERROR':
            err = s.get('error') or {}
            st.error(str(err.get('type', 'ERROR')) + '：' + str(err.get('message', '未知錯誤')))
            st.warning('資料與母池仍保留。修正策略檔後重新上傳即可，不需要碰 Python。')
            with st.expander('除錯資訊', expanded=False):
                st.code(str(err.get('traceback', '')))

        run_path = s.get('fast_run')
        if s.get('status') == 'COMPLETE' and run_path:
            run = Path(run_path)
            result = read_json(run / 'results.json')
            st.success('④ 驗證完成｜下載報告後即可進行下一輪；下輪仍只換一個 JSON。')
            rows = []
            for h, m in result.get('models', {}).items():
                sm = m.get('development_validation') or {}
                rows.append({
                    '週期': str(h) + '日',
                    '母池': m.get('pool'),
                    '勝出策略': (m.get('choice') or {}).get('name'),
                    '模型': (m.get('choice') or {}).get('model_type', 'manual'),
                    '模型選擇分數': _fmt_pct(m.get('choice_score')),
                    '選股能力(vs同池)': _fmt_pct((sm.get('versus_same_pool') or {}).get('mean')),
                    '整體配置(vs0050)': _fmt_pct((sm.get('versus_0050') or {}).get('mean')),
                    '額外成本後': _fmt_pct(sm.get('extra_cost_stress_excess')),
                })
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            if (run / 'V22_TEST_REPORT.zip').is_file():
                st.download_button('⬇ 下載完整驗證報告 ZIP', (run / 'V22_TEST_REPORT.zip').read_bytes(),
                                   'V22_TEST_REPORT.zip', 'application/zip', type='primary', use_container_width=True, on_click='ignore')
            if (run / 'V22_TEST_REPORT.xlsx').is_file():
                st.download_button('⬇ 下載 Excel 驗證報告', (run / 'V22_TEST_REPORT.xlsx').read_bytes(),
                                   'V22_TEST_REPORT.xlsx', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', use_container_width=True, on_click='ignore')
            if (run / 'profiles.json').is_file():
                p = read_json(run / 'profiles.json')
                rmeta = p.get('_release') or {}
                st.caption('報告綁定策略：' + str(rmeta.get('release_id', '舊版')) + '｜SHA ' + str(rmeta.get('release_sha', '—')))

    monitor()

    with st.expander('進階／只有策略正式凍結後才使用', expanded=False):
        st.warning('日常調整不要碰這裡。最後保留期只在策略完全凍結後使用一次。')
        s = guided_state(root)
        if s.get('status') == 'COMPLETE' and s.get('pool_rule') and state_matches_release:
            frozen = st.checkbox('我確認母池、因子、權重都不再修改；現在才要打開最後保留期', key='v225_final_frozen')
            if st.button('執行一次最終確認', disabled=busy or not frozen, use_container_width=True):
                final_settings = copy.deepcopy(release['settings'])
                final_settings['pool_rule'] = s['pool_rule']
                profiles = _guided_profiles(_profiles_for_run(release))
                try:
                    start_v22(root, profiles, final_settings, 'confirm')
                    st.success('已啟動一次性最終確認。')
                except Exception as exc:
                    st.error(str(exc))
