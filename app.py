"""Alpha Radar V23.1 Benchmark-First + compatible V22 legacy studio.

The data layer keeps the V22.2 coverage-safe bootstrap/cache workflow while
the strategy layer adds horizon-specific, walk-forward adaptive models.
V23.1 uses immutable frozen V22 datasets; no automatic orders.
"""
from pathlib import Path
from dataclasses import replace
import hashlib
import importlib
import os
import streamlit as st

from lab_config import VERSION, Config, load_config
from lab_upgrade import run_config
from lab_io import read_json, DataError
from lab_jobs import start, alive, request_pause, workspace_busy, resume_if_interrupted
from lab_transfer import start_transfer
from lab_data import extract_dataset
from lab_runtime import estimated_hours, hours_message, progress_caption, seconds_since, rate
from lab_bulk import finlab_credentials_present, finlab_package_present

HERE = Path(__file__).resolve().parent
ROOT = HERE / 'data' / 'v21_universe_lab'


def secret_token():
    try:
        return str(st.secrets.get('FINMIND_TOKEN', os.getenv('FINMIND_TOKEN', ''))).strip()
    except Exception:
        return os.getenv('FINMIND_TOKEN', '').strip()


def activate_finlab_secrets():
    keys=('FINLAB_REFRESH_TOKEN','FINLAB_SESSION_ID','FINLAB_API_KEY')
    for key in keys:
        if os.getenv(key,'').strip():continue
        try:value=str(st.secrets.get(key,'')).strip()
        except Exception:value=''
        if value:os.environ[key]=value
    return finlab_credentials_present()


def quick_local_check(config):
    """Bounded no-network health check; not market-data validation."""
    config.validate()
    out = {'config': 'OK', 'file_write': 'NOT_CHECKED', 'requirements': {},
           'external_api_reachable': 'NOT_TESTED'}
    for name in ('numpy', 'pandas', 'requests', 'streamlit', 'yfinance'):
        m = importlib.import_module(name)
        out['requirements'][name] = str(getattr(m, '__version__', 'installed'))
    ROOT.mkdir(parents=True, exist_ok=True)
    tmp = ROOT / '.ui_write_test'
    tmp.write_text('ok', encoding='ascii')
    if tmp.read_text(encoding='ascii') != 'ok':
        raise DataError('Cache directory does not support file read/write')
    tmp.unlink(missing_ok=True)
    out['file_write'] = 'OK'
    return out


st.set_page_config(page_title='Alpha Radar V23.1｜Benchmark First', layout='wide')
page=st.radio('工作模式',['V23.1全新選股驗效','V22快速策略驗證','資料準備與備份','舊版母池研究（必要時才開）'],horizontal=True)
if page.startswith('V23.1'):
    from v231_ui import render as render_v231
    render_v231(st, ROOT)
    st.stop()
if page.startswith('V22'):
    # Streamlit reruns the main script in the same Python process.  A changed
    # imported module can therefore remain cached even after GitHub has the new
    # file.  Reload the V22 strategy stack only when an old generation is
    # detected.  Once the expected versions are loaded, normal reruns do not
    # reload active worker modules.
    import sys
    expected_ui = '22.5-hot-swap-ui'
    expected_runner = '22.4.2-validation-refill'
    expected_guided = '22.4.2-guided-validation-refill'

    def _module_version(name, attr):
        mod = sys.modules.get(name)
        return getattr(mod, attr, None) if mod is not None else None

    stale = (
        _module_version('v22_ui', 'UI_VERSION') != expected_ui
        or _module_version('v22_runner', 'V22_VERSION') != expected_runner
        or _module_version('v224_guided', 'GUIDED_VERSION') != expected_guided
    )
    if stale:
        # Dependency order matters: engine -> runner -> guided -> UI.
        for name in ('v22_engine', 'v22_runner', 'v224_guided', 'v22_ui'):
            if name in sys.modules:
                importlib.reload(sys.modules[name])
            else:
                importlib.import_module(name)

    from v22_ui import render
    render(st,ROOT)
    st.stop()
st.title('Alpha Radar V23.1｜資料準備＋V22研究相容頁')
st.caption('Strategy 22.4.2-validation-refill • Data layer ' + VERSION + ' • 一鍵接續驗證，不會自動下單')
st.write('同一日期比較過去一年日均成交金額前100／200／300／500檔；訓練期選因子及權重，後續分開驗證。')

cfg = load_config(HERE / 'config.json')
action='universe_search' if page.startswith('舊版') else 'prepare_data'
cfg=replace(cfg,research_action=action)
st.info('資料準備不搜尋權重；只有明確選母池研究才重跑四種母池。')
st.info('資料層沿用V22.2 Coverage Fix：Yahoo批次先做252日歷史完整度檢查；缺漏ticker小批次修復，仍不足時切TWSE/TPEx，不會默默退回7小時FinMind。策略層已升級V22.4.2一鍵接續驗證（保留KOSTO衍生母池鎖定，新增紀律後補位與雙層驗證分數）。')
access_token = secret_token()
finlab_auth=activate_finlab_secrets()
provider_label=st.radio('歷史資料傳輸', [
    'AUTO_FAST｜有FinLab用實際成交金額；否則Yahoo批次快速建庫（最快，建議開發）',
    'OFFICIAL_BULK｜TWSE＋TPEx逐日全市場整批（免費、實際成交金額）',
    'YAHOO_FAST｜強制Yahoo批次快速建庫（成交金額代理）',
    'FINLAB_BULK｜強制FinLab整表實際成交金額',
    'FINMIND｜逐檔精確fallback（最慢，依帳號額度）'], horizontal=False)
if provider_label.startswith('AUTO_FAST'):provider='AUTO_FAST'
elif provider_label.startswith('OFFICIAL_BULK'):provider='OFFICIAL_BULK'
elif provider_label.startswith('YAHOO_FAST'):provider='YAHOO_FAST'
elif provider_label.startswith('FINLAB'):provider='FINLAB_BULK'
else:provider='FINMIND'
cfg=replace(cfg,history_provider=provider)
with st.expander('第一次使用前請看：資料層與執行時間', expanded=True):
    st.warning('不要再用免費FinMind逐檔建立全市場。AUTO_FAST先走Yahoo批次，並檢查每個月是否真的至少有500檔具備252日歷史；若Yahoo被限流造成稀疏資料，會自動修復或切換官方整市場來源。')
    effective=('FINLAB_BULK' if (provider=='FINLAB_BULK' or (provider=='AUTO_FAST' and finlab_auth)) else ('OFFICIAL_BULK' if provider=='OFFICIAL_BULK' else ('YAHOO_FAST' if provider in ('AUTO_FAST','YAHOO_FAST') else 'FINMIND')))
    st.write(hours_message(2001,0,bool(access_token),provider=effective))
    st.write('**FinLab伺服器憑證：'+('已設定' if finlab_auth else '未設定')+'｜套件：'+('已安裝' if finlab_package_present() else '未安裝')+'**')
    st.write('**FinMind Token：'+('已設定（實際額度將自動查詢）' if access_token else '未設定')+'**')
    st.caption('Yahoo快速模式的流動性指標是((高+低+收)/3)×成交量代理，不是交易所實際成交金額；適合快速開發。需要精確母池時，再用FinLab或FinMind做一次確認。')
    st.caption('母池建立後，完整OHLC/因子只保留歷史母池聯集；日常策略調權重不會重做全市場下載。')

mode = st.radio('資料模式', ['自動／批次：官方現行名冊＋歷史市場資料', '匯入：已整理的全市場資料ZIP'], horizontal=True)
input_dir = None
if mode.startswith('匯入'):
    cfg = replace(cfg, source_mode='import')
    upload = st.file_uploader('標準格式資料集，不是v19_results.json', type=['zip'])
    historical = st.checkbox('另附各月底歷史成分membership.csv')
    market_cap = st.checkbox('納入歷史市值Top-N（需market_caps.csv）')
    optional = st.checkbox('加入已標註公告時間的營收/EPS/法人因子（需evidence.csv）')
    cfg = replace(cfg,
                  requested_scope='HISTORICAL_MEMBERSHIP' if historical else 'CURRENT_OFFICIAL_SNAPSHOT',
                  include_market_cap_pools=market_cap, enable_optional_factors=optional)
    if upload:
        data = upload.getvalue()
        digest = hashlib.sha256(data).hexdigest()
        input_dir = ROOT / 'inputs' / digest
        if not (input_dir / 'manifest.json').exists():
            archive = ROOT / 'uploads' / (digest + '.zip')
            archive.parent.mkdir(parents=True, exist_ok=True)
            archive.write_bytes(data)
            try:
                extract_dataset(archive, input_dir)
            except Exception as exc:
                st.error('資料格式檢查失敗：' + str(exc))
                st.stop()
    else:
        st.info('請參照 DATA_CONTRACT.md 匯入完整市場資料。以前的150檔研究報告不足以建立Top500。')
else:
    st.info('AUTO_FAST：FinLab可用時採實際成交金額；否則Yahoo批次＋歷史完整度修復。Yahoo仍不足會改TWSE/TPEx官方整市場回補；只有你明確選FINMIND才走逐檔慢速模式。')

with st.expander('查看固定研究設定與資料限制', expanded=False):
    st.json(cfg.to_dict())
    st.caption('預設使用現行名冊，不能宣稱已修正歷史下市股倖存者偏差。未驗證的財報/法人因子不會擅自加入。')

# A browser reconnect is not a reason to create another experiment. Resume only
# an interrupted active task, never a user-paused/blocked/completed one.
resume_if_interrupted(ROOT, access_token)
latest = read_json(ROOT / 'latest.json')
status = read_json(Path(latest['run']) / 'state.json') if latest.get('run') else {}
status_age = seconds_since(status.get('heartbeat_at') or status.get('updated_at'))
active = alive(ROOT) or workspace_busy(ROOT)
cols = st.columns(3)
if cols[0].button('▶ 啟動／繼續', type='primary', disabled=active or (cfg.source_mode == 'import' and input_dir is None), use_container_width=True):
    try:
        old = read_json(ROOT / 'latest.json')
        oldstate = read_json(Path(old['run']) / 'state.json') if old.get('run') else {}
        prior_protocol=read_json(Path(old['run'])/'protocol.json') if old.get('run') else {}
        cfg=replace(run_config(cfg,old,prior_protocol),research_action=action)
        if prior_protocol.get('version')==VERSION and old.get('config',{}).get('source_mode')==cfg.source_mode:
            input_dir=old.get('input_dir') or input_dir
        cfg.validate()
        started = start(ROOT, cfg, access_token, input_dir)
        st.success('任務已建立：' + Path(started).name + '。下方會自動更新進度。')
    except Exception as exc:
        st.error('啟動失敗：' + str(exc))
if cols[1].button('暫停並保留快取', disabled=not active, use_container_width=True):
    request_pause(latest['run'])
    st.info('已提出暫停要求；會在下一個可安全中斷點停止。')
if cols[2].button('🔍 本機設定與寫入測試', use_container_width=True):
    try:
        st.json(quick_local_check(cfg))
        st.success('程式依賴與磁碟寫入正常（未測網路/API）。')
    except Exception as exc:
        st.error('本機檢查失敗：' + type(exc).__name__ + ': ' + str(exc))

with st.expander('關閉手機仍繼續：GitHub Actions 背景執行／資料備份', expanded=True):
    st.info('真正不依賴 Streamlit 網頁的方式：將套件內 .github/workflows/v21-background.yml 放到 GitHub 相同路徑，到 Actions 按 Run workflow 一次。')
    st.write('背景機器每30分鐘交棒存檔，自動接續；最多32個內部分段，不是新的研究階段。預設只準備資料；完成後回固定母池工作台調整策略。')
    st.caption('請先暫停本頁下載，避免兩台主機同時消耗同一個FinMind配額。Actions有運算與儲存配額，不保證免費或永久保存。')
    if st.button('製作目前資料庫檢查點（請先暫停）',disabled=active):
        start_transfer(ROOT,'export')
        st.info('正在製作 SQLite 一致性備份，不會重抓行情。')
    checkpoint=st.file_uploader('還原自己的 V21_DATABASE.zip（不是研究報告或程式ZIP）',type=['zip'],key='resume_zip')
    if st.button('合併還原檢查點',disabled=active or checkpoint is None):
        if checkpoint is not None:
            target=ROOT/'incoming'/'checkpoint.zip';target.parent.mkdir(parents=True,exist_ok=True)
            # Stream the UploadedFile into disk, rather than making another getvalue copy.
            import shutil
            checkpoint.seek(0)
            with open(target,'wb') as out:shutil.copyfileobj(checkpoint,out,1024*1024)
            start_transfer(ROOT,'import',source=target)
    def remote_secret(key):
        try:return str(st.secrets.get(key,os.getenv(key,''))).strip()
        except Exception:return os.getenv(key,'').strip()
    repository=remote_secret('GITHUB_REPOSITORY');github_token=remote_secret('GITHUB_READ_TOKEN')
    if repository and github_token:
        if st.button('從 GitHub Actions 直接同步最新資料（不經手機上傳）',disabled=active):
            start_transfer(ROOT,'github',repo=repository,token=github_token)
    else:
        st.caption('選配：在 Streamlit Secrets 設定 GITHUB_REPOSITORY=owner/repo 與僅該倉儲 Actions:read 權限的 GITHUB_READ_TOKEN，即可直接同步，不必先下載到手機。沒設定也可在Actions直接取得報告。')

@st.fragment(run_every='5s')
def monitor():
    transfer=read_json(ROOT/'transfer_state.json')
    if transfer:
        st.caption('檢查點傳輸：'+str(transfer.get('status'))+' | '+str(transfer.get('operation')))
        if transfer.get('status')=='ERROR':st.error(transfer.get('message','Transfer failed'))
        if transfer.get('status')=='COMPLETE':
            st.json(transfer.get('result',{}))
            result=transfer.get('result',{})
            if transfer.get('operation')=='export' and result.get('file'):
                backup=Path(result['file'])
                if backup.is_file() and backup.stat().st_size<150*1024*1024:
                    with open(backup,'rb') as stream:
                        st.download_button('下載資料庫檢查點ZIP',stream,'V21_DATABASE.zip','application/zip',on_click='ignore')
                else:st.info('備份超過150MB，為保護Cloud記憶體，請使用Actions或主機檔案取得。')
    item = read_json(ROOT / 'latest.json')
    if not item.get('run'):
        st.info('尚無任務。先閱讀上方的初次下載時間，再按「啟動／繼續」。')
        return
    run = Path(item['run'])
    state = read_json(run / 'state.json')
    code = state.get('status', 'QUEUED')
    phase = state.get('phase', 'QUEUE')
    age = seconds_since(state.get('heartbeat_at') or state.get('updated_at'))
    st.subheader('目前狀態：' + str(code) + ' ｜ ' + str(phase))
    st.write(state.get('message') or '正在建立背景工作...')
    elapsed = seconds_since(state.get('started_at'))
    if elapsed is not None:
        st.caption('任務已開始約 ' + str(elapsed // 60) + ' 分鐘；最後一次狀態更新：' +
                   (str(age) + ' 秒前' if age is not None else '未知'))
    if state.get('current_symbol'):
        st.caption('目前股票：' + str(state['current_symbol']) +
                   ' ｜ 最後已儲存：' + str(state.get('last_committed_symbol') or '尚無'))
    total, done = state.get('total'), state.get('done')
    if isinstance(total, int) and total > 0 and isinstance(done, int):
        st.progress(max(0., min(1., done / total)), text=f'{min(done, total):,} / {total:,}（此階段）')
        # Use the frozen run's token availability only as a UI estimate;
        # secrets never appear in state files.
        note = progress_caption(state, bool(access_token))
        if note: st.caption(note)
    if state.get('wait_seconds'):
        st.warning('來源配額或限速等待：約 ' + str(state['wait_seconds']) + ' 秒，進度仍會每隔數秒更新。')
    if code == 'RUNNING' and age is not None and age > 120:
        st.error('狀態已超過兩分鐘沒有更新：背景工作可能已中斷／主機休眠。保留data/後按「繼續」，或查看主機日誌。')
    if code == 'QUEUED' and age is not None and age > 45:
        st.error('排隊超過45秒未進入工作，可能是背景執行緒或部署環境有問題。')
    if code in ('BLOCKED', 'ERROR'):
        st.error('資料／執行環境阻擋；不是研究結果失敗。不要刪掉data/，查看上方錯誤後再按繼續。')
    if code == 'COMPLETE':
        st.success('研究完成。下方報告是真實資料回測時才具有研究解讀價值。')
    report_path = run / 'REPORT.txt'
    if report_path.is_file():
        report = report_path.read_text(encoding='utf-8-sig')
        st.text(report)
        st.download_button('下載研究報告 TXT', report.encode('utf-8-sig'),
                           'LAB_REPORT.txt', 'text/plain', on_click='ignore')
        for name in ('selected_models.json', 'results.json', 'LAB_RESULTS.zip'):
            p = run / name
            if p.is_file() and p.stat().st_size < 30 * 1024 * 1024:
                st.download_button('下載 ' + name, p.read_bytes(), name,
                                   'application/zip' if name.endswith('.zip') else 'application/json',
                                   on_click='ignore')
    # Shareable status text, useful when a user reports "10 minutes nothing".
    if state and code != 'COMPLETE':
        lines = [f'version={VERSION}', f'status={code}', f'phase={phase}',
                 f'updated_at={state.get("updated_at", "")}',
                 f'progress={done}/{total}', f'message={state.get("message", "")}']
        st.download_button('下載執行診斷狀態 TXT', '\n'.join(lines).encode('utf-8'),
                           'V21_JOB_DIAGNOSTICS.txt', 'text/plain', on_click='ignore')

monitor()
