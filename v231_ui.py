"""Streamlit V23.1 integrated mobile-first interface.

Two actions: import/collect dated data (optional), run an evidence-gated,
benchmark-first historical research comparison on existing V22 data.
"""
from pathlib import Path
from datetime import datetime
import io
import json
import pandas as pd
import streamlit as st
from lab_io import DataError,read_json
from studio_pool import discover_prepared,adopt_dataset,load_dataset
from v231_data import CONTRACTS,load_events,validate_events,persist_events,fetch_latest_official,export_templates
from v231_engine import VERSION,MODELS,V23Config
from v231_release import load as load_v231_release,parse as parse_v231_release,persist as persist_v231_release,bundled
from v231_runner import start,v231_busy,get_state,job_path


def json_bytes(obj):
    return json.dumps(obj,ensure_ascii=False,indent=2).encode('utf-8')

def _pct(value,decimal=1):
    return f'{value*100:.{decimal}f}%' if value is not None and pd.notna(value) else '—'


def render(st,root):
    root=Path(root)
    st.title('Alpha Radar V23.1｜0050超額報酬研究工作台')
    st.caption(f'CORE {VERSION}｜使用已建好的KOSTO250、只在開發期驗證、T+1開盤進場、每日資金淨值、所有未投個股的資金持有0050。')
    st.warning('研究模型尚未證明打贏0050。本頁完全不自動下單；資料不足不會把缺月補0或宣稱驗收通過。')
    try:
        _,meta=load_dataset(root)
    except Exception:
        prepared=discover_prepared(root)
        if not prepared:
            st.error('尚未找到舊版完整行情資料庫。請先切到「資料準備與備份」還原舊V21_DATABASE.zip，或建置一次。')
            return
        q=st.selectbox('找到可沿用的歷史行情',prepared,format_func=lambda v:v['name']+'｜'+str(v.get('start'))+'～'+str(v.get('end')))
        if st.button('採用這份資料（不重新下載）',type='primary'):
            try:adopt_dataset(root,q['run']);st.rerun()
            except Exception as exc:st.error(str(exc))
        return
    st.success('① 既有行情資料可用｜'+str(meta.get('source_config',{}).get('start'))+' ～ '+str(meta.get('source_config',{}).get('end'))+'｜Dataset '+str(meta['dataset_id']))
    st.caption('舊版資料與策略快取保留；新V23資料獨立存放在 data/v21_universe_lab/v23。資料在Streamlit Cloud本機磁碟，請定期下載備份。')
    evidence=load_events(root)
    rev=len(evidence['revenue']);financial=len(evidence['financial']);flows=len(evidence['flow'])
    st.subheader('② 基本面資料｜公告日可稽核')
    cols=st.columns(3)
    cols[0].metric('月營收事件',f'{rev:,}筆')
    cols[1].metric('季度財報事件',f'{financial:,}筆')
    cols[2].metric('法人資金事件',f'{flows:,}筆')
    if not rev or not financial:
        st.info('現有行情仍能執行「0050＋純價格動能控制組」，但完整40/35/15/10成長模型會被標成基本面覆蓋不足。')
    with st.expander('新增資料：下載欄位範本、上傳帶公告日期的歷史CSV',expanded=False):
        st.download_button('下載三種資料範本ZIP',export_templates(root),file_name='V23_1_PIT_DATA_TEMPLATES.zip',mime='application/zip',on_click='ignore')
        kind=st.selectbox('資料類型',list(CONTRACTS),format_func=lambda x:{'revenue':'歷史月營收（原始金額）','financial':'財報EPS及毛利率','flow':'已正規化法人資金流向'}[x])
        supplied=st.file_uploader('僅接受CSV（具有確切 available_at 和來源）',type=['csv'],key='v231_pit_'+kind)
        if supplied is not None:
            try:
                buf=io.BytesIO(supplied.getvalue())
                if buf.getbuffer().nbytes>25*1024*1024:raise DataError('單份CSV上限25MB')
                parsed=validate_events(pd.read_csv(buf,encoding='utf-8-sig'),kind)
                st.info(f'格式有效，{len(parsed):,}筆｜VERIFIED={int(parsed.quality.eq("VERIFIED").sum())}筆（仍需獨立來源稽核）')
                if st.button('確認匯入這份公告資料',disabled=v231_busy(root),type='primary'):
                    status=persist_events(root,parsed,kind);st.success('已匯入'+str(status['rows'])+'筆');st.rerun()
            except Exception as exc:st.error('無法匯入：'+str(exc))
    with st.expander('官方最新月營收／歷史月營收（研究版）',expanded=False):
        st.caption('TWSE/TPEx最新快照不是歷史公告時間。網站抓取當下才開始可用；歷史MOPS彙總資料採保守次月20日代理日，不能作正式PIT驗收。')
        if st.button('擷取官方最新月份月營收快照（資料不倒填）',disabled=v231_busy(root)):
            try:
                report=fetch_latest_official(root);st.success('官方最新快照'+str(report['rows'])+'筆');st.rerun()
            except Exception as exc:st.error('官方端點抓取失敗：'+str(exc))
        from v231_mops_jobs import start_mops, mops_job_state, mops_busy
        s=mops_job_state(root)
        if s:st.caption('MOPS歷史下載：'+str(s.get('status'))+'｜'+str(s.get('completed',0))+'/'+str(s.get('total',0))+'月｜'+str(s.get('message','')))
        initial=int(str(meta['source_config'].get('start','2021-01-01'))[:4]);last=int(str(meta['source_config'].get('end','2026-10-25') or '2026-10-25')[:4]);
        start_year=st.selectbox('補充歷史從哪年開始（保守代理，研究用）',list(range(max(2013,initial),last+1)),index=0)
        if st.button('▶ 自動分批下載歷史月營收（可中斷後接續）',disabled=mops_busy(root) or v231_busy(root)):
            try:start_mops(root,start_year,last);st.info('歷史補充已開始，請查看下方進度。')
            except Exception as exc:st.error(str(exc))
        if s and s.get('status')=='ERROR':st.warning(s.get('message','MOPS錯誤'))
    st.subheader('③ 一鍵比較：完整模型 vs 0050')
    active_release=load_v231_release(root)
    st.success('目前策略：'+active_release['release_id']+'｜SHA '+active_release['release_sha'])
    st.write('目前共有'+str(len(active_release['models']))+'組候選（含0050基準）；20／60日為主、5日為輔助。')
    with st.expander('後續模型更新：只上傳單一JSON，不用再換Python檔',expanded=False):
        st.download_button('下載目前策略JSON',json_bytes(active_release),'V23_1_strategy_release.json','application/json',on_click='ignore')
        candidate=st.file_uploader('上傳新的V23.1策略JSON',type=['json'],key='v231_strategy')
        if candidate is not None:
            try:
                uploaded=parse_v231_release(candidate.getvalue())
                st.info('可套用：'+uploaded['release_id']+'｜SHA '+uploaded['release_sha']+'｜候選'+str(len(uploaded['models']))+'組')
                if st.button('套用這份策略並立即重新驗效',type='primary',disabled=v231_busy(root)):
                    persist_v231_release(root,uploaded)
                    start(root,V23Config())
                    st.success('策略已套用；不會使用舊策略的驗證快取。')
            except Exception as exc:st.error('策略檔無效：'+str(exc))
    st.caption('V23.1固定驗收門檻：年化淨超額至少+5個百分點、資訊比率≥0.5、成本壓力後仍過關；歷史公告與名冊未獨立稽核時，不能標示正式達標。')
    busy=v231_busy(root)
    if st.button('▶ 沿用現有行情，開始／接續V23.1驗效',type='primary',use_container_width=True,disabled=busy):
        try:start(root,V23Config());st.info('已啟動研究。只新增V23快取，不會重建KOSTO250或碰最後保留期。')
        except Exception as exc:st.error('無法啟動：'+str(exc))
    @st.fragment(run_every='4s')
    def status():
        s=get_state(root)
        if not s:return
        st.progress(max(0.,min(1.,float(s.get('pct') or 0.0))),text=str(s.get('phase',''))+'｜'+str(s.get('message','')))
        st.caption('狀態：'+str(s.get('status'))+'｜最近更新：'+str(s.get('at','')))
        if s.get('status')=='RUNNING' and not v231_busy(root):
            st.warning('Streamlit執行程序已重啟／中斷。可再次按開始，已完成的歷史下載、因子與資料快取會繼續沿用。')
        if s.get('status')=='ERROR':
            st.error(str(s.get('message')))
            with st.expander('錯誤細節'):st.code(str(s.get('trace','')))
        if s.get('status')!='COMPLETE':return
        latest=read_json(job_path(root));run=Path(latest.get('run',''))
        result=read_json(run/'results.json')
        metrics=pd.DataFrame(result.get('models',[]))
        if not metrics.empty:
            show=[k for k in ('horizon','model','status','cagr','benchmark_cagr','excess_cagr_pp','extra_roundtrip_stress_cagr_diff_pp','ir','mdd','data_coverage') if k in metrics]
            st.dataframe(metrics[show],hide_index=True,use_container_width=True)
        st.info('此為「曾經查看的歷史資料」研究結果；若沒有真實公告時點與獨立未見過資料，即使超額為正也不代表可正式投入真金白銀。')
        if (run/'V23_1_VALIDATION.zip').is_file():
            st.download_button('⬇ 下載V23.1完整驗效報告ZIP',(run/'V23_1_VALIDATION.zip').read_bytes(),
                              'V23_1_VALIDATION.zip','application/zip',type='primary',use_container_width=True,on_click='ignore')
        if (run/'V23_1_VALIDATION.xlsx').is_file():
            st.download_button('⬇ 下載Excel研究報告',(run/'V23_1_VALIDATION.xlsx').read_bytes(),
                              'V23_1_VALIDATION.xlsx','application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                              use_container_width=True,on_click='ignore')
    status()
    with st.expander('程式部署自檢（不用檢查哪個檔案）',expanded=False):
        if st.button('🔍 一鍵確認核心／策略版本／0050計算',disabled=v231_busy(root)):
            from v231_health import selfcheck
            try:st.json(selfcheck(root,Path(__file__).resolve().parent))
            except Exception as exc:st.error('自檢未通過：'+str(exc))
    with st.expander('V23版本／資料稽核／限制',expanded=False):
        st.json({'v231_engine':VERSION,'release_id':active_release['release_id'],'release_sha':active_release['release_sha'],'hypothesis_names':list(active_release['models']), 'dataset_id':meta['dataset_id'],
                 'data_scope':meta.get('scope'),'user_verified_source_is_not_independent_audit':True,
                 'final_tail_not_opened':True,'strategy_research_only':True})
