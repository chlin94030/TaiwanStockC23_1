"""Lightweight default page: adopt -> explicitly lock -> offline fast iterations."""
from pathlib import Path
import json
from lab_io import read_json, DataError
from studio_pool import discover_prepared, adopt_dataset, load_dataset, lock_pool, load_pool, report_pool_suggestions
from studio_runner import start_studio, studio_alive, DEFAULT_SPEC, STUDIO_VERSION

HERE=Path(__file__).resolve().parent


def render(st,root):
    root=Path(root)
    st.title('\u56fa\u5b9a\u6bcd\u6c60\u00b7\u7b56\u7565\u5de5\u4f5c\u53f0')
    st.caption(STUDIO_VERSION+' | \u5feb\u901f\u6a21\u5f0f\u4e0d\u4e0b\u8f09\u884c\u60c5\u3001\u4e0d\u91cd\u8dd1\u6bcd\u6c60\u641c\u5c0b\u3001\u4e0d\u67e5\u770b\u6700\u5f8c\u4fdd\u7559\u671f\u3002')
    busy=studio_alive(root)
    from lab_jobs import workspace_busy
    busy=busy or workspace_busy(root)
    prepared=discover_prepared(root)
    dataset=None;locked=None
    try:
        if (root/'studio'/'active_dataset.json').exists():_,dataset=load_dataset(root)
        if (root/'studio'/'active_pool.json').exists():locked=load_pool(root)
    except Exception as exc:st.warning(str(exc))
    with st.expander('\u2460 \u6cbf\u7528\u5df2\u6709\u8cc7\u6599\uff08\u901a\u5e38\u53ea\u9700\u8a2d\u5b9a\u4e00\u6b21\uff09',expanded=dataset is None):
        st.write('\u8acb\u5148\u9084\u539f\u81ea\u5df1\u7684 V21_DATABASE.zip\uff0c\u6216\u9078\u7528\u5df2\u5b8c\u6210\u6e96\u5099\u7684\u8cc7\u6599\u3002LAB_RESULTS.zip \u53ea\u662f\u5831\u544a\uff0c\u4e0d\u662f\u884c\u60c5\u8cc7\u6599\u5eab\u3002')
        if prepared:
            selected=st.selectbox('\u5df2\u6e96\u5099\u7684\u8cc7\u6599\u4efb\u52d9',prepared,format_func=lambda p:p['name']+' | '+str(p['start'])+' ~ '+str(p['end']))
            if st.button('\u767b\u9304\u9019\u4efd\u8cc7\u6599\uff08\u4e0d\u4e0b\u8f09\uff09',disabled=busy):
                try:
                    adopt_dataset(root,selected['run']);st.rerun()
                except Exception as exc:st.error(str(exc))
        else:st.info('\u9084\u6c92\u6709\u5b8c\u6574\u8cc7\u6599\u3002\u8acb\u5230\u4e0a\u65b9\u300c\u8cc7\u6599\u6e96\u5099\u8207\u5099\u4efd\u300d\u9084\u539f\u6aa2\u67e5\u9ede\uff0c\u6216\u57f7\u884c\u4e00\u6b21\u8cc7\u6599\u6e96\u5099\u3002')
        if dataset:
            st.write('\u8cc7\u6599\u671f\u9593\uff1a'+dataset['source_config']['start']+' ~ '+dataset['source_config']['end'])
            st.caption('Dataset ID: '+dataset['dataset_id'])
    if not dataset:return
    with st.expander('\u2461 \u56fa\u5b9a\u6bcd\u6c60\u898f\u5247\uff08\u4e0d\u662f\u56fa\u5b9a\u4eca\u5929\u7684\u540d\u55ae\uff09',expanded=locked is None):
        st.info('\u56fa\u5b9a ADV200 \u4ee3\u8868\u5404\u6b77\u53f2\u6708\u4efd\u90fd\u7528\u7576\u6642\u7684\u524d200\u6a94\u3002\u9019\u662f\u660e\u78ba\u9078\u5b9a\u7684\u7814\u7a76\u7bc4\u570d\uff0c\u4e0d\u4ee3\u8868\u5df2\u8b49\u660e\u5168\u5c40\u6700\u4f73\u3002')
        pools=['ADV'+str(n) for n in dataset['source_config']['pools']]
        if dataset['source_config'].get('include_market_cap_pools'):pools+=['MCAP'+str(n) for n in dataset['source_config']['pools']]
        mapping={}
        for h in dataset['source_config']['horizons']:
            old=(locked or {}).get('mapping',{}).get(str(h),'ADV200')
            mapping[str(h)]=st.selectbox(str(h)+'\u65e5\u6bcd\u6c60',pools,index=pools.index(old) if old in pools else 0,key='pool_'+str(h))
        reason=st.text_input('\u56fa\u5b9a\u7406\u7531\uff0f\u7814\u7a76\u5047\u8aaa',value='USER_FIXED_PROVISIONAL_SCOPE')
        source=root/'runs'/dataset['source_run']
        suggestions=report_pool_suggestions(root)
        if suggestions:
            st.caption('\u820a\u5831\u544a\u7684\u8a13\u7df4\u671f\u9078\u64c7\uff08\u4e0d\u7528\u6700\u5f8c\u6e2c\u8a66\u5831\u916c\u6311\u6bcd\u6c60\uff09')
            st.json(suggestions)
        acknowledged=st.checkbox('\u6211\u77e5\u9053\u9019\u662f\u56fa\u5b9a\u7814\u7a76\u898f\u5247\uff0c\u4e0d\u662f\u5df2\u8b49\u660e\u6700\u9069\u6bcd\u6c60',key='lock_ack')
        if st.button('\u56fa\u5b9a\u6bcd\u6c60\u898f\u5247',disabled=busy or not acknowledged):
            try:lock_pool(root,mapping,reason);st.rerun()
            except Exception as exc:st.error(str(exc))
        if locked:st.caption('Lock ID: '+locked['lock_id']+' | '+json.dumps(locked['mapping']))
    if not locked:return
    st.subheader('\u2462 \u8abf\u6574\u7b56\u7565\uff0c\u76f4\u63a5\u5feb\u901f\u6bd4\u8f03')
    st.write('\u6bcf\u500b\u7b56\u7565\u6700\u591a4\u500b\u56e0\u5b50\uff0c\u6b0a\u91cd\u7e3d\u548c\u70ba1\u3002\u66f4\u6539\u6b0a\u91cd\u4e0d\u91cd\u7b97\u6bcd\u6c60\uff1b\u672a\u8b8a\u7684\u7b56\u7565\u53ef\u91cd\u7528\u7d50\u679c\u3002')
    st.caption('\u66f8\u4e2d\u539f\u5247\u53ea\u4f5c\u7814\u7a76\u555f\u767c\uff1a\u6210\u4ea4\u91d1\u984d\u2260\u8ca8\u5e63\u4f9b\u7d66\uff1b\u50f9\u91cf\u4ee3\u7406\u4e0d\u80fd\u76f4\u63a5\u77e5\u9053\u6301\u6709\u4eba\u5fc3\u7406\u3002')
    editor=read_json(root/'studio'/'editor.json')
    profiles_default=json.dumps(editor.get('profiles',read_json(HERE/'studio_profiles.json')),ensure_ascii=False,indent=2)
    settings_default=json.dumps(editor.get('settings',read_json(HERE/'studio_settings.json')),ensure_ascii=False,indent=2)
    with st.form('strategy_editor'):
        profiles_text=st.text_area('\u7b56\u7565 JSON\uff08\u53ef\u76f4\u63a5\u4fee\u6539\u56e0\u5b50\u6b0a\u91cd\uff09',value=profiles_default,height=300)
        spec_text=st.text_area('\u5feb\u901f\u6e2c\u8a66\u8a2d\u5b9a\uff08\u8cc7\u91d1\u3001\u6301\u80a1\u6578\u3001\u8cbb\u7528\uff09',value=settings_default,height=190)
        execute=st.form_submit_button('\u25b6 \u5feb\u901f\u6e2c\u8a66\uff0f\u7e7c\u7e8c\uff08\u4e0d\u78b0\u6700\u5f8c\u4fdd\u7559\u671f\uff09',disabled=busy)
    def launch(mode):
        try:
            p=json.loads(profiles_text);s=json.loads(spec_text)
            start_studio(root,p,s,mode);st.success('\u4efb\u52d9\u5df2\u555f\u52d5\uff0c\u4e0b\u65b9\u6703\u66f4\u65b0\u9032\u5ea6\u3002')
        except Exception as exc:st.error(str(exc))
    if execute:launch('fast')
    with st.expander('\u2463 \u51cd\u7d50\u5f8c\u624d\u57f7\u884c\u6700\u7d42\u78ba\u8a8d\uff08\u4e0d\u662f\u6bcf\u6b21\u8abf\u6b0a\u91cd\u90fd\u6309\uff09'):
        st.warning('\u540c\u4e00\u6700\u5f8c\u671f\u9593\u50c5\u767c\u5e03\u4e00\u6b21\u7d50\u679c\uff1b\u767c\u5e03\u5f8c\u4e0d\u80fd\u63db\u53c3\u6578\u91cd\u958b\u5b83\u3002\u5df2\u88ab\u820a\u7248\u89c0\u5bdf\u7684\u6b77\u53f2\u4ecd\u4e0d\u662f\u672a\u4f86\u76f2\u6e2c\u3002')
        ok=st.checkbox('\u6211\u5df2\u51cd\u7d50\u672c\u6b21\u5047\u8aaa\u8207\u53c3\u6578\uff0c\u540c\u610f\u767c\u5e03\u4fdd\u7559\u671f\u7d50\u679c')
        if st.button('\u57f7\u884c\u4e00\u6b21\u51cd\u7d50\u78ba\u8a8d',disabled=busy or not ok):launch('confirm')
    if st.button('\u66ab\u505c\u4e26\u4fdd\u7559\u5feb\u53d6',disabled=not busy):
        latest=read_json(root/'studio'/'latest.json')
        if latest.get('run'):(Path(latest['run'])/'PAUSE').touch()
    @st.fragment(run_every='3s')
    def monitor():
        if bool(studio_alive(root) or workspace_busy(root)) != bool(busy):
            st.rerun()
        err=read_json(root/'studio'/'launch_error.json')
        if err:st.error(err.get('type','')+': '+err.get('message',''))
        latest=read_json(root/'studio'/'latest.json')
        if not latest.get('run'):return
        run=Path(latest['run']);state=read_json(run/'state.json')
        st.subheader(str(state.get('status',''))+' | '+str(state.get('phase','')))
        st.write(state.get('message',''))
        if state.get('total'):
            st.progress(min(1.,max(0.,state.get('done',0)/state['total'])))
        result=read_json(run/'results.json')
        if result:
            st.json(result.get('cache',{}))
            for h,m in result.get('models',{}).items():
                st.write(str(h)+'\u65e5 | '+m['pool']+' | '+m['choice']['name']+' | '+m['verdict'])
                if m.get('profile_comparison'):
                    st.dataframe(m['profile_comparison'],use_container_width=True)
            with st.expander('\u76f4\u63a5\u95b1\u8b80\u5831\u544a'):st.text((run/'REPORT.txt').read_text(encoding='utf-8-sig'))
            for fn,mime in [('REPORT.txt','text/plain'),('results.json','application/json'),('STRATEGY_RESULTS.zip','application/zip')]:
                if (run/fn).is_file():st.download_button('\u4e0b\u8f09 '+fn,(run/fn).read_bytes(),fn,mime,on_click='ignore',key='dlstudio'+fn)
    monitor()
