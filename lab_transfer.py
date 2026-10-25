"""Small server-side snapshot tasks; no Streamlit calls from worker threads."""
from pathlib import Path
import threading
from lab_io import atomic_json,read_json,DataError
from lab_snapshot import export_snapshot,import_snapshot
from lab_remote import restore_latest
_LOCK=threading.Lock()
_TASKS={}


def start_transfer(root,operation,source=None,repo=None,token=None):
    root=Path(root);key=str(root.resolve());statepath=root/'transfer_state.json'
    with _LOCK:
        thread=_TASKS.get(key)
        if thread and thread.is_alive():return False
        if operation not in ('export','import','github'):raise ValueError('Unknown transfer')
        atomic_json(statepath,{'status':'QUEUED','operation':operation})
        def work():
            try:
                atomic_json(statepath,{'status':'RUNNING','operation':operation})
                if operation=='export':result=export_snapshot(root,root.parent/'v21_exports'/'V21_DATABASE.zip')
                elif operation=='import':result=import_snapshot(source,root)
                else:result=restore_latest(repo,token,root)
                atomic_json(statepath,{'status':'COMPLETE','operation':operation,'result':result})
            except Exception as exc:
                # Fixed-message exceptions contain no signed URLs or credential values.
                msg=str(exc) if isinstance(exc,(DataError,ValueError)) else type(exc).__name__
                atomic_json(statepath,{'status':'ERROR','operation':operation,'message':msg[:350]})
        thread=threading.Thread(target=work,daemon=True,name='v21-checkpoint-transfer')
        _TASKS[key]=thread;thread.start()
        return True
