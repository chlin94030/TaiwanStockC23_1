"""V23.1 one-JSON strategy releases. Core never executes uploaded code.
Version, factor allowlist, canonical SHA and all weights validated fail-closed.
Every release creates a new experiment ID; it cannot reuse old report cache.
"""
from __future__ import annotations
import json
from pathlib import Path
from lab_io import DataError,atomic_json,read_json
from studio_pool import digest
from v231_engine import MODELS,VERSION

ALLOWED_FACTORS={
 'revenue_yoy_3m','revenue_acceleration','eps_growth_yoy','rs_0050_120',
 'momentum120_skip20','high252','gross_margin_delta','residual_lowvol60','institutional_flow5'
}


def _stamp(obj):
    x={k:v for k,v in obj.items() if k!='release_sha'}
    return {**x,'release_sha':digest(x)[:16]}


def bundled():
    return _stamp({'schema':1,'requires_engine':VERSION,'release_id':'R231-BENCHMARK-FIRST-BASELINE',
                   'note':'Predeclared static hypotheses, not tested optimal weights.',
                   'models':MODELS})


def parse(data):
    try:obj=json.loads(data.decode('utf-8-sig') if isinstance(data,bytes) else str(data))
    except Exception as exc:raise DataError('不是有效JSON策略檔：'+str(exc))
    if not isinstance(obj,dict) or set(obj)-{'schema','requires_engine','release_id','note','models','release_sha'}:
        raise DataError('V23.1策略檔包含未知欄位')
    if obj.get('schema')!=1 or obj.get('requires_engine')!=VERSION:
        raise DataError('策略JSON引擎版本不符：必須使用'+VERSION)
    rid=obj.get('release_id')
    if not isinstance(rid,str) or not 1<=len(rid)<=80:raise DataError('release_id 必須有值')
    models=obj.get('models')
    if not isinstance(models,dict) or not 2<=len(models)<=8 or models.get('0050_ONLY')!={}:
        raise DataError('必須包含 ETF-only 基準，並提供1至7個候選')
    for name,w in models.items():
        if not isinstance(name,str) or not name or len(name)>60 or not isinstance(w,dict):raise DataError('模型名稱／權重格式錯誤')
        if name=='0050_ONLY':continue
        if not (1<=len(w)<=9 and set(w)<=ALLOWED_FACTORS):raise DataError('策略使用不支援的因子')
        try:values=[float(v) for v in w.values()]
        except (ValueError,TypeError):raise DataError('權重不是數值')
        if any(not 0<v<=1 for v in values) or abs(sum(values)-1)>1e-8:
            raise DataError('各模型因子權重必須為正且合計100%')
    stamped=_stamp(obj)
    if obj.get('release_sha') and obj['release_sha']!=stamped['release_sha']:
        raise DataError('策略檔SHA校驗失敗，不能套用混版')
    return stamped


def load(root):
    p=Path(root)/'v23'/'active_strategy_release.json'
    return parse(p.read_bytes()) if p.exists() else bundled()


def persist(root,obj):
    x=parse(json.dumps(obj,ensure_ascii=False).encode('utf-8'))
    p=Path(root)/'v23'/'active_strategy_release.json';p.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(p,x)
    return x
