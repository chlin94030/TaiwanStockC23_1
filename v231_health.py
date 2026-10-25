"""Self-contained deployment integrity and offline smoke diagnostics."""
from pathlib import Path
import numpy as np
import pandas as pd
from lab_io import sha,DataError
from v231_engine import VERSION,MODELS,V23Config,simulate_nav
from v231_release import bundled,parse,load

ESSENTIAL=('app.py','v231_ui.py','v231_runner.py','v231_engine.py','v231_data.py',
           'v231_factors.py','v231_release.py','v231_mops.py','v231_mops_jobs.py','lab_snapshot.py')


def selfcheck(root,repo):
    repo=Path(repo);missing=[f for f in ESSENTIAL if not (repo/f).is_file()]
    if missing:raise DataError('部署缺少核心檔案：'+str(missing))
    app=(repo/'app.py').read_text(encoding='utf-8')
    if 'V23.1全新選股驗效' not in app:raise DataError('目前Streamlit主入口不是V23.1')
    cfg=V23Config().validate();release=load(root)
    if parse(__import__('json').dumps(release))!=release:raise DataError('策略SHA不一致')
    cal=pd.bdate_range('2024-01-01',periods=35)
    bench=pd.DataFrame({'date':cal,'open':100*(1.001)**np.arange(len(cal)),
                        'close':100*(1.001)**np.arange(len(cal))})
    data,mt=simulate_nav(cal,bench,{},pd.DataFrame(),cfg)
    if abs(mt['excess_cagr_pp'])>1e-10:raise DataError('0050純基準報酬計算不一致')
    return {'result':'PASS','core_version':VERSION,'strategy_release':release['release_id'],
            'strategy_sha':release['release_sha'],'core_files_count':len(ESSENTIAL),
            'core_files_hash':{n:sha(repo/n)[:12] for n in ESSENTIAL},
            'etf_only_vs_0050_excess_pp':mt['excess_cagr_pp'],
            'external_market_api_status':'NOT_TESTED_BY_SELF_CHECK',
            'financial_independent_PIT_status':'NOT_TESTED_BY_SELF_CHECK'}
