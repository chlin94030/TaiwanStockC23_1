from pathlib import Path
import argparse,json
from v22_runner import run_experiment,run_pool_study

p=argparse.ArgumentParser(description='Alpha Radar V22 offline stage-first research runner')
p.add_argument('--root',default='data/v21_universe_lab')
p.add_argument('--profiles',default='v22_profiles.json')
p.add_argument('--settings',default='v22_settings.json')
p.add_argument('--mode',choices=['fast','confirm','pool-study'],default='fast')
a=p.parse_args()
profiles=json.load(open(a.profiles,encoding='utf-8'));settings=json.load(open(a.settings,encoding='utf-8'))
if a.mode=='pool-study':run=run_pool_study(Path(a.root),profiles,settings)
else:run=run_experiment(Path(a.root),profiles,settings,a.mode)
print(run)
