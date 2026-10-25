"""Stdlib-only completed-job probe. Does not download the database."""
import hashlib,json,os,urllib.request
from pathlib import Path


def digest():
    h=hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob('*.py')):
        if not p.name.startswith('test_'):h.update(p.name.encode());h.update(p.read_bytes())
    return h.hexdigest()

if __name__=='__main__':
    repo=os.environ['GITHUB_REPOSITORY']
    req=urllib.request.Request('https://api.github.com/repos/'+repo+'/actions/artifacts?per_page=100',
        headers={'Authorization':'Bearer '+os.environ['GH_TOKEN'],'Accept':'application/vnd.github+json',
                 'User-Agent':'AlphaRadarCheckpoint/21.2'})
    with urllib.request.urlopen(req,timeout=30) as r:rows=json.load(r).get('artifacts',[])
    rows=sorted([x for x in rows if x.get('name','').startswith('v21-checkpoint-') and not x.get('expired')],
                key=lambda x:(x.get('created_at',''),int(x['id'])),reverse=True)
    done=bool(rows and rows[0]['name'].startswith('v21-checkpoint-'+digest()[:12]+'-') and rows[0]['name'].endswith('-complete'))
    with open(os.environ['GITHUB_OUTPUT'],'a') as f:f.write('complete='+str(done).lower()+'\n')
    print('Matching completed checkpoint found: '+str(done))
