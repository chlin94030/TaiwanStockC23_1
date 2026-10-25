"""GitHub Actions data transfer. Network only on an explicit command/button.

Actions artifacts are finite-retention checkpoints, not permanent storage.
GH_TOKEN stays in headers; it is never stored in the database or snapshot.
"""
from pathlib import Path
from urllib.parse import urlparse
import json
import os
import re
import shutil
import tempfile
import zipfile
import requests
from lab_io import DataError
from lab_snapshot import import_snapshot

PREFIX='v21-checkpoint-'
MAX_ARCHIVE=1_500_000_000

class GitHub:
    def __init__(self,repo,token):
        if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',repo or ''):
            raise DataError('Set GITHUB_REPOSITORY to owner/repository')
        if not token:raise DataError('GitHub token missing (Actions read is needed for downloads)')
        self.base='https://api.github.com/repos/'+repo
        self.headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json',
                      'X-GitHub-Api-Version':'2022-11-28','User-Agent':'AlphaRadarCheckpoint/21.2'}
    def call(self,method,suffix,**kw):
        r=requests.request(method,self.base+suffix,headers=self.headers,timeout=(10,60),**kw)
        if r.status_code>=400:raise DataError('GitHub HTTP '+str(r.status_code)+'; check repository Actions permissions/retention')
        return r
    def artifacts(self):
        found=[]
        for page in range(1,11):
            a=self.call('GET',f'/actions/artifacts?per_page=100&page={page}').json().get('artifacts',[])
            found += [x for x in a if x.get('name','').startswith(PREFIX) and not x.get('expired')]
            if len(a)<100:break
        return sorted(found,key=lambda x:(x.get('created_at',''),int(x['id'])),reverse=True)
    def archive(self,suffix,destination):
        # Do not send the GitHub bearer token to a signed object-storage URL.
        r=self.call('GET',suffix,allow_redirects=False,stream=True)
        try:
            if r.status_code in (301,302,303,307,308):
                location=r.headers.get('Location','');u=urlparse(location)
                if u.scheme!='https' or not u.hostname:raise DataError('Unsafe artifact redirect')
                host=u.hostname.lower()
                if not (host.endswith('.githubusercontent.com') or host.endswith('.blob.core.windows.net') or host.endswith('.actions.githubusercontent.com')):
                    raise DataError('Unrecognized artifact download host')
                r.close();r=requests.get(location,timeout=(10,90),stream=True)
                if r.status_code>=400:raise DataError('Checkpoint download failed HTTP '+str(r.status_code))
            n=0
            with open(destination,'wb') as out:
                for chunk in r.iter_content(1024*1024):
                    n+=len(chunk)
                    if n>MAX_ARCHIVE:raise DataError('Checkpoint archive too large')
                    out.write(chunk)
        finally:r.close()
    def prune(self,keep=2):
        rows=self.artifacts()
        for a in rows[max(2,keep):]:self.call('DELETE','/actions/artifacts/'+str(a['id']))


def restore_latest(repo,token,root):
    api=GitHub(repo,token);rows=api.artifacts()
    if not rows:return {'restored':False,'reason':'NO_REMOTE_CHECKPOINT'}
    latest=rows[0]
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as td:
        outer=Path(td)/'artifact.zip'
        api.archive('/actions/artifacts/'+str(latest['id'])+'/zip',outer)
        # upload-artifact adds an outer ZIP around our verified inner snapshot.
        with zipfile.ZipFile(outer) as z:
            infos=z.infolist()
            match=[x for x in infos if x.filename=='V21_DATABASE.zip']
            if len(match)!=1 or match[0].file_size>MAX_ARCHIVE:raise DataError('Artifact lacks a unique V21_DATABASE.zip')
            inner=Path(td)/'snapshot.zip'
            with z.open(match[0]) as inp,open(inner,'wb') as out:shutil.copyfileobj(inp,out,1024*1024)
        result=import_snapshot(inner,root)
        result.update(restored=True,artifact_id=latest['id'],artifact_name=latest['name'])
        return result


def restore_release_seed(repo,token,asset_id,root):
    """Optional one-time migration of an existing Streamlit checkpoint.
    User explicitly chooses a release asset in the SAME repository."""
    if not str(asset_id).isdigit() or int(asset_id)<=0:raise DataError('seed_asset_id must be a positive integer')
    api=GitHub(repo,token)
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as td:
        path=Path(td)/'seed.zip'
        api.headers['Accept']='application/octet-stream'
        api.archive('/releases/assets/'+str(asset_id),path)
        return import_snapshot(path,root)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('command',choices=['restore','prune'])
    p.add_argument('--root',default='data/v21_universe_lab');p.add_argument('--seed-asset-id',default='')
    a=p.parse_args();api=GitHub(os.getenv('GITHUB_REPOSITORY',''),os.getenv('GH_TOKEN',''))
    if a.command=='prune':api.prune()
    else:
        result=restore_latest(os.getenv('GITHUB_REPOSITORY',''),os.getenv('GH_TOKEN',''),a.root)
        if not result.get('restored') and a.seed_asset_id:
            result=restore_release_seed(os.getenv('GITHUB_REPOSITORY',''),os.getenv('GH_TOKEN',''),a.seed_asset_id,a.root)
        print(json.dumps(result,ensure_ascii=False))
