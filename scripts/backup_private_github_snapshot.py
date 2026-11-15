"""Private lightweight source backup; original datasets/model assets never uploaded/deleted."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import urllib.request
import urllib.error

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/storage_maintenance_20261007'
SNAPSHOT=OUT/'github_snapshot'
OWNER='vltruong01'
REPO='multimodal_summarization_thesis'
SECRET=re.compile(r'(?:sk-(?:proj-)?|gh[pousr]_|github_pat_|hf_)[A-Za-z0-9_-]{20,}')


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def prepare():
    SNAPSHOT.mkdir(parents=True,exist_ok=True)
    files=[]
    for name in ['README.md','DECISIONS.md','THESIS_PROGRESS.md','thesis_proposal_TruongVuong.pdf','thesis_proposal_v16.docx']:
        files.append(ROOT/name)
    for name in ['configs','docs','requirements','scripts','specs','thesis_sdd_docs','notebooks']:
        files.extend(p for p in (ROOT/name).rglob('*') if p.is_file() and not any(k in p.parts for k in ['__pycache__','.ipynb_checkpoints']) and p.suffix not in ['.pyc','.pyo'])
    manifest=[]
    for source in sorted(files):
        relative=source.relative_to(ROOT);target=SNAPSHOT/relative;target.parent.mkdir(parents=True,exist_ok=True)
        assert source.stat().st_size<80*1024*1024,'Oversize source needs separate review.'
        if source.suffix=='.ipynb':
            notebook=json.loads(source.read_text(encoding='utf-8'))
            for cell in notebook['cells']:
                if cell['cell_type']=='code':cell['outputs']=[];cell['execution_count']=None
                cell.get('metadata',{}).pop('execution',None)
            notebook.get('metadata',{}).pop('widgets',None)
            content=json.dumps(notebook,ensure_ascii=False,indent=1)+'\n'
            assert not SECRET.search(content),'Possible token in notebook source; upload stopped.'
            target.write_text(content,encoding='utf-8')
        else:
            if source.suffix.lower() in ['.py','.md','.yaml','.yml','.json','.txt','.in','.lock','.html']:
                assert not SECRET.search(source.read_text(encoding='utf-8',errors='replace')),'Possible secret; upload stopped without printing it.'
            shutil.copy2(source,target)
        manifest.append({'path':relative.as_posix(),'local_source_sha256':sha(source),'backup_sha256':sha(target),'backup_bytes':target.stat().st_size,'notebook_outputs_removed_in_backup_only':source.suffix=='.ipynb'})
    (SNAPSHOT/'.gitignore').write_text('.venv*/\n__pycache__/\n*.py[cod]\n.env*\n/data/\n/outputs/\n/logs/\n/models/\n/assets/model_cache/\n.ipynb_checkpoints/\n',encoding='utf-8')
    (SNAPSHOT/'.gitattributes').write_text('* -text\n',encoding='utf-8')
    scope='''# Backup scope

Private source/docs/proposals/notebook-source snapshot of the thesis project.

Original Yelp JSON/images, processed datasets, trained/pretrained weights, caption records, scientific output caches and human annotation files are NOT stored in this repository. Their local originals remain intact; do not delete them because this source backup exists. Executed local notebook outputs remain untouched; only remote snapshot outputs are removed to keep this repo lightweight and avoid embedding original review/photo data.

Restore original dataset/cache artifacts separately before running dataset-dependent notebooks. See docs/STORAGE_POLICY.md and local maintenance receipts. No Git LFS/payment/big dataset upload is enabled by this backup.
'''
    (SNAPSHOT/'BACKUP_SCOPE.md').write_text(scope,encoding='utf-8')
    report={'status':'LIGHTWEIGHT_PRIVATE_BACKUP_PREPARED','files':manifest,'total_backup_bytes':sum(r['backup_bytes'] for r in manifest),'original_datasets_backed_up':False,'local_executed_notebooks_unchanged':True,'secrets_scan_passed':True,'snapshot_path':SNAPSHOT.relative_to(ROOT).as_posix(),'prepared_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (OUT/'github_snapshot_manifest.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report


def credential():
    env=os.environ.copy();env.update(GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='never')
    p=subprocess.run(['git','-c','credential.interactive=never','credential','fill'],input='protocol=https\nhost=github.com\n\n',capture_output=True,text=True,env=env,timeout=30)
    data=dict(l.split('=',1) for l in p.stdout.splitlines() if '=' in l)
    assert p.returncode==0 and data.get('password'),'GitHub auth unavailable; no interactive login/token logging.'
    return data['password']


def api(token,path,method='GET',data=None):
    request=urllib.request.Request('https://api.github.com'+path,data=json.dumps(data).encode() if data is not None else None,method=method,
            headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28','User-Agent':'thesis-maintenance','Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=30) as response:return json.load(response)


def git(*args,cwd=SNAPSHOT):
    env=os.environ.copy();env.update(GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='never')
    for attempt in range(3):
        p=subprocess.run(['git',*args],cwd=cwd,env=env,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=180)
        if p.returncode==0 or args[0]!='push' or 'Internal Server Error' not in p.stderr:break
        time.sleep(3)
    if p.returncode:raise RuntimeError('Git command failed: '+args[0]+'; '+p.stderr[-1500:])
    return p.stdout.strip()


def publish(report):
    token=credential();user=api(token,'/user');assert user['login']==OWNER
    try:repo=api(token,'/repos/'+OWNER+'/'+REPO)
    except urllib.error.HTTPError as error:
        if error.code!=404:raise
        repo=api(token,'/user/repos','POST',{'name':REPO,'private':True,'description':'Private multimodal summarization thesis source/docs backup; datasets retained separately.','auto_init':False})
    assert repo['private'] is True and repo['owner']['login']==OWNER,'Never upload to a public or other-owner repository.'
    if not (SNAPSHOT/'.git').exists():
        git('init','-b','main')
        git('config','user.name',user.get('name') or OWNER)
        git('config','user.email',str(user['id'])+'+'+OWNER+'@users.noreply.github.com')
        git('remote','add','origin','https://github.com/'+OWNER+'/'+REPO+'.git')
    git('config','core.autocrlf','false')
    before_commit=git('rev-parse','HEAD') if git('status','--porcelain') is not None and (SNAPSHOT/'.git/refs/heads/main').exists() else None
    git('add','--all')
    git('add','--renormalize','.')
    if git('status','--porcelain'):git('commit','-m','Back up thesis source, plans and notebook code; retain datasets locally')
    previous_report=OUT/'github_backup_report.json'
    previous=json.loads(previous_report.read_text(encoding='utf-8')) if previous_report.exists() else None
    remote_main=git('ls-remote','origin','refs/heads/main')
    own_existing_main=bool(before_commit and remote_main and remote_main.split()[0] in git('rev-list',before_commit).splitlines())
    branch=previous['branch'] if previous and previous['url']==repo['html_url'] else 'main' if repo.get('size',0)==0 or own_existing_main else 'workspace-backup-20261007'
    git('push','origin','HEAD:refs/heads/'+branch)
    commit=git('rev-parse','HEAD')
    assert git('ls-remote','origin','refs/heads/'+branch).split()[0]==commit
    fresh=api(token,'/repos/'+OWNER+'/'+REPO);assert fresh['private']
    restore=OUT/('restore_check_'+commit[:12])
    if not (restore/'.git').exists():git('clone','--depth','1','--branch',branch,'https://github.com/'+OWNER+'/'+REPO+'.git',str(restore),cwd=ROOT)
    for r in report['files']:assert sha(restore/r['path'])==r['backup_sha256'],r['path']
    assert sha(restore/'BACKUP_SCOPE.md')==sha(SNAPSHOT/'BACKUP_SCOPE.md')
    result={'status':'PRIVATE_REPOSITORY_PUSH_AND_FRESH_CLONE_HASH_VERIFIED','url':fresh['html_url'],'private':True,'branch':branch,'commit':commit,'file_count':len(report['files']),'backup_bytes':report['total_backup_bytes'],'all_snapshot_file_hashes_restored':True,'original_datasets_uploaded':False,'datasets_or_notebook_outputs_deleted':False,'LFS_used':False,'credentials_logged_or_committed':False}
    (OUT/'github_backup_report.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--publish',action='store_true');args=p.parse_args()
    report=prepare()
    if args.publish:publish(report)
    else:print(json.dumps({'status':report['status'],'files':len(report['files']),'bytes':report['total_backup_bytes']}))
