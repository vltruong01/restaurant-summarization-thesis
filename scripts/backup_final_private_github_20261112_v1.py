"""Current lightweight private source/report backup on a new dated branch."""
from pathlib import Path
import datetime,hashlib,json,os,re,shutil,subprocess,sys,urllib.request

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/storage_closeout_20261112_v1'
SNAP=OUT/'GITHUB_SNAPSHOT'
REPO='vltruong01/multimodal_summarization_thesis'
BRANCH='research-closeout-20261112'
TOKEN=re.compile(r'(?:sk-(?:proj-)?[A-Za-z0-9_-]{32,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|hf_[A-Za-z0-9]{30,})')
PASSWORD=re.compile(r'''(?im)^\s*["']?(?:password|passwd|api_key|access_token|secret_key|hf_token|github_token)["']?\s*[:=]\s*["']([^"'\r\n]{5,})["']''')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(path,data):path.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
def git(*args,cwd=SNAP):
 env=os.environ.copy();env.update(GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='never')
 result=subprocess.run(['git',*args],cwd=cwd,capture_output=True,text=True,encoding='utf8',errors='replace',env=env,timeout=180)
 if result.returncode:raise RuntimeError('Git operation failed: '+args[0])
 return result.stdout.strip()
def credential():
 env=os.environ.copy();env.update(GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='never')
 result=subprocess.run(['git','-c','credential.interactive=never','credential','fill'],input='protocol=https\nhost=github.com\n\n',capture_output=True,text=True,env=env,timeout=25)
 values=dict(x.split('=',1) for x in result.stdout.splitlines() if '=' in x)
 assert result.returncode==0 and values.get('password');return values['password']
def metadata(token):
 req=urllib.request.Request('https://api.github.com/repos/'+REPO,headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json','User-Agent':'thesis-storage-closeout'})
 with urllib.request.urlopen(req,timeout=30) as response:return json.load(response)
def main():
 assert metadata(credential())['private'] is True
 SNAP.mkdir(exist_ok=True)
 if not (SNAP/'.git').exists():git('clone','--depth','1','https://github.com/'+REPO+'.git',str(SNAP),cwd=ROOT)
 git('config','core.autocrlf','false');git('config','user.name','vltruong01');git('config','user.email','vltruong01@users.noreply.github.com')
 git('checkout','-B',BRANCH)
 sources=[]
 for top in ['approvals','cloud','configs','docs','notebooks','prompts','requirements','schemas','scripts','specs','tests','thesis_sdd_docs']:
  sources.extend(p for p in (ROOT/top).rglob('*') if p.is_file() and not any(x in p.parts for x in ['.git','__pycache__','.ipynb_checkpoints']) and p.suffix not in ['.pyc','.pyo'])
 sources.extend(p for p in ROOT.iterdir() if p.is_file() and p.suffix in ['.md','.py','.sh','.docx','.pdf'])
 for folder in ['chapter4_figures_20261112_v1','chapter4_figures_20261112_v2','final_research_analysis_20261106_v1/PACKAGE']:
  base=ROOT/'outputs'/folder
  sources.extend(p for p in base.iterdir() if p.is_file() and p.suffix.lower() in ['.md','.py','.png','.pdf','.json','.txt'])
 for name in ['README_CLEANUP_VI.md','FINAL_CLEANUP_REPORT.json','FINAL_HF_OFFLOAD_RECEIPT.json','FINAL_PROTECTED_FILES_AND_APP_CHECK.json','HF_CATALOG_RECEIPT.json','ACTUAL_RESTORE_VERIFICATION.json','PROTECTED_FINAL_CHECKPOINTS.json','PROTECTED_FINAL_DELIVERABLES.json','OFFLOAD_PLAN_BINDING.json','LATEST_LIVE_STATE_HF_RECEIPT.json','EXISTING_HF_TRAIN_DATA_BINDING.json','restore_research_from_hf.py']:
  candidate=OUT/name
  if candidate.exists():sources.append(candidate)
 files=[];excluded=[]
 for source in sorted(set(sources)):
  rel=source.relative_to(ROOT).as_posix()
  if source.stat().st_size>40*2**20:excluded.append(dict(path=rel,reason='Large file requires HF research backup'));continue
  if any(x.lower() in {'.env','credentials.json','secrets.json'} for x in source.parts):excluded.append(dict(path=rel,reason='Credential filename; retained locally'));continue
  target_rel='research_reports/'+source.relative_to(ROOT/'outputs').as_posix() if rel.startswith('outputs/') else rel
  target=SNAP/target_rel;target.parent.mkdir(parents=True,exist_ok=True)
  raw=source.read_bytes();changed=False
  if source.suffix=='.ipynb':
   d=json.loads(raw)
   for cell in d['cells']:
    if cell['cell_type']=='code':cell['outputs']=[];cell['execution_count']=None
    cell.get('metadata',{}).pop('execution',None)
   d.get('metadata',{}).pop('widgets',None);raw=(json.dumps(d,ensure_ascii=False,indent=1)+'\n').encode('utf8');changed=True
  text=raw.decode('utf8',errors='replace') if source.suffix.lower() in ['.py','.json','.md','.txt','.yaml','.yml','.sh','.ipynb','.html','.toml','.in','.lock'] else ''
  if TOKEN.search(text) or PASSWORD.search(text):excluded.append(dict(path=rel,reason='Potential credential literal; retained locally, not uploaded'));continue
  target.write_bytes(raw);files.append(dict(source=rel,github_path=target_rel,original_sha256=sha(source),backup_sha256=hashlib.sha256(raw).hexdigest(),backup_bytes=len(raw),notebook_outputs_removed_in_backup=changed))
 save(SNAP/'CURRENT_BACKUP_MANIFEST.json',dict(format='research_source_closeout_v1',created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=files,excluded=excluded))
 (SNAP/'.gitattributes').write_text('* -text\n',encoding='utf8')
 (SNAP/'BACKUP_SCOPE.md').write_text('# Current research source backup\n\nPrivate code, specifications, proposals, source notebooks and compact final reports/figures. Large datasets, original executed notebook outputs, model weights, raw human ratings and account stores are backed up separately on private Hugging Face where verified; this source snapshot alone does not authorize their deletion. Potential credential-containing source files remain local and are listed without their values.\n',encoding='utf8')
 git('add','--all')
 if git('status','--porcelain'):git('commit','-m','Back up completed thesis source, final analysis and Chapter 4 figures')
 git('push','origin','HEAD:refs/heads/'+BRANCH)
 commit=git('rev-parse','HEAD');assert git('ls-remote','origin','refs/heads/'+BRANCH).split()[0]==commit
 restored=OUT/('GITHUB_FRESH_RESTORE_'+commit[:12])
 if not restored.exists():git('clone','--depth','1','--branch',BRANCH,'https://github.com/'+REPO+'.git',str(restored),cwd=ROOT)
 assert git('rev-parse','HEAD',cwd=restored)==commit
 for row in files:assert sha(restored/row['github_path'])==row['backup_sha256']
 assert sha(restored/'CURRENT_BACKUP_MANIFEST.json')==sha(SNAP/'CURRENT_BACKUP_MANIFEST.json')
 assert metadata(credential())['private'] is True
 result=dict(status='PASS_PRIVATE_PUSH_FRESH_CLONE_AND_ALL_FILE_HASHES',repo=REPO,branch=BRANCH,commit=commit,files=len(files),bytes=sum(r['backup_bytes'] for r in files),excluded=excluded,source_files_deleted=0,credentials_logged=False)
 save(OUT/'GITHUB_VERIFIED_RECEIPT.json',result);print(json.dumps({k:v for k,v in result.items() if k!='excluded'},indent=2));print('Excluded potential credential/large files: '+str(len(excluded)))
if __name__=='__main__':
 try:main()
 except Exception as error:print('STOP GitHub: '+type(error).__name__+'; originals retained; no credentials logged.',file=sys.stderr);raise SystemExit(1)
