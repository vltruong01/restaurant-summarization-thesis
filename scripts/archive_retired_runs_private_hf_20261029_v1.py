"""Optional CMD-invoked cold backup. Never deletes original source trees.

Existing PRIVATE HF dataset only, isolated new content-addressed namespace.
Only full freshly streamed archive+every-member SHA verification unlocks deletion.
"""
from pathlib import Path,PurePosixPath
import argparse,os,json,hashlib,tarfile,gzip,datetime,io,sys
R=Path(__file__).resolve().parents[1]
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:json.dump(x,f,indent=2,ensure_ascii=False);f.write('\n')
def need(ok,msg):
    if not ok:raise ValueError(msg)
def safe(relative,root=R):
    need(not Path(relative).is_absolute(),'Absolute source member')
    p=root/relative;need(p.resolve().is_relative_to(root.resolve()) and p.resolve()!=root.resolve(),'Source escapes workspace')
    for q in [p,*p.parents]:
        if q==root.parent:break
        if q.exists():need(not q.is_symlink() and not getattr(q.lstat(),'st_file_attributes',0)&0x400,'Reparse source')
    return p
def snapshot(root,workspace):
    files=[];dirs=[]
    for base,dd,nn in os.walk(root,followlinks=False):
        for name in dd+nn:
            p=Path(base)/name;s=p.lstat();need(not p.is_symlink() and not getattr(s,'st_file_attributes',0)&0x400,'Reparse member')
        d=Path(base);s=d.stat();dirs.append(dict(path=d.relative_to(workspace).as_posix(),mtime_ns=s.st_mtime_ns))
        for name in nn:
            p=d/name;s=p.stat();need(p.is_file(),'Nonregular source file')
            files.append(dict(path=p.relative_to(workspace).as_posix(),bytes=s.st_size,mtime_ns=s.st_mtime_ns))
    return sorted(files,key=lambda r:r['path']),sorted(dirs,key=lambda r:r['path'])
class HashFile:
    def __init__(self,f):self.f=f;self.h=hashlib.sha256();self.n=0
    def read(self,n=-1):
        b=self.f.read(n);self.h.update(b);self.n+=len(b);return b
def pack_group(group,out,workspace=R):
    source=workspace/group['path'];files,dirs=snapshot(source,workspace);need(files,'Empty source group')
    target=out/'UPLOAD_STAGING'/(group['id']+'.tar.gz');target.parent.mkdir(parents=True,exist_ok=True)
    manifest_path=out/'HF_MANIFESTS'/(group['id']+'.json')
    if manifest_path.exists():
        m=read(manifest_path);need([{k:r[k] for k in ['path','bytes','mtime_ns']} for r in m['files']]==files,'Source changed since staging')
        need(target.exists() and sha(target)==m['archive_sha256'],'Staged archive changed/missing')
        return m,target,manifest_path
    need(not target.exists(),'Partial staging exists; preserve it and diagnose before rerun')
    with target.open('xb') as raw,gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0,compresslevel=1) as compressed:
        with tarfile.open(fileobj=compressed,mode='w|',format=tarfile.PAX_FORMAT,dereference=True) as tar:
            for d in dirs:tar.addfile(tar.gettarinfo(workspace/d['path'],arcname=d['path']))
            for row in files:
                p=workspace/row['path'];before=p.stat()
                with p.open('rb') as f:
                    stream=HashFile(f);tar.addfile(tar.gettarinfo(p,arcname=row['path']),stream)
                after=p.stat();need((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)==(row['bytes'],row['mtime_ns']) and stream.n==row['bytes'],'Source changed during packing')
                row['sha256']=stream.h.hexdigest()
    after_files,_=snapshot(source,workspace);need(after_files==[{k:r[k] for k in ['path','bytes','mtime_ns']} for r in files],'Tree changed during packing')
    m=dict(format='RETIRED_RESEARCH_TREE_V1',group_id=group['id'],source_path=group['path'],files=files,directories=dirs,
        archive_sha256=sha(target),archive_bytes=target.stat().st_size,scientific_history_preserved=True,
        note='Every regular file stored as full bytes; no reliance on source hardlinks. Directories/timestamps retained. No original source deletion.')
    save(manifest_path,m);return m,target,manifest_path
class RemoteBytes:
    def __init__(self,chunks):self.chunks=iter(chunks);self.buf=b'';self.h=hashlib.sha256();self.n=0
    def read(self,n):
        need(n>=0,'Unbounded remote read')
        out=bytearray()
        while len(out)<n:
            if not self.buf:
                try:self.buf=next(self.chunks)
                except StopIteration:break
                self.h.update(self.buf);self.n+=len(self.buf)
            take=min(n-len(out),len(self.buf));out+=self.buf[:take];self.buf=self.buf[take:]
        return bytes(out)
def verify_tar(stream,manifest,restore_root=None):
    expected={r['path']:r for r in manifest['files']};dirs={r['path'] for r in manifest['directories']};seen=set()
    with tarfile.open(fileobj=stream,mode='r|gz') as tar:
        for member in tar:
            name=member.name;parts=PurePosixPath(name).parts
            need(not PurePosixPath(name).is_absolute() and '..' not in parts,'Unsafe remote archive member')
            if member.isdir():
                need(name in dirs,'Unexpected directory')
                if restore_root:(restore_root/name).mkdir(parents=True,exist_ok=True)
                continue
            need(member.isfile() and name in expected and name not in seen,'Unexpected/nonregular/duplicate member')
            row=expected[name];need(member.size==row['bytes'],'Remote member size')
            h=hashlib.sha256();n=0;dest=None
            if restore_root:
                p=restore_root/name;need(p.resolve().is_relative_to(restore_root.resolve()),'Restore escape');p.parent.mkdir(parents=True,exist_ok=True);dest=p.open('xb')
            try:
                with tar.extractfile(member) as f:
                    for b in iter(lambda:f.read(4*2**20),b''):
                        h.update(b);n+=len(b)
                        if dest:dest.write(b)
            finally:
                if dest:dest.close()
            need(n==row['bytes'] and h.hexdigest()==row['sha256'],'Remote member SHA256')
            if restore_root:os.utime(restore_root/name,ns=(row['mtime_ns'],row['mtime_ns']))
            seen.add(name)
    need(seen==set(expected),'Missing remote members')
    while stream.read(4*2**20):pass
    need(stream.h.hexdigest()==manifest['archive_sha256'] and stream.n==manifest['archive_bytes'],'Remote full archive SHA256')
    return len(seen)
def remote_bytes(repo,path,revision):
    import requests
    from huggingface_hub import hf_hub_url
    from huggingface_hub.utils import build_hf_headers
    response=requests.get(hf_hub_url(repo,path,repo_type='dataset',revision=revision),headers=build_hf_headers(token=True),stream=True,timeout=(30,120))
    response.raise_for_status();return response
def check_remote(plan,out,group,restore_to=None):
    from huggingface_hub import HfApi
    rec=read(out/'HF_RECEIPTS'/(group['id']+'.json'));mfile=out/'HF_MANIFESTS'/(group['id']+'.json');m=read(mfile)
    need(rec['status']=='PASS_FULL_REMOTE_ARCHIVE_AND_EVERY_MEMBER_SHA256' and rec['source_path']==group['path'] and sha(mfile)==rec['manifest_sha256'],'Invalid proof receipt')
    need(m['group_id']==group['id'] and m['source_path']==group['path'] and all(r['path'].startswith(group['path']+'/') for r in m['files']),'Archive manifest scope mismatch')
    api=HfApi();info=api.repo_info(plan['private_HF_repository'],repo_type='dataset',revision=rec['revision'],files_metadata=True);need(info.private is True,'Backup repository must remain private')
    entry=next(x for x in info.siblings if x.rfilename==rec['archive_remote_path'])
    need(entry.size==m['archive_bytes'] and entry.lfs and entry.lfs.sha256==m['archive_sha256'],'Pinned remote archive identity changed')
    with remote_bytes(plan['private_HF_repository'],rec['manifest_remote_path'],rec['revision']) as response:
        h=hashlib.sha256()
        for b in response.iter_content(4*2**20):h.update(b)
    need(h.hexdigest()==rec['manifest_sha256'],'Pinned remote manifest changed')
    if restore_to:
        need(not restore_to.exists(),'Restore destination must be NEW');restore_to.mkdir(parents=True)
        with remote_bytes(plan['private_HF_repository'],rec['archive_remote_path'],rec['revision']) as response:
            verify_tar(RemoteBytes(response.iter_content(4*2**20)),m,restore_to)
        for d in sorted(m['directories'],key=lambda r:-len(r['path'])):os.utime(restore_to/d['path'],ns=(d['mtime_ns'],d['mtime_ns']))
        save(restore_to/'RESTORE_VERIFICATION.json',dict(status='PASS_RESTORED_EVERY_MEMBER_SHA256',group_id=group['id'],revision=rec['revision'],files=len(m['files'])))
    return rec
def publish(plan,out,group):
    from huggingface_hub import HfApi
    api=HfApi();need(api.repo_info(plan['private_HF_repository'],repo_type='dataset').private is True,'Existing PRIVATE repository required; never create or change visibility')
    receipt=out/'HF_RECEIPTS'/(group['id']+'.json')
    if receipt.exists():
        check_remote(plan,out,group);print('Already remotely verified: '+group['id'],flush=True);return
    print('Packing '+group['path'],flush=True);m,archive,mfile=pack_group(group,out)
    prefix=plan['HF_new_namespace']+'/'+group['id']+'/'+m['archive_sha256']
    archive_remote=prefix+'/bundle.tar.gz';manifest_remote=prefix+'/source_manifest.json'
    # Content-addressed new paths; no existing dataset archive/config/target is overwritten.
    api.upload_file(repo_id=plan['private_HF_repository'],repo_type='dataset',path_in_repo=archive_remote,path_or_fileobj=archive,commit_message='Archive retired S1000 development evidence '+group['id'])
    commit=api.upload_file(repo_id=plan['private_HF_repository'],repo_type='dataset',path_in_repo=manifest_remote,path_or_fileobj=mfile,commit_message='Record original file hashes for '+group['id'])
    revision=commit.oid
    with remote_bytes(plan['private_HF_repository'],archive_remote,revision) as response:
        count=verify_tar(RemoteBytes(response.iter_content(4*2**20)),m)
    with remote_bytes(plan['private_HF_repository'],manifest_remote,revision) as response:
        h=hashlib.sha256()
        for b in response.iter_content(4*2**20):h.update(b)
    need(h.hexdigest()==sha(mfile),'Remote manifest byte SHA')
    need(api.repo_info(plan['private_HF_repository'],repo_type='dataset',revision=revision).private is True,'Private archive check')
    save(receipt,dict(status='PASS_FULL_REMOTE_ARCHIVE_AND_EVERY_MEMBER_SHA256',group_id=group['id'],source_path=group['path'],repo_id=plan['private_HF_repository'],revision=revision,
        archive_remote_path=archive_remote,manifest_remote_path=manifest_remote,archive_sha256=m['archive_sha256'],archive_bytes=m['archive_bytes'],manifest_sha256=sha(mfile),
        verified_members=count,local_archive_relative=archive.relative_to(R).as_posix(),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),original_source_deleted=False))
    print('REMOTE FULL SHA/MEMBER PASS '+group['id'],flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha',required=True);p.add_argument('--publish',action='store_true');p.add_argument('--check-remote',action='store_true');p.add_argument('--group');p.add_argument('--restore-to',type=Path);a=p.parse_args()
    need(sha(a.plan)==a.plan_sha,'Plan SHA mismatch');plan=read(a.plan);need(Path(plan['root']).resolve()==R,'Wrong root');out=a.plan.parent
    groups=[g for g in plan['retired_archive_groups'] if not a.group or g['id']==a.group];need(groups,'Unknown group')
    if a.restore_to:need(len(groups)==1 and a.check_remote,'Restore requires one explicit archived group')
    for g in groups:
        safe(g['path'])
        need(not any(g['path']==h or g['path'].startswith(h+'/') or h.startswith(g['path']+'/') for h in plan['protected_prefixes']),'Protected source group')
        if a.check_remote:check_remote(plan,out,g,a.restore_to)
        elif a.publish:publish(plan,out,g)
        else:print('DRY RUN ONLY: '+g['path']+' -> existing private HF dataset; no data packed/uploaded/deleted')
if __name__=='__main__':
    try:main()
    except Exception as e:
        # Avoid emitting credentials, signed redirect URLs or HTTP Authorization headers.
        print('STOP: '+type(e).__name__+'; original source trees retained. Check plan/receipts and HF permissions/quota. No automatic retry or visibility/payment change.',file=sys.stderr);raise SystemExit(1)
