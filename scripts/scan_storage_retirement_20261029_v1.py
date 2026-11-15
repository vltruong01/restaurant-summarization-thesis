"""Read-only current workspace sizes; no links followed, deletion or upload."""
from pathlib import Path
import os,json,ctypes,collections,datetime,shutil
R=Path(__file__).resolve().parents[1];O=R/'outputs/storage_cleanup/20261029_review_v1'
def main():
    O.mkdir(parents=True,exist_ok=False)
    api=ctypes.WinDLL('kernel32',use_last_error=True).GetCompressedFileSizeW
    api.argtypes=[ctypes.c_wchar_p,ctypes.POINTER(ctypes.c_uint32)];api.restype=ctypes.c_uint32
    def physical(p):
        high=ctypes.c_uint32();ctypes.set_last_error(0);low=api('\\\\?\\'+str(p),ctypes.byref(high))
        return None if low==0xffffffff and ctypes.get_last_error() else (high.value<<32)|low
    files=[];errors=[];links=[];stack=[R];sums=collections.defaultdict(lambda:[0,0,0]);physical_ids={}
    while stack:
        d=stack.pop()
        try:
            with os.scandir(d) as it:
                for e in it:
                    try:
                        p=Path(e.path);s=e.stat(follow_symlinks=False)
                        if e.is_symlink() or getattr(s,'st_file_attributes',0)&0x400:links.append(str(p));continue
                        if e.is_dir(follow_symlinks=False):
                            if p!=O:stack.append(p)
                        elif e.is_file(follow_symlinks=False):
                            rel=p.relative_to(R).as_posix();stored=physical(p)
                            if stored is None:errors.append(dict(path=rel,error='GetCompressedFileSizeW failed'));stored=s.st_size
                            row=dict(path=rel,bytes=s.st_size,storage_bytes=stored,mtime_ns=s.st_mtime_ns,dev=s.st_dev,inode=s.st_ino,nlink=s.st_nlink)
                            files.append(row)
                            for n in range(1,min(5,len(p.relative_to(R).parts))+1):
                                k='/'.join(p.relative_to(R).parts[:n]);sums[k][0]+=s.st_size;sums[k][1]+=stored;sums[k][2]+=1
                            physical_ids.setdefault((s.st_dev,s.st_ino),stored)
                    except OSError as x:errors.append(dict(path=e.path,error=str(x)))
        except OSError as x:errors.append(dict(path=str(d),error=str(x)))
    data=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),root=str(R),mode='READ_ONLY_NO_DELETE_NO_UPLOAD',
        note='Logical versus Windows GetCompressedFileSizeW stored bytes; directory totals overlap. Physical release may differ with filesystem links/compression/cluster allocation.',
        total_logical_bytes=sum(f['bytes'] for f in files),total_storage_bytes=sum(f['storage_bytes'] for f in files),unique_file_identity_storage_bytes=sum(physical_ids.values()),
        disk_free_bytes=shutil.disk_usage(R).free,files=files,groups=[dict(path=k,bytes=v[0],storage_bytes=v[1],files=v[2]) for k,v in sorted(sums.items(),key=lambda t:-t[1][1])],errors=errors,skipped_links=links)
    (O/'CURRENT_STORAGE_INVENTORY.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({k:data[k] for k in ['UTC','total_logical_bytes','total_storage_bytes','unique_file_identity_storage_bytes','disk_free_bytes']},indent=2))
    print('TOP DIRECTORIES depth<=3')
    for g in data['groups']:
        if g['path'].count('/')<=2 and g['files']>1 and g['storage_bytes']>150*2**20:print(f"{g['storage_bytes']/2**30:8.3f} stored GiB | {g['bytes']/2**30:8.3f} logical GiB | {g['path']}")
    print('errors',len(errors),'links skipped',len(links),'files',len(files))
if __name__=='__main__':main()
