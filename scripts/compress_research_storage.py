"""Transparent local NTFS compression; preserve bytes and current paths."""
from pathlib import Path
import argparse, ctypes, hashlib, json, shutil, subprocess, time
from ctypes import wintypes

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/storage_cleanup'
PATTERNS={'.json','.jsonl','.csv','.sqlite','.yaml','.md','.html','.png','.pdf','.txt'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def allocated(path):
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    call=kernel.GetCompressedFileSizeW;call.argtypes=[wintypes.LPCWSTR,ctypes.POINTER(wintypes.DWORD)];call.restype=wintypes.DWORD
    high=wintypes.DWORD();ctypes.set_last_error(0);low=call(str(path.resolve()),ctypes.byref(high))
    if low==0xffffffff and ctypes.get_last_error():raise ctypes.WinError(ctypes.get_last_error())
    return (high.value<<32)|low


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--group',choices=['outputs','data'],required=True);args=parser.parse_args();started=time.perf_counter()
    OUT.mkdir(exist_ok=True)
    if args.group=='outputs':
        directory=ROOT/'outputs/stage_c';files=sorted(p for p in directory.rglob('*') if p.is_file())
        commands=[['compact.exe','/C','/S:'+str(directory),'/I','/Q']]
    else:
        audit=json.loads((OUT/'pre_deletion_audit.json').read_text(encoding='utf-8'));assert audit['status']=='PASS'
        assert all(not (ROOT/name).exists() for name in audit['delete_targets']), 'Delete audited redundant copies before compressing data.'
        raw=sorted((ROOT/'data').glob('yelp_academic_dataset_*.json'))+[ROOT/'data/yelp_photos/photos.json']
        processed=ROOT/'data/processed'
        files=raw+sorted(p for p in processed.rglob('*') if p.is_file() and p.suffix in {'.json','.jsonl','.csv','.sqlite'})
        commands=[['compact.exe','/C','/I','/Q',*map(str,raw)],['compact.exe','/C','/S:'+str(processed),'/I','/Q','*.json','*.jsonl','*.csv','*.sqlite']]
    assert files and all(p.resolve().is_relative_to(ROOT) for p in files)
    free_before=shutil.disk_usage(ROOT).free
    before={p.relative_to(ROOT).as_posix():{'logical_bytes':p.stat().st_size,'storage_bytes_winapi':allocated(p),'mtime_ns':p.stat().st_mtime_ns} for p in files}
    # Raw data have no changing analysis outputs; retain content digests before compression.
    content_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in files if args.group=='data'}
    (OUT/f'compression_{args.group}_before.json').write_text(json.dumps({'files':before,'content_sha256':content_hashes,'disk_free_bytes':free_before},indent=2)+'\n',encoding='utf-8')
    print(f'Compressing {args.group}: {len(files)} files, {sum(v["logical_bytes"] for v in before.values())/1e9:.3f} GB logical.',flush=True)
    results=[]
    for index,command in enumerate(commands):
        result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,errors='replace')
        log=OUT/f'compact_{args.group}_{index}.log';log.write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
        results.append({'command':command,'exit_code':result.returncode,'log':log.relative_to(ROOT).as_posix()})
        print(f'Compact {args.group} command{index}: exit {result.returncode}.',flush=True)
        assert result.returncode==0,log
    after={};unchanged=True
    for p in files:
        key=p.relative_to(ROOT).as_posix();state=before[key]
        assert p.stat().st_size==state['logical_bytes'] and p.stat().st_mtime_ns==state['mtime_ns'],key
        if key in content_hashes:assert sha(p)==content_hashes[key],key
        after[key]={'logical_bytes':p.stat().st_size,'storage_bytes_winapi':allocated(p),'compressed_attribute':bool(p.stat().st_file_attributes&0x800)}
    saved=sum(before[k]['storage_bytes_winapi']-after[k]['storage_bytes_winapi'] for k in before)
    report={'status':'PASS','group':args.group,'files':len(files),'logical_bytes':sum(v['logical_bytes'] for v in before.values()),'storage_bytes_before':sum(v['storage_bytes_winapi'] for v in before.values()),'storage_bytes_after':sum(v['storage_bytes_winapi'] for v in after.values()),'storage_bytes_saved':saved,'file_sizes_mtimes_unchanged':unchanged,'data_content_sha256_unchanged_files':len(content_hashes),'accepted_output_hashes_verified_separately':args.group=='outputs','disk_free_before':free_before,'disk_free_after':shutil.disk_usage(ROOT).free,'commands':results,'elapsed_seconds':round(time.perf_counter()-started,3)}
    (OUT/f'compression_{args.group}_after.json').write_text(json.dumps({'files':after,'report':report},indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
