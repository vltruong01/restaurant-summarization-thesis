"""Incremental byte-verified checkpoint ZIPs; no Drive image reads, deletion or keepalive tricks."""
from pathlib import Path,PurePosixPath
import hashlib,json,os,shutil,signal,subprocess,time,zipfile

RELATIVE='outputs/stage_f/production_caption_construction_d033_colab_v1'


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def state_files(root):
    out=Path(root)/RELATIVE
    for p in out.rglob('*'):
        if not p.is_file() or p.suffix in ['.tmp','.lock','.log']:continue
        if 'scores' in p.relative_to(out).parts and p.parent.name.startswith('chunk_') and not (p.parent/'chunk_receipt.json').exists():continue
        # Only this newly created Colab output namespace is backed up.
        yield p


def restore(root,bank):
    root=Path(root).resolve();bank=Path(bank);known={};archives=sorted(bank.glob('checkpoints_*.zip'))
    for archive in archives:
        with zipfile.ZipFile(archive) as z:
            info=json.loads(z.read('checkpoint_manifest.json'))
            for name,expected in info['files'].items():
                relative=PurePosixPath(name);assert not relative.is_absolute() and '..' not in relative.parts and name.startswith(RELATIVE+'/')
                target=root/name;assert target.resolve().is_relative_to(root/RELATIVE)
                data=z.read(name);assert hashlib.sha256(data).hexdigest()==expected
                if target.exists() and '/caption_checkpoints/' in name:assert digest(target)==expected,'Existing caption differs from saved checkpoint'
                target.parent.mkdir(parents=True,exist_ok=True);temp=target.with_suffix(target.suffix+'.restore.tmp');temp.write_bytes(data);temp.replace(target)
                known[name]=expected
    print(f'Restored {len(archives)} checkpoint bundles; {len(known)} tracked output files')
    return known


def backup(root,bank,known):
    root=Path(root).resolve();bank=Path(bank);bank.mkdir(parents=True,exist_ok=True);changed={};snapshots={}
    for p in state_files(root):
        # Generation updates mutable reports while this foreground cell copies checkpoints.
        # Hash and archive the SAME bytes; never reread a report after hashing it.
        name=p.relative_to(root).as_posix();data=p.read_bytes();h=hashlib.sha256(data).hexdigest()
        if known.get(name)!=h:changed[name]=h;snapshots[name]=data
    if not changed:return
    name='checkpoints_'+str(time.time_ns())+'.zip';local=Path('/content')/name if Path('/content').exists() else root/name
    with zipfile.ZipFile(local,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for relative,h in changed.items():z.writestr(relative,snapshots[relative])
        z.writestr('checkpoint_manifest.json',json.dumps({'namespace':RELATIVE,'files':changed},sort_keys=True))
    partial=bank/(name+'.part');shutil.copyfile(local,partial)
    assert digest(local)==digest(partial),'Drive backup byte verification failed'
    partial.replace(bank/name);known.update(changed)
    print(f'Saved and verified {len(changed)} output files to {name}',flush=True)


def run_cell(command,root,bank,known,interval=120):
    root=Path(root).resolve();log=root/'colab_foreground.log';env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='0',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',RAYON_NUM_THREADS='4')
    with log.open('w',encoding='utf-8') as stream:
        process=subprocess.Popen(command,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=(os.name!='nt'))
        last=time.monotonic();offset=0
        try:
            while process.poll() is None:
                time.sleep(5)
                with log.open(encoding='utf-8',errors='replace') as output:
                    output.seek(offset);text=output.read();offset=output.tell()
                    if text:print(text,end='',flush=True)
                if time.monotonic()-last>=interval:backup(root,bank,known);last=time.monotonic()
            with log.open(encoding='utf-8',errors='replace') as output:output.seek(offset);print(output.read(),end='')
            if process.returncode:raise RuntimeError(f'Cell subprocess failed exit{process.returncode}; inspect output, no automatic method changes')
        finally:
            if process.poll() is None:
                if os.name!='nt':os.killpg(process.pid,signal.SIGTERM)
                else:process.terminate()
                process.wait(timeout=30)
            backup(root,bank,known)
