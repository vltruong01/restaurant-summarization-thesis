"""Small local coordinator primitives. No automatic evaluation or retries."""
from pathlib import Path
import os,time,subprocess,signal,json,errno
from led_large_boundary_dual_contract_v2 import require,write,read,ARMS

class LockBusy(ValueError):pass
class Locks:
    def __init__(self,paths):self.paths=sorted({str(Path(p).resolve()) for p in paths});self.fds=[]
    def __enter__(self):
        try:
            for name in self.paths:
                p=Path(name);p.parent.mkdir(parents=True,exist_ok=True)
                require(not p.is_symlink(),'Unsafe symlink lock')
                fd=os.open(p,os.O_RDWR|os.O_CREAT,0o600)
                try:
                    if os.name=='posix':
                        import fcntl;fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    else:
                        import msvcrt
                        if os.fstat(fd).st_size==0:os.write(fd,b'0')
                        os.lseek(fd,0,0);msvcrt.locking(fd,msvcrt.LK_NBLCK,1)
                except OSError as exc:
                    os.close(fd)
                    if exc.errno in (errno.EACCES,errno.EAGAIN,errno.EDEADLK):raise LockBusy('Cooperative lock unavailable') from exc
                    raise
                except BaseException:os.close(fd);raise
                self.fds.append(fd)
            return self
        except BaseException:self.__exit__(None,None,None);raise
    def __exit__(self,*args):
        for fd in reversed(self.fds):os.close(fd)
        self.fds=[]

def pulse(path,value):
    path=Path(path);temp=path.with_suffix('.pulse.tmp')
    with temp.open('w',encoding='utf-8') as f:json.dump(value,f);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)
def lease_check(suite,run_id,now=None):
    suite=Path(suite);require(not (suite/'PAIR_FAILURE.json').exists(),'Pair cancelled')
    x=read(suite/'LEASE.json');now=time.time() if now is None else now
    require(x['run_id']==run_id and 0<=now-x['time']<=60,'Missing/stale coordinator lease')
    require(now<x['deadline'],'Pair/phase budget expired')
    return x
def barrier(suite,name,arm,run_id,check,timeout=7200):
    folder=Path(suite)/'barriers'/name;write(folder/(arm+'.json'),dict(arm=arm,run_id=run_id))
    start=time.monotonic()
    while True:
        check();lease_check(suite,run_id)
        ready=[folder/(x+'.json') for x in ARMS]
        if all(p.is_file() for p in ready):
            require(all(read(p)['run_id']==run_id for p in ready),'Foreign barrier receipt');return
        require(time.monotonic()-start<timeout,'Barrier timeout');time.sleep(.1)

def stop_owned(process):
    if process.poll() is not None:return
    try:
        if os.name=='posix':os.killpg(process.pid,signal.SIGTERM)
        else:process.terminate()
    except ProcessLookupError:return
    try:process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            if os.name=='posix':os.killpg(process.pid,signal.SIGKILL)
            else:process.kill()
        except ProcessLookupError:return
        process.wait(timeout=5)

def run_workers(suite,commands,deadline,run_id,check_host_fn=lambda:None,poll=.2):
    """Own process groups only; any missing/failed worker invalidates the pair."""
    suite=Path(suite);processes={};logs={};start=time.monotonic()
    try:
        require(commands and set(commands)<=set(ARMS),'Invalid worker set')
        pulse(suite/'LEASE.json',dict(run_id=run_id,time=time.time(),deadline=deadline,coordinator_pid=os.getpid()))
        for arm,spec in commands.items():
            log=(suite/(run_id+'_'+arm+'.log')).open('x',encoding='utf-8');logs[arm]=log
            kwargs=dict(cwd=spec['cwd'],env=spec['env'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            if os.name=='posix':kwargs['start_new_session']=True
            else:kwargs['creationflags']=subprocess.CREATE_NEW_PROCESS_GROUP
            processes[arm]=subprocess.Popen(spec['command'],**kwargs)
        last=0
        while True:
            now=time.time();require(now<deadline,'Coordinator wall budget reached')
            if now-last>=1:
                pulse(suite/'LEASE.json',dict(run_id=run_id,time=now,deadline=deadline,coordinator_pid=os.getpid()))
                check_host_fn();last=now
            require(not (suite/'PAIR_FAILURE.json').exists(),'Worker reported pair failure')
            codes={arm:p.poll() for arm,p in processes.items()}
            require(not any(v is not None and v!=0 for v in codes.values()),'Worker failure: '+str(codes))
            if all(v==0 for v in codes.values()):return dict(returncodes=codes,seconds=time.monotonic()-start)
            time.sleep(poll)
    except BaseException as exc:
        try:
            if not (suite/'PAIR_FAILURE.json').exists():write(suite/'PAIR_FAILURE.json',dict(status='PAIR_INCOMPLETE_STOP_NO_RETRY',error=str(exc),run_id=run_id))
        finally:
            # A full disk or failed receipt must never prevent killing sibling jobs.
            for p in processes.values():
                try:stop_owned(p)
                except BaseException:pass  # Original failure retained; never report success.
        raise
    finally:
        for log in logs.values():log.close()

from contextlib import contextmanager
@contextmanager
def checkpoint_io(path,check):
    while True:
        check()
        lock=Locks([path])
        try:lock.__enter__();break
        except LockBusy:time.sleep(.1)
    try:yield
    finally:lock.__exit__(None,None,None)

def parent_death_guard(expected_parent,register=None,get_parent=os.getppid):
    """Linux kernel kills this OWN worker if coordinator exits, even with blocked GIL."""
    require(get_parent()==expected_parent,'Worker already orphaned or wrong coordinator')
    if register is None:
        require(os.name=='posix','Production worker requires Linux parent-death protection')
        import ctypes
        libc=ctypes.CDLL(None,use_errno=True)
        def register(sig):
            require(libc.prctl(1,sig,0,0,0)==0,'PR_SET_PDEATHSIG unavailable')
    register(getattr(signal,'SIGKILL',9))  # 9 is Linux SIGKILL; permits CPU mock tests on Windows.
    require(get_parent()==expected_parent,'Coordinator exited during parent-death registration')
