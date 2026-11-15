"""Shared timing/identity/VRAM helpers; no model configuration changes."""
import datetime as dt,hashlib,json,os,platform,subprocess,threading,uuid
from pathlib import Path
GPU='GPU-2df66163-ec71-02c3-6ab0-dc432c0cb759'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def put(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def tsha(x):return hashlib.sha256(x.encode()).hexdigest()
def canon(x):return json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
def digest(x):return hashlib.sha256(canon(x)).hexdigest()
def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def guard(torch):
    assert os.environ.get('CONTAINER_ID')=='59313861' and platform.node()=='239abaa00671'
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    p=torch.cuda.get_device_properties(0);assert str(uuid.UUID(str(p.uuid).removeprefix('GPU-')))==str(uuid.UUID(GPU.removeprefix('GPU-')))
    assert 'RTX 3090' in p.name
    return p
def mem(torch):
    f,t=torch.cuda.mem_get_info();return dict(allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),device_free_bytes=f,device_total_bytes=t)
class Sampler:
    def __init__(self):self.samples=[];self.errors=[];self.stop=threading.Event();self.thread=threading.Thread(target=self.work,daemon=True)
    def work(self):
        while not self.stop.is_set():
            try:
                r=subprocess.run(['nvidia-smi','-i',GPU,'--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3,check=True)
                m,u=map(float,r.stdout.strip().split(','));self.samples.append(dict(at_utc=now(),device_used_MiB=m,utilization_pct=u))
            except Exception as e:self.errors.append(str(e))
            self.stop.wait(.25)
    def start(self):self.thread.start()
    def close(self,dest):
        self.stop.set();self.thread.join(timeout=4);put(dest,dict(samples=self.samples,errors=self.errors,peak_device_used_MiB=max((r['device_used_MiB'] for r in self.samples),default=None),interval_target_seconds=.25,whole_device_includes_context=True,sampling_may_miss_transients=True))
