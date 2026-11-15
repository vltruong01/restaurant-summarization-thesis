"""Per-UUID provider and workload checks. No CUDA API, SSH or lifecycle code."""
from pathlib import Path
import os,subprocess,json,urllib.request,time
from led_large_gpu_execution_simple_v1 import parse_inventory,validate_processes
from led_large_boundary_dual_contract_v2 import require,uuid,write

def allocation(data,instance):
    obj=data.get('instances',data.get('instance',data)) if isinstance(data,dict) else data
    if isinstance(obj,list):
        matches=[x for x in obj if str(x.get('id'))==str(instance)]
        require(len(matches)==1,'Ambiguous provider instance');obj=matches[0]
    require(isinstance(obj,dict) and str(obj.get('id'))==str(instance),'Provider instance differs')
    require(obj.get('actual_status')=='running' and obj.get('intended_status')=='running','Instance not running')
    require(obj.get('num_gpus')==2 and obj.get('gpu_name')=='RTX 3090','Two allocated RTX3090 cards required')
    return {k:obj[k] for k in ('id','actual_status','intended_status','num_gpus','gpu_name','gpu_ram','gpu_frac','machine_id') if k in obj}
def provider(instance):
    require(os.environ.get('CONTAINER_ID')==str(instance),'Wrong container instance')
    key=os.environ.get('CONTAINER_API_KEY');require(bool(key),'Provider allocation authentication missing')
    url='https://console.vast.ai/api/v0/instances/'+str(instance)+'/'
    req=urllib.request.Request(url,headers={'Authorization':'Bearer '+key},method='GET')
    with urllib.request.urlopen(req,timeout=20) as r:
        require(r.status==200,'Provider verification failed');obj=json.load(r)
    return dict(source=url,allocation=allocation(obj,instance),authenticated=True,
        limitation='Provider assurance, cooperative locks and sampled continuity; not authenticated host/container PID mapping')

class DeviceSafety:
    def __init__(self,device,record,query=subprocess.run):
        self.device=uuid(device);self.record=record;self.query=query;self.expected=None;self.loaded=False;self.counter=0
    def check(self,loaded=None):
        if loaded is not None:self.loaded=loaded
        commands=[['nvidia-smi','-i',self.device,'--query-gpu=uuid,name,memory.total,memory.used,driver_version','--format=csv,noheader,nounits'],
                  ['nvidia-smi','-i',self.device,'--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits']]
        raw=[]
        for cmd in commands:
            p=self.query(cmd,capture_output=True,text=True,timeout=15)
            raw.append(dict(command=cmd,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr))
        self.record('DEVICE_QUERY_'+str(self.counter)+'.json',dict(queries=raw));self.counter+=1
        require(all(x['returncode']==0 for x in raw),'Management query failed')
        snap=parse_inventory(raw[0]['stdout'],raw[1]['stdout'],dict(GPU_uuid=self.device))
        if self.loaded:self.expected=validate_processes(snap['compute_pids'],self.expected)
        else:validate_processes(snap['compute_pids'],before_model=True)
        return snap

def host_snapshot(root):
    import psutil
    vm=psutil.virtual_memory();total=vm.total;available=vm.available
    base=Path('/sys/fs/cgroup')
    if (base/'memory.max').is_file():
        limit=(base/'memory.max').read_text().strip()
        if limit!='max':
            maximum=int(limit);current=int((base/'memory.current').read_text())
            total=min(total,maximum);available=min(available,max(0,maximum-current))
    elif (base/'memory/memory.limit_in_bytes').is_file():
        maximum=int((base/'memory/memory.limit_in_bytes').read_text());current=int((base/'memory/memory.usage_in_bytes').read_text())
        total=min(total,maximum);available=min(available,max(0,maximum-current))
    cores=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else (os.cpu_count() or 1)
    if (base/'cpu.max').is_file():
        quota,period=(base/'cpu.max').read_text().split()
        if quota!='max':cores=min(cores,int(quota)/int(period))
    elif (base/'cpu/cpu.cfs_quota_us').is_file():
        quota=int((base/'cpu/cpu.cfs_quota_us').read_text());period=int((base/'cpu/cpu.cfs_period_us').read_text())
        if quota>0:cores=min(cores,quota/period)
    import shutil
    return dict(effective_RAM_total=total,effective_RAM_available=available,effective_CPU_cores=cores,
        free_disk_bytes=shutil.disk_usage(root).free,UTC_seconds=time.time())
def check_host(snap,initial=False):
    require(snap['effective_CPU_cores']>=8,'Fewer than 8 effective CPU cores')
    require(snap['effective_RAM_total']>0 and snap['effective_RAM_available']/snap['effective_RAM_total']>=.15,'Host RAM headroom below 15%')
    require(snap['free_disk_bytes']>=(25 if initial else 1)*1024**3,'Insufficient free disk')
    if initial:require(snap['effective_RAM_available']>=16*1024**3,'At least 16GiB available host RAM required before launch')
    return snap
