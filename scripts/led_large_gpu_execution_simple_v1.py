"""Provider-allocation + idle/continuity checks. No host/container PID comparison.

This is an operational check, not kernel-level proof of ownership/exclusivity.
Only one cooperating experiment worker may use the common advisory lock.
"""
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import urllib.request


def require(ok,message):
    if not ok:raise ValueError(message)


def validate_allocation(item,config):
    require(isinstance(item,dict),'Provider allocation response is not an object')
    require(str(item.get('id'))==str(config['instance_id']),'Provider instance differs')
    require(item.get('actual_status')=='running' and item.get('intended_status')=='running','Provider instance not running')
    require(item.get('num_gpus')==1 and item.get('gpu_name')=='RTX 3090','Provider single RTX3090 allocation differs')
    require(item.get('gpu_ram')==24576,'Provider full GPU VRAM allocation differs')
    fraction=item.get('gpu_frac')
    require(isinstance(fraction,(int,float)) and 0<fraction<=1,'Invalid machine resource fraction')
    # gpu_frac is the instance fraction of the host's aggregate resources, not a VRAM-sharing claim.
    return {k:item[k] for k in ('id','actual_status','intended_status','num_gpus','gpu_name','gpu_ram','gpu_frac','machine_id') if k in item}


def provider_allocation(config):
    key=os.environ.get('CONTAINER_API_KEY')
    require(bool(key),'Provider allocation credentials unavailable; STOP')
    url='https://console.vast.ai/api/v0/instances/'+str(config['instance_id'])+'/'
    request=urllib.request.Request(url,headers={'Authorization':'Bearer '+key},method='GET')
    with urllib.request.urlopen(request,timeout=20) as response:
        require(response.status==200,'Provider allocation query failed')
        data=json.load(response)
    require(isinstance(data,dict),'Unexpected provider response')
    item=data.get('instances')
    if isinstance(item,list):
        found=[x for x in item if str(x.get('id'))==str(config['instance_id'])]
        require(len(found)==1,'Ambiguous provider instance')
        item=found[0]
    return dict(source=url,authenticated_HTTP=True,allocation=validate_allocation(item,config),
        provider_policy='https://docs.vast.ai/guides/instances/docker-environment',
        assurance='Provider running-instance GPU allocation contract; not host-kernel PID authentication')


def parse_inventory(gpu_text,process_text,config):
    rows=list(csv.reader(io.StringIO(gpu_text.strip()),skipinitialspace=True))
    require(len(rows)==1 and len(rows[0])==5,'Single visible GPU required; malformed GPU query')
    uuid,name,total,used,driver=[x.strip() for x in rows[0]]
    require(uuid==config['GPU_uuid'] and name=='NVIDIA GeForce RTX 3090','Assigned GPU UUID/name differs')
    total,used=int(total),int(used)
    require(total==24576 and 0<=used<=total,'Full VRAM/device measurement differs')
    processes=[]
    for row in csv.reader(io.StringIO(process_text.strip()),skipinitialspace=True):
        require(len(row)==2,'Malformed compute-process query')
        gpu,pid=[s.strip() for s in row]
        require(gpu==uuid and pid.isdecimal() and int(pid)>0,'Unknown GPU/process output')
        processes.append(int(pid))
    require(len(processes)==len(set(processes)),'Duplicate/ambiguous compute process rows')
    return dict(uuid=uuid,name=name,memory_total_MiB=total,memory_used_MiB=used,driver=driver,compute_pids=processes)


def validate_processes(observed,expected=None,before_model=False):
    if before_model:
        require(not observed,'Existing GPU compute workload; STOP before model load')
        return None
    require(len(observed)==1,'Expected one compute context owner; conflicting or unavailable GPU process inventory')
    current=frozenset(observed)
    require(expected is None or current==expected,'GPU compute-process set changed; STOP without retry')
    return current


def acquire_run_lock(path):
    import fcntl
    path=Path(path)
    fd=os.open(path,os.O_RDWR|os.O_CREAT|os.O_CLOEXEC,0o600)
    try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BaseException:
        os.close(fd);raise ValueError('Another cooperating LED experiment holds the common GPU run lock')
    return fd


class SimpleSafety:
    def __init__(self,config,record):
        self.config=config;self.record=record;self.expected=None;self.counter=0

    def snapshot(self):
        uuid=self.config['GPU_uuid'];outputs=[];commands=[
            ['nvidia-smi','--query-gpu=uuid,name,memory.total,memory.used,driver_version','--format=csv,noheader,nounits'],
            ['nvidia-smi','-i',uuid,'--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits']]
        for command in commands:
            p=subprocess.run(command,capture_output=True,text=True,timeout=15)
            outputs.append(dict(command=command,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr))
        self.record('SAFETY_RAW_'+str(self.counter)+'.json',dict(queries=outputs))
        self.counter+=1
        require(all(x['returncode']==0 for x in outputs),'NVIDIA management query failed; STOP')
        return parse_inventory(outputs[0]['stdout'],outputs[1]['stdout'],self.config)

    def before_model(self):
        proof=provider_allocation(self.config)
        self.record('PROVIDER_ALLOCATION.json',proof)
        snap=self.snapshot();validate_processes(snap['compute_pids'],before_model=True)
        self.record('SAFETY_POLICY.json',dict(raw_host_container_PID_equality_removed=True,custom_PID_mapping=False,
            provider_allocation_verified=True,common_run_lock_required=True,preexisting_compute_workloads_rejected=True,
            post_model_policy='one stable observed compute PID; not authenticated mapping',
            limitation='Provider assurance + cooperative lock + sampled workload/resource checks; not continuous exclusivity or adversarial host protection'))
        return {k:snap[k] for k in ('uuid','name','memory_total_MiB','driver')}

    def check(self):
        snap=self.snapshot();self.expected=validate_processes(snap['compute_pids'],self.expected)
        return {k:snap[k] for k in ('uuid','name','memory_total_MiB','driver')}
