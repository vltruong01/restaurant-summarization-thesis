"""Read-only NVIDIA availability/idle check before the authorized model pipeline."""
import argparse
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import subprocess

def inspect_host(run=subprocess.run):
    result={'recorded_at_utc':datetime.now(timezone.utc).isoformat(), 'status':'FAIL',
            'model_operations':0,'reset_or_signals':False,'commands':[]}
    def query(args):
        p=run(args,capture_output=True,text=True,timeout=20,check=False)
        result['commands'].append(dict(argv=args,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr))
        if p.returncode:
            raise ValueError('GPU query failed; restore host availability before model execution')
        return p.stdout
    try:
        query(['nvidia-smi','-L'])
        raw=query(['nvidia-smi','--query-gpu=name,uuid,pci.bus_id,driver_version,memory.total',
                   '--format=csv,noheader,nounits'])
        rows=list(csv.reader(io.StringIO(raw)))
        if len(rows)!=1 or len(rows[0])!=5:
            raise ValueError('Exactly one NVIDIA GPU must be visible to this instance')
        name,uuid,bus,driver,total=[x.strip() for x in rows[0]]
        if name!='NVIDIA GeForce RTX 3090' or float(total)<23000:
            raise ValueError('Required RTX3090/24GB differs')
        result['GPU']=dict(name=name,uuid=uuid,bus=bus,driver=driver,memory_MiB=float(total))
        apps=query(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'])
        active=[row for row in csv.reader(io.StringIO(apps)) if row and row[0].strip()==uuid]
        if active:
            raise ValueError('Allocated GPU still has compute processes; preserve evidence, do not kill or launch')
        result['status']='PASS_READONLY_GPU_VISIBLE_IDLE_NOT_A_MODEL_FEASIBILITY_PASS'
    except Exception as error:
        result['error']=type(error).__name__+': '+str(error)
    return result

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--report',type=Path,required=True)
    a=ap.parse_args();report=inspect_host()
    with a.report.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(report,indent=2),flush=True)
    if report['status'].startswith('FAIL'):raise SystemExit(1)

if __name__=='__main__':main()
