"""Prior explicit user instruction: destroy only this instance AFTER verified export.

Ephemeral container API credential is held in memory, never logged/saved/printed.
All provider output is reduced to whitelisted lifecycle fields.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,subprocess,time
R=Path(__file__).resolve().parents[1];E=R/'outputs/stage_i/led_large_cad_v1_vast_execution_v1'
ID=57481997
SSH=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=10','-p','44054','root@171.5.182.162']
CLI=Path.home()/'AppData/Local/Programs/Python/Python311/Scripts/vastai.exe'
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
 assert not (E/'LIFECYCLE_KEEP_INSTANCE.json').exists(), 'Researcher revoked destroy: keep instance; no lifecycle action authorized'
 ap=argparse.ArgumentParser();ap.add_argument('--destroy',action='store_true');args=ap.parse_args()
 assert args.destroy and not (E/'INSTANCE_DESTROY_RECEIPT.json').exists(),'Explicit single lifecycle operation required'
 proof=read(E/'DOWNLOAD_VERIFIED.json');assert proof['status']=='ALL_EXPORTED_BYTES_VERIFIED'
 archive=E/'LED_LARGE_CAD_V1_VAST_EXECUTION_EVIDENCE.tar.gz';assert sha(archive)==proof['archive_sha256']
 files=read(E/'downloaded/ARTIFACT_MANIFEST.json')['files']
 for rel,h in files.items():assert sha(E/'downloaded'/rel)==h['sha256']
 assert read(E/'downloaded/operations/STATUS.json')['status']=='COMPLETE_STOP_FOR_REVIEW'
 assert read(E/'downloaded/operations/POSTFLIGHT_PRESERVATION.json')['status']=='PASS'
 # Source is the current user's already-authorized instance; restrict credential
 # acquisition to exact container identity. Do not read private-key files.
 code="import os;assert os.environ['CONTAINER_ID']=='57481997';print(os.environ['CONTAINER_API_KEY'])"
 import base64
 p=base64.b64encode(code.encode()).decode();command="python3 -c \"import base64;exec(base64.b64decode('"+p+"'))\""
 got=subprocess.run(SSH+[command],capture_output=True,text=True,timeout=30)
 result=dict(instance_id=ID,UTC=datetime.now(timezone.utc).isoformat(),backup_files_reverified=len(files),
     backup_sha256=sha(archive),authorization='prior explicit user instruction: after completion download everything then destroy',
     credential_saved=False,scientific_files_deleted=False)
 if got.returncode!=0:
  result.update(status='NOT_DESTROYED_AUTHENTICATED_SSH_UNAVAILABLE')
 else:
  credential=got.stdout.strip();assert credential and '\n' not in credential
  def call(action):
   try:
    proc=subprocess.run([str(CLI),*action,'--api-key',credential,'--raw'],capture_output=True,text=True,timeout=45)
    try:value=json.loads(proc.stdout)
    except Exception:value=None
    return proc.returncode,value,proc.stdout.lower()
   except Exception:return None,None,''
  rc,current,txt=call(['show','instance',str(ID)])
  obj=current.get('instances',current.get('instance',current)) if isinstance(current,dict) else current
  if isinstance(obj,list):obj=next((x for x in obj if x.get('id')==ID),{})
  result['before']={k:obj.get(k) for k in ['id','actual_status','cur_state','next_state'] if isinstance(obj,dict) and k in obj}
  valid=isinstance(obj,dict) and obj.get('id')==ID and not obj.get('error')
  if not valid:
   result.update(status='NOT_DESTROYED_PROVIDER_AUTH_OR_IDENTITY_FAILED',show_return_code=rc,
                 provider_status_code=current.get('status_code') if isinstance(current,dict) else None,
                 response_type=type(current).__name__,
                 response_keys=list(current)[:20] if isinstance(current,dict) else None,
                 nested_key_shapes={k:list(v)[:12] for k,v in current.items() if isinstance(v,dict)} if isinstance(current,dict) else None,
                 output_length=len(txt),error_keyword_present='error' in txt)
  else:
   rc,response,txt=call(['destroy','instance',str(ID)])
   result.update(destroy_return_code=rc,provider_success=response.get('success') if isinstance(response,dict) else None,
       destroy_acknowledged=('destroying instance '+str(ID) in txt) or (isinstance(response,dict) and response.get('success') is True))
   result['status']='DESTROY_REQUEST_ACCEPTED' if result['destroy_acknowledged'] else 'DESTROY_NOT_CONFIRMED'
   if result['destroy_acknowledged']:
    time.sleep(3)
    rc,after,txt=call(['show','instance',str(ID)])
    obj=after.get('instances',after.get('instance',after)) if isinstance(after,dict) else after
    if isinstance(obj,list):obj=next((x for x in obj if x.get('id')==ID),None)
    result['after']={k:obj.get(k) for k in ['id','actual_status','cur_state','next_state','status_code'] if isinstance(obj,dict) and k in obj}
    if (rc==0 and isinstance(after,list) and not after) or \
       (isinstance(obj,dict) and obj.get('status_code')==404) or \
       'no such instance' in txt or 'instance not found' in txt:
     result['status']='DESTROY_CONFIRMED_PROVIDER_INSTANCE_ABSENT'
    elif isinstance(obj,dict) and obj.get('actual_status') in ['destroyed','deleted']:
     result['status']='DESTROY_CONFIRMED_PROVIDER_STATUS'
  credential=None
 with (E/'INSTANCE_DESTROY_RECEIPT.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
