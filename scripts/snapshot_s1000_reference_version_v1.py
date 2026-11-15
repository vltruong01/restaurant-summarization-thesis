"""Archive an explicitly supplied human-authored reference version. Never approves it."""
import argparse,datetime
from pathlib import Path
from s1000_final_eval_core_v1 import *
def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--history-directory',type=Path,required=True);p.add_argument('--human-id',required=True);p.add_argument('--output-version',type=Path,required=True);a=p.parse_args()
    r=read(a.reference);require(r.get('review_consensus','').strip() and r.get('visual_evidence','').strip(),'Human must supply both components')
    h=digest(reference_content(r));previous=r.get('history',[])
    event=dict(version=len(previous)+1,content_sha256=h,supplied_by_human_id=a.human_id,recorded_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        parent_reference_file_sha256=file_hash(a.reference),event='IMMUTABLE_VERSION_SNAPSHOT_NOT_APPROVAL')
    r['history']=previous+[event]
    # Text changes always invalidate any approval bound to an earlier content hash.
    if r.get('checker',{}).get('reviewed_content_sha256')!=h:r['status']='DRAFT'
    a.history_directory.mkdir(parents=True,exist_ok=True)
    write(a.history_directory/f"{r['business_id']}_v{event['version']:03d}_{h[:12]}.json",r)
    write(a.output_version,r);print(json.dumps(event,indent=2))
if __name__=='__main__':main()
