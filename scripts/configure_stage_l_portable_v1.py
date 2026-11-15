"""Create a NEW source-only participant workspace from the review package."""
import argparse
from pathlib import Path
import shutil
from stage_l_reference_app_v1 import Store,read,new_json
def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True,help='extracted WORKSPACE directory');p.add_argument('--photos',type=Path,required=True);p.add_argument('--destination',type=Path,required=True);a=p.parse_args()
    if a.destination.exists():raise ValueError('Destination must be new')
    c=read(a.bundle/'CONFIG.json');c['project_root']=None;c['portable_photo_root']=str(a.photos.resolve())
    a.destination.mkdir(parents=True)
    shutil.copytree(a.bundle/'SOURCE_PACKAGE',a.destination/'SOURCE_PACKAGE')
    for name in ['HUMAN_REFERENCE_ANNOTATION_GUIDELINES.md','REFERENCE_AUTHOR_CHECKER_WORKFLOW.md','EVIDENCE_LINKING_SCHEMA.json']:
        shutil.copy2(a.bundle/name,a.destination/name)
    new_json(a.destination/'CONFIG.json',c);store=Store(a.destination)
    new_json(a.destination/'PORTABLE_SOURCE_VERIFICATION.json',store.verify_sources(True));print('Verified source-only workspace. Register eligible real participants before use.')
if __name__=='__main__':main()
