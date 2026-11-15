"""Copy the exact original-byte-equivalent TEST photos into a NEW portable source folder."""
import argparse,shutil
from pathlib import Path
from s1000_final_eval_core_v1 import *
def main():
    a=argparse.ArgumentParser();a.add_argument('--workspace',type=Path,required=True);a.add_argument('--annotation-package',type=Path,required=True);a.add_argument('--destination',type=Path,required=True);x=a.parse_args()
    m=read(x.annotation_package/'ANNOTATION_PACKAGE_MANIFEST.json');require(not x.destination.exists(),'Refuse overwrite')
    source=[]
    for row in m['frozen_sources']:
        s=read(x.annotation_package/row['source_file']);require(digest(s)==row['sha256'],'Source changed')
        for ph in s['photos']:
            p=x.workspace/ph['path'];require(file_hash(p)==ph['sha256']==ph['original_source_sha256'],'Original image mismatch')
            source.append((p,ph))
    require(len(source)==1136 and len({ph['photo_id'] for _,ph in source})==1136,'Image frame mismatch')
    x.destination.mkdir(parents=True)
    for p,ph in source:
        dest=x.destination/(ph['photo_id']+'.jpg');shutil.copy2(p,dest);require(file_hash(dest)==ph['sha256'],'Copied image changed')
    write(x.destination/'PHOTO_COPY_RECEIPT.json',dict(status='PASS',photos=1136,unchanged_bytes=True,annotation_manifest_sha256=file_hash(x.annotation_package/'ANNOTATION_PACKAGE_MANIFEST.json')))
    print('Verified1136 original-byte-equivalent photos; no source modification.')
if __name__=='__main__':main()
