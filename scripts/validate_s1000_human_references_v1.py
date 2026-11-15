"""Objective validator/exporter. Does not write prose or grant human approval."""
import argparse,json
from pathlib import Path
from s1000_final_eval_core_v1 import *

def validate_all(package,annotations,approved=True):
    package=Path(package);annotations=Path(annotations);m=read(package/'ANNOTATION_PACKAGE_MANIFEST.json')
    require(m['businesses']==100 and len(m['frozen_sources'])==100,'Source frame')
    errors={};refs=[]
    expected={x['business_id'] for x in m['frozen_sources']}
    actual={p.stem for p in annotations.glob('*.json')}
    require(actual==expected,'Missing or additional reference files')
    for row in m['frozen_sources']:
        b=row['business_id'];s=read(package/row['source_file']);require(digest(s)==row['sha256'],'Frozen source file changed')
        r=read(annotations/f'{b}.json');ee=validate_reference(r,s,approved=approved)
        if ee:errors[b]=ee
        refs.append(r)
    return refs,dict(status='PASS_OBJECTIVE_APPROVED_REFERENCE_RULES' if approved and not errors else 'PASS_DRAFT_STRUCTURE_ONLY' if not errors else 'NOT_READY',
        businesses=100,require_approved=approved,errors=errors,semantic_factuality_certified=False,
        authenticity_note='Identity attestations and history bindings are integrity evidence, not independent authentication of the humans')

def main():
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--annotations',type=Path,required=True)
    p.add_argument('--draft',action='store_true');p.add_argument('--export',type=Path);p.add_argument('--content-hash',type=Path)
    a=p.parse_args()
    if a.content_hash:print(digest(reference_content(read(a.content_hash))));return
    refs,r=validate_all(a.package,a.annotations,approved=not a.draft);print(json.dumps(r,indent=2))
    if r['errors']:raise SystemExit(2)
    if a.export:
        require(not a.draft,'Cannot export blank/draft references as final gold')
        a.export.mkdir(parents=True,exist_ok=False)
        write(a.export/'APPROVED_REFERENCES.json',refs)
        write(a.export/'REFERENCE_FREEZE.json',dict(businesses=100,source_manifest_sha256=file_hash(a.package/'ANNOTATION_PACKAGE_MANIFEST.json'),
            reference_file_sha256=file_hash(a.export/'APPROVED_REFERENCES.json'),human_written_and_second_checked=True,
            objective_validation=r,content_hashes={r['business_id']:digest(reference_content(r)) for r in refs}))

if __name__=='__main__':main()
