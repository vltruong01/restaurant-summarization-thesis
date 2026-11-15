"""Broad file/ZIP reference search, without executing code or touching human DB.

Evidence-shaped matches are candidates, never automatically accepted as references.
Binary model/training tar archives are catalogued, not decompressed/reinterpreted.
"""
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import io
import json
import re
import zipfile
from stage_m1_existing_teacher_metrics_v1 import ROOT,OUT,PREP
from s1000_final_eval_core_v1 import read,write,write_csv,text_hash

IDS=set(read(PREP/'COHORT100_FREEZE.json')['business_ids'])
BUSINESS=re.compile(rb'"business_id"\s*:\s*"([^"\\]+)"')
REFERENCE_KEYS={'target','targets','review_consensus','visual_evidence','review_reference','visual_reference','reference','reference_text','teacher_target'}
KEY_BYTES=tuple(('"'+k+'"').encode() for k in REFERENCE_KEYS)
LIMIT=64*1024*1024

def inspect_blob(blob,name):
    if not any(k in blob for k in KEY_BYTES):return []
    ids={m.group(1).decode('utf8',errors='replace') for m in BUSINESS.finditer(blob)}&IDS
    if not ids:return []
    try:
        if name.lower().endswith('.jsonl'):objects=[json.loads(x) for x in blob.decode('utf-8-sig').splitlines() if x.strip()]
        else:objects=[json.loads(blob.decode('utf-8-sig'))]
    except (ValueError,UnicodeError):return [{'file':name,'business_id':b,'classification':'UNPARSEABLE_CANDIDATE_REQUIRES_MANUAL_PROVENANCE','field':'','text_sha256':None,'status':None} for b in sorted(ids)]
    findings=[];stack=objects
    while stack:
        obj=stack.pop()
        if isinstance(obj,dict):
            if obj.get('business_id') in ids:
                for key in sorted(REFERENCE_KEYS&set(obj)):
                    value=obj[key]
                    strings=[(key,value)] if isinstance(value,str) else [(key+'/'+str(k),v) for k,v in value.items() if isinstance(v,str)] if isinstance(value,dict) else []
                    for field,text in strings:
                        if text.strip():findings.append({'file':name,'business_id':obj['business_id'],'classification':'NONEMPTY_REFERENCE_SHAPED_CANDIDATE_NOT_APPROVED','field':field,
                            'text_sha256':text_hash(text),'status':obj.get('status'),'origin':obj.get('origin',obj.get('target_content_origin')),'source_bundle_sha256':obj.get('source_bundle_sha256'),
                            'human_approved':obj.get('status')=='APPROVED','note':'Candidate structure is not semantic/provenance verification; never score automatically.'})
            stack.extend(v for v in obj.values() if isinstance(v,(dict,list)))
        elif isinstance(obj,list):stack.extend(v for v in obj if isinstance(v,(dict,list)))
    return findings

def scan_file(name):
    p=ROOT/name
    try:
        size=p.stat().st_size
        if size>LIMIT:return {'file':name,'status':'CATALOGUED_OVER64MiB_NOT_PARSED','bytes':size},[]
        blob=p.read_bytes();return {'file':name,'status':'SCANNED_REFERENCE_FIELDS_AND_TEST_IDS','bytes':size},inspect_blob(blob,name)
    except (OSError,ValueError) as e:return {'file':name,'status':'ERROR_'+type(e).__name__,'bytes':None},[]

def scan_zip(name):
    stats={'archive':name,'status':'ZIP_INDEX_AND_MACHINE_READABLE_JSON_MEMBERS_SCANNED','json_members_scanned':0,'json_members_over64MiB':0};findings=[]
    try:
        with zipfile.ZipFile(ROOT/name) as z:
            for member in z.infolist():
                if Path(member.filename).suffix.lower() not in ['.json','.jsonl']:continue
                if member.file_size>LIMIT:stats['json_members_over64MiB']+=1;continue
                findings+=inspect_blob(z.read(member),name+'::'+member.filename);stats['json_members_scanned']+=1
    except (OSError,ValueError,zipfile.BadZipFile,RuntimeError) as e:stats['status']='ERROR_'+type(e).__name__
    return stats,findings

def main():
    paths=(OUT/'WORKSPACE_FILE_INVENTORY.txt').read_text(encoding='utf8').splitlines()
    # Exclude only our in-progress results; package copies elsewhere remain searched.
    jsonfiles=[p for p in paths if Path(p).suffix.lower() in ['.json','.jsonl'] and not (ROOT/p).is_relative_to(OUT)]
    zips=[p for p in paths if Path(p).suffix.lower()=='.zip' and not (ROOT/p).is_relative_to(OUT)]
    scans=[];findings=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,(row,found) in enumerate(pool.map(scan_file,jsonfiles),1):
            scans.append(row);findings+=found
            if i%20000==0:print(f'Loose reference search {i}/{len(jsonfiles)}',flush=True)
        archive_stats=[]
        for row,found in pool.map(scan_zip,zips):archive_stats.append(row);findings+=found
    write_csv(OUT/'WORKSPACE_JSON_SEARCH_LEDGER.csv',scans)
    write(OUT/'WORKSPACE_ZIP_SEARCH_LEDGER.json',archive_stats)
    write(OUT/'TEST_REFERENCE_CANDIDATES.json',findings)
    # Targets require generated text, correct cohort and source/model provenance;
    # no bare candidate gets promoted to a valid Teacher reference.
    write(OUT/'WORKSPACE_REFERENCE_SEARCH_SUMMARY.json',{'status':'SEARCH_COMPLETED_CANDIDATES_REQUIRE_PROVENANCE_REVIEW','workspace_file_inventory':len(paths),'loose_JSON_JSONL_files_considered':len(jsonfiles),
        'loose_search_status_counts':dict(Counter(r['status'] for r in scans)),'ZIP_archives_considered':len(zips),'ZIP_status_counts':dict(Counter(r['status'] for r in archive_stats)),
        'ZIP_JSON_members_scanned':sum(r['json_members_scanned'] for r in archive_stats),'ZIP_JSON_members_over64MiB_not_parsed':sum(r['json_members_over64MiB'] for r in archive_stats),
        'nonempty_TEST_reference_shaped_candidates':len(findings),'candidate_businesses':len({r['business_id'] for r in findings}),
        'TRAIN_VAL_targets_never_reassigned_to_TEST':True,'scope_limits':['All inventoried loose JSON/JSONL (up to64MiB/file) and every ZIP JSON/JSONL member up to64MiB were searched by TEST business IDs/reference fields.',
            'Binary training/model tar/tar.gz archives, arbitrary prose/CSV, encrypted members and external accounts are not certified reference datasets by this scan.',
            'Existing source/example packages, schemas, prediction files and blank templates are not generated Teacher references. Raw provider responses alone need source/model/generation validation.'],
        'API_calls':0,'model_calls':0,'human_DB_opened':False})
    print(json.dumps({'loose_files':len(jsonfiles),'ZIPs':len(zips),'candidate_rows':len(findings),'candidate_businesses':len({r['business_id'] for r in findings})}),flush=True)

if __name__=='__main__':main()
