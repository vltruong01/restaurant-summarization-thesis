"""Stage D1 read-only file checks; no unfiltered deduplication or captions."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import hashlib,json,time
import pandas as pd,yaml
import PIL,numpy as np
from photo_integrity_audit import ROOT,CONFIG,inspect_photo,digest_file,duplicate_diagnostics


def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def main():
    started=time.perf_counter();cfg=yaml.safe_load(CONFIG.read_text(encoding='utf-8'));out=ROOT/cfg['output_directory'];cohort=ROOT/cfg['cohort_directory']
    assert not (out/'completion_report.json').exists(),'Existing completed audit; do not overwrite.'
    source=pd.read_csv(ROOT/cfg['photo_manifest'],keep_default_na=False);assert len(source)==30027 and source.photo_id.is_unique and source.business_id.nunique()==2300
    assert source.split.value_counts().to_dict()=={'train':26387,'validation':2504,'test':1136}
    source.insert(0,'photo_index',range(len(source)))
    checksums=json.loads((cohort/'checksums.json').read_text(encoding='utf-8'))
    assert digest_file(ROOT/cfg['photo_manifest'])==checksums['photo_manifest.csv']['sha256']
    for row in source.to_dict('records'):
        expected=checksums[row['relative_path']];assert expected['sha256']==row['sha256'] and expected['bytes']==row['bytes']
    protected={name:digest_file(ROOT/name) for name in [cfg['photo_manifest'],cfg['cohort_directory']+'/business_split.csv','configs/research_pending.yaml','configs/project.yaml','outputs/stage_c/led_source_token_allocation_audit_v1/acceptance_report.json','outputs/stage_c/review_selection_source_length_expanded_v1/acceptance_report.json']}
    identity={'config_sha256':digest_file(CONFIG),'protected_sha256':protected,'scripts_sha256':{name:digest_file(ROOT/'scripts'/name) for name in ['photo_integrity_audit.py','run_photo_integrity_audit.py']},'versions':{'pillow':PIL.__version__,'numpy':np.__version__}}
    out.mkdir(parents=True,exist_ok=True);write(out/'scientific_identity.json',identity);(out/'config_at_run.yaml').write_bytes(CONFIG.read_bytes())
    reference=sorted(source[source.split.isin(['train','validation'])].photo_id,key=lambda p:hashlib.sha256(('42|D1_reference|'+p).encode()).hexdigest())[:256]
    write(out/'precommitted_reference_photo_ids.json',reference)
    results=[]
    with ProcessPoolExecutor(max_workers=cfg['execution']['workers']) as pool:
        jobs=((row,str(cohort)) for row in source.to_dict('records'))
        for result in pool.map(inspect_photo,jobs,chunksize=cfg['execution']['worker_chunksize']):
            results.append(result)
            if len(results)%1000==0:print(f'Image integrity/decode/hash {len(results)}/30027.',flush=True)
    manifest=pd.DataFrame(results);manifest.to_csv(out/'image_manifest.csv',index=False)
    manifest[~manifest.readable].to_csv(out/'unreadable_images.csv',index=False)
    manifest[manifest.readable][['photo_index','photo_id','business_id','split','label','relative_path','raw_sha256','pixel_sha256','phash64_hex']].to_csv(out/'unfiltered_readable_photo_pool.csv',index=False)
    print('Computing all432699 source within-business pairs (readable subset if failures).',flush=True)
    tables=duplicate_diagnostics(manifest,cfg['near_duplicates']['diagnostic_hamming_thresholds'])
    filenames={'pairs':'near_pair_candidates.csv','groups':'within_business_duplicate_groups.csv','memberships':'within_business_group_memberships.csv','global_groups':'global_duplicate_collision_groups.csv','global_memberships':'global_group_memberships.csv','businesses':'business_photo_counts.csv'}
    for key,filename in filenames.items():tables[key].to_csv(out/filename,index=False)
    for name,digest in protected.items():assert digest_file(ROOT/name)==digest,name
    files=[p for p in out.iterdir() if p.is_file()];size=sum(p.stat().st_size for p in files);assert size<cfg['storage']['maximum_scientific_output_bytes'],size
    report={'status':'PASS','businesses':2300,'photos':30027,'readable':int(manifest.readable.sum()),'unreadable':int((~manifest.readable).sum()),'raw_bytes_matched_accepted_manifest':True,'all_within_business_pairs_possible':432699,'all_within_business_readable_pairs_evaluated':int(sum(r.readable_photos*(r.readable_photos-1)//2 for r in tables['businesses'].itertuples())),'near_candidate_pairs_max12':len(tables['pairs']),'within_groups':len(tables['groups']),'global_groups':len(tables['global_groups']),'near_thresholds_exploratory':[0,4,8,12],'test_use':'fixed_integrity_hashing_only_no_tuning','unfiltered_readable_pool_retains_all_valid_original_IDs':True,'physical_images_deleted':False,'businesses_replaced_or_excluded':False,'final_duplicate_policy':None,'final_near_threshold':None,'caption_generation_started':False,'model_GPU_API_rental_started':False,'scientific_output_bytes':size,'elapsed_seconds':round(time.perf_counter()-started,3),'artifact_sha256':{p.name:digest_file(p) for p in files}}
    write(out/'completion_report.json',report);print(json.dumps({k:v for k,v in report.items() if k!='artifact_sha256'},indent=2),flush=True)


if __name__=='__main__':main()
