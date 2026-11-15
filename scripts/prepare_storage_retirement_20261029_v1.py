"""Curate immutable keep/delete/archive manifests. No deletion or HF upload."""
from pathlib import Path
import json,collections,hashlib,concurrent.futures,datetime,csv
R=Path(__file__).resolve().parents[1];O=R/'outputs/storage_cleanup/20261029_review_v1'
HOT=['data','scripts','configs','requirements','approvals','schemas','specs','prompts','notebooks','docs','thesis_sdd_docs',
    'outputs/stage_k','outputs/final_evaluation','outputs/pre_evaluation','outputs/data_integrity','outputs/stage_f','outputs/stage_g','outputs/storage_cleanup',
    'outputs/stage_i/led_large_ro_u_source_separated_s1000_matched_v1','outputs/stage_i/led_large_source_separated_v1_two_epoch_preparation_v1',
    'outputs/stage_i/led_large_decoder_policy_v2_execution_v1','outputs/stage_i/led_large_ro_u_ce3_direct_training_execution_v1',
    'outputs/stage_i/led_large_filtered_validation200_selection_v1','outputs/stage_i/led_large_source_separated_epoch1_diagnostic_v1',
    'outputs/stage_i/led_large_source_separated_epoch2_development_v1','outputs/stage_i/stage_k_workflow_preparation_v1',
    'outputs/cloud_handoff/LED_LARGE_DECODER_POLICY_V2_WORKSPACE','assets/tokenizers']
def under(p,prefix):return p==prefix or p.startswith(prefix+'/')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(n,x):(O/n).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
def main():
    scan=json.loads((O/'CURRENT_STORAGE_INVENTORY.json').read_text(encoding='utf8'));files=scan['files']
    prefixes=['led_large_boundary','led_large_section_wce','led_large_cad','led_large_exploratory','led_large_contrastive','led_large_nbest','led_large_gradient_gate','led_large_full_train_v5']
    explicit={'led_large_vast_d059_v1_initial_transport_check','led_large_source_separated_grounded_v2_training_execution_v1','led_large_source_separated_v1_s1500_continuation_v1'}
    roots=sorted({g['path'] for g in scan['groups'] if g['path'].startswith('outputs/stage_i/') and g['path'].count('/')==2 and (any(g['path'].split('/')[-1].startswith(t) for t in prefixes) or g['path'].split('/')[-1] in explicit)})
    roots+=['outputs/stage_c/review_selection_source_length_expanded_v1/business_runs','outputs/stage_c/review_selection_source_length_expanded_v1/support_v4','outputs/stage_c/review_selection_source_length_expanded_v1/support_v4_1','outputs/stage_c/review_selection_train_validation_v1','outputs/stage_c/review_selection_train_validation_v3']
    cold=[]
    for rel in roots:
        ff=[f for f in files if under(f['path'],rel)]
        if not ff:continue
        assert not any(under(rel,h) or under(h,rel) for h in HOT),rel
        cold.append(dict(id='cold_'+hashlib.sha256(rel.encode()).hexdigest()[:12],path=rel,files=len(ff),logical_bytes=sum(f['bytes'] for f in ff),storage_bytes=sum(f['storage_bytes'] for f in ff),
            reason='Retired development variant, execution/operational evidence, or historical selector witnesses; not used by next S1000 reference/metric phase. Archive complete tree; no assertion that every variant failed.',
            delete_status='BLOCKED_UNTIL_PRIVATE_HF_FULL_ARCHIVE_AND_MEMBER_SHA_VERIFIED'))
    coldpaths=[g['path'] for g in cold]
    groups=collections.defaultdict(list)
    for f in files:
        if f['bytes']>=8*2**20:groups[f['bytes']].append(f)
    candidates=[f for ff in groups.values() if len(ff)>1 for f in ff]
    print('Fresh-hashing',len(candidates),'same-size large files, logical GiB',round(sum(f['bytes'] for f in candidates)/2**30,3),flush=True)
    def hashfile(f):
        p=R/f['path'];s=p.stat();h=sha(p);after=p.stat();assert (s.st_size,s.st_mtime_ns)==(after.st_size,after.st_mtime_ns)==(f['bytes'],f['mtime_ns']),f['path']
        return f,h,s.st_nlink,s.st_ino
    hashed=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        for i,result in enumerate(ex.map(hashfile,candidates),1):
            hashed.append(result)
            if i%50==0:print('hashed',i,'/',len(candidates),flush=True)
    same=collections.defaultdict(list)
    for f,h,nlink,inode in hashed:same[h].append((f,nlink,inode))
    canonical_base='outputs/cloud_handoff/LED_LARGE_DECODER_POLICY_V2_WORKSPACE/assets/models/led-large-16384/pytorch_model.bin'
    canonical_bert='outputs/stage_k/s1000_test100_phase1_v1/METRIC_RUNTIME/roberta-large/pytorch_model.bin'
    def score(f):
        p=f['path']
        return (0 if p in [canonical_base,canonical_bert] else 1 if any(under(p,h) for h in HOT) else 5 if any(under(p,g) for g in coldpaths) else 2,len(p),p)
    duplicates=[]
    for h,ff in same.items():
        if len(ff)<2:continue
        ff.sort(key=lambda t:score(t[0]));kept=ff[0][0]
        # Keeper must remain outside every later cold-delete tree.
        if any(under(kept['path'],g) for g in coldpaths):continue
        for f,nlink,inode in ff[1:]:
            p=f['path'];exception=p.startswith('outputs/stage_k/s1000_test100_phase1_v1/METRIC_CACHE/') and p.endswith('/pytorch_model.bin')
            if (any(under(p,h) for h in HOT) and not exception) or any(under(p,g) for g in coldpaths):continue
            assert p!=kept['path']
            duplicates.append(dict(path=p,keep=kept['path'],bytes=f['bytes'],storage_bytes=f['storage_bytes'] if nlink==1 else 0,sha256=h,mtime_ns=f['mtime_ns'],
                nlink=nlink,inode=inode,classification='EXACT_BYTE_DUPLICATE',reason='Fresh full SHA256/length equal to retained file; no unique file bytes discarded',operational_protected_exception=exception))
    duplicates.sort(key=lambda f:-f['storage_bytes'])
    plan=dict(version='STORAGE_RETIREMENT_20261029_V1',created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),root=str(R),
        mode='PREPARED_ONLY_NOT_EXECUTED',inventory_sha256=sha(O/'CURRENT_STORAGE_INVENTORY.json'),workspace_logical_bytes=scan['total_logical_bytes'],workspace_storage_bytes=scan['total_storage_bytes'],
        protected_prefixes=HOT,duplicate_deletions=duplicates,retired_archive_groups=cold,
        private_HF_repository='vltruong01/multimodal_summarization_thesis',HF_new_namespace='cold_storage/s1000_retired_20261029_v1',
        current_science_kept_local=['RO/U/F-S1000 checkpoints and their training receipts','Original CE3/RO750/U750 ancestry','Pinned LED backbone and BERTScore model/tokenizers','All TEST100 predictions/raw tokens/diagnostics and frozen Shuffled mapping','100 blank reference annotation templates and original photos/reviews','Teacher targets and canonical caption/filter manifests','Scripts/configs/proposal/current documentation'],
        immediate_verified_duplicate_storage_bytes=sum(f['storage_bytes'] for f in duplicates),archive_then_remove_storage_bytes=sum(g['storage_bytes'] for g in cold),
        files_deleted_now=0,uploads_executed_now=0)
    save('CLEANUP_PLAN.json',plan);save('LARGE_FILE_SHA_VERIFICATION.json',[dict(path=f['path'],bytes=f['bytes'],sha256=h,nlink=n,inode=i) for f,h,n,i in hashed])
    with (O/'VERIFIED_DUPLICATE_FILES.csv').open('w',encoding='utf8',newline='') as out:
        w=csv.DictWriter(out,fieldnames=list(duplicates[0]));w.writeheader();w.writerows(duplicates)
    with (O/'RETIRED_HF_ARCHIVE_GROUPS.csv').open('w',encoding='utf8',newline='') as out:
        w=csv.DictWriter(out,fieldnames=list(cold[0]));w.writeheader();w.writerows(cold)
    print(json.dumps(dict(duplicate_files=len(duplicates),immediate_verified_duplicate_GiB=plan['immediate_verified_duplicate_storage_bytes']/2**30,cold_groups=len(cold),archive_then_delete_GiB=plan['archive_then_remove_storage_bytes']/2**30,total_disjoint_potential_GiB=(plan['immediate_verified_duplicate_storage_bytes']+plan['archive_then_remove_storage_bytes'])/2**30),indent=2))
    for f in duplicates[:25]:print(round(f['storage_bytes']/2**30,3),f['path'],'KEEP',f['keep'])
if __name__=='__main__':main()
