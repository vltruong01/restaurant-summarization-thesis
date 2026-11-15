"""Finalize portable V3 review, preserving historical and current artifacts."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,zipfile
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/stage_i/led_large_exploratory_contrastive_v3_execution_v1'
PREP=ROOT/'outputs/stage_i/led_large_exploratory_native_ce_v3'


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')


def main():
    download=read(OUT/'DOWNLOAD_VERIFIED.json');assert download['manifest_PASS'] and download['files']==113
    audit=read(OUT/'INDEPENDENT_CPU_EXECUTION_AUDIT.json');cases=read(OUT/'ALL_PROBE_OUTPUTS_AND_METRICS.json')
    assert audit['arms']['ce']['successful_updates']==audit['arms']['pair']['successful_updates']==16
    assert len(cases)==4 and all(len(c['outputs'])==3 for c in cases)
    assert all(o['metrics']['nonempty'] and o['metrics']['terminal_EOS'] and o['metrics']['BOS_after_start']==0 for c in cases for o in c['outputs'].values())
    source={x['business_id']:x for x in read(OUT/'LOCKED_TRAIN_PROBE_SOURCES.json')}
    quotes=read(OUT/'PROBE_SOURCE_EVIDENCE_QUOTES.json')
    for q in quotes:assert source[q['business_id']]['source'][q['source_start']:q['source_end']]==q['quote']
    c=read(PREP/'CONFIG.json')
    for rel,digest in c['files'].items():assert sha(ROOT/rel)==digest,rel
    prior=read(ROOT/'outputs/stage_i/led_large_contrastive_v2_1_root_cause_review_v1/completed_cpu_check/PRESERVATION_BEFORE.json')['files']
    for rel,digest in prior.items():assert sha(ROOT/rel)==digest,rel
    history_zip=ROOT/'outputs/cloud_handoff/LED_LARGE_CONTRASTIVE_V2_1_ROOT_CAUSE_V1_OFFLINE_REVIEW.zip'
    assert sha(history_zip)=='cf617ebc83e16e58e6f80af746c32c147b11766b56b94bf16d2bef220d6a7964'
    base=ROOT/'outputs/cloud_handoff/LED_LARGE_DECODER_POLICY_V2_WORKSPACE/assets/models/led-large-16384/pytorch_model.bin'
    assert sha(base)==c['host_backbone_sha256']
    write(OUT/'FINAL_PRESERVATION.json',dict(UTC=datetime.now(timezone.utc).isoformat(),
        historical_files_checked=len(prior),all_unchanged=True,historical_files=prior,
        original_backbone_sha256=sha(base),previous_root_cause_review_sha256=sha(history_zip),
        original_adapter_sha256=c['adapter_sha256'],historical_equivalence='FAIL_UNRESOLVED',
        current_execution_bindings_unchanged=len(c['files']),primary_RO_U_F_Shuffled_modified=False))
    write(OUT/'SCIENTIFIC_RECOMMENDATION.json',dict(
        assessor='Implementation author; model-assisted, unblinded, provisional',
        researcher_quality_decision='PENDING',recommendation='NO_GO_FOR_FURTHER_CONTRASTIVE_DEVELOPMENT',
        promotion_authorized=False,current_Filtered_checkpoint='Original LED-large CE epoch3',
        historical_equivalence='FAIL_UNRESOLVED',explicit_post_failure_deviation=True,
        case_judgments={
            '_anAV7MH8kZ-KXI6hNjtHQ':'MIXED: restores meat disagreement versus CE16; unsupported wait/zero-caption visual claim persists',
            'h0tUOsL3sRYVqSQlxWjW5g':'REGRESSION: loses buffet/value specificity and does not restore important food complaints',
            'gTKoJH4BWX67pVu-bRdsmw':'REGRESSION: unsupported steaks-on-cherry visual relationship',
            'd8N5KyvlAm4jT-f8EbV5mQ':'MIXED: review/value organization gains; visual merge/redundancy shared with CE16'},
        not_validated_generalization=True,no_automatic_continuation=True))
    memory=read(OUT/'downloaded/runs/CF_EXPLORATORY_NATIVE_CE_V3/MEMORY_EVIDENCE.json')['summary']
    write(OUT/'EXPERIMENT_RECEIPT.json',dict(status='COMPLETE_EXPLORATORY_POST_FAILURE_DEVIATION_STOP_FOR_REVIEW',
        authorization=read(OUT/'V3_RESEARCHER_AUTHORIZATION.json'),execution=download['remote_status'],
        live_safety='PASS_FINITE_NATIVE_TRAINING_PATH_ONLY_NOT_EQUIVALENCE',
        historical_equivalence='FAIL_UNRESOLVED',CPU_execution_audit_status=audit['status'],
        updates_per_arm={'ce':16,'pair':16},CE_exposures_per_arm=128,fit_businesses=c['fit'],probe_businesses=c['probe'],
        approved_pairs=c['pair_ids'],excluded_pairs=c['excluded_pairs'],generation_calls=12,outputs=12,
        source_quotes_verified=len(quotes),memory=memory,backoffs=0,skipped_steps=0,
        checkpoints=audit['checkpoints'],execution_bindings=read(PREP/'EXACT_BINDINGS.json'),
        download=download,lifecycle=read(OUT/'REMOTE_CLOSEOUT_AND_PAUSE_ATTEMPT.json'),
        provider_final_state=read(OUT/'PAUSE_VERIFICATION.json'),
        quality_recommendation='PROVISIONAL_NO_GO',researcher_quality_decision='PENDING',
        original_checkpoint_modified=False,teacher_targets_modified=False,validation_or_heldout_used=False,
        pretrained_parent_hashing_only=True,automatic_retry=False,additional_experiments=0))
    packet=OUT/'portable_review';packet.mkdir(exist_ok=False)
    def copy(src,rel):
        dest=packet/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest);assert sha(src)==sha(dest)
    for p in OUT.iterdir():
        if p.is_file() and p.suffix in ('.md','.json','.jsonl','.txt','.log','.py'):copy(p,p.name)
    for p in (OUT/'downloaded').rglob('*'):
        if p.is_file() and p.suffix not in ('.pt','.pth','.safetensors','.bin') and not p.name.endswith('.tar.gz'):
            copy(p,Path('execution')/p.relative_to(OUT/'downloaded'))
    for p in PREP.iterdir():
        if p.is_file():copy(p,Path('locked_preparation')/p.name)
    for name in ('led_large_contrastive_native_joint_v3.py','run_led_large_exploratory_native_ce_v3.py',
                 'review_native_ce_contrastive_v3_execution.py','package_native_ce_contrastive_v3_review.py',
                 'download_contrastive_native_v3_evidence.py'):
        copy(ROOT/'scripts'/name,Path('code')/name)
    copy(ROOT/'tests/test_contrastive_native_joint_v3.py',Path('code/test_contrastive_native_joint_v3.py'))
    hist=ROOT/'outputs/stage_i/led_large_exploratory_contrastive_v2_1_execution_v1'
    for rel in ['INDEPENDENT_CPU_RECOMPUTATION.json','REVIEW_REPORT.md']:
        copy(hist/rel,Path('historical_V2_1_FAIL')/rel)
    for rel in ['ROOT_CAUSE_ANALYSIS.md','PATCH_DECISION.md']:
        copy(ROOT/'outputs/stage_i/led_large_contrastive_v2_1_root_cause_review_v1'/rel,Path('historical_offline_root_cause')/rel)
    # Preserve the accepted training inputs/labels and pair evidence for review.
    qualification=ROOT/'outputs/stage_i/led_large_contrastive_pair_qualification_v1'
    for bid in c['fit']:copy(qualification/'positives'/f'{bid}.json',Path('locked_fit_data/positives')/f'{bid}.json')
    for pid in c['pair_ids']:copy(qualification/'pairs'/f'{pid}.json',Path('locked_fit_data/pairs')/f'{pid}.json')
    write(packet/'FULL_BACKUP_REFERENCE.json',dict(filename='LED_LARGE_CF_NATIVE_CE_V3_EXECUTION_EVIDENCE.tar.gz',
        sha256=download['archive_sha256'],bytes=download['archive_bytes'],files=download['files'],
        local_path=str((OUT/'LED_LARGE_CF_NATIVE_CE_V3_EXECUTION_EVIDENCE.tar.gz').relative_to(ROOT)),
        checkpoint_binaries_omitted_from_review_ZIP=True,
        all_checkpoint_optimizer_scaler_RNG_binaries_verified_in_local_full_backup=True))
    entries={p.relative_to(packet).as_posix():sha(p) for p in sorted(packet.rglob('*')) if p.is_file()}
    write(packet/'SHA256_MANIFEST.json',dict(files=entries))
    archive=ROOT/'outputs/cloud_handoff/LED_LARGE_NATIVE_CE_CONTRASTIVE_V3_CHATGPT_REVIEW.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(packet.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(packet).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert all(hashlib.sha256(z.read(rel)).hexdigest()==digest for rel,digest in entries.items())
    with archive.with_suffix('.zip.sha256').open('x',encoding='utf-8') as f:f.write(sha(archive)+'  '+archive.name+'\n')
    receipt=dict(archive=str(archive.relative_to(ROOT)),sha256=sha(archive),bytes=archive.stat().st_size,
        portable_manifest_entries=len(entries),manifest_PASS=True,full_backup_sha256=download['archive_sha256'],
        result='EXECUTION_COMPLETE_PROVISIONAL_QUALITY_NO_GO',historical_equivalence='FAIL_UNRESOLVED')
    write(OUT/'DELIVERY_RECEIPT.json',receipt);print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
