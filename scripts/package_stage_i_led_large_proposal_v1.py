"""Offline proposal/report packaging only. Does not create a GPU execution runner."""
import json
import shutil
import zipfile
from prepare_stage_i_led_large_v1 import *


def main():
    cfg=read(CONFIG);meta=read(PREP/'META_LORA_INVENTORY.json');data=read(PREP/'TOKENIZER_DATA_COMPATIBILITY.json')
    log=(PREP/'OFFLINE_TESTS.log').read_text(encoding='utf-8')
    require('Ran 18 tests' in log and log.strip().endswith('OK'),'Focused offline tests must pass')
    require(not (PREP/'PREFLIGHT_MANIFEST.json').exists(),'Proposal already frozen')
    own=['scripts/prepare_stage_i_led_large_v1.py','scripts/finalize_stage_i_led_large_offline_v1.py',
        'scripts/package_stage_i_led_large_proposal_v1.py','tests/test_stage_i_led_large_preparation_v1.py',
        'configs/stage_i_led_large_proposal_v1.json','thesis_sdd_docs/LED_LARGE_READINESS_AND_EXPERIMENT_PROPOSAL_V1.md']
    old=ROOT/'outputs/stage_i/unlikelihood_preparation_v2/PREFLIGHT_MANIFEST.json'
    protected=dict(read(old)['files'])
    for rel,h in protected.items():require(sha(Path(rel) if Path(rel).is_absolute() else ROOT/rel)==h,'Frozen history changed: '+rel)
    # Bind actual stopped UL evidence separately, without rewriting its historical recommendation.
    for base in ['outputs/stage_i/unlikelihood_execution_v2','outputs/stage_i/unlikelihood_execution_report_v2',
                 'outputs/stage_i/unlikelihood_paired_run_v2']:
        for p in (ROOT/base).rglob('*'):
            if p.is_file():protected[p.relative_to(ROOT).as_posix()]=sha(p)
    incident_zip=ROOT/'outputs/stage_i/stage_i_unlikelihood_execution_review_v2.zip'
    protected[incident_zip.relative_to(ROOT).as_posix()]=sha(incident_zip)
    report={'status':'OFFLINE_PROTOCOL_REVIEW_READY_NOT_GPU_EXECUTION_READY',
        'recommended_initialization':{'model':cfg['model_id'],'revision':cfg['revision']},
        'secondary_candidate':'assessed_only_not_selected_no_fallback_trial',
        'architecture_verified':'actual installed meta LED/PEFT, no backbone weights loaded',
        'unique_backbone_parameters':meta['unique_backbone_parameters'],'LoRA_modules':228,
        'trainable_parameters':4915200,'CPU_untrained_adapter_roundtrip_exact':True,
        'full_train_validation_targets_checked':data['business_targets'],'candidate_encodes':data['candidate_source_or_label_encodes'],
        'tokenizer_policy':'retain frozen tokenizer/packs; explicit no truncation; raw Amazon192backend mismatch preserved',
        'offline_tests':18,'GPU_operations':0,'model_forward_backward_generation':0,'new_teacher_calls':0,
        'held_out_content_access':False,'selected_reviews_captions_targets_references_modified':False,
        'checkpoint_binary_downloaded_or_tensor_values_verified':False,
        'GPU_memory_gradient_and_throughput_verified':False,
        'proposal_F_exposure':cfg['training'],'proposal_quality_generations':48,
        'proposal_technical_gate_generations_max':2,
        'full_matched_RO_U_family_requires_later_approval':True,
        'historical_base_family_stays_primary':True,
        'unlikelihood_closure':'numerical feasibility failure,quality NOT EVALUATED; no patch/retry',
        'D051_strict_replay':'FAIL_UNCHANGED','GPU_execution_approved':False,
        'remaining_before_execution':['researcher protocol/initialization approval',
            'confirm defense schedule and explicit new deadline preserving>=14days',
            'prepare/pin reviewed large runtime training runner from D059; no automatic reuse of base hardcoded114module guards',
            'download exact official checkpoint and verify LFS hash/tensor inventory on CPU',
            'explicit bounded GPU feasibility approval on selected hardware; measured memory/ETA before training'],
        'no_automatic_additional_model_search':True}
    write(PREP/'READINESS_REPORT.json',report)
    write(PREP/'RESOURCE_ESTIMATES.json',{'backbone_FP32_bytes':meta['unique_backbone_parameters']*4,
        'LoRA_FP32_params_grads_and_Adam_moments_bytes':4915200*16,
        'decoder_768_by50265_FP32_logits_bytes':768*50265*4,
        'recommended_first_GPU_VRAM_GB':24,'host_RAM_GB_minimum':32,'host_RAM_GB_preferred':64,
        'free_disk_GB':40,'VRAM_12_to20_GiB_envelope_is_unmeasured':True,
        'D059_base_measured_training_seconds':20553.6271674,'rough_width_depth_compute_factor':(12/6)*(1024/768)**2,
        'large_training_time_measured':False,'F_gate_training_and_quality_budget_hours_max_proposed':48,
        'family_freeze_UTC_proposed':cfg['timeline']['proposed_absolute_model_family_freeze_utc'],
        'historical_authorization_deadlines_modified':False,'paid_GPU_or_data_upload_authorized':False})
    write(PREP/'OFFLINE_TEST_RESULT.json',{'status':'PASS','tests':18,'model_forward_backward_generation':0,'GPU_operations':0})
    for rel in own:protected[rel]=sha(ROOT/rel)
    for p in PREP.rglob('*'):
        if p.is_file():protected[p.relative_to(ROOT).as_posix()]=sha(p)
    write(PREP/'PREFLIGHT_MANIFEST.json',{'files':protected,'status':'OFFLINE_PROPOSAL_BINDINGS_ONLY',
        'GPU_execution_approved':False,'binary_tensor_value_verification_pending':True,'executable_GPU_training_runner_present':False})
    write(PREP/'PROPOSAL_BINDINGS.json',{'config_sha256':sha(CONFIG),'preflight_manifest_sha256':sha(PREP/'PREFLIGHT_MANIFEST.json'),
        'proposal_sha256':sha(ROOT/'thesis_sdd_docs/LED_LARGE_READINESS_AND_EXPERIMENT_PROPOSAL_V1.md'),
        'GPU_execution_approved':False,'do_not_treat_as_GPU_authorization':True})
    bundle=ROOT/'outputs/stage_i/led_large_proposal_review_bundle_v1';bundle.mkdir(exist_ok=False)
    def cp(src,rel):
        dest=bundle/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
    for rel in own:cp(ROOT/rel,rel)
    for p in PREP.rglob('*'):
        if p.is_file() and 'documentation_before' not in str(p) and p.suffix!='.safetensors':cp(p,p.relative_to(ROOT))
    cp(ROOT/'configs/stage_i_d059_expanded_lora_v2.json','configs/stage_i_d059_expanded_lora_v2.json')
    for name in ['vocab.json','merges.txt']:
        cp(ROOT/'assets/tokenizers/led-base-16384'/name,Path('assets/tokenizers/led-base-16384')/name)
    write(bundle/'README_FIRST.md','''# LED-large offline readiness and ONE experiment proposal

Read thesis_sdd_docs/LED_LARGE_READINESS_AND_EXPERIMENT_PROPOSAL_V1.md.
Actual pinned metadata/configs, static public training code (never executed),
228-module meta/CPU-adapter serialization audit, full6600pair token-ID results and
18offline tests are included. No full backbone checkpoint, GPU forward/training,
Teacher generation or held-out contents accessed. This is a proposal, not a GPU
execution handoff. Checkpoint LFS hashes are advertised, not binary-verified yet.

Researcher stopped UL as numerical feasibility failure; do not reinterpret as a
scientific qualityNO-GO or silently patch/retry. Original evidence remains unchanged.
Selected initialization:officialLED-large only; Amazon assessed as secondary.
Keep historical matched baseRO/U/F primary until a complete accepted large family.
Protect>=14days for evaluation/humanstudy/writing using the conservative proposed
Oct26freeze against the reported >3week horizon; exact defense date pending.

From extracted root with pinned dependencies:
    python -m unittest discover -s tests -p test_stage_i_led_large_preparation_v1.py -v

The preparation scripts require original workspace frozen packs and must not be
rerun to overwrite preserved evidence. Small untrained adapter tensors and full
weights/packs are omitted; verified inventories and hashes are provided.
''')
    write(bundle/'CHATGPT_REVIEW_REQUEST.md','''# Researcher review request: ONE LED-large proposal

Review the choice of officialLED-large vs Amazon review-finetuned initialization,
the author-reported provenance/history limits, rawAmazon truncation192 incident,
explicit full-token equivalence and exact228module/4915200LoRA coverage.
Assess one full-data2000/200 F run,5epochs/1250updates,epochs3/5quality review,
unchanged data/targets/budgets/beam4, and the proposed actual hardware feasibility
gate. The inherited D059 hyperparameters are an engineering starting choice, not
claimed optimal. No128update experiment is used to reject LED-large capability.
GO must concern usable,Core-rich,balanced,grounded,nonredundant complete summaries,
not loss/headers alone. If promising, RO/U require separate approval with identical
architecture/init/schedule/commonepoch;Shuffled must reuseF. Historical matched
basefamily stays primary until complete. Confirm the14day defense reserve.
No weights/GPU execution approved yet. No new model search or UL patch/retry.
''')
    files={p.relative_to(bundle).as_posix():sha(p) for p in bundle.rglob('*') if p.is_file()}
    write(bundle/'SHA256_MANIFEST.json',{'files':files,'payload_files':len(files)})
    archive=ROOT/'outputs/stage_i/stage_i_led_large_proposal_review_v1.zip'
    require(not archive.exists(),'Preserve delivered proposal archive')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(bundle).as_posix())
    receipt={'archive':archive.relative_to(ROOT).as_posix(),'sha256':sha(archive),'bytes':archive.stat().st_size,
        'payload_files':len(files),'proposal_bindings':read(PREP/'PROPOSAL_BINDINGS.json'),
        'GPU_execution_approved':False,'total_protected_bindings':len(protected)}
    write(PREP/'DELIVERY_RECEIPT.json',receipt)
    print(json.dumps(receipt))


if __name__=='__main__':main()
