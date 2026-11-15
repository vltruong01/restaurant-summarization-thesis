"""Prepare a researcher-bound PENDING receipt; never grant or execute GPU scope."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]
PREP=ROOT/'outputs/stage_i/led_large_section_wce_v1'
OUT=ROOT/'outputs/stage_i/led_large_section_wce_v1_authorization_v1'
EXPECTED={
    'config_sha256':'fbc0617d6832ca82b43646b677cbf6d21275ff9b29b4cbd4eae7954c45f5a2e4',
    'runner_sha256':'2cf3a59e866ef7a3de19ca59fb5fc74780ef6c437c423b49e75aec383640e8e4',
    'helper_sha256':'a1890b9c30707d895cdfe5f02ef9a98b6282dec30abe431f32ec57ef9fb50960',
    'loss_module_sha256':'ae7b55564266dd982988fd5d7d3ecd1db632f62bdcde1da5b5e890c918b8c921',
    'section_mask_sha256':'dc723c6c01bd95f571cfa8c0cdfc216cbd80c385387cdeb75c0f8b5978baf83d',
    'preflight_manifest_sha256':'e13a390a96d46fe52b66db46790b87d4639020a1ab6ec0bbc303255d018a6149',
}
PATHS={
    'config_sha256':'configs/stage_i_led_large_section_wce_v1.json',
    'runner_sha256':'scripts/run_stage_i_led_large_section_wce_v1.py',
    'helper_sha256':'scripts/stage_i_led_large_section_wce_v1.py',
    'loss_module_sha256':'scripts/led_section_weighted_ce_v1.py',
    'section_mask_sha256':'outputs/stage_i/led_large_section_wce_v1/SECTION_MASK_VIEWS.npz',
    'preflight_manifest_sha256':'outputs/stage_i/led_large_section_wce_v1/PREFLIGHT_MANIFEST.json',
}


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def save(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:
        if isinstance(obj,str):f.write(obj)
        else:json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n')


def main():
    assert not OUT.exists(),'Versioned receipt already exists; do not overwrite'
    now=datetime.now(timezone.utc).isoformat()
    accepted=read(PREP/'EXECUTION_BINDINGS.json')
    stage=ROOT/'outputs/cloud_handoff/LED_LARGE_DECODER_POLICY_V2_WORKSPACE'
    actual={k:sha(ROOT/p) for k,p in PATHS.items()}
    assert actual==EXPECTED
    assert all(accepted[k]==h for k,h in EXPECTED.items())
    assert all(sha(stage/p)==h for k,p in PATHS.items() for h in [EXPECTED[k]])
    parent=read(PREP/'BASELINE_INPUT_BINDINGS.json')['files']
    assert all(sha(ROOT/p)==h for p,h in parent.items())
    review=read(PREP/'DELIVERY_RECEIPT.json')
    archive=Path(review['archive']);assert sha(archive)==review['sha256']
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        files=json.loads(z.read('SHA256_MANIFEST.json'))['files']
        assert len(files)==147 and set(z.namelist())==set(files)|{'SHA256_MANIFEST.json'}
        for n,h in files.items():assert hashlib.sha256(z.read(n)).hexdigest()==h,n
    # Retain all operational guards from the exact technically accepted template.
    receipt=read(PREP/'AUTHORIZATION_TEMPLATE_NOT_APPROVED.json')
    receipt.update(EXPECTED)
    receipt.update(
        receipt_version='SECTION_WCE_RESEARCHER_BOUND_AUTHORIZATION_V1',
        receipt_status='PENDING_EXPLICIT_GPU_AUTHORIZATION',
        preparation_recorded_at_utc=now,
        offline_technical_preparation_accepted=True,
        technical_acceptance_source='researcher_message_in_current_conversation',
        technical_acceptance_platform_timestamp_utc=None,
        technical_acceptance_is_GPU_authorization=False,
        scientific_quality_GO=None,
        run_root='runs/F_large_section_wce_v1',
        GPU='NVIDIA GeForce RTX 3090',
        minimum_measured_memory_headroom_fraction=0.15,
        intended_trainable_LoRA_tensors=456,
        nominal_feasibility_microbatches=24,
        checkpoint_steps=[250,500,750,1000,1250],
        final_prediction_epochs=[3,5],
        validation_businesses_per_epoch=24,
        fixed_validation_subsets={'random':20,'targeted':4},
        baseline_regeneration_authorized=False,
        target_modification_authorized=False,
        decoding_change_authorized=False,
        weight_search_authorized=False,
        additional_experiments_authorized=False,
        use_weighted_vs_unweighted_validation_loss_as_quality_metric=False,
        quality_review_required_before_any_family_continuation=True,
        protected_thesis_evaluation_period={
            'researcher_schedule_response_verbatim':'không cần quan tâm tới ngày',
            'schedule_management':'researcher_managed_no_thesis_date_inferred',
            'protected_period_start_utc':None,
            'protected_period_end_utc':None,
            'explicit_14_day_reserve_confirmation_received':False,
            'accepted_runner_time_policy_unchanged':True,
            'note':'No academic deadline is inferred. Actual GPU approval/time fields stay unset; the accepted runner still requires its original clock/reserve checks before GPU work.'},
        required_quality_checks=[
            'Important source-grounded Core coverage and distinctive dishes/facts',
            'Same-aspect positive/negative opinion synthesis',
            'Factual grounding and no unsupported claims',
            'Zero-caption visual compliance, including Brooklyn Water Bagel',
            'Useful visual evidence; specifically protect Aquarium fish-tank Core',
            'Semantic repetition, coherence, completeness and overall usability'],
        preserve=['Original Teacher text/token IDs and all frozen source artifacts',
                  'Original LED-large V2 model checkpoints and saved baseline predictions',
                  'All five new full-state checkpoints and integrity receipts if execution is approved',
                  'Optimizer/scaler/gradient/section-loss/AMP/memory evidence and known failures',
                  'D-051 strict replay FAIL and all historical decisions/deadlines'],
        future_review_bundle={
            'new_generated_summaries':48,'saved_original_CE_baseline_summaries':48,
            'source_evidence_and_approved_references':True,
            'training_checkpoint_integrity_receipts':True,
            'raw_token_ids_and_full_decoded_text':True,
            'quality_diagnostics_and_known_errors':True,
            'model_assisted_assessment_provisional_until_researcher_confirmation':True},
        pre_execution_requirements_pending=[
            'Explicit new researcher GPU authorization; technical acceptance alone is insufficient',
            'Actual UTC execution authorization/deadline and reserve checks required by unchanged accepted runner',
            'Complete exact parent workspace/runtime/data/mask preflight on the execution host',
            'Fresh output namespace and intended RTX3090 hardware verification before first gate operation'],
    )
    assert not receipt['GPU_execution_approved'] and not receipt['training_authorized']
    assert not receipt['scientific_amendment_approved'] and not receipt['early_sanity_generation_approved']
    assert receipt['execution_authorized_at_utc'] is receipt['execution_deadline_utc'] is receipt['researcher_absolute_stop_utc'] is None
    assert receipt['defense_reserve_at_least_14_days_confirmed'] is False
    save(OUT/'SECTION_WCE_RESEARCHER_AUTHORIZATION_PENDING.json',receipt)
    save(OUT/'TECHNICAL_ACCEPTANCE_VERIFICATION.json',dict(
        checked_at_utc=now,accepted_six_bindings=EXPECTED,all_six_local_and_staged_match=True,
        accepted_offline_review_zip_sha256=review['sha256'],portable_members_verified=147,
        original_target_source_label_baseline_bindings_unchanged=parent,
        complete_parent_manifest_sha256=sha(stage/'VAST_SHA256_MANIFEST.json'),
        fresh_local_output_namespace_absent=not (stage/'runs/F_large_section_wce_v1').exists(),
        execution_host_runtime_GPU_and_output_status='NOT_VERIFIED_NO_REMOTE_OR_GPU_ACTION',
        GPU_authorization=False,new_training_updates=0,new_generation_calls=0))
    save(OUT/'READ_ME_VI.md','''# Receipt gắn với researcher — chưa được phép chạy GPU

Technical preparation đã được researcher chấp nhận, đúng sáu SHA-256. Không thay
đổi config/code/masks, không đóng gói lại hoặc ghi đè ZIP147file đã review.
SECTION_WCE_RESEARCHER_AUTHORIZATION_PENDING.json gắn scope, dữ liệu, đối chứng,
budget và tiêu chí review vào đúng bindings. Các cờ GPU/train/sanity vẫn false;
các mốc thực thi và xác nhận thời gian dành cho hoàn tất luận văn chưa được tự điền.

Researcher trả lời lịch: “không cần quan tâm tới ngày”. Ghi nguyên phản hồi và để
lịch luận văn do researcher quản lý. Không suy ra một hạn luận văn hay xác nhận
14ngày từ câu này. Policy thời gian trong helper đã được chấp nhận giữ nguyên;
không bỏ check hoặc đổi hashes để vượt qua. Receipt này không vượt qua guard GPU.

Khi có explicit researcher GPU authorization, mới lưu một receipt ĐƯỢC DUYỆT riêng
SECTION_WCE_RESEARCHER_APPROVAL.json, gắn actualUTC/scope theo contract hiện hữu.
Thời gian tạo receipt pending không phải thời gian researcher cấp quyền GPU.
Không tự bật cờ, không mượn approval lịch sử, không chạy gate/training lúc này.

Sau authorization hợp lệ: một gate3successfulupdates, >=15%headroom,456LoRAtensors,
boundedAMP đúng policy; PASS mới freshF5epochs/1250updates. Earlysanity4 giữ nguyên;
48finaloutputs và savedbaseline48, không regeneratebaseline. Giữ5checkpoints và
fullstate evidence. Kiểm tra visualCore Aquarium, zero-caption compliance và
Core/balance/grounding/usability; không chọn theo weightedloss/length.
STOP cho independent quality review trước mọi matchedRO/U continuation.
''')
    save(OUT/'SHA256_MANIFEST.json',dict(files={p.name:sha(p) for p in OUT.iterdir() if p.is_file()}))
    dest=Path.home()/'Downloads/SECTION_WCE_RESEARCHER_AUTHORIZATION_PENDING.json'
    assert not dest.exists();shutil.copyfile(OUT/dest.name,dest)
    assert sha(dest)==sha(OUT/dest.name)
    save(OUT/'DELIVERY_RECEIPT.json',dict(file=str(dest),sha256=sha(dest),bytes=dest.stat().st_size,
        researcher_technical_acceptance_recorded=True,GPU_authorized=False,
        execution_authorized_at_utc=None,deadline_utc=None,status='PREPARED_STOP_NO_GPU'))
    print(json.dumps(read(OUT/'DELIVERY_RECEIPT.json'),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
