"""Bind offline audit/candidate implementation and package for independent review."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import zipfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/stage_i/led_large_contrastive_pair_qualification_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(name,v):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    require_files=[OUT/'CPU_SEQUENTIAL_TESTS.log',OUT/'CPU_RUNNER_TESTS.log']
    for p in require_files:
        raw=p.read_bytes();text=raw.decode('utf-16') if raw.startswith(b'\xff\xfe') else raw.decode('utf-8')
        assert '\nOK' in text.replace('\r',''),str(p)
        (OUT/(p.stem+'_UTF8.txt')).write_text(text,encoding='utf-8')
    now=datetime.now(timezone.utc).isoformat()
    f=read(OUT/'FROZEN_BINDINGS.json');pilot=read(ROOT/'outputs/stage_i/led_large_contrastive_feasibility_v1/portable_review/TRAIN_ONLY_PILOT_MANIFEST.json')
    parent=read(ROOT/'configs/stage_i_led_large_nbest_v2.json')
    original_manifest=read(ROOT/'outputs/stage_h/led_dataset_v1/manifest.json')
    chain={}
    for line in (ROOT/f['train_jsonl']['path']).open(encoding='utf-8'):
        r=json.loads(line)
        if r['business_id'] in pilot['fit']:
            assert sha(ROOT/r['target_artifact'])==r['target_artifact_sha256']
            chain[r['business_id']]=dict(path=r['target_artifact'],sha256=r['target_artifact_sha256'],
                access='bytes hashed only; archived photo judgments not used for source qualification')
    assert len(chain)==8
    write('TEACHER_ARTIFACT_CHAIN.json',chain)
    bindings={}
    for p in list((OUT/'positives').glob('*'))+list((OUT/'pairs').glob('*.json')):
        bindings[p.relative_to(ROOT).as_posix()]=sha(p)
    code=['scripts/led_large_contrastive_sequential_v1.py','scripts/run_led_large_contrastive_candidate_v1.py',
          'scripts/prepare_led_large_contrastive_pairs_v1.py','tests/test_led_large_contrastive_sequential_v1.py',
          'tests/test_led_large_contrastive_runner_v1.py']
    for rel in code:bindings[rel]=sha(ROOT/rel)
    for rel,digest in parent['parent_bindings'].items():assert sha(ROOT/rel)==digest;bindings[rel]=digest
    for rel in ['scripts/led_large_dynamic_amp_v2.py','scripts/led_large_nbest_rng_v2.py',
                'scripts/stage_i_led_large_runtime_v2.py','scripts/led_large_decoder_label_view_v1.py',
                'scripts/stage_hi_led_data.py','scripts/stage_i_paired_pilot.py','scripts/run_stage_i_d050_feasibility_v1.py',
                'scripts/led_large_gradient_evidence_v3.py','outputs/stage_i/led_large_preparation_v1/EXPECTED_EFFECTIVE_GENERATION_CONFIG.json',
                'outputs/stage_i/led_large_nbest_reranking_v1/TRAIN_SOURCE_ONLY_CASES.json']:
        bindings[rel]=sha(ROOT/rel)
    for key in ['train_jsonl','train_npz']:
        meta=f[key];assert sha(ROOT/meta['path'])==meta['sha256'];bindings[meta['path']]=meta['sha256']
    labels=f['corrected_train_labels'];assert sha(ROOT/labels['path'])==labels['sha256'];bindings[labels['path']]=labels['sha256']
    for name,digest in f['tokenizer_manifest']['sha256'].items():
        rel=f['tokenizer_manifest']['relative_directory']+'/'+name;assert sha(ROOT/rel)==digest;bindings[rel]=digest
    cp=ROOT/'outputs/stage_i/led_large_decoder_policy_v2_execution_v1/final_evidence/runs/F_large_decoder_policy_v2/train/checkpoint-750'
    receipt=read(cp/'checkpoint_integrity.json')
    for n,digest in receipt['files'].items():assert sha(cp/n)==digest;bindings[(cp/n).relative_to(ROOT).as_posix()]=digest
    config=dict(experiment='LED_LARGE_CONTRASTIVE_PAIR_QUALIFICATION_V1',GPU_execution_approved=False,training_authorized=False,
        implementation_status='CANDIDATE_FULL_PRETRAINED_CUDA_FP16_NOT_TESTED',
        objective=dict(pair_weight=.1,margin=.1,temperature=1.),files=bindings,runtime_pins=parent['runtime_pins'],
        fit=pilot['fit'],probe=pilot['probe'],pair_ids=[f'P{i:02d}' for i in range(1,17)],
        pilot_updates=16,gate_updates=3,accumulation=8,learning_rate=2e-5,warmup_steps=1,
        optimizer='AdamW torch',betas=[.9,.999],epsilon=1e-8,weight_decay=.01,max_grad_norm=1.,
        sampler='All8 fixed fit businesses in order per update; rotate approved pair IDs per business',
        initialization='Identical original CE3 weights, fresh optimizer/scheduler/scaler per arm and gate/pilot restart',
        adapter=(cp/'adapter_model.safetensors').relative_to(ROOT).as_posix(),adapter_sha256=receipt['files']['adapter_model.safetensors'],
        TRAIN_source_packets='outputs/stage_i/led_large_nbest_reranking_v1/TRAIN_SOURCE_ONLY_CASES.json',
        effective_generation_config='outputs/stage_i/led_large_preparation_v1/EXPECTED_EFFECTIVE_GENERATION_CONFIG.json',
        gate_max_examples=88,pilot_max_examples=192,total_successful_update_ceiling=38,total_generation_calls_ceiling=16,
        diagnosis_generation_calls=0,diagnosis_successful_optimizer_updates=0,diagnosis_model_forward_calls_expected=40,
        generation_num_beams=4,generation_num_return_sequences=1,headroom_min=.15,
        AMP=dict(init_scale=65536,growth_factor=2.,backoff_factor=.5,growth_interval=2000,min_scale=256,max_backoffs_per_job=8),
        production_scope='Matched TRAIN-only pilot remains disabled; future diagnostic-only authorization first',
        checkpoint_policy='Gate step3, pilot steps8/16; no save rotation, best-loss reload or resume',
        frozen_Teacher_targets_modified=False,held_out_access=False,validation_access=False,RO_U_authorized=False,
        all_researcher_adjudications='PENDING',accepted_pairs=0,independent_numerical_review=False,
        execution_root='runs/CF_QUALIFICATION_V1',max_suite_wall_clock_hours=1,
        prior_approval_reusable=False,prior_UTC_deadline_reusable=False)
    write('PRODUCTION_CONFIG_NOT_APPROVED.json',config)
    preflight_files=dict(bindings)
    preflight_files[(OUT/'PRODUCTION_CONFIG_NOT_APPROVED.json').relative_to(ROOT).as_posix()]=sha(OUT/'PRODUCTION_CONFIG_NOT_APPROVED.json')
    write('PREFLIGHT_MANIFEST.json',dict(files=preflight_files,scope='TRAIN-only offline execution inputs and exact candidate code'))
    decision=dict(decider_role='researcher',independently_reviewed=False,
        positives={bid:dict(decision='PENDING',sha256=sha(OUT/'positives'/f'{bid}.json'),notes=None) for bid in pilot['fit']},
        pairs={f'P{i:02d}':dict(decision='PENDING',sha256=sha(OUT/'pairs'/f'P{i:02d}.json'),notes=None) for i in range(1,19)})
    # Submission must contain exactly the16 admitted IDs; rejected controls remain independently reviewable here.
    decision['submission_instruction']='Keep this full audit form unchanged as a template; create a separate signed submission with all8 positives and exactly P01-P16 accepted only if independently justified. P17/P18 are rejected controls, never automatic replacements.'
    write('RESEARCHER_ADJUDICATION_FORM_PENDING.json',decision)
    write('FUTURE_DIAGNOSTIC_AUTHORIZATION_REQUEST_NOT_APPROVED.json',dict(
        experiment=config['experiment'],GPU_execution_approved=False,training_authorized=False,approval_source=None,
        scopes=['diagnose'],arm='pair',successful_optimizer_updates_max=0,generation_calls_max=0,
        full_pretrained_numerical_checks_only=True,automatic_gate_or_pilot=False,
        authorized_utc=None,deadline_utc=None,protected_thesis_days_confirmed=False,
        config_sha256=sha(OUT/'PRODUCTION_CONFIG_NOT_APPROVED.json'),
        runner_sha256=sha(ROOT/'scripts/run_led_large_contrastive_candidate_v1.py'),
        objective_sha256=sha(ROOT/'scripts/led_large_contrastive_sequential_v1.py'),
        preflight_sha256=sha(OUT/'PREFLIGHT_MANIFEST.json'),researcher_decisions_sha256=None,
        prerequisites=['Independent acceptance of8 full positives and16 exact pair candidates',
            'Independent code/CPU-evidence review', 'Explicit fresh diagnostic-only GPU approval and real deadline/reserve confirmation'],
        known_risks=['Full pretrained CUDA score/gradient equivalence and dropout replay unproven',
            'Native FP16 effective-batch finiteness unproven; diagnostic stops on natural overflow without retry',
            'RTX3090 joint-reference/memory headroom unproven; no fallback or source shortening'],
        subsequent_training_requires='New independently accepted numerical receipt and separate researcher authorization; not implied by diagnosis'))
    write('NUMERICAL_EVIDENCE_AND_BLOCKERS.json',dict(
        new_tests=dict(sequential=10,runner_and_Trainer=6,passed=16,failed=0),
        inherited_V1_mathematical_tests=11,inherited_tests_rerun=False,
        pinned_CPU_fixture=dict(implementation='Transformers4.51.3 LED',encoder_layers=12,decoder_layers=12,
            hidden_width=16,attention_heads=1,ffn_width=32,attention_window=1024,vocabulary=50265,
            LoRA_r=8,LoRA_tensors=456,dropout=.1,gradient_checkpointing='non_reentrant',
            input='Exact untruncated frozen Tempo TRAIN source and original/candidate labels',
            joint_vs_sequential_max_gradient_abs_error=0.,lambda0_loss_and_gradients_bit_equal=True,
            source_tokens_padded=4096,pretrained_weights_loaded=False),
        native_Trainer_CPU_accumulation=True,eight_example_CPU_FP32_scaled_arithmetic=True,
        full_pretrained_LED_equivalence='UNTESTED',native_CUDA_FP16_accumulation='UNTESTED',
        native_CUDA_GradScaler_overflow='UNTESTED',GPU_memory='UNMEASURED',
        full_training_checkpoint_recovery='UNTESTED_NO_AUTOMATIC_RESUME_ALLOWED',
        GPU_execution_ready=False,accepted_researcher_pairs=0,
        warning='Reduced-width CPU software tests are not full-checkpoint or CUDA feasibility evidence.',
        GPU_operations=0,generation_calls=0,optimizer_updates=0,
        source_and_token_audit=dict(positives=8,exact_original_target_tokenizations=8,exact_BOS_free_views=8,
            exact_source_id_matches=8,complete_target_claim_coverage=8,all_original_evidence_quotes_relocated=True),
        CPU_logs={p.name:sha(p) for p in require_files}))
    stats=read(OUT/'PAIR_QUALIFICATION_STATUS.json')
    packets=[read(OUT/'positives'/f'{bid}.json') for bid in pilot['fit']]
    cases={p['business_id']:p for p in packets}
    text=['# Eight immutable positive targets — source qualification', '',
        'All assessments below are model-assisted and provisional. Every researcher decision is PENDING. Original Teacher/photo-verdict labels are not independent source certification. No original images, validation or test sources were inspected.', '',
        'Each JSON supplies full model-visible source, exact original positive, original/corrected labels, complete claim-to-quote map and immutable hashes. Review claims were checked as attributed reviewer reports; visual support uses the visible caption only, not historical photo explanations.', '']
    for p in packets:
        text += [f"## {p['name']} — {p['business_id']}", '',
            f"Original target SHA256: `{p['positive_utf8_sha256']}`. Researcher status: **PENDING**.", '',
            '```text',p['positive_text'],'```','',
            f"Audit: {len([x for x in p['claim_audit'] if x['domain']=='review'])} review claim groups; {len([x for x in p['claim_audit'] if x['domain']=='visual'])} visual atomic claims; {len(p['captions'])} visible captions.",
            'Model assessment: source-supported as reviewer-attributed synthesis; positive/negative sides are retained where stated, and no cross-caption relationship is asserted. This is not a human approval or a finding that every possible Core detail is covered.', '']
        text.extend('- '+w for w in p['warnings'])
        if not p['captions']:text+=['- The original visual body is empty: zero-caption compliant.']
        if p['business_id']=='5uhG34VRsD0A6lOY8F41Dg':text+=['- Favorable fry comments also exist; the target attributes the unfavorable comments to several reviewers rather than claiming unanimity. Assess the adequacy of that emphasis independently.']
        if p['business_id']=='guxzC4GOY1MYY5KqfItwOQ':text+=['- The serious cleanliness allegation remains explicitly attributed to one review; the source report is not independently verified reality.']
        text+=['']
    (OUT/'EIGHT_POSITIVES_SOURCE_AUDIT.md').write_text('\n'.join(text),encoding='utf-8')
    pairs=[read(OUT/'pairs'/f'P{i:02d}.json') for i in range(1,19)]
    text=['# Proposed edits and admissibility audit','',
        '18 proposed edits, not18 accepted negatives. Sixteen are provisional candidates:6 Core omissions,6 same-aspect disagreement losses,4 grounding errors. Two shorter-but-useful controls are recommended for rejection. **Accepted pairs:0; all researcher fields PENDING.**', '',
        'Core omission is an adequacy judgment, not automatically factual error. Grounding edits are unsupported by the visible source, not necessarily false in the real world. Source quotations and exact character offsets appear in each pair JSON.', '']
    for pair in pairs:
        text += [f"## {pair['pair_id']} — {cases[pair['business_id']]['name']} — {pair['category']}",'',
            f"Model recommendation: {pair['model_assisted_recommendation']}. Researcher: **PENDING**.",
            f"Property: {pair['negative_property']}. Tokens: {pair['diagnostics']['positive_tokens']} → {pair['diagnostics']['negative_tokens']}.",
            '',pair['source_reason'],'','Removed:', '```text',pair['minimal_edit']['removed'],'```','Inserted:', '```text',pair['minimal_edit']['inserted'],'```','']
        if pair['evidence']:
            text += [f"- {x['domain']} {x['marker']}: {x['quote']}" for x in pair['evidence']]
        else:text+=['- Filtered captions are empty; the exact source packet contains no visual evidence.']
        text+=['']
    text+=['## Remaining risks','',
        '- All pairs and this first audit share an author. Independent acceptance must not be inferred from the JSON recommendation.',
        '- Core deletions are shorter; visual insertions are longer. Consensus edits often contain words such as consistently, unanimously or without. Length and surface-word shortcuts are plausible.',
        '- P17/P18 explicitly demonstrate that shorter can remain useful; they are not training negatives.',
        '- Fixed positives may omit other useful Core details. They may not be rewritten; this objective cannot supply absent factual content.',
        '- If any complete positive fails, stop. If the independent decisions do not yield exactly16 accepted P01-P16 across8 fit cases with locked quotas, stop without replacing cases or adding another edit search.','']
    (OUT/'PAIR_ADMISSIBILITY_REVIEW.md').write_text('\n'.join(text),encoding='utf-8')
    (OUT/'IMPLEMENTATION_AND_NUMERICAL_LIMITS.md').write_text('''# Candidate implementation and numerical limits

The objective remains CE +0.1 softplus(0.1-s_positive+s_negative), temperature1. Native model CE supplies mean valid-token log probability, with independent once-shifted decoder prefixes. EOS and headers are included, padding ignored. Weight0 delegates the existing model CE exactly.

The new autograd function scouts both sequences without storing graphs, captures Python/NumPy/CPU/CUDA RNG separately, and recomputes one differentiable graph at a time. It returns the exact sequential gradient identity to the outer native GradScaler. The upstream effective-batch scaling is retained. It logs the true objective, not the linear surrogate. Autocast cast caching is disabled inside scout/replay forwards to prevent no_grad casts contaminating replay. State mismatch is a hard failure.

10 sequential CPU tests and6 authorization/Trainer CPU tests passed. The reduced-width LED fixture uses all12 encoder and12 decoder layers, the original1024 attention window,456 LoRA tensors and a real untruncated TRAIN batch. Its maximum joint-versus-sequential gradient error was0.0. Hidden width16/head1/FFN32 and random weights are test fixtures, not the approved1024-wide pretrained model. No scientific backbone or hyperparameter was changed by these tests. The8-example/native Trainer test verifies accumulation exactly once on CPU; it is not native CUDA FP16 evidence.

The production candidate reuses the existing Trainer, bounded native-scaler recovery implementation, memory sampler, V2 technical-sanity screen, V2 eval RNG report, and full optimizer/adapter continuity context. It has fixed namespaces for one diagnostic, two gates and two pilot arms, no resume, no save rotation, no validation loader and no automatic follow-on. GPU modes require exact bindings, fresh approval/deadline/reserve confirmation, and all8 positive/16 pair researcher acceptances. Training additionally requires independently accepted full-model numerical receipts from the same GPU UUID/driver and current code. Pilot additionally requires the same-arm approved gate receipt. The whole fixed suite has a conservative one-hour wall deadline, bounded by the48-hour researcher deadline; expired windows stop, not silently extend.

Full pretrained LED/CUDA dropout replay, gradient equivalence, native FP16 effective-batch accumulation, scaler overflow behavior and measured15% headroom remain UNTESTED. Checkpoint write/restore and the complete GPU runner path remain unexecuted; there is no automatic checkpoint recovery. These are blockers to calling the candidate GPU-qualified. CPU success must not be substituted for them.

The proposed diagnostic mode performs no successful model optimizer step and no generation. It compares existing CE and lambda0, joint versus sequential gradients on an untruncated short-target TRAIN case, then an eight-example native-FP16 window and an explicitly injected native overflow skip. It stops on natural nonfinite gradients or memory/replay failure without retry. Injected overflow is a separate software test, not a measured natural numerical failure. All of this requires a new explicit diagnostic-only GPU approval; none has been granted.
''',encoding='utf-8')
    (OUT/'RESOURCE_AND_EXECUTION_REQUEST.md').write_text('''# Resource estimate, stop conditions and future authorization request

Current status: OFFLINE ONLY.16 candidate pairs are pending; accepted count0. No GPU, decoding or optimizer update has run. Original CE3 remains the Filtered checkpoint.

Historical full CE training conservatively peaked at16.47GiB with30.10%headroom. Inference's roughly4.6GiB is not a training-memory estimate. Sequential recomputation retains one activation graph plus small gradient/RNG buffers, but actual full-model memory and allocator behavior remain unmeasured. The joint-graph numerical reference might not fit; an OOM or <15%headroom stops without shortening sources or changing precision/batch.

Two scouts plus two replay/backward contributions suggest roughly3–5 times CE's per-example cost for the contrastive arm. This is a planning range, not a GPU benchmark. Use15–45minutes for the matched small pilot, plus setup/diagnostic overhead, under the locked one-hour suite wall stop and fresh48-hour researcher window. If setup or independent review consumes the suite window, it expires: no automatic extension. No rental-price estimate is claimed.

Requested next GPU scope, only after independent pair and code review: ONE diagnostic-only attempt, zero generation calls and zero successful model optimizer updates. Fresh actual UTC approval/deadline and protected evaluation-period confirmation are required. The attached request is NOT APPROVED. After the diagnostic, stop for independent numerical review. No conditional training is inferred from diagnostic approval.

Later, separately approved matched pilot: both arms start from identical original CE3 adapter weights with fresh matched optimizer/scheduler/scaler. LR2e-5, one warmup update, linear16-update schedule, physical batch1/accum8, same source order, original targets, original beam4/top1 decoding. Three successful gate updates per arm are discarded before each16-update pilot. Four fixed TRAIN probes after each gate/pilot produce16 outputs total;38 successful updates include the gates. Save gate3/pilot8,16. No best-loss selection or silent resume. These probes are already seen TRAIN businesses, not generalization evidence;3/4 preference is only a provisional resource-allocation signal.

Hard stops: altered source/target/code, missing independent decisions, absent/expired authorization, wrong GPU/runtime, replay mismatch, nonfinite forward loss, missing LoRA gradients, exhausted bounded native AMP recovery, <15%memory headroom, BOS collapse, blank/special-token-dominated generation or state-continuity failure. Isolated BOS and cap hits are warnings under the unchanged V2 screen. Preserve failure evidence; do not automatically retry.

CPU commands (no GPU operations):

    python scripts/run_led_large_contrastive_candidate_v1.py preflight
    python -m unittest discover -s tests -p test_led_large_contrastive_sequential_v1.py -v
    python -m unittest discover -s tests -p test_led_large_contrastive_runner_v1.py -v

Future command template, deliberately blocked until explicit signed decisions and diagnostic approval exist:

    python scripts/run_led_large_contrastive_candidate_v1.py diagnose --arm pair --execute --approval APPROVED_DIAGNOSTIC.json --decisions RESEARCHER_DECISIONS_ACCEPTED.json --output runs/CF_QUALIFICATION_V1/diagnose_pair

Do not run it now. Gate/pilot approval is separately required after numerical evidence is independently accepted. There is no training authorization in this package.
''',encoding='utf-8')
    (OUT/'MATCHED_FAMILY_READINESS.md').write_text('''# Matched-family readiness preserved

No model comparison is replaced by this preparation. Keep Original CE DecoderPolicyV2 epoch3 as the current Filtered checkpoint; preserve all its five full-state checkpoints and existing outputs. The historical matched LED-base RO/U/F family remains the primary established thesis baseline until a complete accepted large family exists.

If contrastive qualification/numerical feasibility fails, or source-grounded quality does not justify continuation, retain CE3 and prioritize separately authorized matched original-CE LED-large RO/U using the same official backbone, corrected label policy, Expanded-LoRA, frozen condition-specific sources/targets and common checkpoint rule. Shuffled reuses the exact matched F adapter and its fixed input protocol; it is not retrained.

Only an independently supported contrastive quality GO could justify a separately approved matched contrastive RO/U/F protocol. That would require equivalent condition-aware pair policy, adjudication budget, CE warm-start rule, continuation schedule and checkpoint selection across all conditions. RO has no visual evidence section in its original target format: the current Filtered-only pair constructor is not automatically RO/U ready. Do not transplant only an improved F model into the caption-filtering comparison or attribute objective effects to filtering. No RO/U training or Shuffled inference is authorized now. D051 strict replay FAIL and historical stopped experiments remain unchanged.
''',encoding='utf-8')
    (OUT/'README_FIRST_VI.md').write_text('''# Contrastive V1 — chuẩn bị offline, chưa được chạy GPU

Đã lấy nguyên văn8 Teacher target Filtered gốc, đối chiếu source/token IDs và corrected BOS-free views. Audit nguồn là đánh giá hỗ trợ bởi mô hình; mọi quyết định researcher vẫn PENDING.

Có18 đề xuất chỉnh sửa:16 ứng viên(6mất Core,6mất cân bằng cùng khía cạnh,4grounding),2ca rút ngắn nhưng vẫn hữu ích được khuyến nghị loại. **Số cặp đã chấp nhận:0.** Không sửa positive, đổi doanh nghiệp hay tạo thêm corpus.

16test CPU mới PASS:10cho objective/replay và6cho authorization/Trainer accumulation. Fixture LED giữ12+12layers,456LoRA tensors,source TRAIN nguyên độ dài nhưng giảm chiều ẩn xuống16 và dùng trọng số ngẫu nhiên. Kết quả này không chứng minh checkpoint lớn thật chạy đúng với CUDA/FP16.

Đọc EIGHT_POSITIVES_SOURCE_AUDIT.md, PAIR_ADMISSIBILITY_REVIEW.md và từng JSON để duyệt. Điền quyết định bằng một file researcher riêng; không tự đổi PENDING thành ACCEPT theo verdict của mô hình. Nếu bất kỳ positive không đạt, hoặc không đủ16cặp thực sự được duyệt theo quota, dừng.

Code objective và runner là candidate chưa được GPU-qualify. Full pretrained/CUDA/FP16/memory còn UNTESTED. Request đi kèm chỉ đề nghị một lần kiểm chứng số học GPU sau khi researcher duyệt; hiện request KHÔNG ĐƯỢC PHÊ DUYỆT. Training cần approval riêng sau review số học. Không chạy lệnh GPU trong gói này lúc này.

Giữ nguyên CE epoch3 làm checkpoint Filtered hiện tại. Không validation/test access, Teacher calls, RO/U training, beam/sampling search hay reranker PhaseB/C.
''',encoding='utf-8')
    # A compact portable copy. Original files and raw CPU logs are retained in OUT.
    port=OUT/'portable_review';assert not port.exists();port.mkdir()
    names=['README_FIRST_VI.md','EIGHT_POSITIVES_SOURCE_AUDIT.md','PAIR_ADMISSIBILITY_REVIEW.md',
        'IMPLEMENTATION_AND_NUMERICAL_LIMITS.md','RESOURCE_AND_EXECUTION_REQUEST.md','MATCHED_FAMILY_READINESS.md',
        'POSITIVE_AUDIT_SUMMARY.json','PAIR_QUALIFICATION_STATUS.json','FROZEN_BINDINGS.json','TEACHER_ARTIFACT_CHAIN.json',
        'PRODUCTION_CONFIG_NOT_APPROVED.json','PREFLIGHT_MANIFEST.json','RESEARCHER_ADJUDICATION_FORM_PENDING.json',
        'FUTURE_DIAGNOSTIC_AUTHORIZATION_REQUEST_NOT_APPROVED.json','NUMERICAL_EVIDENCE_AND_BLOCKERS.json',
        'CPU_SEQUENTIAL_TESTS_UTF8.txt','CPU_RUNNER_TESTS_UTF8.txt']
    for name in names:(port/name).write_bytes((OUT/name).read_bytes())
    for sub in ['positives','pairs']:
        (port/sub).mkdir()
        for p in (OUT/sub).iterdir():
            if p.is_file():(port/sub/p.name).write_bytes(p.read_bytes())
    (port/'code').mkdir()
    for rel in code+['scripts/finalize_led_large_contrastive_pair_review_v1.py']:
        (port/'code'/Path(rel).name).write_bytes((ROOT/rel).read_bytes())
    prior=ROOT/'outputs/cloud_handoff/LED_LARGE_CONTRASTIVE_FEASIBILITY_V1_OFFLINE_REVIEW.zip'
    with zipfile.ZipFile(prior) as z:
        m=json.loads(z.read('SHA256_MANIFEST.json'))['files']
        assert len(m)==17 and all(hashlib.sha256(z.read(k)).hexdigest()==v for k,v in m.items())
    write('PREVIOUS_V1_CHAIN_PRESERVED.json',dict(path=prior.relative_to(ROOT).as_posix(),sha256=sha(prior),entries_verified=17))
    (port/'PREVIOUS_V1_CHAIN_PRESERVED.json').write_bytes((OUT/'PREVIOUS_V1_CHAIN_PRESERVED.json').read_bytes())
    payload={p.relative_to(port).as_posix():sha(p) for p in sorted(port.rglob('*')) if p.is_file()}
    (port/'SHA256_MANIFEST.json').write_text(json.dumps(dict(files=payload,created_utc=now),indent=2)+'\n',encoding='utf-8')
    archive=ROOT/'outputs/cloud_handoff/LED_LARGE_CONTRASTIVE_PAIR_QUALIFICATION_V1_OFFLINE_REVIEW.zip'
    assert not archive.exists()
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(port.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(port).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for k,v in payload.items():assert hashlib.sha256(z.read(k)).hexdigest()==v
    archive.with_suffix('.zip.sha256').write_text(sha(archive)+'  '+archive.name+'\n',encoding='ascii')
    report=dict(archive=archive.relative_to(ROOT).as_posix(),sha256=sha(archive),bytes=archive.stat().st_size,
        manifest_entries=len(payload),positives=8,proposed_edits=18,provisional_candidates=16,researcher_accepted_pairs=0,
        new_CPU_tests_passed=16,GPU_operations=0,generation_calls=0,optimizer_updates=0,
        full_pretrained_CUDA_FP16_verified=False,status='OFFLINE_PREPARED_STOP_FOR_INDEPENDENT_REVIEW',
        execution_bindings=dict(config_sha256=sha(OUT/'PRODUCTION_CONFIG_NOT_APPROVED.json'),
            runner_sha256=sha(ROOT/'scripts/run_led_large_contrastive_candidate_v1.py'),
            objective_sha256=sha(ROOT/'scripts/led_large_contrastive_sequential_v1.py'),
            preflight_sha256=sha(OUT/'PREFLIGHT_MANIFEST.json')))
    write('DELIVERY_RECEIPT.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
