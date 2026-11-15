"""Readiness report/portable offline patch; never executes training or generation."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,zipfile,shutil

ROOT=Path(__file__).resolve().parents[1]
PREP=ROOT/'outputs/stage_i/led_large_decoder_policy_v1'
PARENT=ROOT/'outputs/cloud_handoff/LED_LARGE_VAST_D059_F_V1'
STAGE=ROOT/'outputs/cloud_handoff/LED_LARGE_DECODER_POLICY_V1_WORKSPACE'
DEST=ROOT/'outputs/cloud_handoff/LED_LARGE_DECODER_POLICY_V1_OFFLINE_REVIEW'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    result=read(PREP/'FINAL_VALIDATION.json');assert result['status']=='PASS' and result['tests']==19
    binding=read(PREP/'EXECUTION_BINDINGS.json');assert result['bindings']==binding
    manifest=read(PREP/'PREFLIGHT_MANIFEST.json')['files']
    for rel,digest in manifest.items():
        original=(PARENT/rel) if rel=='VAST_SHA256_MANIFEST.json' else (ROOT/rel)
        assert sha(original)==sha(STAGE/rel)==digest,rel
    old_stage_h=read(ROOT/'outputs/stage_h/led_dataset_v1/manifest.json')
    audited=read(PREP/'LABEL_AUDIT.json')
    assert audited['rows_total']==6600
    unchanged={}
    for name,info in old_stage_h['datasets'].items():
        assert sha(ROOT/info['path'])==info['sha256']==audited['datasets'][name]['original_npz_sha256']
        assert sha(ROOT/info['stage_g_jsonl'])==info['stage_g_jsonl_sha256']==audited['datasets'][name]['original_jsonl_sha256']
        unchanged[name]=dict(NPZ='UNCHANGED',Teacher_source_JSONL='UNCHANGED',rows=info['rows'])
    tm=read(ROOT/old_stage_h['tokenizer_manifest'])
    for name,digest in tm['sha256'].items():assert sha(ROOT/tm['relative_directory']/name)==digest
    write(PREP/'FINAL_SCIENTIFIC_INTEGRITY.json',dict(status='PASS',datasets=unchanged,
        Stage_H_manifest_sha256=sha(ROOT/'outputs/stage_h/led_dataset_v1/manifest.json'),
        Stage_H_status_unchanged=old_stage_h['status'],tokenizer_assets_unchanged=True,
        parent_payload_files=read(PREP/'PREFLIGHT_MANIFEST.json')['parent_payload_files'],
        new_preflight_files=len(manifest),historical_checkpoints_not_written=True,
        held_out_contents_accessed=False,Teacher_calls=0,new_pretrained_forward_calls=0,
        new_generate_calls=0,new_backward_calls=0,new_optimizer_updates=0))
    report=f'''# LED-large decoder-label policy: offline readiness — 22/10/2026 ICT

**OFFLINE READY FOR RESEARCHER REVIEW. EXECUTION NOT APPROVED.**

One scientific correction is proposed: remove only tokenizer-added initial BOS0 from the training/validation label view, retaining all content tokens and finalEOS2. The pinned checkpoint is official allenai/led-large-16384 revision cd59d11c3528415b7dda4dfc95cc8f138aceda2e. No decoding, architecture, LoRA rank, optimizer, numerical policy, source or Teacher change is proposed.

## A–C. Verified pinned runtime and label semantics

Transformers4.51.3 LED shifts labels once when decoder inputs are absent. CE compares each logit position with the unshifted label; it does not shift the targets a second time. Padding remains ignored as-100. Pure runtime helper tests confirm generation without a decoder prompt begins with decoder-start2 only; forced_bos_token_id remains null. Cache is disabled for labeled training/validation and enabled by the unchanged generation config for eval-mode decoding. Pinned LED/generation/Trainer/Accelerator Python source hashes are bound and checked by the new preflight.

All6 frozen train/validation datasets were audited locally: RO/U/F each2000train+200validation =6600 labels. Every old sequence equals `[BOS]+tokenizer.encode(Teacher_text,add_special_tokens=False)+[EOS]`. Each has exactly one BOS and one terminalEOS, no interior BOS/EOS/PAD/MASK, zero unknown tokens. Removing BOS yields exactly the unchanged content encoding plusEOS and decodes byte-for-text identically to the original Teacher string. No first word, header fragment, punctuation, newline or EOS is removed. Original source IDs are retained from the byte-verified original NPZ; no source reconstruction/compression is introduced.

The initial BOS is therefore a tokenizer wrapper rather than business evidence. Dropping it from the LABEL VIEW is semantically justified for the proposed decoder policy. It is not a claim that the old StageH data are corrupted or that BOS removal is proven to fix all generation quality. Source/encoder BOS0 and its global attention stay intact. The historical StageH status remains `{old_stage_h['status']}`; this new proposed policy has its own receipt and is not retrospectively marked approved.

## D–E. Exact, non-destructive correction

Two new F-only label NPZs contain business IDs, label offsets and corrected labels, without any rewritten input IDs. The versioned dataset wrapper pairs them with original sources and verifies every row equals old_labels[1:]. The generic transformation is applicable consistently to RO/U later, but those arms are not prepared for execution or authorized here.

The new collator reconstructs `[0]+view_labels` only as an internal input to the original trusted padding routine, then returns its label tensor shifted left by exactly one and pads the vacated final slot with-100. Thus source tensors, attention/global masks, source padding and even original label padding SHAPES remain identical. Model loss targets contain `[content...,EOS]`; model-auto-shift yields `[EOS,first_content,...]`. It does not merely mask a BOS target while leaving BOS in the decoder prefix.

`EXACT_PROPOSED_PATCH.diff` shows the complete old-to-versioned helper/runner/reporter/config diff and new label/sanity modules. No old code or scientific file is overwritten. The new fixed run namespace is `runs/F_large_decoder_policy_v1`; resume or import of old large/base/gate adapter state is forbidden. Parent files and new overlay files are separately bound.

19 focused tests PASS, including raw-label rejection cases, full2200F-view loading, unchanged source/masks/padding, exact pinned shift, synthetic CE alignment and ignored padding, generation-start helper/configuration, scope/hash/deadline rejection, old approval rejection, early-sanity failure propagation and mocked mode/cache/RNG restoration. Tests use tensors/helpers/mocks; no pretrained model forward, generation, backward or optimizer runs. Full portable-root CPU preflight PASS, including all9961 parent payload bindings and{len(manifest)} new manifest entries. Final scientific integrity recheck PASS on all6 original NPZ/JSONL datasets and tokenizer assets.

## F. Competing defects and limits

- Adapter loading: old actual run/earlier CPU evidence verifies saved tensor equality; all5 large adapters are finite/nonzero. Original loader requires228modules/456tensors/4915200trainable parameters, active LoRA scaling2 and frozen backbone/hash integrity. No evidence of an entirely untrained or skipped adapter was found.
- Generation/mode: actual old eval config matches the pinned expected config; no forced BOS, token suppression or new repetition constraints. Runner sets eval mode and omits labels/decoder prompts from generate inputs. PEFT's LoRA path delegates labels to the base seq2seq model, not a prompt-learning branch.
- Masks/shift: encoder global BOS anchor is supplied and verified. No double label shift or PAD-vs-content error was identified. The prior tests correctly checked implementation of the old policy but did not establish checkpoint appropriateness.
- Cache: static behavior is consistent with labeled/no-cache and eval/generation/cache. No new cache-on/off model parity experiment was performed; a runtime-specific defect is not ruled out solely by static inspection.
- AMP: unchanged65536 initial scale with bounded native backoff. Historical D059 saved scalers also remained65536. Prior FP32CPU prefix probes reproduce the large-specific initial-position anomaly. These facts weaken a scale-only explanation; they do not prove every numerical path is ideal.
- Learning capacity/hyperparameters and lexical/semantic repetition remain possible additional issues. They are not searched or modified here.

## G. Evidence calibration

CONFIRMED: exact old label/decoder prefix, all6600 text/token identities, old48 raw BOS-loop prefixes, null forced BOS, versioned view/collator alignment, preserved frozen artifacts and passed offline checks.

STRONGLY SUPPORTED HYPOTHESIS: interaction of this LED-large checkpoint with the old EOS/BOS prefix contributes to collapse. Earlier FP32CPU probes on2synthetic sources found near-identical first-two-position logits for pretrained/epoch3/5 large, while D059baseepoch3 distinguished them. This is prior diagnostic evidence, not a new model call in this preparation. Historical maintainer/user discussion of a related workaround is a supporting lead: https://github.com/huggingface/transformers/issues/18190 .

UNPROVEN: causal recovery under the corrected view; full real-case FP16 behavior; elimination of phrase/sentence/semantic repetition; improved Core coverage, opinion balance, factual/visual grounding and usability. No scientific GO, researcher labels or GPU-readiness PASS is inferred from offline tests or loss.

## H. ONE proposed bounded fresh F verification

Future approval must bind this exact new config/runner/preflight and the decoder-policy amendment, carry actual approval UTC, a maximum48hour execution-and-review window plus researcher absolute stop, and confirm final-phase reserve. No historical deadline is changed. Actual defense date is not supplied here, so reserve cannot be independently certified from this package.

1. Verify all parent/new hashes, pinned runtime source/dependency versions and healthy idle RTX3090/24GB. New approval is required; the embedded template is FALSE and old execution approvals are rejected.
2. Run one fresh corrected-label feasibility gate:3successful updates/24nominal, at most88actual microbatches under the unchanged8-overflow bound;2existing TRAIN-input beam4 memory traces, no replay updates. Require all original gradient/AMP/frozen-backbone/save-restore checks and >=15% measured headroom. Gate states are discarded.
3. Start ONE fresh F2000/200 run from pinned original pretrained weights and fresh named-seeded LoRA. Same228modules/4915200params, AdamW2e-4, batch1/accum8, seed42/order, FP32storage/FP16AMP, r8/alpha16/dropout0, linear schedule/warmup63, maximum5epochs/1250successful updates. No inherited optimizer/scaler/RNG/adapters. Save all5epoch checkpoints.
4. After epoch1 checkpoint250, generate exactly4predeclared development sanity outputs: first2random cases plus zero-caption targeted case and KFC targeted case. No case replacement. Require nonempty text, no BOS after decoder start, >=16ordinary tokens and naturalEOS before768cap on all4. Raw outputs are kept. Preserve/verify adapter state and restore all module modes, cache setting and RNG before continuing. Failure stops this run without retry/fallback/setting changes. It is a technical generation-verification failure, not proof of inherent large incapability or a substantive quality NO-GO after250updates.
5. Only after technical PASS continue the SAME fresh run to1250. Generate the existing24random20/targeted4 cases at epoch3 and5:48final outputs. No epoch1 quality checkpoint selection. Total maximum54generation calls:2gate technical+4early sanity+48final. This transparently adds4early validation outputs to the prior48-output final review scope and requires explicit approval; it is not hidden inside the old budget.

## I. Quality decision and matched-family rule

Early sanity is not quality GO. Review actual final outputs against D059Expandedepoch3 as the principal practical benchmark, also historicalF/D058B/D059epoch5. Keep random20 and targeted4 separate; do not credit empty output as reduced repetition. Require meaningful business-specific Core coverage, balanced opinion synthesis, grounded useful visual details, zero-caption compliance, semantic repetition reduction and usable completion without severe regressions. Corrected-label loss has a changed token denominator/alignment; raw loss values alone are not directly interchangeable with the old-label run.

AI assessments remain provisional. Only researcher-confirmed substantive success could support separately approved matched corrected-policy largeRO/U/F with identical initialization/architecture/optimization and a single common epoch3or5 selection rule. Shuffled reuses the exact matchedF checkpoint. Do not replace onlyF in the main caption-effect comparison. Historical matched LED-baseRO/U/F remain primary until a complete matched large family is accepted. Preserve D051FAIL, all previous runs and stopped Unlikelihood evidence. No further optimization experiment is initiated by this proposal.

## Exact bindings

```json
{json.dumps(binding,indent=2)}
```

The approval template is deliberately NOT approved and contains no fabricated approval timestamp. This package requests review and approval of the scope, not retrospective authorization.
'''
    write(PREP/'READINESS_AND_AUTHORIZATION_PLAN.md',report)
    write(ROOT/'thesis_sdd_docs/LED_LARGE_DECODER_POLICY_CORRECTION_V1.md',report)
    DEST.mkdir(exist_ok=False,parents=True)
    oldfiles=read(PARENT/'VAST_SHA256_MANIFEST.json')['files']
    overlay=list(manifest)
    overlay.remove('VAST_SHA256_MANIFEST.json')
    overlay += ['outputs/stage_i/led_large_decoder_policy_v1/'+n for n in ('PREFLIGHT_MANIFEST.json','EXECUTION_BINDINGS.json','AUTHORIZATION_TEMPLATE_NOT_APPROVED.json')]
    assert not (set(overlay)&set(oldfiles)),'Overlay must not overwrite a historical file'
    for rel in overlay:
        dest=DEST/'overlay'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,dest)
    for name in ('READINESS_AND_AUTHORIZATION_PLAN.md','EXACT_PROPOSED_PATCH.diff','EXECUTION_BINDINGS.json','LABEL_AUDIT.json',
                 'FINAL_SCIENTIFIC_INTEGRITY.json','FINAL_VALIDATION.json','FOCUSED_TESTS_FINAL.log','CPU_PREFLIGHT_FINAL.log','PINNED_RUNTIME_CONTRACT.json','AUTHORIZATION_TEMPLATE_NOT_APPROVED.json'):
        shutil.copyfile(PREP/name,DEST/name)
    prior=ROOT/'outputs/cloud_handoff/LED_LARGE_VAST_D059_F_GPT_REVIEW_V2.zip'
    with zipfile.ZipFile(prior) as z:
        for name in z.namelist():
            if name.startswith(('original_review/','bos_diagnosis/')) and not name.endswith('/'):
                p=DEST/'prior_completed_run_evidence'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(name))
    write(DEST/'README_FIRST.md','''# LED-large decoder-policy V1: OFFLINE INVESTIGATION/PREPARATION ONLY

Read READINESS_AND_AUTHORIZATION_PLAN.md, EXACT_PROPOSED_PATCH.diff, LABEL_AUDIT.json, FINAL_VALIDATION.json and EXECUTION_BINDINGS.json. The actual code and two materialized F label-only views are under overlay/ using the original workspace-relative paths. No overlay path replaces any file in the preserved parent package. The parent full-data/model ZIP is NOT duplicated here.

GPU execution is NOT approved. AUTHORIZATION_TEMPLATE_NOT_APPROVED.json is false and has no approval UTC. Do not run either the old parent training launcher or the new proposed launcher during review. Prior approvals concern historical runs only and are rejected by the corrected-policy runner. This preparation performed no pretrained model forward, generate, backward or optimizer update.

All6train/validation RO/U/F sets were audited locally (6600labels). Only F label views are materialized for the proposed experiment. The portable parent contains F inputs only; do not rerun the local all-condition preparation builder on a F-only cloud workspace. Parent original StageH files, Teacher texts, tokenizer, sources, references, model weights and checkpoints remain unchanged.

The future proposal is one fresh F run, unchanged scientific/numerical settings except decoder-label representation. It adds an explicit4-output epoch1 technical sanity screen to the existing48final validation outputs, plus2gate memory traces, maximum54generation calls. This is a NEW proposed budget requiring explicit hash-bound approval, not permission to rerun the old experiment. See the plan for fail-stop, quality and matched-family rules.

Prior raw48outputs, source/captions/photos, approved references and old CPU diagnosis are included under prior_completed_run_evidence/ for independent assessment. These records are historical and unmodified; their true old authorization fields confer no new permission. Human quality decision remains pending. No held-out content, credentials, full model or optimizer binaries are included.
''')
    write(DEST/'GPT_REVIEW_REQUEST.md','''Independently review this OFFLINE decoder-policy correction; respond in Vietnamese.

1. Verify against exact pinned source/label sequences that stripping only initialBOS preserves all content/EOS and makes loss/teacher-forcing/generation-start alignment correct. Challenge the proposed bridge collator: source/masks/padded shapes must remain unchanged, loss labels must really excludeBOS, and the model must shift only once.
2. Distinguish confirmed implementation/evidence, the strongly supported BOS-collapse hypothesis, and unproven causal/substantive recovery. Do not assume the historical issue's workaround solves every repetition/grounding defect. Review competing adapter/config/mode/mask/cache/loading concerns with evidence and limits.
3. Review the proposed fresh F-only gate/training/early-sanity protocol and exactly54generation-call ceiling. Confirm4early sanity outputs do not imply quality GO, do not change the RNG/optimizer/schedule/data order or create a hidden extra experiment. Old states cannot initialize the new run.
4. Check that frozen StageH/Teacher/source/reference/tokenizer artifacts and historical decisions remain intact. The StageH pending status is historical; new policy is not approved yet. All-condition matched-label consistency is required before replacing the primary baseline family.
5. Recommend approve/revise/reject this PREPARATION with exact reasons. Do not execute code, authorize training, create researcher-confirmed labels, access held-out data, propose a search or treat loss/structure/BOS freedom alone as scientific success. State remaining uncertainty and the minimum needed approval scope.
''')
    write(DEST/'SHA256_MANIFEST.json',dict(files={p.relative_to(DEST).as_posix():sha(p) for p in sorted(DEST.rglob('*')) if p.is_file()}))
    archive=DEST.with_suffix('.zip')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(DEST.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(DEST).as_posix())
    with zipfile.ZipFile(archive) as z:
        m=json.loads(z.read('SHA256_MANIFEST.json'))['files']
        assert len(z.namelist())==len(set(z.namelist()))==len(m)+1
        for rel,digest in m.items():assert hashlib.sha256(z.read(rel)).hexdigest()==digest,rel
        for rel in overlay:assert z.read('overlay/'+rel)==(ROOT/rel).read_bytes()
        for ep in (3,5):assert len(json.loads(z.read(f'prior_completed_run_evidence/original_review/predictions/epoch{ep}.json')))==24
        import re
        html=z.read('prior_completed_run_evidence/original_review/RESULTS_REVIEW.html').decode('utf-8')
        for ref in re.findall(r'<img src="([^"]+)"',html):assert 'prior_completed_run_evidence/original_review/'+ref in z.namelist()
    receipt=dict(status='OFFLINE_PREPARATION_COMPLETE_GPU_NOT_APPROVED',archive=str(archive),bytes=archive.stat().st_size,
        sha256=sha(archive),members=len(m)+1,archive_SHA256_CRC='PASS',overlay_overwrites_parent_files=False,
        audited_labels=6600,focused_tests=19,preflight='PASS',bindings=binding,
        new_pretrained_forward_calls=0,new_generation_calls=0,new_optimizer_updates=0)
    write(PREP/'DELIVERY_RECEIPT.json',receipt)
    archive.with_suffix('.zip.sha256').write_text(receipt['sha256']+'  '+archive.name+'\n',encoding='utf-8')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
