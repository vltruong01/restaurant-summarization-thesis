"""Package the minimal offline revision and its existing validation evidence."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / 'outputs/stage_i/led_large_decoder_policy_v2'
V1 = ROOT / 'outputs/stage_i/led_large_decoder_policy_v1'
STAGE = ROOT / 'outputs/cloud_handoff/LED_LARGE_DECODER_POLICY_V2_WORKSPACE'
PREVIOUS = ROOT / 'outputs/cloud_handoff/LED_LARGE_DECODER_POLICY_V1_OFFLINE_REVIEW'
DEST = ROOT / 'outputs/cloud_handoff/LED_LARGE_DECODER_POLICY_V2_OFFLINE_REVIEW'


def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))


def put(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(value if isinstance(value, str) else json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


def main():
    checks = read(PREP/'CHECK_RESULTS.json')
    assert set(checks) == {'FOCUSED_TESTS.log','CPU_PREFLIGHT.log'}
    assert all(v['returncode'] == 0 and sha(PREP/k) == v['sha256'] for k,v in checks.items())
    tests = int(re.search(r'Ran (\d+) tests', (PREP/'FOCUSED_TESTS.log').read_text(encoding='utf-8'))[1])
    assert tests == 35
    binding = read(PREP/'EXECUTION_BINDINGS.json')
    manifest = read(PREP/'PREFLIGHT_MANIFEST.json')
    for rel, digest in manifest['files'].items(): assert sha(ROOT/rel) == sha(STAGE/rel) == digest, rel
    accepted = read(V1/'EXECUTION_BINDINGS.json')
    assert sha(V1/'PREFLIGHT_MANIFEST.json') == accepted['preflight_manifest_sha256']
    assert sha(ROOT/'scripts/led_large_decoder_label_view_v1.py') == accepted['label_module_sha256']
    original_zip = ROOT/'outputs/cloud_handoff/LED_LARGE_DECODER_POLICY_V1_OFFLINE_REVIEW.zip'
    assert sha(original_zip) == '3f3aab4f4307f5f01cec115a44dd3257fc9d006e486f18fd18ac8b11d0dedd87'
    # Check original train/validation artifacts; never read held-out contents.
    stage_h = read(ROOT/'outputs/stage_h/led_dataset_v1/manifest.json')
    frozen = {}
    for name, info in stage_h['datasets'].items():
        assert name.split('_')[0] in ('train','validation')
        for path_key, hash_key in (('path','sha256'),('stage_g_jsonl','stage_g_jsonl_sha256')):
            assert sha(ROOT/info[path_key]) == info[hash_key], name
        frozen[name] = dict(rows=info['rows'], original_npz_sha256=info['sha256'], original_JSONL_sha256=info['stage_g_jsonl_sha256'])
    tok = read(ROOT/stage_h['tokenizer_manifest'])
    for name, digest in tok['sha256'].items(): assert sha(ROOT/tok['relative_directory']/name) == digest
    integrity = dict(status='PASS', accepted_V1_bindings=accepted, accepted_V1_review_zip_sha256=sha(original_zip),
        accepted_label_module_unchanged=True, original_datasets=frozen,
        Stage_H_status_unchanged=stage_h['status'], tokenizer_assets_unchanged=True,
        label_views_and_2200_content_EOS_rows_unchanged=True,
        source_inputs_Teacher_targets_references_checkpoint_artifacts_not_modified=True,
        held_out_contents_accessed=False, new_pretrained_forward_calls=0, new_generation_calls=0,
        new_backward_calls=0, new_optimizer_updates=0, GPU_execution=False)
    put(PREP/'FINAL_SCIENTIFIC_INTEGRITY.json', integrity)
    validation = dict(status='PASS_OFFLINE_ONLY', tests=tests, checks=checks, bindings=binding,
        scope='Actual pinned portable-root CPU preflight and tensor/helper/fake-inference tests; no pretrained model execution',
        parent_payload_files=9961, accepted_V1_manifest_entries=18, V2_manifest_entries=len(manifest['files']))
    put(PREP/'FINAL_VALIDATION.json', validation)
    report = f'''# LED-large decoder policy V2: minimal early-sanity revision

**OFFLINE READY. Corrected labels accepted by the researcher. GPU execution remains unapproved.**

This supersedes only the V1 epoch-1 technical screen and adds state-continuity evidence. The accepted V1 label module, two F label views (2,200 rows), all source IDs/masks/padded shapes and original scientific artifacts remain byte-identical. The V1 archive and its five bindings are preserved. No additional model investigation or hyperparameter search was performed.

## Predeclared technical screen

| Observation after decoder-start EOS2 | Result |
|---|---|
| At least **5 consecutive BOS0** at any position | Hard FAIL: sustained BOS loop |
| Decoded text empty or whitespace only | Hard FAIL: blank output |
| Fewer than **16 ordinary tokens**, **and at least 90% special tokens** among tokens after decoder start | Hard FAIL: near-empty special-token-dominated output |
| Wrong decoder-start ID, missing start, or invalid token ID | Hard FAIL: protocol collapse/integrity error |
| BOS without a run of 5, including a single BOS followed by useful text | Warning; no failure by itself |
| Generation cap reached, including forced EOS at the cap | Warning; no failure by itself |
| No terminal EOS but no other collapse | Warning |

Special IDs are 0,1,2,3,50264, matching the unchanged tokenizer. Fractions exclude the initial decoder-start token and include terminal EOS when present. The near-empty rule is conjunctive: fewer than16 content tokens alone does not fail a readable short output. Raw BOS count, longest BOS run, content/special counts, fraction, terminal EOS, cap flag, warnings and hard-fail reasons are recorded per case. These transparent thresholds identify technical collapse; they do not certify semantic quality.

The same four predeclared cases and order are retained at step250/epoch1. Classification does not short-circuit their loop: all four raw token sequences, decoded texts, source/checkpoint hashes and diagnostics are saved before aggregate technical PASS/FAIL. Warnings permit continuation. A runtime/IO/deadline failure still stops immediately; missing predictions are never fabricated and no automatic retry is introduced. Neither technical PASS nor low loss is substantive quality GO.

## Inference continuity

The existing callback now passes the live Trainer into the versioned sanity function. A narrow guard saves RNG (Python/NumPy/CPU/CUDA), each individual module's train/eval flag, use_cache and exact adapter tensor values. In finally it restores RNG, individual modes and use_cache, then checks exact equality.

Read-only fingerprints before/after also check optimizer state/parameter groups and object identity, scheduler, GradScaler, AMP recovery counters, Trainer state/control, skipped-step flag, parameter identities/requires_grad, existing gradients and adapter activation/merge/disable flags. No optimizer, scheduler or scaler step/reset/restore is performed by sanity. Unexpected mutation fails continuity. STATE_CONTINUITY.json preserves both sides and checks, including when inference raises; the original exception remains primary. Runtime equality checks will be executed on the real epoch1 run; offline fixtures cannot establish actual GPU restoration in advance.

**{tests}/{tests} focused tests PASS**, including the eleven unchanged accepted label/shift/source tests, single/four/five BOS boundaries, blanks, near-empty special dominance, cap warnings with/without EOS, four raw records on collapse, warning continuation, fake inference exception, exact RNG/mode/cache continuity, synthetic CUDA RNG round-trip, and mutation detection for optimizer/scheduler/scaler/counters/gradients/adapters. No real model forward/generation/backward/update occurred. Full pinned-runtime portable-root CPU preflight PASS: 9,961 original payload files, the18 accepted V1 entries and {len(manifest['files'])} V2 manifest entries are checked through the bound chain. Frozen train/validation/Teacher/tokenizer integrity PASS.

## Exact patch and bindings

EXACT_PROPOSED_PATCH.diff contains V1-to-V2 helper/runner/reporter/launcher/config/sanity/test changes. New execution namespace: runs/F_large_decoder_policy_v2. The accepted label module is reused unchanged; only technical criteria, continuity verification and versioned execution paths differ.

```json
{json.dumps(binding, indent=2)}
```

## Minimal future execution authorization

Approve these exact V2 config/runner/preflight hashes for **one fresh Filtered run** on RTX3090, from the same pinned official LED-large backbone and fresh LoRA. Reject old adapters, interrupted Colab states and previous gate states as initialization. Preserve the existing fresh feasibility gate:3 successful updates,24 nominal microbatches (88 maximum with the already bounded AMP policy),2 technical memory generation traces, at least15% headroom and all original integrity/numerical checks. Gate states are discarded before fresh training.

Only after gate PASS: unchanged F2,000train/200validation,5epochs/1,250successful updates, same228 LoRA modules/456 tensors, seed42, order, AdamW settings, accumulation8, FP16/FP32 storage and dynamic65536 policy. Same five epoch checkpoints. At epoch1 run exactly4 technical sanity cases under the above criteria; then exactly24 outputs from each of epoch3/epoch5, same20random+4targeted. Beam4/max_length768 and all generation settings stay unchanged. Total maximum remains **54 calls =2gate+4sanity+48final**.

Approval must contain actual UTC start and a stop no later than48h afterward under the accepted absolute-stop/reserve fields. No historical deadline is reset by this preparation; no approval timestamp is fabricated. AUTHORIZATION_TEMPLATE_NOT_APPROVED.json remains false. The accepted scientific label policy does not itself authorize GPU execution.

After completion or expiry, stop for independent substantive review against D-059 Expanded-LoRA epoch3. Core information, opinion balance, factual/visual grounding, useful visual description, harmful semantic repetition, zero-caption compliance and usable completion still determine quality. Existing matched LED-base RO/U/F remain primary. No automatic RO/U, Teacher regeneration, held-out access, decoder/architecture/optimization change, retry or follow-on experiment is authorized. D051 strict replay FAIL and all stopped Unlikelihood/D058/D059/D060 evidence stay preserved.

Remaining uncertainty: the label correction's causal effectiveness and substantive output quality are still unproven. This revision addresses overstrict technical rejection, not those scientific questions. No other preparation is needed before the researcher decides on these execution hashes.
'''
    put(PREP/'READINESS_AND_AUTHORIZATION_PLAN.md', report)
    put(ROOT/'thesis_sdd_docs/LED_LARGE_DECODER_POLICY_EARLY_SANITY_V2.md', report)
    DEST.mkdir(parents=True, exist_ok=False)
    # Include accepted overlay dependencies unchanged: a single review ZIP can
    # overlay the preserved original full-data/model package without rewriting it.
    shutil.copytree(PREVIOUS/'overlay', DEST/'overlay')
    parent = read(STAGE/'VAST_SHA256_MANIFEST.json')['files']
    new_files = [r for r in manifest['files'] if not r.startswith('outputs/stage_i/led_large_decoder_policy_v1/')]
    new_files += ['outputs/stage_i/led_large_decoder_policy_v2/'+n for n in (
        'PREFLIGHT_MANIFEST.json','EXECUTION_BINDINGS.json','AUTHORIZATION_TEMPLATE_NOT_APPROVED.json')]
    for rel in new_files:
        assert rel not in parent and not (DEST/'overlay'/rel).exists(), rel
        p=DEST/'overlay'/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,p)
    shutil.copytree(PREVIOUS/'prior_completed_run_evidence', DEST/'prior_completed_run_evidence')
    for name in ('LABEL_AUDIT.json','PINNED_RUNTIME_CONTRACT.json','EXECUTION_BINDINGS.json','FINAL_VALIDATION.json'):
        p=DEST/'accepted_v1'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(V1/name,p)
    for name in ('READINESS_AND_AUTHORIZATION_PLAN.md','EXACT_PROPOSED_PATCH.diff','EXECUTION_BINDINGS.json',
                 'FINAL_VALIDATION.json','FINAL_SCIENTIFIC_INTEGRITY.json','FOCUSED_TESTS.log','CPU_PREFLIGHT.log',
                 'AUTHORIZATION_TEMPLATE_NOT_APPROVED.json'):
        shutil.copyfile(PREP/name,DEST/name)
    put(DEST/'README_FIRST.md', '''# Minimal V2 revision — OFFLINE REVIEW ONLY

Read READINESS_AND_AUTHORIZATION_PLAN.md, EXACT_PROPOSED_PATCH.diff and EXECUTION_BINDINGS.json. GPU training/inference has not been authorized or executed. Corrected labels are accepted; real generation recovery and substantive quality remain unproven.

overlay/ contains the unchanged accepted V1 dependencies plus the new V2 technical-sanity code/config. It uses original workspace-relative paths. Apply only to the preserved full LED_LARGE_VAST_D059_F_V1 payload (or its V1-overlay workspace), checking every hash; existing V1 paths must already have identical bytes. No original-parent path is replaced. Do not rerun the preparation builders on the cloud: the overlay is already materialized. No model/data re-download or all-condition audit is required for this revision. The full 2GB backbone/data ZIP is not duplicated in this small review archive.

Future execution entry point, ONLY after new V2 approval: bash RUN_DECODER_POLICY_V2.sh <new_approval.json>. The separate false approval template is deliberately non-executable. Old approvals and V1 launcher paths cannot authorize the V2 run. Preserve old archives/results. accepted_v1/ and prior_completed_run_evidence/ are historical evidence, not current technical criteria or permission.

The revised screen saves all four epoch1 raw predictions/diagnostics, warns on isolated BOS/cap hits and fails sustained >=5 BOS loops, blank or near-empty special-dominated output. Early sanity remains technical only. Final substantive review and the unchanged 54-call ceiling apply. No unrelated change, new experiment, automatic RO/U, Teacher or held-out access is requested.
''')
    put(DEST/'GPT_REVIEW_REQUEST.md', '''Review only the requested minimal V1-to-V2 early-sanity revision. Confirm >=5 consecutive BOS hard failure, conjunctive near-empty rule, isolated BOS/cap warnings, all four raw records, technical-only decisions, and exact state restoration/optimizer continuity checks. Verify hashes and tests. Accepted labels/collator, sources, generation, adapter/optimizer/numerical protocol and54-call ceiling must remain unchanged. Distinguish offline mock evidence from future real GPU checks and unproven quality. Return approval/revision decision and exact V2 execution hashes. Do not execute GPU work or propose another search/framework.
''')
    put(DEST/'release_tools/finalize_led_large_decoder_policy_review_v2.py', Path(__file__).read_text(encoding='utf-8'))
    file_hashes={p.relative_to(DEST).as_posix():sha(p) for p in sorted(DEST.rglob('*')) if p.is_file()}
    put(DEST/'SHA256_MANIFEST.json',dict(files=file_hashes))
    archive=DEST.with_suffix('.zip')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(DEST.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(DEST).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert len(z.namelist())==len(set(z.namelist()))==len(file_hashes)+1
        for rel,digest in file_hashes.items():assert hashlib.sha256(z.read(rel)).hexdigest()==digest,rel
        overlay={n.removeprefix('overlay/') for n in z.namelist() if n.startswith('overlay/')}
        assert not (overlay & set(parent))
        for rel in overlay:assert z.read('overlay/'+rel)==(STAGE/rel).read_bytes(),rel
        for ep in (3,5):assert len(json.loads(z.read(f'prior_completed_run_evidence/original_review/predictions/epoch{ep}.json')))==24
        html=z.read('prior_completed_run_evidence/original_review/RESULTS_REVIEW.html').decode('utf-8')
        for ref in re.findall(r'<img src="([^"]+)"',html):assert 'prior_completed_run_evidence/original_review/'+ref in z.namelist()
    receipt=dict(status='OFFLINE_READY_GPU_NOT_APPROVED',archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),
        manifest_files=len(file_hashes),members=len(file_hashes)+1,SHA256_CRC='PASS',overlay_overwrites_original_parent=False,
        focused_tests=tests,portable_root_CPU_preflight='PASS',bindings=binding,
        accepted_V1_preserved=True,new_pretrained_forward_calls=0,new_generation_calls=0,new_optimizer_updates=0)
    put(PREP/'DELIVERY_RECEIPT.json',receipt)
    archive.with_suffix('.zip.sha256').write_text(receipt['sha256']+'  '+archive.name+'\n',encoding='utf-8')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
