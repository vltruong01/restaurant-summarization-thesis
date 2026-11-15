"""Package existing D055 offline evidence only; no model or Teacher execution."""
import argparse,hashlib,json,pathlib,shutil,zipfile
R=pathlib.Path(__file__).resolve().parents[1]
O=R/'outputs/stage_i/d055_root_cause_v2'
B=R/'outputs/stage_i/stage_i_d055_chatgpt_review_bundle_v1'
C=R/'outputs/stage_i/cloud_return_20261015_v1/stage_hi_v2'
def sh(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def cp(src,dst):dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='\n') as f:f.write((x if isinstance(x,str) else json.dumps(x,ensure_ascii=False,indent=2))+'\n')
def main():
    global B
    parser=argparse.ArgumentParser();parser.add_argument('--version',type=int,default=1);args=parser.parse_args()
    B=R/f'outputs/stage_i/stage_i_d055_chatgpt_review_bundle_v{args.version}'
    if B.exists():raise ValueError('Preserve previous bundle; use a new version')
    B.mkdir(parents=True)
    for p in O.rglob('*'):
        if p.is_file():cp(p,B/'reports'/p.relative_to(O))
    originals=R/'outputs/stage_i/d053_researcher_reference_finalization_v1/source_bundle/stage_i_d053_reference_verification_chatgpt_bundle_v2'
    cp(originals/'source_cases.json',B/'evidence/source_cases.json')
    cases=json.loads((originals/'source_cases.json').read_text(encoding='utf-8'))
    for c in cases:
        for cap in c['captions']:cp(originals/cap['image_file'],B/'evidence'/cap['image_file'])
    for name in ['evaluation_inputs.jsonl','expected_generation_config.json','pilot_manifest.json']:
        cp(R/'outputs/stage_i/paired_lora_pilot_preparation_v1'/name,B/'evidence'/name)
    cp(R/'assets/tokenizers/led-base-16384/tokenizer.json',B/'evidence/tokenizer.json')
    cp(R/'outputs/stage_i/d054_paired_generation_v1/predictions.jsonl',B/'evidence/D054_predictions.jsonl')
    for name in ['CODEX_SOURCE_BASED_ASSESSMENT.json','SOURCE_BASED_QUALITY_ASSESSMENT.md','STAGE_I_D054_PAIRED_GENERATION_REPORT.md','PREFLIGHT_REFERENCES_AND_FILES.json','BOTH_MODELS_PREFLIGHT_PASS.json','AUTOMATIC_EVALUATION.json','TOKENIZATION_ROUNDTRIP_CHECK.json']:
        cp(R/'outputs/stage_i/d054_paired_generation_v1'/name,B/'evidence/D054'/name)
    cp(R/'outputs/stage_i/d053_final_references_v1/FINAL_RESEARCHER_APPROVED_REFERENCES.json',B/'evidence/D053_final_references.json')
    for p in R.joinpath('scripts').glob('*.py'):
        if p.name in ['audit_stage_i_d055_offline.py','finalize_stage_i_d055_offline_report.py','verify_stage_i_d055_review_bundle.py','package_stage_i_d055_review.py',
            'stage_i_d053_evaluation.py','stage_i_decoding_diagnostics.py','stage_hi_led_data.py','stage_i_paired_pilot.py',
            'run_stage_i_paired_lora_pilot_v1.py','run_stage_i_d052_paired_pilot_v2.py','run_stage_i_d054_generation_v1.py','stage_i_d054.py']:
            cp(p,B/'code/project'/p.name)
    cp(C/'scripts/run_stage_i_lora_led_cloud_v2.py',B/'code/historical/run_stage_i_lora_led_cloud_v2.py')
    cp(C/'scripts/stage_hi_led_data.py',B/'code/historical/stage_hi_led_data.py')
    for p in ['configs/stage_i_paired_lora_pilot_v1.json','configs/stage_i_d053_evaluation_v1.json','specs/009j_d055_root_cause_investigation.md','tests/test_stage_i_d055_offline.py',
        'thesis_sdd_docs/THESIS_PROGRESS.md','thesis_sdd_docs/CODEX_THESIS_IMPLEMENTATION_PLAN.md','DECISIONS.md']:
        cp(R/p,B/'project_snapshot'/p)
    for p in ['configs/stage_i_lora_led_training_approved_v2.json','outputs/stage_i/cloud_training_v2/run_bindings.json','outputs/stage_i/runs_v3/filtered/completion_report.json','outputs/stage_i/runs_v3/filtered/trainer_state.json']:
        cp(C/p,B/'evidence/historical'/p)
    cp(R/'scripts/verify_stage_i_d055_review_bundle.py',B/'verify_bundle.py')
    write(B/'README_FIRST.md',"""# D055 offline root-cause review

Start with reports/STAGE_I_D055_ROOT_CAUSE_REPORT.md, then CODE_PATH_AUDIT.md,
RANKED_HYPOTHESES.md and PROPOSED_MINIMAL_GPU_DIAGNOSTIC.md.

This contains72 EXISTING outputs:24 historical F,24 PilotA,24 PilotB; original24
validation sources/reviews/captions/photos, teacher targets and frozen D053 units.
No predictions were generated or repaired in D055. Source-based Codex output
assessments are not researcher-confirmed human labels. Old source-reference human
approval remains distinct from human approval of generated summaries.

Only offline investigation is authorized. Proposed GPU diagnosis, forced-prefix
traces, precision comparisons and all further training need separate approval.
There is no runnable new GPU diagnostic in this package.

Portable verification (Python3.11, tokenizers0.21.1):
`python verify_bundle.py .`

The verifier checks every SHA256, original72 prediction bindings,24 source/target
tokenizations,72 exact BPE decodes and216 diagnostics, plus photo paths. It needs
no torch, model weights, GPU, API key or network. It checks source data, not factual
truth. Full6600-target audit reproduction needs the accepted six datasets in the
original workspace; these large datasets/weights are intentionally not duplicated.
Audit scripts preserve original project imports for source inspection, not a claim
that new training/inference is runnable from this review package.

reports/OFFLINE_TESTS.log records14 CPU regression tests in the original workspace.
The original partial D055v1 engineering attempt is preserved there; completedv2
is authoritative. ZIP SHA is in the adjacent .sha256 receipt (not self-referential
inside the archive). SHA256_MANIFEST.json covers every other bundled file.
""")
    write(B/'CHATGPT_REVIEW_REQUEST.md',"""# Independent review request — D055

Please assess whether the offline evidence justifies the recommendation for a
bounded read-only GPU diagnosis before more training. Do not assume a cause from
header counts or validation loss. Check:

1. Historical21/24 versus A/B0/24 header compliance with random20 and targeted4
   separate; historical1250 versus pilot128updates and the unmatched confounds.
2. Actual LED/Trainer/Accelerate/PEFT label shift, loss, masks, cache and generation
   paths, including whether any concrete implementation defect is demonstrated.
3. B's correct marker text18/24 followed by token11 newline-transition collapse,
   and A's early source-marker confusion. Distinguish structure from information.
4. Teacher-target audit: exact encodes/markers, zero repeated sentences, adjacent
   word false positives, zero-caption behavior; do not demand silent label edits.
5. Ranked H1–H7 evidence/counterevidence and the minimal discriminating protocol.
   Are the three diagnostic cases, four read-only states, forward/trace ceilings,
   precision/cache equivalence and numerical screens appropriate before approval?
6. Do not infer a LoRA capacity ceiling or architectural causal winner; B doubles
   trainable parameters. Do not treat Codex source assessments as human truth.

Return any substantive error or a revised next-step recommendation. No new
training, GPU execution, dataset changes, Teacher calls or held-out evaluation
are approved by this review request.
""")
    manifest={p.relative_to(B).as_posix():sh(p) for p in sorted(B.rglob('*')) if p.is_file()}
    write(B/'SHA256_MANIFEST.json',manifest)
    archive=B.with_suffix('.zip')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(B.rglob('*')):
            if p.is_file():z.write(p,B.name+'/'+p.relative_to(B).as_posix())
    digest=sh(archive);archive.with_suffix('.zip.sha256').write_text(digest+'  '+archive.name+'\n',encoding='ascii')
    report={'zip_path':str(archive),'bytes':archive.stat().st_size,'sha256':digest,'files_in_manifest':len(manifest)}
    receipt_name='DELIVERED_ZIP_RECEIPT.json' if args.version==1 else f'DELIVERED_ZIP_RECEIPT_v{args.version}.json'
    write(O/receipt_name,report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
