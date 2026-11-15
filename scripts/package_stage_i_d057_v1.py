"""Package the completed or stopped D057 evidence; no model/Teacher calls."""
import json,pathlib,shutil,zipfile,hashlib
R=pathlib.Path(__file__).resolve().parents[1];O=R/'outputs/stage_i/d057_bounded_gpu_diagnostic_v1'
B=R/'outputs/stage_i/stage_i_d057_review_bundle_v1'

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def copy(a,b):b.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(a,b)

def main():
    assert (O/'D057_SUMMARY.json').exists(),'Finish offline analysis first'
    assert (O/'STAGE_I_D057_SCIENTIFIC_ASSESSMENT.md').exists(),'Finish evidence-based assessment first'
    B.mkdir(exist_ok=False)
    for p in O.rglob('*'):
        if p.is_file():copy(p,B/'reports'/p.relative_to(O))
    cfg=json.loads((R/'configs/stage_i_d056_minimal_diagnostic_v2.json').read_text(encoding='utf-8'))
    raw=R/cfg['future_output']
    if raw.exists():
        for p in raw.rglob('*'):
            if p.is_file():copy(p,B/'raw_gpu'/p.relative_to(raw))
    prep=R/'outputs/stage_i/d056_diagnostic_preparation_v2'
    for name in ['cases.json','TOKEN_POSITION_ALIGNMENT.json','RUNTIME_SOURCE_MASK_ADAPTER_BINDINGS.json',
        'HISTORICAL_FINGERPRINT_COMPATIBILITY.json','D056_V2_AMENDMENT.md','LITERATURE_INFORMED_RATIONALE.md',
        'REVISED_MINIMAL_GPU_PROTOCOL.md','INTERPRETATION_AND_NEXT_EXPERIMENT_TREE.md']:
        copy(prep/name,B/'evidence'/name)
    copy(prep/'PREFLIGHT_MANIFEST.json',B/'evidence/D056_V2_PREFLIGHT_MANIFEST.json')
    copy(R/'assets/tokenizers/led-base-16384/tokenizer.json',B/'evidence/tokenizer.json')
    for name in ['RANKED_HYPOTHESES.md','PROPOSED_GPU_DIAGNOSTIC_CONFIG.json','STAGE_I_D055_ROOT_CAUSE_REPORT.md']:
        copy(R/'outputs/stage_i/d055_root_cause_v2'/name,B/'evidence/D055'/name)
    copy(R/'outputs/stage_i/d057_authorization_gate_v1/SCIENTIFIC_ACCEPTANCE.json',B/'evidence/PRIOR_SCIENTIFIC_ACCEPTANCE_ONLY.json')
    copy(R/'configs/stage_i_d056_minimal_diagnostic_v2.json',B/'configs/stage_i_d056_minimal_diagnostic_v2.json')
    for name in ['run_stage_i_d056_gpu_diagnostic_v2.py','stage_i_d056_v2.py','validate_stage_i_d056_v2.py',
        'execute_stage_i_d057_authorized_v1.py','report_stage_i_d057_v1.py','package_stage_i_d057_v1.py',
        'check_stage_i_d056_offline_v2.py','stage_hi_led_data.py','stage_i_paired_pilot.py']:
        copy(R/'scripts'/name,B/'code'/name)
    copy(R/'scripts/verify_stage_i_d057_review_bundle_v1.py',B/'verify_bundle.py')
    for name in ['README_FIRST.md','CHATGPT_REVIEW_REQUEST.md']:copy(O/name,B/name)
    for rel in ['DECISIONS.md','thesis_sdd_docs/THESIS_PROGRESS.md','thesis_sdd_docs/CHATGPT_NEW_CONVERSATION_HANDOFF.md']:
        copy(R/rel,B/'project_snapshot'/rel)
    manifest={p.relative_to(B).as_posix():sha(p) for p in sorted(B.rglob('*')) if p.is_file()}
    (B/'SHA256_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    zip_path=B.with_suffix('.zip')
    with zipfile.ZipFile(zip_path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(B.rglob('*')):
            if p.is_file():z.write(p,B.name+'/'+p.relative_to(B).as_posix())
    receipt={'zip':str(zip_path),'bytes':zip_path.stat().st_size,'sha256':sha(zip_path),'manifest_files':len(manifest),
        'raw_evidence_included':True,'backbone_and_checkpoints_not_duplicated':True}
    zip_path.with_suffix('.zip.sha256').write_text(receipt['sha256']+'  '+zip_path.name+'\n',encoding='ascii')
    with (O/'DELIVERED_REVIEW_ZIP_RECEIPT.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
