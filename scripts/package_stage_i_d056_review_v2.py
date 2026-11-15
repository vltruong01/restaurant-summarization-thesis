"""Package D056 preparation evidence; no execution permission or model code calls."""
import hashlib,json,pathlib,shutil,zipfile
R=pathlib.Path(__file__).resolve().parents[1];O=R/'outputs/stage_i/d056_diagnostic_preparation_v2'
B=R/'outputs/stage_i/stage_i_d056_review_bundle_v2'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def cp(a,b):b.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(a,b)
def main():
    B.mkdir(exist_ok=False)
    for p in O.rglob('*'):
        if p.is_file():cp(p,B/'reports'/p.relative_to(O))
    for name in ['README_FIRST.md','CHATGPT_REVIEW_REQUEST.md']:cp(O/name,B/name)
    for name in ['stage_i_d056_v2.py','prepare_stage_i_d056_offline_v2.py','run_stage_i_d056_gpu_diagnostic_v2.py','check_stage_i_d056_offline_v2.py','test_stage_i_d056_preparation_v2.py','validate_stage_i_d056_v2.py','document_stage_i_d056_v2.py','create_stage_i_d056_amendment_v2.py','stage_hi_led_data.py','stage_i_paired_pilot.py','run_stage_i_d050_feasibility_v1.py']:
        cp(R/'scripts'/name,B/'code'/name)
    cp(R/'tests/test_stage_i_d056_offline_v2.py',B/'code/tests/test_stage_i_d056_offline_v2.py')
    cp(R/'tests/test_stage_i_d056_offline.py',B/'code/tests/test_stage_i_d056_offline.py')
    cp(R/'scripts/verify_stage_i_d056_review_bundle_v2.py',B/'verify_bundle.py')
    cp(R/'configs/stage_i_d056_minimal_diagnostic_v2.json',B/'configs/stage_i_d056_minimal_diagnostic_v2.json')
    cp(R/'assets/tokenizers/led-base-16384/tokenizer.json',B/'evidence/tokenizer.json')
    cp(R/'outputs/stage_i/d055_root_cause_v2/PROPOSED_GPU_DIAGNOSTIC_CONFIG.json',B/'evidence/D055_PROPOSED_GPU_DIAGNOSTIC_CONFIG.json')
    for name in ['PREFLIGHT_MANIFEST.json','TOKEN_POSITION_ALIGNMENT.json','GPU_AUTHORIZATION_TEMPLATE.json']:
        cp(R/'outputs/stage_i/d056_diagnostic_preparation_v1'/name,B/'evidence/parent_v1'/name)
    cp(R/'configs/stage_i_d056_minimal_diagnostic_v1.json',B/'evidence/parent_v1/config.json')
    for rel in ['thesis_sdd_docs/THESIS_PROGRESS.md','DECISIONS.md','thesis_sdd_docs/CHATGPT_NEW_CONVERSATION_HANDOFF.md','specs/009k_d056_amendment_v2.md']:
        cp(R/rel,B/'project_snapshot'/rel)
    m={p.relative_to(B).as_posix():sha(p) for p in sorted(B.rglob('*')) if p.is_file()}
    (B/'SHA256_MANIFEST.json').write_text(json.dumps(m,indent=2)+'\n',encoding='utf-8')
    z=B.with_suffix('.zip')
    with zipfile.ZipFile(z,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as f:
        for p in sorted(B.rglob('*')):
            if p.is_file():f.write(p,B.name+'/'+p.relative_to(B).as_posix())
    receipt={'zip':str(z),'bytes':z.stat().st_size,'sha256':sha(z),'manifest_files':len(m)}
    z.with_suffix('.zip.sha256').write_text(receipt['sha256']+'  '+z.name+'\n',encoding='ascii')
    (O/'DELIVERED_REVIEW_ZIP_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
