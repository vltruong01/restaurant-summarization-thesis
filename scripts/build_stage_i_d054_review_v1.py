"""Package existing D054 evidence only. Never calls a model or edits accepted data."""
import ast,hashlib,json,shutil,zipfile
from pathlib import Path
from stage_i_d054 import ROOT,OUT,read,write,sha,preflight,require

BUNDLE=OUT/'stage_i_d054_chatgpt_review_bundle_v3'
ARCHIVE=ROOT/'outputs/stage_i/stage_i_d054_chatgpt_review_bundle_v3.zip'

def copy(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True);require(not dst.exists(),'Refuse overwrite '+str(dst));shutil.copy2(src,dst)

def build():
    ctx,cfg,prep,ec,inputs,training,protected,ref=preflight()
    require(read(OUT/'GENERATION_COMPLETION.json')['outputs']==48,'Generation incomplete')
    require(not BUNDLE.exists() and not ARCHIVE.exists(),'Refuse overwrite existing review package')
    BUNDLE.mkdir()
    reference_zip=ROOT/'outputs/stage_i/stage_i_d053_final_references_chatgpt_bundle_v1.zip'
    with zipfile.ZipFile(reference_zip) as z:
        for name in z.namelist():
            if name.endswith('/'):continue
            rel=Path(*Path(name).parts[1:]);dest=(BUNDLE/'reference_evidence'/rel).resolve()
            require(dest.is_relative_to(BUNDLE.resolve()),'Unsafe reference archive path')
            dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(z.read(name))
    references=read(ctx['final']/'FINAL_RESEARCHER_APPROVED_REFERENCES.json')
    predictions=[json.loads(s) for s in (OUT/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    key=read(prep/'blinding_key_keep_separate.json');payload={'bundle_id':'D054_PAIRED48_V1','cases':[]}
    for c in ctx['cases']:
        q={k:c[k] for k in ['business_id','display_name','subset','source','reviews']}
        q['captions']=[{**p,'image_file':'reference_evidence/'+p['image_file']} for p in c['captions']]
        q['units']=[u for u in references['units'] if u['business_id']==c['business_id']]
        q['outputs']=[]
        for position,side in [('left','X'),('right','Y')]:
            arm=key['mapping'][c['business_id']][position];p=next(p for p in predictions if p['business_id']==c['business_id'] and p['arm']==arm)
            q['outputs'].append({'side':side,'text':p['text'],'text_sha256':p['text_sha256'],'tokens':len(p['token_ids'])})
        payload['cases'].append(q)
    page=(ROOT/'scripts/stage_i_d054_review_ui.html').read_text(encoding='utf-8')
    encoded=json.dumps(payload,ensure_ascii=False).replace('<','\\u003c')
    write(BUNDLE/'D054_BLINDED_REVIEW.html',page.replace('__PAYLOAD__',encoded))
    copy(prep/'blinding_key_keep_separate.json',BUNDLE/'private/BLINDING_KEY_UNBLINDS_REVIEW.json')
    copy(ROOT/'configs/stage_i_d053_evaluation_v1.json',BUNDLE/'configs/FROZEN_EVALUATION_CONFIG.json')
    copy(ROOT/'configs/stage_i_paired_lora_pilot_v1.json',BUNDLE/'configs/FROZEN_TRAINING_PILOT_CONFIG.json')
    copy(ROOT/'specs/009i_d054_paired_generation_evaluation.md',BUNDLE/'reports/D054_EXECUTION_SPEC.md')
    copy(prep/'expected_generation_config.json',BUNDLE/'configs/EXPECTED_GENERATION_CONFIG.json')
    copy(prep/'evaluation_inputs.jsonl',BUNDLE/'evidence/FROZEN_EVALUATION_INPUTS.jsonl')
    tokenizer=ROOT/cfg['frozen_training_root']/'assets/tokenizers/led-base-16384'
    for p in tokenizer.iterdir():
        if p.is_file():copy(p,BUNDLE/'assets/tokenizer'/p.name)
    copy(ROOT/'outputs/stage_i/d052_paired_pilot_v1/D052_TRAINING_VALIDATION.json',BUNDLE/'evidence/D052_TRAINING_VALIDATION.json')
    for p in OUT.glob('*.json'):
        if p.name.startswith('D054_ARCHIVE_RECEIPT') or p.name=='OFFLINE_SOFTWARE_TESTS.json':continue
        copy(p,BUNDLE/'results'/p.name)
    for p in OUT.glob('*.jsonl'):copy(p,BUNDLE/'results'/p.name)
    for p in OUT.glob('*.csv'):copy(p,BUNDLE/'results'/p.name)
    for p in OUT.glob('*.md'):copy(p,BUNDLE/'reports'/p.name)
    copy(OUT/'RESEARCHER_REQUEST.txt',BUNDLE/'evidence/RESEARCHER_REQUEST.txt')
    for arm in ['A_decoder_only','B_encoder_decoder']:
        for p in (OUT/arm).rglob('*.json'):copy(p,BUNDLE/'execution'/arm/p.relative_to(OUT/arm))
        adapter=ROOT/ec['arms'][arm]['adapter_path']
        copy(adapter.parent/'adapter_config.json',BUNDLE/'configs'/arm/'adapter_config.json')
    for name in ['stage_i_d054.py','stage_i_d054_text.py','run_stage_i_d054_generation_v1.py','evaluate_stage_i_d054_v1.py','assemble_stage_i_d054_source_assessment.py','write_stage_i_d054_report_v1.py','verify_stage_i_d054_bundle.py','build_stage_i_d054_review_v1.py','stage_i_d054_review_ui.html','test_stage_i_d054_offline.py','stage_i_d053_evaluation.py','stage_i_decoding_diagnostics.py','stage_i_paired_pilot.py','stage_hi_led_data.py','run_stage_i_d050_feasibility_v1.py']:
        copy(ROOT/'scripts'/name,BUNDLE/'code'/name)
    # Preserve the full local Python source dependency closure for review.
    queue=list((BUNDLE/'code').glob('*.py'));seen={};dependency_rows=[]
    while queue:
        p=queue.pop(0)
        if p.name in seen:continue
        seen[p.name]=True;deps=set()
        for node in ast.walk(ast.parse(p.read_text(encoding='utf-8-sig'))):
            modules=[node.module] if isinstance(node,ast.ImportFrom) else [a.name for a in node.names] if isinstance(node,ast.Import) else []
            for module in modules:
                if not module:continue
                name=module.split('.')[0]+'.py';original=ROOT/'scripts'/name
                if not original.exists():continue
                deps.add(name);target=BUNDLE/'code'/name
                if not target.exists():copy(original,target);queue.append(target)
        dependency_rows.append({'file':p.name,'sha256':sha(p),'local_python_dependencies':sorted(deps)})
    write(BUNDLE/'CODE_DEPENDENCY_MANIFEST.json',{'purpose':'All local Python imports included for source review; full GPU reexecution still requires original project layout and frozen weights','files':dependency_rows})
    write(BUNDLE/'README_FIRST.md', '''# D-054 paired pilot quality review

Open **D054_BLINDED_REVIEW.html** locally in Chrome/Edge. All24 businesses, full selected reviews, original LED-visible captions, original photos and576 approved Core/Supporting reference units are available offline. X/Y follows the previously frozen blinding key. Read both outputs and source evidence before opening unblinded reports or private/.

Hướng dẫn nhanh: giải nén toàn bộ ZIP rồi mở D054_BLINDED_REVIEW.html. Chọn business ở thanh trên, đọc hai output X/Y và mở phần reviews/captions/references để đối chiếu. Bấm ảnh để mở bản gốc. Có thể ghi chú và Export/Import notes; ghi chú chưa được xem là bộ đánh giá human hoàn chỉnh. Chỉ mở báo cáo không làm phát sinh inference, training hay API call.

The main report is reports/STAGE_I_D054_PAIRED_GENERATION_REPORT.md. results/predictions.jsonl contains exactly48 raw outputs and generated token IDs. No target, source, checkpoint or reference was edited. Code, effective configs, adapter/checkpoint hashes, both model preflights and protected-artifact checks are included. The weights themselves are not duplicated; rerunning GPU generation requires the original D052 adapters and pinned pretrained backbone. Offline audit verification needs no weights, GPU, API or original project.

Three distinct assessment layers: deterministic metrics; explicitly unconfirmed Codex source-based assessments; pending researcher output evaluations. Approved SOURCE references are not human verification of STUDENT outputs. Core and Supporting coverage are separate.88 exclusions and two uncertain parent candidates never become scoring denominators. Supporting omission alone is not an error. Generic correct content is not factual error. Photos cannot justify details absent from LED-visible captions.

## Offline verification after extraction

Use Python3.11 with numpy1.26.4 and jsonschema (pip install numpy==1.26.4 jsonschema if needed). From this extracted folder:

    python code/verify_stage_i_d054_bundle.py .

This verifies every SHA256 file, the full unchanged accepted D053 v2 reference proof, exactly48 matched output/source bindings, deterministic metrics, paired seeded bootstrap, original photos, blinded display and pending human annotations. No network/model execution. The final ZIP receipt lives OUTSIDE the archive to avoid self-referential hashes. Historical D053 documents preserve their original stage-specific authorization status; the current D054 authorization is results/RESEARCHER_AUTHORIZATION.json.

Researcher notes export/import is optional and always remains pending; the HTML does not assign human labels or generate formal human-quality scores. Full atomic annotation shells are in results/PENDING_RESEARCHER_OUTPUT_ANNOTATIONS.json. Reviewers should check all source evidence and important omissions, not just teacher-selected candidates.

Stop after this independent review. No additional inference/training is authorized by opening or verifying this package.
''')
    write(BUNDLE/'CHATGPT_REVIEW_REQUEST.md','''# Independent D054 quality review

Compare matched A decoder-only LoRA and B encoder+decoder LoRA on20 random validation businesses and4 separately reported targeted cases. Both trained512 businesses for2 epochs, one seed, with different adapter locations AND parameter counts. Source-reference decisions were finalized before generation. Frozen beam4/max_length768 decoding was identical.

First inspect blinded X/Y outputs, full source reviews/captions/photos, Core/Supporting references and omissions. Then consult the unblinded report and metrics. Assess repetition, semantic redundancy, useful restaurant-specific information, positive/negative contrasts, unsupported or wrong-domain claims, caption utilization, structure, EOS and the targeted zero-caption case. Correct-but-simple is not a factual error. Do not count photo-only details as caption support. Do not treat malformed/unparseable outputs as perfect factual precision.

Audit Codex judgments, exact quotation/claim/evidence bindings and uncertain cases independently. No student-output judgment is researcher-confirmed yet. Automated metrics and software PASS do not prove factual correctness. Do not treat teacher-target agreement or lower validation loss as grounded quality. Report meaningful improvements/regressions with business IDs and exact output/evidence quotes; distinguish minor repetition shifts from useful information recovery. Do not force an architectural winner. Recommend a next decision; do not authorize or execute retraining.
''')
    files=[{'path':p.relative_to(BUNDLE).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(BUNDLE.rglob('*')) if p.is_file()]
    write(BUNDLE/'SHA256_MANIFEST.json',{'algorithm':'SHA-256','self_exclusion':'This manifest only; final ZIP receipt is external','files':files})
    with zipfile.ZipFile(ARCHIVE,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(BUNDLE.rglob('*')):
            if p.is_file():z.write(p,Path(BUNDLE.name)/p.relative_to(BUNDLE))
    receipt={'archive':str(ARCHIVE),'bytes':ARCHIVE.stat().st_size,'sha256':sha(ARCHIVE),'files_in_archive':len(files)+1,'reference_source_archive_sha256':sha(reference_zip)}
    write(OUT/'D054_ARCHIVE_RECEIPT_v3.json',receipt);print(json.dumps(receipt,indent=2))

if __name__=='__main__':build()
