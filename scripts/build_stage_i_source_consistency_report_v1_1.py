"""Offline audit/report after one CPU derived-output experiment; no model load."""
import base64,hashlib,html,json,statistics,zipfile,shutil
from collections import Counter
from pathlib import Path
from stage_i_repetition_pilot_v1 import ROOT,read,rows,sha,write
from stage_i_repetition_pilot_v2 import preflight
from stage_i_source_consistency_v1 import ALLOWED_INPUT_KEYS
from stage_i_decoding_diagnostics import text_diagnostics,structure


def build():
    run=ROOT/'outputs/stage_i/source_consistency_postprocess_v1';completion=read(run/'COMPLETION.json')
    assert not (run/'FAILURE_STOP.json').exists()
    prep=ROOT/'outputs/stage_i/source_consistency_preparation_v1';inputs=rows(prep/'INFERENCE_ONLY_INPUTS.jsonl')
    m=read(prep/'PREFLIGHT_MANIFEST.json')
    for rel,d in m['files'].items():
        p=Path(rel);assert sha(p if p.is_absolute() else ROOT/p)==d
    _,_,frozen,_,proof=preflight(require_fresh=False)
    result=read(run/'predictions.json');assert len(result)==len(inputs)==24
    assert [p['business_id'] for p in result]==[i['business_id'] for i in inputs]==[c['business_id'] for c in frozen]
    out=ROOT/'outputs/stage_i/source_consistency_report_v1_1';out.mkdir(exist_ok=False)
    records=[];copy_kept=0;evidence_checks=0
    for i,p in zip(inputs,result):
        assert set(i)==ALLOWED_INPUT_KEYS
        assert p==read(run/'cases'/(p['business_id']+'.json'))
        assert p['source_sha256']==i['source_sha256'] and p['raw_prediction_sha256']==i['raw_prediction_sha256']
        assert len(p['serialized_token_ids'])<=768
        for group in p['selected_sentences'].values():
            for u in group:
                assert u['text']==i['raw_prediction'][u['start']:u['end']]
                assert u['decision']=='KEEP_UNCHANGED_RAW_SENTENCE';copy_kept+=1
                for ev in u['evidence']:
                    assert ev['text']==i['source'][ev['source_start']:ev['source_end']];evidence_checks+=1
        if not i['selected_photo_ids']:assert not p['selected_sentences']['visual']
        for arm,text in [('raw_guard',i['raw_prediction']),('derived_postprocess',p['text'])]:
            records.append({'business_id':p['business_id'],'subset':p['subset'],'arm':arm,
                'software':text_diagnostics(text),'structure_by_serialization_not_model_learning':structure(text,'filtered',len(i['selected_photo_ids'])),
                'canonical_marker_presence':text.startswith('[REVIEW_CONSENSUS]\n') and text.count('[REVIEW_CONSENSUS]')==1 and text.count('[VISUAL_EVIDENCE]')==1,'human_verified':False})
    groups={}
    for subset in ('random','targeted'):
        for arm in ('raw_guard','derived_postprocess'):
            rr=[r for r in records if r['subset']==subset and r['arm']==arm]
            st=[r['structure_by_serialization_not_model_learning'] for r in rr]
            groups[arm+'/'+subset]={'n':len(rr),
                'exact_repeated_sentence_outputs':sum(r['software']['exact_repeated_sentence'] for r in rr),
                'legacy_adjacent_word_flags':sum(r['software']['consecutive_repeated_word'] for r in rr),
                'canonical_markers_present':sum(r['canonical_marker_presence'] for r in rr),
                'legacy_full_structure_compliance_including_required_nonempty_bodies':sum(s['section_headers_compliant'] for s in st),
                'zero_caption_cases':sum(s['empty_visual_required'] for s in st),
                'zero_caption_empty_visual_compliant':sum(s['empty_visual_required'] and s['empty_visual_compliant'] for s in st)}
    known=[]
    for name,key in [('AI_QUALITY_REVIEW.json','exact_new_claim_or_excerpt'),('AI_TARGETED_REVIEW.json','exact_new_claim')]:
        for f in read(ROOT/'outputs/stage_i/repetition_guard_quality_assessment_v2'/name)['findings']:
            p=next(p for p in result if p['business_id']==f['business_id']);quote=f[key]
            known.append({'business_id':f['business_id'],'original_AI_classification':f['classification'],'exact_excerpt':quote,
                'excerpt_retained_verbatim_after_processing':quote in p['text'],
                'interpretation':'Deletion status of previously reviewed examples, NOT a factual accuracy metric or independent evaluation.','human_verified':False})
    summary={'completion':completion,'software_groups':groups,'whole_raw_sentences_retained':copy_kept,'exact_source_evidence_spans_checked':evidence_checks,
        'empty_review_outputs':[p['business_id'] for p in result if not p['selected_sentences']['review']],
        'fewer_than_three_review_sentences':[p['business_id'] for p in result if len(p['selected_sentences']['review'])<3],
        'mean_retained_review_sentences':statistics.mean(len(p['selected_sentences']['review']) for p in result),
        'mean_retained_visual_sentences':statistics.mean(len(p['selected_sentences']['visual']) for p in result),
        'all_raw_and_frozen_bindings_PASS':True,'preflight':proof,
        'Core_coverage':'NOT_ESTABLISHED; no numerical Core score from this same NLI scorer',
        'selection_inputs_exclude_gold_targets_and_references':True,'new_LED_generation_or_training':False,
        'human_quality_acceptance':'PENDING','scope':'Repeatedly inspected24development cases; no independent validation/test claim'}
    write(out/'SOFTWARE_AND_RETENTION_REPORT.json',summary)
    write(out/'KNOWN_EXCERPT_RETENTION_NOT_ACCURACY.json',known)
    write(out/'PER_CASE_SOFTWARE.json',records)
    write(out/'PROVISIONAL_RESEARCH_ASSESSMENT.json',{'status':'NOT_RECOMMENDED_AS_A_RESCUE_PIPELINE','basis':'17/24 empty review sections;23/24 retain fewer than3 review sentences; mean0.42 review sentences. Content loss outweighs cleaner surface metrics. This is an AI recommendation, not a researcher formal GO/NO-GO.','false_positive':{'business_id':'gcQSAQIx-KQOdf5E8HF6Lw','exact_retained_claim':'There is a donut sitting next to a donut.','caption_id':'IMG_CAP_1','PHOTO':'9YRB33hF9nyyv4C8xckbJg','caption':'two cups of coffee and a donut sitting next to each other.','NLI_entailment':0.9886206984519958,'evidence_based_AI_judgment':'Unsupported multiplication: caption and original inspected photograph show two coffee cups and one donut. High model score is not calibrated factual certainty.','human_verified':False},'two_markers_present_in_all24_is_by_serialization':True,'legacy_full_structure_compliance':6,'Core_coverage_not_established':True,'frozen_data_or_teacher_modified':False,'no_further_variant_automatically_executed':True})
    write(out/'INTEGRITY_AUDIT.json',{'status':'DERIVED_OUTPUT_MECHANICAL_AUDIT_PASS_NOT_FACTUAL_GROUND_TRUTH',
        'cases':24,'kept_raw_sentence_spans':copy_kept,'source_evidence_span_checks':evidence_checks,
        'no_in_sentence_rewrite':True,'no_gold_in_selection':True,'protected_files_unchanged':True,
        'config_sha256':sha(ROOT/'configs/stage_i_source_consistency_postprocess_v1.json'),'D051_strict_replay':'FAIL_UNCHANGED'})
    intro=f"""# CPU source-consistency/MMR experiment

Applied exactly one preset to24existing guard outputs, with0LED generation,
0training,0GPU operations,0Teacher API calls and0held-out cases.
NLI retrieval/filtering + mutual-entailment semantic deduplication + MMR selection
are implemented. Fixed thresholds are exploratory, not calibrated accuracy.

Result:whole raw sentences retained={copy_kept}; empty review outputs=
{len(summary['empty_review_outputs'])}/24; fewer than3review sentences=
{len(summary['fewer_than_three_review_sentences'])}/24. Mean review/visual sentences:
{summary['mean_retained_review_sentences']:.2f}/{summary['mean_retained_visual_sentences']:.2f}.
All retained sentences and their source evidence offsets passed byte-exact checks.

Canonical headers and empty zero-caption visual sections are enforced explicitly
by postprocessing. These are not learned LED improvements or actual new EOS
termination. No missing Core information is restored; no sentence is rewritten
and no Teacher/reference content fills gaps. Lower repetition can accompany
severe information loss. Empty or short review sections count as limitations.

NLI is fallible. Passing a text-support threshold does not independently verify
photos, correct original BLIP2 errors, guarantee review attribution/counts, or
prove factual correctness. Model support decisions must not become human labels.
The same24cases were repeatedly examined; they are development examples.
There is no new scientific GO or automatic deployment to RO/U/F/Shuffled/test.

Compare before/after in RESULTS_REVIEW.html; inspect every keep/drop reason,
original source and model evidence. Known-excerpt removal is retrospective
diagnostic evidence, not a factual-accuracy denominator. References are bundled
for post-hoc review only and never entered the selector.

Assessment: NOT RECOMMENDED AS A RESCUE. Empty reviews17/24;short reviews23/24.
Canonical wrappers are present24/24 by construction, while the unchanged full
structure check passes only6/24 because nonempty content is also required.
The preserved donut-next-to-donut claim receives NLI entailment0.9886 despite
IMG_CAP1/PHOTO9YRB33hF9nyyv4C8xckbJg showing two coffees and one donut.
This separates software execution from factual and informative usability.

Sources and rationale:source_consistency_preparation_v1/METHOD_AND_RESEARCH.md.
Original artifacts and D051 strict replay FAIL are preserved. Stop for review.
"""
    write(out/'REPORT.md',intro)
    source_path=ROOT/'outputs/stage_i/d053_reference_verification_v1/source_cases.json'
    src={s['business_id']:s for s in read(source_path)};cards=[]
    for i,p in zip(inputs,result):
        pictures=[]
        for cap in src[p['business_id']]['captions']:
            data=(source_path.parent/cap['image_file']).read_bytes();mime='image/png' if data.startswith(b'\x89PNG') else 'image/jpeg'
            pictures.append(f"<figure><img loading='lazy' src='data:{mime};base64,{base64.b64encode(data).decode()}'><figcaption>{html.escape(cap['caption_id']+' / PHOTO '+cap['photo_id']+' / '+cap['text'])}</figcaption></figure>")
        cards.append('<article><h2>'+html.escape(p['business_id']+' / '+p['subset'])+'</h2><div class="grid"><section><h3>Raw guard output — unchanged</h3><pre>'+html.escape(i['raw_prediction'])+'</pre></section><section><h3>Experimental derived output — not a new LED generation</h3><pre>'+html.escape(p['text'])+'</pre><b>'+html.escape(p['coverage_warning'])+'</b></section></div><details><summary>Exact input source</summary><pre>'+html.escape(i['source'])+'</pre></details><details><summary>Original captions/photos</summary><div class="photos">'+''.join(pictures)+'</div></details><details><summary>Every keep/drop reason, source evidence and fallible model scores</summary><pre>'+html.escape(json.dumps(p['decisions'],ensure_ascii=False,indent=2))+'</pre></details></article>')
    page="<!doctype html><meta charset='utf-8'><title>Source consistency postprocessing</title><meta name='viewport' content='width=device-width,initial-scale=1'><style>body{font:16px system-ui;margin:24px;background:#f4f6f8}article,section,details{background:white;margin:12px 0;padding:16px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.photos{display:flex;flex-wrap:wrap}figure{max-width:300px}img{max-width:100%;max-height:300px}pre{white-space:pre-wrap;overflow-wrap:anywhere}summary{cursor:pointer}@media(max-width:900px){.grid{grid-template-columns:1fr}}</style><h1>24development cases: source-consistency/MMR postprocessing</h1><pre>"+html.escape(intro)+'</pre>'+''.join(cards)
    write(out/'RESULTS_REVIEW.html',page)
    bundle=ROOT/'outputs/stage_i/source_consistency_review_bundle_v1_1';bundle.mkdir(exist_ok=False)
    files={}
    def copy(p):
        rel=p.relative_to(ROOT);d=bundle/rel;d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,d);files[rel.as_posix()]=sha(d)
    for directory in (run,prep,out):
        for p in directory.rglob('*'):
            if p.is_file():copy(p)
    for name in ('stage_i_source_consistency_v1.py','run_stage_i_source_consistency_v1.py','build_stage_i_source_consistency_report_v1.py','build_stage_i_source_consistency_report_v1_1.py','stage_i_repetition_pilot_v1.py','stage_i_repetition_pilot_v2.py','stage_i_decoding_diagnostics.py','download_source_consistency_scorer_v1.py','stage_i_repetition_guard_v1.py'):
        copy(ROOT/'scripts'/name)
    for name in ('configs/stage_i_source_consistency_postprocess_v1.json','tests/test_stage_i_source_consistency_v1.py','outputs/stage_i/d053_final_references_v1/FINAL_RESEARCHER_APPROVED_REFERENCES.json'):
        copy(ROOT/name)
    copy(source_path)
    for s in src.values():
        for cap in s['captions']:copy(source_path.parent/cap['image_file'])
    for p in (ROOT/'assets/experimental_scorers/nli_deberta_small_e989068').iterdir():
        if p.is_file() and p.suffix!='.safetensors' and p.name!='spm.model':copy(p)
    write(bundle/'README_FIRST.md','Open outputs/stage_i/source_consistency_report_v1_1/RESULTS_REVIEW.html. This is explicit postprocessing, not a trained-model improvement. Check omissions/empty reviews and NLI false positives. Model weights are excluded; public revisions and hashes are recorded. SHA256_MANIFEST.json verifies all payloads except itself. No human quality approval or main methodology change is implied.')
    write(bundle/'CHATGPT_REVIEW_REQUEST.md','Independently assess24before/after outputs. Prioritize information loss, preserved contrasting opinions, exact source bindings, caption-only visual scope, zero-caption behavior and semantic loops. Do not equate NLI confidence, shorter outputs, forced headers/empty visual sections or deletion of known errors with factual correctness or model learning. Check whether any pipeline benefit is outweighed by lost Core information. All AI assessments are provisional; no held-out inference or new training is authorized.')
    for name in ('README_FIRST.md','CHATGPT_REVIEW_REQUEST.md'):files[name]=sha(bundle/name)
    write(bundle/'SHA256_MANIFEST.json',files)
    archive=ROOT/'outputs/stage_i/stage_i_source_consistency_review_v1_1.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(bundle))
    write(out/'DELIVERY_RECEIPT.json',{'archive':str(archive.relative_to(ROOT)),'sha256':sha(archive),'bytes':archive.stat().st_size,'payload_files':len(files)})
    print(json.dumps({'empty_review_outputs':len(summary['empty_review_outputs']),'fewer_than3':len(summary['fewer_than_three_review_sentences']),'known_excerpt_retained':sum(k['excerpt_retained_verbatim_after_processing'] for k in known),'archive_sha256':sha(archive)}))


if __name__=='__main__':build()
