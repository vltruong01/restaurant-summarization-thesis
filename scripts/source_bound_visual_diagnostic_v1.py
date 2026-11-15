"""One fixed 12-TRAIN saved-output diagnostic; standard library only, no model calls."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import statistics

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'outputs/stage_i/led_large_nbest_v2_execution_v1/portable_review'
OUT=ROOT/'outputs/stage_i/source_bound_visual_diagnostic_v1'
ADAPTER_SHA='507dcdd92333dc0625559570f06140b84a14eb202191779ebd03393224090a60'
REVIEW='[REVIEW_CONSENSUS]'
VISUAL='[VISUAL_EVIDENCE]'


def require(ok,message):
    if not ok:raise ValueError(message)


def sha_bytes(raw):return hashlib.sha256(raw).hexdigest()
def sha_file(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def words(text):return len(re.findall(r"\b\w+(?:['’-]\w+)*\b",text))
def write(name,obj):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n')


def section_parts(text):
    require(text.count(REVIEW)==text.count(VISUAL)==1,'Require exactly two original section headers')
    require(text.startswith(REVIEW) and text.index(REVIEW)<text.index(VISUAL),'Unexpected section order')
    at=text.index(VISUAL);end=at+len(VISUAL)
    if text[end:].startswith('\r\n'):end+=2
    elif text[end:].startswith('\n'):end+=1
    else:raise ValueError('Visual header must have its original newline')
    return text[:at],text[:end],text[end:]


def exact_captions(case):
    source=case['source'];caps=case['captions']
    require(case['split']=='train' and case['condition']=='filtered','TRAIN Filtered only')
    require(sha_bytes(source.encode('utf-8'))==case['source_sha256'],'Source hash mismatch')
    require(isinstance(caps,dict),'Expected frozen ordered caption dictionary')
    markers=list(re.finditer(r'\[(IMG_CAP_\d+)\]',source))
    require([m.group(1) for m in markers]==list(caps),'Caption order/identity mismatch')
    require(len(set(caps))==len(markers),'Duplicate caption markers')
    results=[]
    for i,m in enumerate(markers):
        ident=m.group(1);text=caps[ident];limit=markers[i+1].start() if i+1<len(markers) else len(source)
        require(isinstance(text,str) and bool(text),'Empty/malformed selected caption')
        require(source[m.end():limit].strip()==text,'Caption body differs from frozen source')
        require(REVIEW not in text and VISUAL not in text and '[IMG_CAP_' not in text,'Ambiguous caption delimiters')
        at=source.find(text,m.end(),limit);require(at>=m.end(),'Caption not located exactly')
        results.append(dict(caption_id=ident,text=text,caption_text_sha256=sha_bytes(text.encode('utf-8')),
            source_char_start=at,source_char_end=at+len(text),
            source_UTF8_byte_start=len(source[:at].encode('utf-8')),
            source_UTF8_byte_end=len(source[:at+len(text)].encode('utf-8'))))
    return results


def render(case,original):
    """No rewriting or selection: copy every exact frozen caption once with its ID label."""
    review,prefix,old_visual=section_parts(original)
    caps=exact_captions(case)
    lines=[];trace=[];position=len(prefix)
    for c in caps:
        number=c['caption_id'].removeprefix('IMG_CAP_')
        label='Caption '+number+': '
        line=label+c['text']+'\n'
        start=position+len(label);end=start+len(c['text'])
        trace.append(dict(c,alternative_char_start=start,alternative_char_end=end,
            attribution_label=label.rstrip()))
        lines.append(line);position+=len(line)
    alternative=prefix+''.join(lines)
    new_review,new_prefix,new_visual=section_parts(alternative)
    require(review.encode('utf-8')==new_review.encode('utf-8'),'Review bytes changed')
    require(prefix.encode('utf-8')==new_prefix.encode('utf-8'),'Original prefix bytes changed')
    for t in trace:
        require(alternative[t['alternative_char_start']:t['alternative_char_end']]==t['text'],'Caption output trace mismatch')
        t['alternative_UTF8_byte_start']=len(alternative[:t['alternative_char_start']].encode('utf-8'))
        t['alternative_UTF8_byte_end']=len(alternative[:t['alternative_char_end']].encode('utf-8'))
    require(bool(caps) or not new_visual.strip(),'Zero-caption alternative has visual assertions')
    return alternative,trace,dict(review_section_utf8_sha256=sha_bytes(review.encode('utf-8')),
        review_bytes_exact=True,prefix_bytes_exact=True,original_visual_words=words(old_visual),
        alternative_visual_words=words(new_visual),raw_caption_words=sum(words(c['text']) for c in caps),
        caption_count=len(caps),caption_texts_copied_exactly=True,
        original_visual_body_nonempty=bool(old_visual.strip()),alternative_visual_body_empty=not new_visual.strip())


def main():
    require(not OUT.exists() or (OUT.is_dir() and not any(OUT.iterdir())),
            'Versioned diagnostic already contains artifacts; do not overwrite')
    manifest=read(SOURCE/'SHA256_MANIFEST.json')['files']
    for rel,expected in manifest.items():
        p=(SOURCE/rel).resolve()
        require(p.is_relative_to(SOURCE.resolve()) and p.is_file(),'Unsafe/missing input artifact')
        require(sha_file(p)==expected,'Accepted TRAIN bundle changed: '+rel)
    sample=read(ROOT/'outputs/stage_i/led_large_nbest_reranking_v2/TRAIN_SAMPLE_MANIFEST.json')
    sources=read(SOURCE/'TRAIN_SOURCE_ONLY_CASES.json')
    predictions=read(SOURCE/'execution/CANDIDATES.json')
    require(len(sources)==len(predictions)==12,'Report shortage; never create replacement model predictions')
    require([x['business_id'] for x in sources]==sample['case_ids'],'Locked TRAIN case order differs')
    require(len(set(sample['case_ids']))==12,'Duplicate business identity')
    checkpoint=read(SOURCE/'CHECKPOINT_BINDING.json')
    require(checkpoint['global_step']==750 and checkpoint['files']['adapter_model.safetensors']==ADAPTER_SHA,'Wrong checkpoint')
    by={p['business_id']:p for p in predictions}
    require(len(by)==12 and set(by)==set(sample['case_ids']),'Missing/duplicate saved prediction')
    used=[SOURCE/'SHA256_MANIFEST.json',SOURCE/'TRAIN_SOURCE_ONLY_CASES.json',SOURCE/'execution/CANDIDATES.json',
          SOURCE/'CHECKPOINT_BINDING.json',ROOT/'outputs/stage_i/led_large_nbest_reranking_v2/TRAIN_SAMPLE_MANIFEST.json']
    before={p.relative_to(ROOT).as_posix():sha_file(p) for p in used}
    OUT.mkdir(exist_ok=True)
    rows=[]
    for case in sources:
        bid=case['business_id'];prediction=by[bid];single=prediction['single']
        require(prediction['split']=='train' and prediction['condition']=='filtered','Wrong prediction split/condition')
        require(prediction['source_sha256']==single['source_sha256']==case['source_sha256'],'Prediction/source mismatch')
        require(prediction['checkpoint_sha256']==single['checkpoint_sha256']==ADAPTER_SHA,'Not Original CE3')
        require(single['candidate_id']=='c0' and single['generation_rank']==0,'Require saved N1, not selected N4 alternative')
        call=SOURCE/f'execution/calls/{bid}_single.json';raw=read(call)
        # The immutable runner flushes raw evidence before writing a separate PASS receipt.
        # EVIDENCE_FLUSHED alone is not proof of successful completion.
        pass_path=SOURCE/f'execution/calls/{bid}_single_PASS.json'
        passed=read(pass_path)
        require(raw['status']=='EVIDENCE_FLUSHED' and passed['status']=='PASS' and not raw['errors'],
                'Saved N1 lacks its separate successful completion receipt')
        require(raw['rng']['status']=='PASS' and raw['raw_token_ids'][0]==single['token_ids'],
                'Saved N1 RNG/token identity differs')
        require(raw['metadata']['source_sha256']==case['source_sha256'] and
                raw['metadata']['checkpoint_sha256']==ADAPTER_SHA,'Raw call binding differs')
        alt,trace,checks=render(case,single['text'])
        name=re.search(r'^name=(.*)$',case['source'],re.M).group(1)
        row=dict(business_id=bid,name=name,split='train',condition='filtered',source_sha256=case['source_sha256'],
            checkpoint_sha256=ADAPTER_SHA,original_N1_call_path=call.relative_to(ROOT).as_posix(),original_N1_call_sha256=sha_file(call),
            original_N1_pass_receipt_sha256=sha_file(pass_path),
            original_text=single['text'],alternative_text=alt,original_text_sha256=sha_bytes(single['text'].encode('utf-8')),
            alternative_text_sha256=sha_bytes(alt.encode('utf-8')),original_model_token_ids=single['token_ids'],
            alternative_model_token_ids=None,alternative_origin='OFFLINE_CAPTION_COPY_DIAGNOSTIC_NOT_MODEL_PREDICTION',
            alternative_token_ids_note='No new tokenizer/model call; not an LED generation',
            caption_trace=trace,checks=checks,caption_photo_truth='UNVERIFIED_ORIGINAL_PHOTOS_NOT_INSPECTED',
            review_quality_improvement='NONE_REVIEW_BYTES_UNCHANGED',human_or_researcher_quality_confirmation=False)
        rows.append(row)
        folder=OUT/'cases'/bid;folder.mkdir(parents=True)
        (folder/'ORIGINAL_CE3.txt').write_bytes(single['text'].encode('utf-8'))
        (folder/'SOURCE_BOUND_ALTERNATIVE.txt').write_bytes(alt.encode('utf-8'))
        (folder/'SOURCE_MODEL_VISIBLE.txt').write_bytes(case['source'].encode('utf-8'))
        write(f'cases/{bid}/PROVENANCE.json',{k:v for k,v in row.items() if k not in ('original_text','alternative_text','original_model_token_ids')})
    nzero=sum(r['checks']['caption_count']==0 for r in rows)
    require(nzero==3 and len(rows)-nzero==9,'Existing strata differ; do not resample')
    write('ALL_PAIRED_OUTPUTS.json',rows)
    write('CAPTION_PROVENANCE.json',dict(source_packet_path=used[1].relative_to(ROOT).as_posix(),source_packet_sha256=sha_file(used[1]),
        cases=[dict(business_id=r['business_id'],source_sha256=r['source_sha256'],captions=r['caption_trace']) for r in rows],
        photo_ID_mapping_not_reconstructed=True,photo_note='This accepted source-only packet supplies CAP IDs/text, not authenticated photo identities; no photo ID inferred and no photo opened.'))
    sums={}
    for group,part in [('all12',rows),('captioned9',[r for r in rows if r['checks']['caption_count']]),('zero_caption3',[r for r in rows if not r['checks']['caption_count']])]:
        a=[r['checks']['original_visual_words'] for r in part];b=[r['checks']['alternative_visual_words'] for r in part]
        sums[group]=dict(N=len(part),original_visual_words_total=sum(a),alternative_visual_words_total=sum(b),
            original_visual_words_mean=statistics.mean(a),alternative_visual_words_mean=statistics.mean(b),
            original_visual_words_median=statistics.median(a),alternative_visual_words_median=statistics.median(b),
            alternative_shorter_cases=sum(y<x for x,y in zip(a,b)),alternative_equal_cases=sum(y==x for x,y in zip(a,b)),
            alternative_longer_cases=sum(y>x for x,y in zip(a,b)))
    write('AUTOMATIC_CHECKS.json',dict(status='PASS_STRUCTURAL_NOT_SEMANTIC_QUALITY_GO',cases=12,captioned=9,zero_caption=3,
        selected_captions=sum(r['checks']['caption_count'] for r in rows),review_bytes_identical_cases=12,
        exact_source_to_alternative_caption_traces=sum(len(r['caption_trace']) for r in rows),
        zero_caption_visual_sections_empty=3,all_caption_texts_preserved_once_in_original_order=True,
        word_count_definition="Unicode regex \\b\\w+(?:['’-]\\w+)*\\b; alternative includes Caption N labels; no section headers",lengths=sums,
        caption_semantic_redundancy_not_automatically_removed=True,photo_ground_truth_not_established=True))
    write('POLICY.json',dict(version='SOURCE_BOUND_VISUAL_OFFLINE_DIAGNOSTIC_V1',case_limit=12,
        case_selection='Exactly existing locked N-best TRAIN12; no selection using outputs',baseline='Existing N1 Original CE3 only',
        review_prefix='Byte-identical original prefix through original visual-header newline',
        visual_policy='Each selected caption exactly once, original source order, Caption N: attribution; no editing, semantic merging/dedup, new selection, new relations or model-generated prose',
        zero_caption_policy='Empty visual body; original review untouched',
        quality_judgments='Model-assisted provisional, not human evaluation or researcher confirmation',
        GPU_calls=0,new_model_predictions=0,new_training_updates=0,derived_offline_alternatives=12,held_out_access=False,
        reference_or_human_evaluation_activity=False,other_experiment_execution=False))
    for rel,expected in before.items():require(sha_file(ROOT/rel)==expected,'Frozen artifact changed during diagnostic')
    write('INPUT_INTEGRITY.json',dict(accepted_bundle_manifest_entries_verified=len(manifest),files=before,
        checkpoint_step=750,adapter_sha256=ADAPTER_SHA,shortage=0,before_after_used_inputs_unchanged=True,
        checkpoint_weights_loaded_or_modified=False,Teacher_targets_opened_or_modified=False))
    evidence=OUT/'EXISTING_EVIDENCE';evidence.mkdir()
    for src,name in [(SOURCE/'TRAIN_SOURCE_ONLY_CASES.json','TRAIN_SOURCE_ONLY_CASES_UNCHANGED.json'),
                     (SOURCE/'execution/CANDIDATES.json','ORIGINAL_NBEST_EXECUTION_UNCHANGED.json'),
                     (SOURCE/'CHECKPOINT_BINDING.json','CHECKPOINT_BINDING_UNCHANGED.json'),
                     (SOURCE/'SHA256_MANIFEST.json','ORIGINAL_MANIFEST_UNCHANGED.json'),
                     (used[4],'TRAIN_SAMPLE_MANIFEST_UNCHANGED.json')]:shutil.copyfile(src,evidence/name)
    calls=evidence/'N1_CALL_RECEIPTS';calls.mkdir()
    for case in sources:
        for suffix in ('_single.json','_single_PASS.json'):
            src=SOURCE/'execution/calls'/(case['business_id']+suffix)
            shutil.copyfile(src,calls/src.name)
    write('PREPARATION_CORRECTION.json',dict(
        initial_offline_preparation_stopped_before_any_alternative_output=True,
        initial_assertion="raw call status == PASS",
        confirmed_schema="Raw status is EVIDENCE_FLUSHED; success is recorded in separate *_single_PASS.json",
        correction="Require both receipts, no errors, RNG PASS, exact raw-token identity and source/checkpoint bindings",
        all12_raw_token_lists_exactly_match_saved_N1=True,
        frozen_artifacts_changed=False,new_model_calls=0))
    shutil.copyfile(Path(__file__),OUT/'DIAGNOSTIC_SOURCE.py')
    print(json.dumps(dict(cases=12,captions=sum(r['checks']['caption_count'] for r in rows),lengths=sums,
        rows=[dict(name=r['name'],old_words=r['checks']['original_visual_words'],new_words=r['checks']['alternative_visual_words']) for r in rows]),indent=2))


if __name__=='__main__':main()
