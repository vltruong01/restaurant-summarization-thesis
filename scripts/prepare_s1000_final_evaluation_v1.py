"""Build TEST source-only evaluation preparation. No inference or reference drafting."""
from pathlib import Path
import sys,json,csv,collections,shutil,datetime,zipfile,xml.etree.ElementTree as ET,importlib.metadata
from s1000_final_eval_core_v1 import *
R=Path(__file__).resolve().parents[1]
O=R/'outputs/final_evaluation/s1000_preparation_v1'
F=R/'outputs/stage_f/production_caption_construction_d033_colab_v1'
M=R/'outputs/stage_i/led_large_ro_u_source_separated_s1000_matched_v1'

def build():
    from tokenizers import Tokenizer
    require(not O.exists(),'Preparation already exists; do not overwrite frozen evidence')
    O.mkdir(parents=True);E=O/'EVIDENCE';E.mkdir();bindings={}
    def bind(p):
        p=Path(p);bindings[p.relative_to(R).as_posix()]=dict(bytes=p.stat().st_size,sha256=file_hash(p));return bindings[p.relative_to(R).as_posix()]['sha256']
    def jl(p):
        bind(p);return [json.loads(l) for l in p.read_text(encoding='utf8').splitlines() if l.strip()]
    request=Path('C:/Users/VUONGLOCTRUONG/.codex/attachments/79e4ba9e-ed75-4419-b066-b9bce66d63a1/Pasted text.txt')
    shutil.copy2(request,E/'RESEARCHER_PREPARATION_REQUEST.txt')
    write(E/'SCOPE_RECEIPT.json',dict(source_attachment_sha256=file_hash(request),scope='Preparation only; no human reference drafting, TEST inference, scores, training or paid resources',current_inference_authorized=False))
    split=list(csv.DictReader((R/'data/processed/yelp_2300_v1/business_split.csv').open(encoding='utf8')));bind(R/'data/processed/yelp_2300_v1/business_split.csv')
    test=sorted(r['business_id'] for r in split if r['split']=='test');require(len(test)==len(set(test))==100,'TEST100')
    stage=read(F/'completion_report.json')
    for rel,h in stage['artifact_sha256'].items():require(bind(F/rel)==h,'Stage F changed: '+rel)
    tokpath=M/'COMMON/assets/tokenizers/led-base-16384';tok=Tokenizer.from_file(str(tokpath/'tokenizer.json'))
    for n,h in stage['pinned_tokenizer']['sha256'].items():require(bind(tokpath/n)==h,'Tokenizer changed')
    shared={r['business_id']:r for r in jl(F/'manifests/shared_review_inputs.jsonl') if r['split']=='test'}
    sources={c:{r['business_id']:r for r in jl(F/f'manifests/test/{c}_sources.jsonl')} for c in ['reviews_only','unfiltered','filtered']}
    for g in [shared,*sources.values()]:require(set(g)==set(test),'TEST cohort source differs')
    rows=[{k:sources['filtered'][b][k] for k in ['business_id','split','source','source_sha256','selected_review_ids','selected_photo_ids']} for b in test]
    oldpolicy=read(R/'outputs/stage_i/led_large_matched_signoff_v3/SHUFFLED_POLICY_PROPOSED_UNCHANGED.json');bind(R/'outputs/stage_i/led_large_matched_signoff_v3/SHUFFLED_POLICY_PROPOSED_UNCHANGED.json')
    policy=dict(version='S1000_TEST_ONE_TO_ONE_V1',seed=42,cuisine_subcategory_allowlist=oldpolicy['cuisine_subcategory_allowlist'],
        category_normalization=oldpolicy['category_normalization'],assignment='scipy linear_sum_assignment within exact positive caption counts; globally minimize category fallback count then seeded integer cost',
        cost='(0 if shared exact allowlisted category else 1000000000) + int(SHA256(canonical [42,version,A,B])[:12],16)%1000000; forbidden=10000000000000',
        deterministic_order='sorted business IDs, sorted count strata; scipy==1.15.2',donor_reuse=False,self_matching=False,
        budgets=dict(full_source=5000,review_block=4000,caption_increment=750,isolated_visual=750,K=5),
        no_perfect_assignment='STOP; do not freeze a relaxed mapping',zero_captions='Exact unchanged empty task; donor null; intervention N/A',
        original_policy_preserved=True,change_reason='Latest researcher explicitly requests one-to-one where feasible; previous proposed with-replacement code remains historical',
        mapping_frozen_before_any_predictions=True,approval_status='Prepared under attached researcher request; final protocols pending review')
    write(O/'SHUFFLED_POLICY.json',policy)
    mapping=shuffled_mapping(rows,tok,policy);write(O/'TEST100_SHUFFLED_INPUT_MANIFEST.json',mapping)
    mapping2=shuffled_mapping(list(reversed(rows)),tok,policy);require(mapping==mapping2,'Mapping not order invariant')
    csvrows=[]
    for x in mapping['mappings']:
        csvrows.append({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in x.items() if k not in ['source','review_task_source','visual_task_source']})
    write_csv(O/'TEST100_SHUFFLED_MAPPING.csv',csvrows)
    positive=[x for x in mapping['mappings'] if x['caption_count']];active=[x for x in positive if x['visual_source_changed']]
    write(E/'SHUFFLED_VALIDATION.json',dict(status='PASS_SOURCE_ONLY_ONE_TO_ONE',businesses=100,positive_caption_recipients=len(positive),actual_visual_source_changes=len(active),zero_caption_businesses=[x['recipient_id'] for x in mapping['mappings'] if not x['caption_count']],
        same_category_assignments=sum(x['matching_tier']=='SHARED_CATEGORY' for x in positive),fallback_assignments=sum(x['matching_tier']=='SAME_COUNT_FALLBACK' for x in positive),
        one_to_one=True,no_self=True,review_inputs_identical_100=True,token_budgets_verified_all100=True,reverse_order_repeat_identical=True,TEST_predictions=0))
    # Verify model/tokenizer/backbone files without constructing a torch model or CUDA context.
    freeze=read(R/'outputs/pre_evaluation/s1000_readiness_v1/EVIDENCE/MODEL_FREEZE_AUDIT.json');bind(R/'outputs/pre_evaluation/s1000_readiness_v1/EVIDENCE/MODEL_FREEZE_AUDIT.json')
    for a,m in freeze['models'].items():
        cp=R/m['path'];require(bind(cp/'adapter_model.safetensors')==m['adapter_sha256'],'Adapter changed');bind(cp/'adapter_config.json')
    require(bind(Path(freeze['backbone']['local_file']))==freeze['backbone']['sha256'],'Backbone changed')
    write(O/'MODEL_AND_DECODER_LOCK.json',freeze)
    # Original-source byte equivalence from immutable HF backup manifest; no raw-image resizing/caption regeneration.
    bm=R/'outputs/hf_dataset_backup_20261008/upload/backup_manifest.json';backup=read(bm);bind(bm)
    require(file_hash(bm)=='f8025f67cc4ef0ba9d91ff72e0d405700d5e56f2de8365cd466c2ecd52dc1d56','Original source backup manifest changed')
    original={x['path']:x for x in backup['files'] if x['path'].startswith('data/yelp_photos/photos/')}
    pm=R/'data/processed/yelp_2300_v1/photo_manifest.csv';bind(pm)
    photos=collections.defaultdict(list)
    for ph in csv.DictReader(pm.open(encoding='utf8')):
        if ph['split']!='test':continue
        pid=ph['photo_id'];p=R/'data/processed/yelp_2300_v1'/ph['relative_path'];h=file_hash(p)
        orig=original['data/yelp_photos/photos/'+pid+'.jpg'];require(h==ph['sha256']==orig['sha256'] and p.stat().st_size==int(ph['bytes'])==orig['bytes'],'Original photo identity mismatch')
        photos[ph['business_id']].append(dict(photo_id=pid,path=p.relative_to(R).as_posix(),bytes=p.stat().st_size,sha256=h,
            original_source_path=orig['path'],original_source_sha256=orig['sha256']))
    require(sum(map(len,photos.values()))==1136,'Original TEST photos incomplete')
    # Include exact raw selected reviews only. Do not open Teacher or TEST reference datasets.
    selected={rid for b in test for rid in shared[b]['selected_review_ids']}
    reviews={r['review_id']:r for r in jl(R/'data/processed/yelp_2300_v1/reviews/test.jsonl') if r['review_id'] in selected}
    A=O/'TEST100_REFERENCE_ANNOTATION_PACKAGE';A.mkdir();templates=[];track=[];source_bundles=[]
    for b in test:
        sh=shared[b];rr=[reviews[rid] for rid in sh['selected_review_ids']]
        require(all(r['business_id']==b for r in rr),'Foreign review source')
        require([text_hash(r['text']) for r in rr]==sh['review_text_sha256_in_order'],'Review text changed')
        bundle=dict(business_id=b,split='test',business_metadata=sh['metadata'],selected_review_ids=sh['selected_review_ids'],
            selected_review_block=sh['review_block'],review_block_sha256=sh['review_block_sha256'],
            reviews=[{k:r[k] for k in ['review_id','business_id','stars','date','text']} for r in rr],photos=sorted(photos[b],key=lambda x:x['photo_id']),
            photo_scope='All original photos belonging to this business, sorted by photo ID; not restricted to U/F selected photos; no donor photos',
            contains_predictions=False,contains_teacher_targets=False,captions_provided=False)
        source_bundles.append(bundle);write(A/'SOURCES'/f'{b}.json',bundle)
        ref=dict(business_id=b,source_bundle_sha256=digest(bundle),review_consensus='',visual_evidence='',review_claims=[],visual_claims=[],status='NOT_STARTED',
            author=dict(human_id='',at_utc='',human_written_attestation=False,predictions_seen=False),
            checker=dict(human_id='',at_utc='',independent_check_attestation=False,predictions_seen=False,reviewed_content_sha256=''),history=[],notes='')
        templates.append(ref);write(A/'TEST100_REFERENCE_TEMPLATE'/f'{b}.json',ref)
        track.append(dict(business_id=b,source_bundle_sha256=digest(bundle),status='NOT_STARTED',author_id='',checker_id='',authored_at_utc='',checked_at_utc='',approved_content_sha256='',approved_reference_file_sha256=''))
        lines=[f'# Reference source case {b}',f"Business: {sh['metadata']['name']}",'','Reference fields are blank in the corresponding JSON template. Review only these selected reviews. Inspect ALL original photos listed below; no model outputs or Teacher targets are provided.','']
        for r in rr:lines.extend([f"## Review {r['review_id']} | stars {r['stars']} | {r['date']}",r['text'],''])
        lines+=['## Original photos','Photo links resolve in the thesis workspace. Use materialize_reference_photos_v1.py for a portable copy; check the manifest SHA first.','']
        for ph in bundle['photos']:lines.append(f"- [{ph['photo_id']}]({(R/ph['path']).as_posix()}) — SHA256 `{ph['sha256']}`")
        p=A/'CASES'/f'{b}.md';p.parent.mkdir(exist_ok=True);p.write_text('\n'.join(lines)+'\n',encoding='utf8')
    write(A/'TEST100_REFERENCE_TEMPLATE.json',templates);write_csv(A/'REFERENCE_APPROVAL_TRACKER.csv',track)
    write(A/'ANNOTATION_PACKAGE_MANIFEST.json',dict(businesses=100,selected_reviews=len(selected),original_photos=1136,photos_not_embedded=True,model_predictions_and_identity_key_excluded=True,
        frozen_sources=[dict(business_id=s['business_id'],sha256=digest(s),source_file=f"SOURCES/{s['business_id']}.json",reviews=len(s['reviews']),photos=len(s['photos'])) for s in source_bundles],
        original_backup=dict(repo=backup['repository'],revision='7e38650c9d93effab7c33d1ec3dd4c4ececf0965',manifest_sha256=file_hash(bm)),
        human_written=0,independently_approved=0,status='PREPARATION_ONLY_BLANK_REFERENCES'))
    # Freeze original model-visible calls and all source/token/checkpoint identities.
    calls=[];mapby={x['recipient_id']:x for x in mapping['mappings']}
    for condition in ['RO','U','F','S']:
        arm='F' if condition=='S' else condition
        for b in test:
            raw=sources[{'RO':'reviews_only','U':'unfiltered','F':'filtered','S':'filtered'}[condition]][b]
            if condition=='RO':rs='[TASK_REVIEW_CONSENSUS]\n'+raw['source'];vs=None
            elif condition=='S':rs=mapby[b]['review_task_source'];vs=mapby[b]['visual_task_source']
            else:rs,vs=isolated(raw['source'])
            require(rs=='[TASK_REVIEW_CONSENSUS]\n'+sources['reviews_only'][b]['source'],'Review not matched')
            for task,source in [('review',rs)]+([] if vs is None else [('visual',vs)]):
                ids=tok.encode(source).ids;require(len(ids)<=5000 and ids[0]==0 and ids[-1]==2,'Task tokenization')
                reuse='F' if condition=='S' and (task=='review' or not mapby[b]['visual_source_changed']) else None
                calls.append(dict(condition=condition,checkpoint_condition=arm,business_id=b,task=task,source=source,source_sha256=text_hash(source),
                    input_ids=ids,input_ids_sha256=digest(ids),global_attention='BOS_ONLY',padding='right to multiple 1024, PAD=1',
                    selected_review_ids=raw['selected_review_ids'] if task=='review' else [],
                    selected_photo_ids=(mapby[b]['donor_photo_ids'] if condition=='S' else raw['selected_photo_ids']) if task=='visual' else [],
                    checkpoint_sha256=freeze['models'][arm]['adapter_sha256'],reuse_condition=reuse,
                    reuse_reason='Same F checkpoint, exact source/token/decoder identities; no independent generation' if reuse else None))
    require(len(calls)==700,'Logical calls')
    write(O/'TEST100_INFERENCE_PLAN.json',dict(records=400,logical_components=700,new_generation_calls=sum(x['reuse_condition'] is None for x in calls),
        F_Review_reuse_for_S=100,additional_identical_Visual_reuse=sum(x['reuse_condition'] is not None and x['task']=='visual' for x in calls),calls=calls,
        execute_now=False,reference_requirement='All100 human-written and second-human checked, protocol reviewed, explicit subsequent execution approval and available compute'))
    # Prospective simple random 30-business human panel, independent of sources/results.
    ranked=sorted(test,key=lambda b:(digest([20261028,'HUMAN_PANEL',b]),b));panel=sorted(ranked[:30])
    write(O/'HUMAN_EVALUATION_COHORT30.json',dict(seed=20261028,rule='Smallest30 SHA256(canonical [seed,HUMAN_PANEL,business_id]), ties business ID; source/outcome independent',business_ids=panel,cohort100_sha256=digest(test),status='DESIGN_ONLY_NO_SCORING'))
    private=[];blanks=[]
    criteria=['review_faithfulness','aspect_coverage','coherence','non_redundancy','usefulness','visual_grounding_and_contribution']
    for annotator in ['A01','A02','A03']:
        units=sorted([(b,c) for b in panel for c in ['RO','U','F','S']],key=lambda v:digest([42,'presentation',annotator,*v]))
        for i,(b,c) in enumerate(units,1):
            alias=digest([42,'blind',annotator,i,b,c])[:20]
            private.append(dict(alias=alias,annotator=annotator,position=i,business_id=b,condition=c))
            blanks.append(dict(alias=alias,annotator=annotator,position=i,**{k:'N/A' if c=='RO' and k.startswith('visual_') else '' for k in criteria},evidence_notes='',submitted_at_utc=''))
    write_csv(O/'PRIVATE_HUMAN_DESIGN_KEY.csv',private);write_csv(O/'HUMAN_RATING_TEMPLATE_DO_NOT_DISTRIBUTE_YET.csv',blanks)
    # Direct proposal extraction, including rubric cells. Only source method paragraphs.
    p=R/'thesis_proposal_v16.docx';bind(p);ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    with zipfile.ZipFile(p) as z:root=ET.fromstring(z.read('word/document.xml'))
    ps=[''.join(t.text or '' for t in p.findall('.//w:t',ns)) for p in root.findall('.//w:p',ns)]
    write(E/'PROPOSAL_V16_ALIGNMENT.json',dict(proposal_sha256=file_hash(p),index='zero-based w:p including table cells',excerpts=[dict(paragraph=i,text=ps[i]) for i in [732,898,900,902,929,936,938,940,942,944,946,*range(948,978)]]))
    write(O/'COHORT100_FREEZE.json',dict(split='test',business_ids=test,cohort_sha256=digest(test),business_split_sha256=file_hash(R/'data/processed/yelp_2300_v1/business_split.csv'),removed=0,replacement83=False))
    write(E/'INPUT_BINDINGS.json',bindings)
    write(E/'PREPARATION_COUNTS.json',dict(businesses=100,selected_reviews=len(selected),original_photos=1136,source_cases=len(source_bundles),blank_references=100,
        human_authored=0,human_checked=0,logical_components=len(calls),prospective_new_generation_calls=sum(x['reuse_condition'] is None for x in calls),
        mapping_active_source_changes=len(active),human_panel=30,human_rating_sets=360,TEST_generations=0,TEST_scores=0,GPU_work=0))
    print(json.dumps(read(E/'PREPARATION_COUNTS.json'),indent=2))

if __name__=='__main__':build()
