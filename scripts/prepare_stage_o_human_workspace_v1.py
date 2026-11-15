"""Prepare the adapted Stage N application only after all real Stage O outputs exist.

No placeholders, ratings, approvals or reference summaries are generated.
"""
from pathlib import Path
import hashlib,json,random,secrets,shutil
from stage_o_tripadvisor_feasibility_v1 import ROOT,OUT,save,digest
from stage_o_frozen_pipeline_v1 import verify_freeze,jl
from prepare_stage_o_tripadvisor_cohort_v1 import h


def prepare():
    frozen=verify_freeze();planpath=OUT/'LED_INFERENCE_CALL_PLAN_75.json'
    if not planpath.exists():raise RuntimeError('PENDING: real captions and frozen 75-call plan are required. No human workspace or ratings were created.')
    plan=json.loads(planpath.read_text(encoding='utf8'));calls={(c['condition'],c['business_id'],c['task']):c for c in plan['calls']}
    assert len(calls)==75
    rows=[]
    for condition in ['RO','U','F']:
        file=OUT/'LED_OUTPUTS'/condition/'raw_predictions.json'
        if not file.exists():raise RuntimeError('PENDING: complete real '+condition+' outputs are required; no placeholder summaries.')
        rows.extend(json.loads(file.read_text(encoding='utf8')))
    assert len(rows)==75
    from source_budget_audit_template import pinned_tokenizer
    tok,_=pinned_tokenizer();bykey={}
    for r in rows:
        key=(r['condition'],r['business_id'],r['task']);assert key in calls and key not in bykey
        c=calls[key];assert r['checkpoint_sha256']==c['checkpoint_sha256'] and r['source_sha256']==c['source_sha256'] and r['input_ids_sha256']==c['input_ids_sha256']
        assert hashlib.sha256(r['text'].encode()).hexdigest()==r['text_sha256']
        assert tok.decode(r['token_ids'],skip_special_tokens=True)==r['text']
        assert r['generation_settings']==c['generation']
        bykey[key]=r
    workspace=OUT/'HUMAN_APP/WORKSPACE'
    assert not workspace.exists(),'Do not overwrite an existing human study or ratings.'
    (workspace/'COORDINATOR_ONLY').mkdir(parents=True)
    shared={r['business_id']:r for r in jl(OUT/'PREPROCESSING/shared_review_inputs.jsonl')}
    sources={}
    for item in frozen['restaurants']:
        b=item['business_id'];original=json.loads((OUT/item['source_file']).read_text(encoding='utf8'));selected=shared[b]
        photos=[]
        for p in original['photos']:
            dest=workspace/'PHOTOS'/(p['photo_id']+'.jpg');dest.parent.mkdir(exist_ok=True)
            shutil.copyfile(OUT/p['file'],dest)
            assert digest(dest)==p['sha256']
            photos.append(dict(photo_id=p['photo_id'],bytes=p['bytes'],sha256=p['sha256'],original_source_sha256=p['sha256']))
        source=dict(business_id=b,split='external_tripadvisor',business_metadata=original['business_metadata'],selected_review_ids=selected['selected_review_ids'],
                    selected_review_block=selected['review_block'],review_block_sha256=selected['review_block_sha256'],reviews=selected['reviews'],photos=photos,
                    photo_scope='All accessible frozen publisher-linked source photographs, common across blinded conditions.',contains_predictions=False,contains_teacher_targets=False,captions_provided=False)
        path=workspace/'SOURCES'/(b+'.json');save(path,source)
        sources[b]={'file':path.relative_to(workspace).as_posix(),'canonical_sha256':h(source)}
    private=[];seeds={}
    for rater in ['A01','A02']:
        seeds[rater]=secrets.token_hex(32);rng=random.Random(int(seeds[rater],16))
        assignment=[(x['business_id'],c) for x in frozen['restaurants'] for c in ['RO','U','F']];rng.shuffle(assignment)
        for pos,(b,c) in enumerate(assignment,1):
            review=bykey[c,b,'review'];visual=bykey.get((c,b,'visual'))
            private.append(dict(alias=hashlib.sha256((seeds[rater]+'|'+b+'|'+c).encode()).hexdigest()[:20],annotator=rater,position=pos,business_id=b,condition=c,
                                review_text=review['text'],review_text_sha256=review['text_sha256'],
                                visual_text=visual['text'] if visual else None,visual_text_sha256=visual['text_sha256'] if visual else None))
    assert len(private)==90 and len({r['alias'] for r in private})==90
    save(workspace/'COORDINATOR_ONLY/PRIVATE_ITEMS.json',private);save(workspace/'COORDINATOR_ONLY/RANDOMIZATION_SEEDS.json',seeds)
    save(workspace/'RUBRIC.json',json.loads((OUT/'PROTOCOL/RUBRIC.json').read_text(encoding='utf8')))
    shutil.copyfile(OUT/'PROTOCOL/HUMAN_EVALUATION_PROTOCOL.md',workspace/'TWO_RATER_PROTOCOL_AMENDMENT.md')
    config=dict(format='stage_o_reference_free_human_v1',expected_records=90,conditions=['RO','U','F'],raters=['A01','A02'],businesses=15,
                sources=sources,synthetic=False,cohort_freeze_sha256=digest(OUT/'COHORT15_FREEZE.json'),file_bindings={})
    config['file_bindings']={p.relative_to(workspace).as_posix():digest(p) for p in [workspace/'COORDINATOR_ONLY/PRIVATE_ITEMS.json',workspace/'RUBRIC.json',workspace/'TWO_RATER_PROTOCOL_AMENDMENT.md']}
    save(workspace/'CONFIG.json',config)
    print('Prepared 90 blinded assignments. LIVE_RELEASE absent: no ratings can be collected before researcher inspection/release. No accounts or ratings generated.')


if __name__=='__main__':prepare()
