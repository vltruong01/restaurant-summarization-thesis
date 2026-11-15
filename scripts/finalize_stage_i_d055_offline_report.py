"""Assemble D055 reports and proposal only. No GPU, model, inference or API code."""
import ast, collections, hashlib, json, pathlib, re, shutil, struct, sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from stage_i_d053_evaluation import repetition_diagnostics
from stage_i_decoding_diagnostics import words
OUT=ROOT/'outputs/stage_i/d055_root_cause_v2'
CLOUD=ROOT/'outputs/stage_i/cloud_return_20261015_v1/stage_hi_v2'
PREP=ROOT/'outputs/stage_i/paired_lora_pilot_preparation_v1'
REQUEST=pathlib.Path('C:/Users/VUONGLOCTRUONG/.codex/attachments/87acd965-b3fe-46d5-8382-013f7088e7c3/Pasted text.txt')
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,x):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='\n') as f:f.write((x if isinstance(x,str) else json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))+'\n')
def require(c,m):
    if not c:raise ValueError(m)
def main():
    receipt=read(OUT/'EXECUTION_RECEIPT.json');require(receipt['all_unchanged'],'Audit incomplete')
    t=read(OUT/'TARGET_AUDIT.json');summary=read(OUT/'HISTORICAL_VS_PILOT_SUMMARY.json');exposure=read(OUT/'TRAINING_EXPOSURE_AND_PROVENANCE.json')
    outputs=read(OUT/'EXISTING_72_OUTPUTS_DIAGNOSTICS.json')
    source=ROOT/'outputs/stage_i/d053_researcher_reference_finalization_v1/source_bundle/stage_i_d053_reference_verification_chatgpt_bundle_v2'
    cases=read(source/'source_cases.json');byid={c['business_id']:c for c in cases}
    # Preserve bytes as well as a convenient normalized text view.
    (OUT/'RESEARCHER_REQUEST_ORIGINAL.txt').write_bytes(REQUEST.read_bytes())
    write('RESEARCHER_REQUEST.txt',REQUEST.read_text(encoding='utf-8'))
    write('RESEARCHER_AUTHORIZATION.json',{'decision':'D-055','approval_source':'Explicit user request in attached text, not a fabricated UI action','request_sha256':sha(REQUEST),
        'offline_investigation_authorized':True,'GPU_execution_authorized':False,'training_authorized':False,'teacher_calls_authorized':False,
        'input_caveat':'Original supplied text ends mid-sentence in deliverable8; substantive scope and restrictions are explicit and were preserved.'})
    # Snapshot actual trained adapter file geometry using the safetensors JSON header only.
    adapters={}
    for arm,x in exposure.items():
        p=ROOT/x['adapter_path']
        with p.open('rb') as f:header=json.loads(f.read(int.from_bytes(f.read(8),'little')))
        tensors={k:v for k,v in header.items() if k!='__metadata__'}
        n=sum(__import__('math').prod(v['shape']) for v in tensors.values())
        require(n==(589824 if arm=='B_encoder_decoder' else 294912),'Actual adapter shape mismatch')
        require(all(v['dtype']=='F32' for v in tensors.values()),'Unexpected adapter storage precision')
        adapters[arm]={'adapter_sha256':sha(p),'actual_tensor_count':len(tensors),'actual_parameter_count':n,'tensors':tensors,
            'interpretation':'Header geometry verified without creating model or evaluating tensors.'}
    write('ACTUAL_ADAPTER_GEOMETRY.json',adapters)
    # Verify original cloud bindings and actual completed pilot source hashes.
    bindings=read(CLOUD/'outputs/stage_i/cloud_training_v2/run_bindings.json')
    proofs={p:sha(CLOUD/p)==h for p,h in bindings.items()};require(all(proofs.values()),'Historical cloud binding mismatch')
    pilot={}
    for arm in ['A_decoder_only','B_encoder_decoder']:
        bind=read(ROOT/'outputs/stage_i/paired_lora_pilot_runs_v1'/arm/'train/completion.json')['binding']
        pilot[arm]={'runner_matches_training_binding':sha(ROOT/'scripts/run_stage_i_paired_lora_pilot_v1.py')==bind['runner_sha256'],
            'helper_matches_training_binding':sha(ROOT/'scripts/stage_i_paired_pilot.py')==bind['helper_sha256']}
        require(all(pilot[arm].values()),'Pilot source binding mismatch')
    write('CODE_PATH_BINDINGS.json',{'historical':proofs,'pilots':pilot,'actual_pilot_wrapper':'scripts/run_stage_i_d052_paired_pilot_v2.py',
        'limitation':'Historical package versions match; original installed library source and live Accelerator accumulation state were not snapshotted at training time.'})
    peftroot=pathlib.Path(sys.executable).resolve().parent.parent/'Lib/site-packages/peft'
    # The caption venv shares the global Python site-packages. Explicit known installed location.
    if not peftroot.exists():peftroot=pathlib.Path('C:/Users/VUONGLOCTRUONG/AppData/Local/Programs/Python/Python311/Lib/site-packages/peft')
    idx={}
    for rel,fn in [('tuners/lora/layer.py','reset_lora_parameters'),('tuners/lora/model.py','_mark_only_adapters_as_trainable')]:
        p=peftroot/rel;txt=p.read_text(encoding='utf-8');tree=ast.parse(txt)
        node=next(x for x in ast.walk(tree) if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) and x.name==fn)
        write('implementation/PEFT_'+fn+'.txt',f'# {p}; file SHA256 {sha(p)}; lines{node.lineno}-{node.end_lineno}\n'+'\n'.join(txt.splitlines()[node.lineno-1:node.end_lineno]))
        idx[fn]={'source_path':str(p),'sha256':sha(p),'first_line':node.lineno,'last_line':node.end_lineno}
    write('PEFT_IMPLEMENTATION_INDEX.json',idx)
    # Whole accepted teacher pool surface diagnostics, reusing accepted rules unchanged.
    manifest=read(ROOT/'outputs/stage_h/led_dataset_v1/manifest.json');pool={};pair_examples=[]
    for k,info in manifest['datasets'].items():
        data=[json.loads(s) for s in (ROOT/info['stage_g_jsonl']).read_text(encoding='utf-8').splitlines() if s.strip()]
        reps=[repetition_diagnostics(c['target']) for c in data]
        phrase=[v['lexical_ngrams']['4']['duplicate_fraction'] for v in reps if v['lexical_ngrams']['4']['duplicate_fraction'] is not None]
        pool[k]={'n':len(data),'exact_sentence_repetition':sum(v['exact_repeated_sentence'] for v in reps),
            'adjacent_word_flags':sum(v['consecutive_repeated_word'] for v in reps),'mean_lexical4_duplicate_fraction':sum(phrase)/len(phrase),
            'replacement_character_targets':sum('\ufffd' in c['target'] for c in data),
            'replacement_character_target_ids':[c['business_id'] for c in data if '\ufffd' in c['target']],
            'target_nbsp_characters':sum(c['target'].count('\u00a0') for c in data)}
        if k=='train_reviews_only':
            for c,v in zip(data,reps):
                if v['consecutive_repeated_word'] and len(pair_examples)<8:
                    w=words(c['target']);pair_examples.append({'business_id':c['business_id'],'adjacent_pairs':[w[i] for i in range(len(w)-1) if w[i]==w[i+1]]})
    write('TEACHER_SURFACE_DIAGNOSTICS.json',{'by_dataset':pool,'adjacent_examples':pair_examples,
        'interpretation':'All395 adjacent-word flags need contextual reading; Dan Dan, so-so, 50-50 and repeated nouns across punctuation can be legitimate. None of6600 targets has an exact repeated sentence under the accepted definition. Replacement characters are inherited accepted content, not silently repaired here; this audit does not establish their impact on training.'})
    # Descriptive degeneration counts, not new success thresholds.
    taxonomy={}
    for arm in exposure:
        z=[x for x in outputs if x['arm']==arm]
        taxonomy[arm]={'outputs':len(z),'starts_with_correct_marker_text':sum(x['text'].startswith('[REVIEW_CONSENSUS]') for x in z),
            'correct_marker_and_newline':sum(x['prefix']['required_header_prefix_complete'] for x in z),
            'first_header_mismatch_positions':summary[arm]['all24']['header_mismatch_positions'],
            'nonbreaking_space_outputs':sum(x['nonbreaking_space_count']>0 for x in z),
            'max_nbsp_characters':max(x['nonbreaking_space_count'] for x in z),
            'source_only_exact_markers':summary[arm]['all24']['outputs_with_source_only_markers'],
            'whitespace_fraction_over_point95':sum(x['whitespace_fraction']>.95 for x in z),
            'interpretation':'Overlapping descriptive tags; .95 whitespace is a descriptive display cutoff, not a tuned scientific acceptance gate. Full fractions and raw text are provided.'}
    write('OUTPUT_DEGENERATION_TAXONOMY.json',taxonomy)
    # Source-linked illustrative findings. Not researcher-confirmed output annotations.
    findings=[]
    def add(arm,bid,claim,label,reason,evidence):
        p=next(x for x in outputs if x['arm']==arm and x['business_id']==bid)
        start=p['text'].find(claim);require(start>=0,'Illustrative claim missing')
        findings.append(dict(arm=arm,business_id=bid,claim=claim,output_span=[start,start+len(claim)],classification=label,
            explanation=reason,evidence=evidence,human_verified=False,assessment_origin='Codex source-based inspection; no new human factual labels or references'))
    c=byid['HhD8DLES5ZeULKDXmfkOOg'];cap=next(x for x in c['captions'] if x['caption_id']=='IMG_CAP_5')
    add('historical_full_F',c['business_id'],'A white plate is topped with shrimp and grits.','caption_supported_specific_information',
        'Ordinary visual content is supported by the LED-visible caption. Subsequent repetitions reduce usefulness. No new photograph factual adjudication is claimed.',[cap])
    c=byid['j3kGicU3mPG_ciIKG1Vuyg']
    add('historical_full_F',c['business_id'],'The tray contains cream cheese.','wrong_evidence_domain_no_visual_input',
        'Filtered caption pool is empty. A review mentioning bagels cannot authorize a visual claim. All three models fail empty-visual compliance.',[])
    c=byid['JcdOQvvx3BqZ5Ga2_9n_Bg']
    add('historical_full_F',c['business_id'],'The bar has a red liquid.','unsupported_relationship',
        'IMG_CAP_3 describes wine poured into a glass, not liquid possessed by a bar. No supplied caption establishes this relationship.',c['captions'])
    c=byid['-Wd5KqFszREg0XdXfSv5tg'];cap=next(x for x in c['captions'] if x['caption_id']=='IMG_CAP_2')
    add('historical_full_F',c['business_id'],'A man is pouring coffee into a cup.','caption_supported_specific_information',
        'The action and cup are in IMG_CAP_2. Capturing one supported fact does not validate unrelated statements or generalization in the summary.',[cap])
    c=byid['aZN5R7Xfea0GkDBwBZOQrQ'];rev=next(x for x in c['reviews'] if x['marker']=='REV_26')
    # Locate exact produced wording instead of silently paraphrasing it.
    p=next(x for x in outputs if x['arm']=='B_encoder_decoder' and x['business_id']==c['business_id'])
    m=re.search(r'[^.!?]*overcooked[^.!?]*[.!?]?',p['text'],re.I)
    require(m is not None,'Expected existing D054 example missing')
    add('B_encoder_decoder',c['business_id'],m.group().strip(),'unsupported_or_reversed_review_detail',
        'The identified selected review says Still undercooked. Existing D054 inspection found no support for repeatedly generalizing overcooked fish. This is a source-based Codex assessment, not a newly confirmed human verdict.',[rev])
    write('SOURCE_LINKED_QUALITY_EXAMPLES.json',findings)
    # Proposed diagnostic cases are selected from existing fixed24, never from test.
    ids=['JcdOQvvx3BqZ5Ga2_9n_Bg','b1pdnbYvBdtTxJOOom5WmA','j3kGicU3mPG_ciIKG1Vuyg']
    inputs=[json.loads(s) for s in (PREP/'evaluation_inputs.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    selected=[]
    for bid in ids:
        c=next(x for x in inputs if x['business_id']==bid)
        selected.append({'business_id':bid,'subset':c['subset'],'source_sha256':c['source_sha256'],'caption_count':len(c['selected_photo_ids']),
            'input_ids_sha256':hashlib.sha256(json.dumps(c['input_ids'],separators=(',',':')).encode()).hexdigest(),
            'label_ids_sha256':hashlib.sha256(json.dumps(c['label_ids'],separators=(',',':')).encode()).hexdigest()})
    write('PROPOSED_GPU_DIAGNOSTIC_CONFIG.json',{'status':'PROPOSED_NOT_AUTHORIZED_NOT_EXECUTED','cases':selected,
        'model_states':['pretrained_no_adapter','historical_full_F','A_decoder_only','B_encoder_decoder'],
        'adapter_sha256':{k:v['adapter_sha256'] for k,v in exposure.items()},
        'baseline_generation_config_sha256':sha(PREP/'expected_generation_config.json'),
        'weights':'FP32','compute_comparison':['FP16 CUDA autocast','FP32 no autocast'],'batch_size':1,'seed':42,'tf32':False,
        'source_padding_multiple':1024,'global_attention':'token0 only','teacher_forced_limit':768,'source_limit':5000,
        'teacher_forced_forward_budget':24,'cache_equivalence_prefix_lengths':[3,11,12],
        'cache_setup_plus_incremental_forward_budget':144,'cache_off_logits':'Reuse the same full teacher-forced forward at equivalent positions',
        'short_generation_budget':36,'new_token_trace_cap':64,'generation_modes':['greedy_unprefixed','beam4_unprefixed','greedy_correct_header_prefix'],
        'trace_censoring':'Keep effective max_length768; separate diagnostic stopping criterion stops after64 newly produced tokens, without fabricating EOS. This is proposed, requires explicit approval.',
        'cache_allclose_screen':{'FP32':{'atol':0.0001,'rtol':0.0001},'FP16':{'atol':0.05,'rtol':0.005}},
        'cross_precision_screen':'Report absolute/RMS/relative errors, KL, correct-token probability, top10 and argmax changes. Max absolute logit drift>0.1 or argmax changes trigger investigation, not automatic proof of failure.',
        'automatic_fallback':False,'no_training':True,'authorization_required':True})
    # Preserve documentation before appending current D055 state.
    for p in [ROOT/'DECISIONS.md',ROOT/'thesis_sdd_docs/THESIS_PROGRESS.md',ROOT/'thesis_sdd_docs/CODEX_THESIS_IMPLEMENTATION_PLAN.md']:
        dest=OUT/'documentation_before'/p.name;dest.parent.mkdir(exist_ok=True);shutil.copyfile(p,dest)
    print(json.dumps({'status':'SUPPLEMENTAL_OFFLINE_EVIDENCE_READY','taxonomy':taxonomy,'teacher_surface':{k:{n:v for n,v in x.items() if n!='replacement_character_target_ids'} for k,x in pool.items()}},ensure_ascii=True))
if __name__=='__main__':main()
