"""Build a compact OFFLINE proposal; no model/torch import and no inference."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import numpy as np
from tokenizers import Tokenizer
from stage_i_led_large_nbest_v1 import EXPERIMENT,POLICY,source_case,sha,read,generation_pair

ROOT=Path(__file__).resolve().parents[1]
PREP=ROOT/'outputs/stage_i/led_large_nbest_reranking_v1'
WORKSPACE=ROOT/'outputs/cloud_handoff/LED_LARGE_DECODER_POLICY_V2_WORKSPACE'
OLD=ROOT/'outputs/stage_i/led_large_decoder_policy_v2_execution_v1/final_evidence/runs/F_large_decoder_policy_v2'


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    PREP.mkdir(parents=True,exist_ok=True)
    cfg=read(ROOT/'configs/stage_i_led_large_decoder_policy_v2.json')
    parent_paths=['configs/stage_i_led_large_decoder_policy_v2.json','scripts/run_stage_i_led_large_decoder_policy_v2.py',
        'scripts/stage_i_led_large_decoder_policy_v2.py','scripts/led_large_decoder_early_sanity_v2.py',
        'scripts/led_large_decoder_label_view_v1.py','outputs/stage_i/led_large_decoder_policy_v2/PREFLIGHT_MANIFEST.json']
    parent={p:sha(ROOT/p) for p in parent_paths}
    accepted=['d079f41aa5ae5b1a680da5174eb3ef83235cb9b82e54c96b3d5d487215a168f4',
        'bea436309fadb632a04db82549ad63139f31bd77365e95271b77c1ff68470a61',
        'a37f7ee37047df01051d9e13cd45ab4ee0d189f3167f2b4167d384ae581a4777',
        '10975bf25d559c248320e7070ea45387bce6cf2f8dde48dc02f7974e64b39306',
        '413a408ed384bdd5d40ef21ea7c99891a2ee42e88554a1f6e543d1ad6f535b96',
        '782de3373434ff01b934818cd143e0d6c0be32fc0658e924c19ba2711401a3d3']
    assert list(parent.values())==accepted
    assert all(sha(WORKSPACE/p)==h for p,h in parent.items())
    checkpoint=OLD/'train/checkpoint-750';proof=read(checkpoint/'checkpoint_integrity.json')
    assert proof['global_step']==750 and proof['binding']['config_sha256']==accepted[0]
    assert all(sha(checkpoint/n)==h for n,h in proof['files'].items())
    save(PREP/'CHECKPOINT_BINDING.json',proof)
    original=read(ROOT/cfg['effective_generation_config_file'])
    assert original==read(OLD/'evaluate/EFFECTIVE_GENERATION_CONFIG.json')
    single,n4=generation_pair(original)
    save(PREP/'ORIGINAL_GENERATION_CONFIG.json',single);save(PREP/'NBEST_GENERATION_CONFIG.json',n4)
    tok=Tokenizer.from_file(str(ROOT/'assets/tokenizers/led-base-16384/tokenizer.json'))
    all_cases={};input_bindings={};zero_train=[]
    for split in ('train','validation'):
        rel=f'outputs/stage_g/teacher_production_d042_v1/guarded_production6_retry_v1/dataset/{split}_filtered.jsonl'
        packrel=f'outputs/stage_h/led_dataset_v1/{split}_filtered.npz'
        input_bindings.update({rel:sha(ROOT/rel),packrel:sha(ROOT/packrel)})
        rows=[json.loads(x) for x in (ROOT/rel).read_text(encoding='utf-8').splitlines()]
        with np.load(ROOT/packrel,allow_pickle=False) as pack:
            arrays={k:pack[k] for k in ('business_ids','input_ids','input_offsets','caption_counts')}
        assert arrays['business_ids'].tolist()==[r['business_id'] for r in rows]
        # Input projection only. Original target text/labels never enter any selection file.
        byid={}
        for i,row in enumerate(rows):
            a,b=arrays['input_offsets'][i:i+2]
            byid[row['business_id']]=source_case(row,arrays['input_ids'][a:b].tolist())
        all_cases[split]=byid
    assert len(all_cases['train'])==2000 and len(all_cases['validation'])==200
    assert not set(all_cases['train'])&set(all_cases['validation'])
    order=lambda x:hashlib.sha256(('LED_NBEST_V1_TRAIN_ONLY:'+x['business_id']).encode()).hexdigest()
    zeros=sorted([x for x in all_cases['train'].values() if not x['captions']],key=order)
    other=sorted([x for x in all_cases['train'].values() if x['captions']],key=order)
    pilot=zeros[:3]+other[:9]
    for case in pilot:
        assert tok.encode(case['source']).ids==case['input_ids']
    save(PREP/'TRAIN_SOURCE_ONLY_CASES.json',pilot)
    save(PREP/'TRAIN_SAMPLE_MANIFEST.json',dict(selection='SHA256(LED_NBEST_V1_TRAIN_ONLY:+business_id), first3 zero-caption then first9 nonzero-caption; source metadata only',
        train_population=2000,zero_caption_population=len(zeros),n=12,strata={'zero_captions':3,'nonzero_captions':9},
        representative_population_estimate=False,case_ids=[x['business_id'] for x in pilot],
        names=[x['source'].split('name=',1)[1].split('\n')[0] for x in pilot]))
    baseline=read(OLD/'quality_review/predictions/epoch3.json')
    validation=[all_cases['validation'][x['business_id']] for x in baseline]
    for case,row in zip(validation,baseline):
        assert case['source_sha256']==row['source_sha256']
        assert row['checkpoint_sha256']==proof['files']['adapter_model.safetensors']
        assert tok.encode(case['source']).ids==case['input_ids']
    save(PREP/'VALIDATION_SOURCE_ONLY_CASES.json',validation)
    save(PREP/'identity_only/HISTORICAL_CE_EPOCH3.json',baseline)
    save(PREP/'RERANKING_POLICY.json',dict(experiment=EXPERIMENT,status='PRESPECIFIED_PROPOSAL_NOT_VALIDATION_LOCKED',
        weights=POLICY,selection_information='model-visible reviews and condition-specific captions only',
        frozen_before_validation_required=True,prohibited=['Teacher targets','D053 references','validation labels','photos','held-out data'],
        backend='Conditional source-judgment interface; manual or model-assisted scorer NOT yet validated/bound',
        no_candidate_editing=True,all_rejected='ABSTAIN_AND_REPORT_NO_SILENT_FALLBACK',
        tie_break='original generation order',quality_GO_automatic=False,
        diversity_gate=dict(n=12,materially_diverse_min=6,oracle_material_gain_min=4),
        verifier_gate=dict(claim_audit_max=96,min_supported=20,min_errors=20,precision_min=.9,recall_min=.8,
                           supported_false_rejection_max=.1,uncertain_fraction_max=.2)))
    config=dict(experiment=EXPERIMENT,preparation=PREP.relative_to(ROOT).as_posix(),
        parent_bindings=parent,runtime_pins=cfg['runtime_pins'],backbone_id=cfg['model_id'],backbone_revision=cfg['revision'],
        backbone_sha256=cfg['weights_sha256'],adapter_sha256=proof['files']['adapter_model.safetensors'],
        original_checkpoint='Original CE DecoderPolicyV2 epoch3 step750; never WCE or base',
        generation_override={'num_return_sequences':4},unchanged_num_beams=4,
        output_names={'training_pilot':'F_CE3_nbest_training_v1','validation_pair':'F_CE3_nbest_validation_v1'},
        budgets={'training_pilot':{'businesses':12,'generate_calls':24,'returned_sequences':60},
                 'validation_pair':{'businesses':24,'generate_calls':48,'returned_sequences':120}},
        GPU_inference_approved=False,training_authorized=False,held_out_access=False,contrastive_authorized=False)
    save(ROOT/'configs/stage_i_led_large_nbest_v1.json',config)
    # Static pinned-source inspection without importing the incompatible global Transformers runtime.
    dist=importlib.metadata.distribution('transformers')
    runtime={}
    for name in ('generation/utils.py','generation/beam_search.py'):
        path=Path(dist.locate_file('transformers/'+name));runtime[name]=sha(path)
    code=Path(dist.locate_file('transformers/generation/beam_search.py')).read_text(encoding='utf-8')
    assert 'sorted_hyps.pop()' in code and 'self.num_beam_hyps_to_keep' in code
    save(PREP/'OFFLINE_INTEGRITY.json',dict(parent_six_bindings_verified=True,checkpoint_components_verified=len(proof['files']),
        checkpoint_receipt_sha256=sha(checkpoint/'checkpoint_integrity.json'),frozen_input_bindings=input_bindings,
        runtime_static_source_sha256=runtime,models_loaded=0,inference_calls=0,optimizer_updates=0,
        held_out_content_access=False,historical_reproduction='NOT_RUN_REQUIRES_APPROVAL',
        semantic_diversity='UNMEASURED',oracle_quality='UNMEASURED',reranker_accuracy='UNMEASURED',
        actual_inference_cost='UNMEASURED'))
    save(PREP/'TRAIN_DIVERSITY_REVIEW_TEMPLATE.json',dict(split='train',assessor_kind='researcher',
        source_only_review_confirmed=False,cases=[dict(business_id=x['business_id'],materially_distinct_profiles=None,
        oracle_material_gain_without_grounding_regression=None,rationale='',oracle_acceptable_candidate_ids=[],
        core_or_salient_coverage={},opinion_balance={},grounding={},visual_usefulness={},overall_usability={}) for x in pilot]))
    save(PREP/'VERIFIER_AUDIT_TEMPLATE.json',dict(split='train',assessor_kind='researcher',
        independent_of_provisional_verifier=False,claims=[],sampling='Up to96 deterministic claim keys across all12 TRAIN cases; include zero-caption assertions; deduplicate repeated same-source claims; uncertain separate; no outcome cherry-picking'))
    save(PREP/'POLICY_LOCK_TEMPLATE_NOT_LOCKED.json',dict(experiment=EXPERIMENT,bindings=None,policy_sha256=None,
        training_only=False,researcher_approved_for_validation=False,reviewer=None,
        pilot_completion={'file':None,'sha256':None},pilot_candidates={'file':None,'sha256':None},
        training_assessment={'file':None,'sha256':None},verifier_audit={'file':None,'sha256':None},
        verifier_backend_lock={'kind':None,'identity_version':None,'prompt_sha256':None,'training_only_verified':False}))
    save(PREP/'AUTHORIZATION_TEMPLATE_NOT_APPROVED.json',dict(experiment=EXPERIMENT,GPU_inference_approved=False,
        researcher_explicit_authorization=False,bindings=None,phases=['training_pilot'],authorized_at_utc=None,
        deadline_utc=None,generation_calls_max=24,returned_sequences_max=60,training_authorized=False,
        held_out_access=False,Teacher_calls=False,automatic_retry=False))
    print('OFFLINE INPUTS PREPARED: 12 TRAIN / 24 validation; no model operation',flush=True)


if __name__=='__main__':
    main()
