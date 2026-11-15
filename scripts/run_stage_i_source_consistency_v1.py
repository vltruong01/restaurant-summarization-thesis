"""One exploratory CPU source-support/MMR pass on existing24 validation outputs.

prepare -> fixed manifest -> run once. No LED generation/training/Teacher/test.
"""
import argparse,hashlib,json,os,subprocess,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
from stage_i_repetition_pilot_v1 import ROOT,read,rows,sha,write,deadline
from stage_i_repetition_pilot_v2 import preflight as previous_preflight

PREP=ROOT/'outputs/stage_i/source_consistency_preparation_v1'
OUT=ROOT/'outputs/stage_i/source_consistency_postprocess_v1'
CONFIG=ROOT/'configs/stage_i_source_consistency_postprocess_v1.json'
NLI=ROOT/'assets/experimental_scorers/nli_deberta_small_e989068'
EMBED=Path('C:/Users/VUONGLOCTRUONG/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41')


def prepare():
    previous_preflight(require_fresh=False)
    PREP.mkdir(exist_ok=False)
    c={'experiment':'source_consistency_postprocess_v1','role':'EXPLORATORY_POSTPROCESSOR_NOT_MODEL_LEARNING',
       'businesses':24,'random':20,'targeted':4,'raw_inputs':'outputs/stage_i/repetition_guard_pilot_v2',
       'embedding_id':'sentence-transformers/all-MiniLM-L6-v2','embedding_revision':EMBED.name,'embedding_directory':str(EMBED),
       'NLI_id':'cross-encoder/nli-deberta-v3-small','NLI_revision':'e9890682d9e4279b7ae6d0fcfb435a43206280ec','NLI_directory':str(NLI),
       'CPU_threads':8,'embedding_batch':16,'NLI_batch':8,'CPU_seconds_max':1800,'NLI_pairs_max':5000,
       'evidence_embedding_tokens':220,'review_retrieval_top_k':3,
       'review_entailment_min':.8,'visual_entailment_min':.9,'review_entailment_margin':.6,'visual_entailment_margin':.8,
       'semantic_cosine_min':.85,'mutual_entailment_min':.9,'MMR_lambda':.65,
       'review_sentences_max':10,'visual_sentences_max':5,'output_LED_tokens_max':768,
       'threshold_status':'Fixed exploratory engineering presets; not calibrated probabilities or published optimal parameters. No tuning after scoring.',
       'generation_calls':0,'training_updates':0,'Teacher_calls':0,'held_out_access':False,'GPU_operations':False,
       'gold_targets_or_D053_references_used_for_selection':False,
       'source_and_checkpoint_changes':False,'new_content_generation':False,
       'scope_authorization':'Latest researcher request to research and apply techniques, interpreted as a reversible CPU-only derived-output experiment. No approval of a primary methodology change or new GPU training is inferred.',
       'deadline_UTC':'2026-10-22T07:11:22.846767+00:00','D051_strict_replay':'FAIL_UNCHANGED'}
    write(CONFIG,c)
    _,_,cases,_,_=previous_preflight(require_fresh=False)
    public=[]
    for case in cases:
        p=ROOT/c['raw_inputs']/'raw_cases'/(case['business_id']+'.json');pred=read(p)
        assert pred['source_sha256']==case['source_sha256']
        public.append({k:case[k] for k in ('business_id','subset','source','source_sha256','selected_photo_ids')}|
                      {'raw_prediction':pred['text'],'raw_prediction_sha256':sha(p)})
    write(PREP/'INFERENCE_ONLY_INPUTS.jsonl','\n'.join(json.dumps(p,ensure_ascii=False) for p in public))
    write(PREP/'METHOD_AND_RESEARCH.md',"""# Source-support + semantic selection: exploratory derived outputs

Researcher requested further technical research and application after finding
the guard inadequate. This authorizes one reversible CPU-only experiment on
saved24guard outputs, not new LED training, GPU generation or official adoption.
Same20random+4targeted cases have been repeatedly inspected and are now development
cases; no unbiased validation or held-out generalization claim follows.

Applied components: sentence-level source-support NLI, single-caption visual
binding, mutual-entailment semantic deduplication and MMR sentence selection.
If no captions exist, no visual sentence is admitted. Canonical section wrappers
are explicitly serialized; this is not a model header/EOS improvement.
No words inside a retained sentence are rewritten. No missing content is filled
from sources, Teacher targets or approved references. Empty/insufficient outputs
must be flagged, not called a quality success. Original raw files remain intact.

Only LED-visible review/caption text and raw predictions enter the scorer. Export
whitelists fields and strips Teacher targets, evidence verdicts and D053 references.
Review evidence is retrieved in whole-sentence windows with exact source offsets;
oversized windows/sentences are logged and excluded, never silently truncated.
Review clauses may be compared with a bounded multi-review premise; visual claims
must be compared with ONE caption. Source sentence windows lose document context;
this and all threshold/retrieval errors remain limitations.

Frozen operational thresholds/config are exploratory, not paper-reported optima
or calibrated confidence. NLI/mutual entailment can fail, especially for opinions,
quantifiers and object relations. Text agreement does not independently verify
the original photo or repair BLIP2 caption errors. Embeddings are similarity,
not truth. Preserve positive/negative contrasts; report omissions and abstentions.

Primary sources:
- Laban et al.(2022), SummaC: sentence granularity matters for NLI consistency.
  https://aclanthology.org/2022.tacl-1.10/
- Falke et al.(2019): off-the-shelf NLI reranking can be inadequate.
  https://aclanthology.org/P19-1213/
- Carbonell & Goldstein(1998), MMR: relevance/redundancy selection.
  https://doi.org/10.1145/290941.291025
- Welleck et al.(2020), Unlikelihood, and Xu et al.(2022), DITTO address learned
  repetition through new training objectives; they are distinct methods, not
  implemented by this postprocessor. Additional training/loss changes would need
  a separately reviewed bounded scientific protocol, not a silent continuation.
  https://arxiv.org/abs/1908.04319
  https://proceedings.neurips.cc/paper_files/paper/2022/file/148c0aeea1c5da82f4fa86a09d4190da-Paper-Conference.pdf

This is inspired by NLI evidence comparison/MMR, not a reproduction of trained
SummaC-Conv. Auxiliary models: pinned MiniLM sentence encoder and DeBERTa NLI;
no replacement of the LED backbone, quantization or paid inference provider.
No reference labels enter decisions. Evaluate only AFTER all decisions are saved,
separating automated model scores from evidence inspection/researcher acceptance.
No automatic further variant, threshold search, main RO/U/F/test deployment or
deadline reset. Preserve D051 strict replay FAIL and all scientific artifacts.
""")
    with (PREP/'OFFLINE_TESTS.log').open('wb') as f:
        result=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_stage_i_source_consistency_v1.py','-v'],stdout=f,stderr=subprocess.STDOUT)
    assert result.returncode==0
    write(PREP/'TEST_RESULT.json',{'passed':12,'failed':0,'initial_synthetic_test_failure':'Case-sensitive fake NLI fixture missed opposing sentiment; fixed fake scorer only before model execution. Original algorithm unchanged.','GPU':False})
    files=[CONFIG,Path(__file__).resolve(),ROOT/'scripts/stage_i_source_consistency_v1.py',ROOT/'tests/test_stage_i_source_consistency_v1.py']
    files+=list(p for p in PREP.iterdir() if p.is_file())
    files+=list(p for p in NLI.iterdir() if p.is_file())
    files+=list(p for p in EMBED.iterdir() if p.is_file())
    files+=[ROOT/'outputs/stage_i/repetition_guard_pilot_v2/predictions.json',ROOT/'outputs/stage_i/repetition_guard_pilot_v2/COMPLETION.json']
    bindings={str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):sha(p) for p in files}
    write(PREP/'PREFLIGHT_MANIFEST.json',{'files':bindings,'parameter_search':False,'training':False,'Teacher_calls':0})
    write(PREP/'RESEARCHER_SCOPE_RECORD.json',{'approval_source':'researcher_message','message':'h\u00e3y t\u00ecm hi\u1ec3u th\u00eam k\u1ef9 thu\u1eadt v\u00e0 \u00e1 d\u1ee5ng \u00e1c k\u1ef9 thu\u1eadt \u0111\u00f3 \u0111\u1ec3 c\u1ea3i thi\u1ec7n output',
        'interpretation':c['scope_authorization'],'recorded_at_UTC':datetime.now(timezone.utc).isoformat(),'config_sha256':sha(CONFIG),
        'manifest_sha256':sha(PREP/'PREFLIGHT_MANIFEST.json'),'runner_sha256':sha(Path(__file__)),
        'independent_advance_hash_review_claimed':False,'GPU_approved':False,'training_approved':False,'Teacher_calls_approved':False})
    print(json.dumps({'preparation':'PASS','tests':12,'bindings':len(bindings),'config_sha256':sha(CONFIG),'manifest_sha256':sha(PREP/'PREFLIGHT_MANIFEST.json')}))


def run():
    os.environ['CUDA_VISIBLE_DEVICES']='-1';os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    c=read(CONFIG);m=read(PREP/'PREFLIGHT_MANIFEST.json')
    for rel,digest in m['files'].items():
        p=Path(rel);assert sha(p if p.is_absolute() else ROOT/p)==digest,rel
    previous_preflight(require_fresh=False);deadline()
    OUT.mkdir(exist_ok=False);start=time.perf_counter()
    write(OUT/'CONFIG_AND_SCOPE.json',{'config':c,'approval':read(PREP/'RESEARCHER_SCOPE_RECORD.json')})
    try:
        from transformers import LEDTokenizerFast
        from stage_i_source_consistency_v1 import CPUScorers,select
        scorer=CPUScorers(c,OUT)
        tok=LEDTokenizerFast.from_pretrained(ROOT/'assets/tokenizers/led-base-16384',local_files_only=True)
        controls=[('The service was friendly.','The service was friendly.'),
                  ('The service was not friendly.','The service was friendly.'),
                  ('A woman is sitting at a table.','A woman is standing at a table.'),
                  ('One drink is visible.','Two drinks are visible.'),
                  ('Wine bottles are on a wooden rack.','Drinking glasses are on a wooden rack.')]
        screen=scorer.nli(controls)
        write(OUT/'NLI_COMPONENT_CONTROLS.json',[{'premise':a,'hypothesis':b,**s,'model_prediction_not_ground_truth':True} for (a,b),s in zip(controls,screen)])
        assert screen[0]['entailment']>screen[0]['contradiction'] and screen[1]['contradiction']>screen[1]['entailment'],'Basic NLI identity/negation mapping failed; stop'
        predictions=[]
        for record in rows(PREP/'INFERENCE_ONLY_INPUTS.jsonl'):
            deadline();scorer.check_budget();result=select(record,scorer,c,tok)
            write(OUT/'cases'/(record['business_id']+'.json'),result);predictions.append(result)
            print(json.dumps({'completed':len(predictions),'of':24,'business_id':record['business_id'],
                'review_sentences':len(result['selected_sentences']['review']),'visual_sentences':len(result['selected_sentences']['visual']),
                'NLI_pairs':len(scorer.pairs),'seconds':round(time.perf_counter()-start,2)}),flush=True)
        assert len(predictions)==24
        for rel,digest in m['files'].items():
            p=Path(rel);assert sha(p if p.is_absolute() else ROOT/p)==digest,rel
        previous_preflight(require_fresh=False)
        write(OUT/'predictions.json',predictions)
        write(OUT/'COMPLETION.json',{'status':'CPU_DERIVED_OUTPUT_PASS_NOT_SCIENTIFIC_QUALITY_GO','businesses':24,
            'seconds':time.perf_counter()-start,'NLI_pairs':len(scorer.pairs),'auxiliary_CPU_forward_batches':scorer.model_forward_batches,
            'LED_generation_calls':0,'training_updates':0,'Teacher_calls':0,'GPU_operations':False,'held_out_access':False,
            'parameters_tuned_after_scoring':False,'raw_inputs_and_frozen_artifacts_unchanged':True,'D051_strict_replay':'FAIL_UNCHANGED','human_quality_acceptance':'PENDING'})
    except Exception as exc:
        write(OUT/'FAILURE_STOP.json',{'error':str(exc),'traceback':traceback.format_exc(),'seconds':time.perf_counter()-start,'no_automatic_retry':True});raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--run',action='store_true');a=p.parse_args()
    if a.prepare:prepare()
    elif a.run:run()
    else:p.error('Choose --prepare or --run')
