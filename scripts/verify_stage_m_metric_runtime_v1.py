"""CPU technical checks on synthetic text and existing VALIDATION only."""
from pathlib import Path
import json
import time
from s1000_final_eval_core_v1 import read,write,file_hash,text_hash,require,digest
from stage_m_evaluation_v1 import metric_runtime,score_components,strip_header

ROOT=Path(__file__).resolve().parents[1]
def main():
    out=ROOT/'outputs/stage_m/s1000_final_automatic_v1/TECHNICAL_RUNTIME_VERIFICATION.json';require(not out.exists(),'Refuse overwrite technical receipt')
    prep=ROOT/'outputs/final_evaluation/s1000_preparation_v1';model=ROOT/'outputs/stage_k/s1000_test100_phase1_v1/METRIC_RUNTIME/roberta-large'
    started=time.perf_counter();scorer,runtime=metric_runtime(prep,model)
    texts=[('A red plate is on a table.','A red plate is on a table.'),('Quantum mechanics describes electrons.','A red plate is on a table.'),('','A red plate.'),('A red plate.',''),('','')]
    raw=[];calls={};refs={}
    for i,(candidate,reference) in enumerate(texts):
        b=f'SYNTHETIC_ONLY_{i}';r={'condition':'F','business_id':b,'task':'review','text':'[REVIEW_CONSENSUS]\n'+candidate,'text_sha256':text_hash(candidate),'source_sha256':'SYNTHETIC','checkpoint_sha256':'NONE','token_ids':[2,0,2]};raw.append(r)
        calls['F',b,'review']={'source':'SYNTHETIC'};refs[b]={'review_consensus':reference,'visual_evidence':'','reference_sha256':text_hash(reference)}
    synthetic,cache=score_components(raw,calls,refs,scorer,'human')
    require(synthetic[0]['bertscore_f1']>.999 and synthetic[1]['bertscore_f1']<synthetic[0]['bertscore_f1'],'BERT identity/disjoint check failed')
    require(all(all(r[m]==0 for m in ['rouge1_fmeasure','rouge2_fmeasure','rougeL_fmeasure','bertscore_precision','bertscore_recall','bertscore_f1']) for r in synthetic[2:]),'Empty score policy failed')
    from bert_score.utils import sent_encode
    long='word '*800;untruncated=scorer._tokenizer.encode(long.strip(),add_special_tokens=True,add_prefix_space=True,truncation=False);prefix=sent_encode(scorer._tokenizer,long)
    require(len(untruncated)>512 and len(prefix)==512 and prefix[0]==0 and prefix[-1]==2,'Prefix policy changed')
    # Existing approved historical development artifacts, not new LED inference.
    dp=ROOT/'outputs/stage_g/teacher_production_d042_v1/guarded_production6_retry_v1/dataset/validation_filtered.jsonl'
    pp=ROOT/'outputs/stage_i/cloud_return_20261015_v1/stage_i_chatgpt_review_bundle/validation/filtered/validation_predictions.jsonl'
    dev=[json.loads(s) for s in dp.read_text(encoding='utf8').splitlines()];pred={r['business_id']:r for r in map(json.loads,pp.read_text(encoding='utf8').splitlines())}
    testids=set(read(prep/'COHORT100_FREEZE.json')['business_ids']);require(all(r['split']=='validation' and r['business_id'] not in testids and text_hash(r['source'])==r['source_sha256'] for r in dev),'Development frame differs')
    selected=next(r for r in sorted(dev,key=lambda r:r['business_id']) if r['business_id'] in pred)
    b=selected['business_id'];marker='\n\n[VISUAL_EVIDENCE]\n'
    candidate=pred[b]['text'];target=selected['target'];require(marker in candidate and marker in target,'Existing development component markers')
    cr,cv=candidate.split(marker,1);tr,tv=target.split(marker,1)
    raw=[];calls={};refs={b:{'review_consensus':strip_header(tr,'review'),'visual_evidence':tv,'reference_sha256':text_hash(target)}}
    for task,text in [('review',cr),('visual','[VISUAL_EVIDENCE]\n'+cv)]:
        raw.append({'condition':'F','business_id':b,'task':task,'text':text,'text_sha256':text_hash(text),'source_sha256':selected['source_sha256'],'checkpoint_sha256':'HISTORICAL_VALIDATION_TECHNICAL_ONLY','token_ids':pred[b]['token_ids']})
        calls['F',b,task]={'source':selected['source']}
    development,_=score_components(raw,calls,refs,scorer,'teacher')
    require(len(development)==2 and all(0<=r['rouge1_fmeasure']<=1 for r in development),'Development integration failed')
    write(out,{'status':'PASS_ACTUAL_CPU_SYNTHETIC_AND_EXISTING_VALIDATION_TECHNICAL_CHECKS','runtime':runtime,
       'synthetic_scores':[{'case':i,**{m:r[m] for m in ['rouge1_fmeasure','rouge2_fmeasure','rougeL_fmeasure','bertscore_precision','bertscore_recall','bertscore_f1']}} for i,r in enumerate(synthetic)],
       'prefix_check':{'untruncated_tokens':len(untruncated),'scored_prefix_tokens':len(prefix),'BOS':prefix[0],'EOS':prefix[-1]},
       'development_check':{'businesses':1,'components':2,'business_id':b,'source_dataset_sha256':file_hash(dp),'prediction_file_sha256':file_hash(pp),'reference_target_sha256':text_hash(target),'component_scores_technical_only':development,'historical_AI_teacher_not_human':True,'not_current_model_selection':True},
       'TEST_reference_based_scores_computed':0,'LED_generation_calls':0,'API_calls':0,'device':'cpu','seconds':time.perf_counter()-started})
    print('PASS actual pinned CPU metric checks; genuine TEST references scored: 0')

if __name__=='__main__':main()
