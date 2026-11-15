"""Actual pinned BERTScore/ROUGE forward checks on synthetic text ONLY, on CPU."""
from pathlib import Path
import sys,importlib.metadata,time,psutil
from s1000_final_eval_core_v1 import *
R=Path(__file__).resolve().parents[1];O=R/'outputs/stage_k/s1000_test100_phase1_v1';P=R/'outputs/final_evaluation/s1000_preparation_v1'
def main():
    lock=read(P/'METRIC_LOCK.json');wheel=P/'METRIC_ASSETS/bert_score-0.3.13-py3-none-any.whl'
    require(file_hash(wheel)=='bbbb4c7fcdaa46d7681aff49f37f96faa09ed74e1b150e659bdc6b58a66989b9','BERTScore wheel drift')
    sys.path.insert(0,str(wheel));require(all(importlib.metadata.version(k)==v for k,v in lock['packages'].items()),'Pinned metric versions unavailable')
    model=O/'METRIC_RUNTIME/roberta-large';require(file_hash(model/'pytorch_model.bin')==lock['bertscore']['weights_sha256'],'Wrong model')
    for f,h in lock['bertscore']['small_file_sha256'].items():require(file_hash(model/f)==h,'Wrong tokenizer/config')
    import torch
    from bert_score import BERTScorer
    from bert_score.utils import sent_encode
    from rouge_score.rouge_scorer import RougeScorer
    from stage_k_bert_loader_compat_v1 import legacy_binary_cpu_load
    torch.set_num_threads(1);torch.manual_seed(42);start=time.perf_counter()
    with legacy_binary_cpu_load(model/'pytorch_model.bin'):
        scorer=BERTScorer(model_type=str(model),num_layers=17,batch_size=8,nthreads=1,all_layers=False,idf=False,device='cpu',lang='en',rescale_with_baseline=False,use_fast_tokenizer=False)
    scorer._tokenizer.model_max_length=512
    candidates=['A red plate is on a table.','Quantum mechanics describes electrons.','']
    references=['A red plate is on a table.','A red plate is on a table.','A red plate is on a table.']
    with torch.no_grad():pr,re,f=scorer.score(candidates,references,batch_size=8,verbose=False)
    require(float(f[0])>.999 and float(f[1])<float(f[0]) and abs(float(f[2]))<1e-6,'Synthetic BERTScore properties failed')
    long='word '*800;untruncated=scorer._tokenizer.encode(long.strip(),add_special_tokens=True,add_prefix_space=True,truncation=False);prefix=sent_encode(scorer._tokenizer,long)
    require(len(untruncated)>512 and len(prefix)==512 and prefix[0]==0 and prefix[-1]==2,'Pinned prefix behavior differs')
    rouge=RougeScorer(['rouge1','rouge2','rougeL'],use_stemmer=True)
    identical=rouge.score('Red plate on table.','Red plate on table.');disjoint=rouge.score('red plate','blue sky')
    require(all(v.fmeasure==1. for v in identical.values()) and disjoint['rouge1'].fmeasure==0,'Synthetic ROUGE checks')
    stat=paired([1,2,3,4],[0,1,2,3],B=500,P=1000);require(stat['CI95']==[1.,1.],'Synthetic paired CI check')
    report=dict(status='PASS_ACTUAL_PINNED_METRIC_RUNTIME_ON_SYNTHETIC_TEXT_ONLY',packages=lock['packages'],model_id=lock['bertscore']['model_id'],revision=lock['bertscore']['revision'],model_sha256=lock['bertscore']['weights_sha256'],
        tokenizer_small_file_sha256=lock['bertscore']['small_file_sha256'],layers=17,idf=False,rescale_with_baseline=False,use_fast_tokenizer=False,device='cpu',batch_size=8,
        synthetic_candidate_reference_pairs=[dict(candidate=candidates[i],reference=references[i],precision=float(pr[i]),recall=float(re[i]),F1=float(f[i])) for i in range(3)],
        prefix_test=dict(original_tokens=len(untruncated),encoded_tokens=len(prefix),max_length=512),synthetic_ROUGE_identity_and_disjoint=True,synthetic_paired_CI=True,
        seconds=time.perf_counter()-start,process_peak_working_set_bytes=psutil.Process().memory_info().peak_wset,TEST_businesses_scored=0,reference_workspace_opened=False,
        loader_compatibility='Exact SHA-pinned legacy binary: meta storage probe deserializes on CPU, weights_only=True; same FP32 tensors/settings',
        limitation='Synthetic technical checks only; prefix BERTScore is not full-document or factuality verification')
    write(O/'EVIDENCE/METRIC_SYNTHETIC_RUNTIME_VERIFICATION.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
