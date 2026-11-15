"""Static/numpy audit only: frozen F train/validation, saved tokens and adapters."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
from pathlib import Path
from datetime import datetime,timezone
import json,zipfile,hashlib,statistics
import numpy as np
from safetensors.numpy import load,load_file

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'outputs/stage_i/led_large_vast_bos_cause_audit_v1'
FROZEN=ROOT/'outputs/cloud_handoff/LED_LARGE_VAST_D059_F_V1'

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def model_stats(weights):
    modules=[]
    assert all(np.isfinite(w).all() for w in weights.values())
    for k,a in weights.items():
        if not k.endswith('.lora_A.weight'):continue
        b=weights[k.replace('.lora_A.weight','.lora_B.weight')]
        squared=float(np.sum((b.T@b)*(a@a.T)))
        modules.append(dict(name=k.removesuffix('.lora_A.weight'),A_norm=float(np.linalg.norm(a)),
            B_norm=float(np.linalg.norm(b)),deltaW_Frobenius=2*max(squared,0.)**.5))
    assert len(modules)*2==len(weights)
    return dict(tensors=len(weights),parameters=sum(w.size for w in weights.values()),finite=True,
        zero_B=sum(m['B_norm']==0 for m in modules),
        deltaW_norm_median=statistics.median(m['deltaW_Frobenius'] for m in modules),modules=modules)

def main():
    DEST.mkdir(parents=True,exist_ok=True)
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),scope='CPU static/NumPy audit; no GPU, model forward, generation, training, held-out content',datasets={},checkpoints={})
    manifest=read(ROOT/'outputs/stage_h/led_dataset_v1/manifest.json')
    for name in ('train_filtered','validation_filtered'):
        info=manifest['datasets'][name];path=ROOT/info['path'];assert sha(path)==info['sha256']
        with np.load(path,allow_pickle=False) as z:
            labels=z['labels'];offsets=z['label_offsets']
            starts=labels[offsets[:-1]];ends=labels[offsets[1:]-1]
            bos_counts=[int(np.sum(labels[a:b]==0)) for a,b in zip(offsets[:-1],offsets[1:])]
            assert np.all(starts==0) and np.all(ends==2)
            assert set(bos_counts)=={1}
            result['datasets'][name]=dict(rows=len(offsets)-1,file_sha256=sha(path),all_labels_start_BOS0=True,
                all_labels_end_EOS2=True,BOS_occurrences_per_label=1,label_first_12=labels[:12].tolist(),
                decoder_first_12_after_standard_shift=[2]+labels[:11].tolist(),
                internal_or_repeated_BOS_in_gold_labels=False)
    base=read(FROZEN/'configs/stage_i_d059_expanded_lora_v2.json')
    large=read(FROZEN/'configs/stage_i_led_large_vast_d059_v1.json')
    result['method_equality']={k:base[k]==large[k] for k in ('optimization','generation','budgets','training','stage_h_manifest','final_references')}
    assert all(result['method_equality'].values())
    archive=ROOT/'outputs/stage_i/led_large_vast_d059_v1_download_audit/downloaded_archives/F_large_d059_v1.zip'
    with zipfile.ZipFile(archive) as z:
        prefix='F_large_d059_v1/'
        for step in (250,500,750,1000,1250):
            name=prefix+f'train/checkpoint-{step}/adapter_model.safetensors'
            w=load(z.read(name));result['checkpoints']['large_step'+str(step)]=model_stats(w)
            assert result['checkpoints']['large_step'+str(step)]['tensors']==456
            assert result['checkpoints']['large_step'+str(step)]['zero_B']==0
        result['AMP_events']={}
        for phase in ('feasibility','train'):
            rows=[json.loads(x) for x in z.read(prefix+phase+'/AMP_DYNAMIC.jsonl').decode().splitlines()]
            updates=[r['successful_updates'] for r in rows if r['event']=='successful_update']
            assert updates==list(range(1,(3 if phase=='feasibility' else 1250)+1))
            result['AMP_events'][phase]=[r for r in rows if r['event']=='native_skip_backoff']
        result['gate_technical_traces']=[]
        for i in (1,2):
            trace=json.loads(z.read(prefix+f'feasibility/train_memory_trace_{i}.json'))
            result['gate_technical_traces'].append(dict(business_id=trace['business_id'],token_count=len(trace['token_ids']),
                leading_ids=trace['token_ids'][:30],scope=trace['scope']))
    bpath=ROOT/'outputs/stage_i/d059_expanded_filtered_v2/train/checkpoint-750/adapter_model.safetensors'
    result['checkpoints']['D059_base_epoch3']=model_stats(load_file(bpath))
    result['raw_generation_prefixes']={}
    for model,relative in [('large_epoch3','predictions/epoch3.json'),('large_epoch5','predictions/epoch5.json'),('D059_base_epoch3','benchmarks/D059_Expanded_epoch3.json')]:
        rows=read(ROOT/'outputs/stage_i/led_large_vast_d059_v1_quality_review/portable_review'/relative)
        counts=[]
        for r in rows:
            n=0
            for x in r['token_ids'][1:]:
                if x!=0:break
                n+=1
            counts.append(dict(business_id=r['business_id'],subset=r['subset'],leading_BOS=n,empty=not r['text'].strip()))
        result['raw_generation_prefixes'][model]=counts
    result['conclusions']=['Gold F labels contain exactly one initial BOS, not 766 repeats.',
        'Standard LED shift constructs decoder prefix [2,0,first_text_token,...].',
        'All five trained large adapters have finite nonzero B tensors; no whole adapter stayed at zero initialization.',
        'Method settings and frozen input manifest match D059, not proof of optimality for large.',
        'No causal attribution to loss scale is established by these static checks.']
    (DEST/'STATIC_ARTIFACT_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('checkpoints','raw_generation_prefixes')},indent=2))
    print('CHECKPOINT_SUMMARY',json.dumps({k:{x:v[x] for x in ('tensors','parameters','finite','zero_B','deltaW_norm_median')} for k,v in result['checkpoints'].items()},indent=2))

if __name__=='__main__':main()
