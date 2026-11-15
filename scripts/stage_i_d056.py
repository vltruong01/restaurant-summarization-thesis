"""D056 pure offline bindings, alignment, budget and numerical helpers.

No torch/transformers model import, forward, generation, network or CUDA calls.
"""
import hashlib,json,math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/stage_i/d056_diagnostic_preparation_v1'
CONFIG=ROOT/'configs/stage_i_d056_minimal_diagnostic_v1.json'
PREP=ROOT/'outputs/stage_i/paired_lora_pilot_preparation_v1'
CASES=['JcdOQvvx3BqZ5Ga2_9n_Bg','b1pdnbYvBdtTxJOOom5WmA','j3kGicU3mPG_ciIKG1Vuyg']
STATES=['pretrained_no_adapter','historical_full_F','A_decoder_only','B_encoder_decoder']
REVIEW='[REVIEW_CONSENSUS]'
VISUAL='[VISUAL_EVIDENCE]'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def rows(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines() if s.strip()]
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ids_sha(v):return hashlib.sha256(json.dumps(v,separators=(',',':')).encode()).hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='\n') as f:f.write((v if isinstance(v,str) else json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False))+'\n')
def require(v,m):
    if not v:raise ValueError(m)
def match_positions(seq,pat):return [i for i in range(len(seq)-len(pat)+1) if seq[i:i+len(pat)]==pat]
def alignment(case,tok):
    y=case['label_ids'];require(y[0]==0 and y[-1]==2 and len(y)<=768,'Labels incomplete/overflow')
    require(y.count(0)==y.count(2)==1,'Internal label special tokens')
    marker=tok.encode(REVIEW,add_special_tokens=False).ids
    vi=tok.encode(VISUAL,add_special_tokens=False).ids
    prefix=[2,0]+marker+[50118];require(len(prefix)==12,'Frozen review prefix geometry changed')
    z=[2]+y[:-1];require(z[:12]==prefix,'Actual teacher prefix mismatch')
    at=match_positions(y,vi);require(len(at)==1,'Visual section marker ambiguity')
    v=at[0];boundary=list(range(max(0,v-2),min(len(y),v+len(vi)+3)))
    critical=sorted(set(range(12))|set(boundary)|{len(y)-1})
    positions=[{'logit_row':i,'label_index':i,'expected_id':y[i],'decoder_input_id':z[i],
        'decoder_prefix_length':i+1,'generated_sequence_index':i+1,
        'role':('initial_BOS' if i==0 else 'review_marker' if i<=9 else 'review_newline' if i==10 else 'first_review_body' if i==11 else 'target_EOS' if i==len(y)-1 else 'visual_boundary')} for i in critical]
    if not case['selected_photo_ids']:
        require(case['target'].split(VISUAL)[-1].strip()=='','Zero-caption target has visual content')
    return {'business_id':case['business_id'],'teacher_decoder_input_ids':z,'rescue_prefix_ids':prefix,
        'label_length':len(y),'source_length':len(case['input_ids']),'source_padded_length':math.ceil(len(case['input_ids'])/1024)*1024,
        'visual_marker_label_start':v,'visual_marker_label_end_exclusive':v+len(vi),
        'critical_logit_rows':critical,'token_positions':positions,'zero_caption':not case['selected_photo_ids'],
        'cache_probes':[{'prefix_length':p,'reference_logit_row':p-1,'expected_label_index':p-1,
            'expected_id':y[p-1],'cached_prefix_length':p-1,'incremental_decoder_id':z[p-1]} for p in [11,12]]}

class Budget:
    def __init__(self,limits):self.limits=dict(limits);self.used={k:0 for k in limits}
    def take(self,kind,n=1):
        require(type(n) is int and n>0,'Invalid budget increment')
        require(kind in self.used and self.used[kind]+n<=self.limits[kind],'Execution ceiling exceeded: '+kind)
        self.used[kind]+=n
def log_probs(x):
    a=np.asarray(x,dtype=np.float64);require(np.isfinite(a).all(),'Nonfinite raw logits')
    v=a-a.max();return v-np.log(np.exp(v).sum())
def vector_comparison(reference,actual,correct_id,atol,rtol):
    r=np.asarray(reference,dtype=np.float64);a=np.asarray(actual,dtype=np.float64)
    require(r.shape==a.shape and r.ndim==1 and r.size>1,'Logit shape mismatch')
    require(np.isfinite(r).all() and np.isfinite(a).all(),'NaN/Inf logits')
    delta=np.abs(a-r);allowed=atol+rtol*np.abs(r);bad=delta>allowed
    lr,la=log_probs(r),log_probs(a)
    return {'atol':atol,'rtol':rtol,'max_abs':float(delta.max()),'rms':float(np.sqrt(np.mean(delta**2))),
        'max_relative_with_floor1e8':float((delta/np.maximum(np.abs(r),1e-8)).max()),
        'rms_relative':float(np.sqrt(np.mean(delta**2))/max(np.sqrt(np.mean(r**2)),1e-8)),
        'failed_elements':int(bad.sum()),'failed_fraction':float(bad.mean()),'allclose_screen_pass':not bool(bad.any()),
        'argmax_reference':int(r.argmax()),'argmax_actual':int(a.argmax()),'argmax_changed':bool(r.argmax()!=a.argmax()),
        'correct_id':int(correct_id),'correct_logprob_reference':float(lr[correct_id]),'correct_logprob_actual':float(la[correct_id]),
        'correct_probability_reference':float(np.exp(lr[correct_id])),'correct_probability_actual':float(np.exp(la[correct_id])),
        'KL_reference_actual':float(np.sum(np.exp(lr)*(lr-la))),
        'top10_reference':np.argsort(-r,kind='stable')[:10].tolist(),'top10_actual':np.argsort(-a,kind='stable')[:10].tolist()}
def tensor_header(p):
    with Path(p).open('rb') as f:n=int.from_bytes(f.read(8),'little');h=json.loads(f.read(n))
    return {k:v for k,v in h.items() if k!='__metadata__'}
def expected_adapter_keys(state):
    require(state in STATES[1:],'Unexpected adapter state')
    modules=[f'led.decoder.layers.{i}.{a}.{q}' for i in range(6) for a in ['self_attn','encoder_attn'] for q in ['q_proj','v_proj']]
    if state=='B_encoder_decoder':modules += [f'led.encoder.layers.{i}.self_attn.longformer_self_attn.{q}' for i in range(6) for q in ['query','value','query_global','value_global']]
    return {f'base_model.model.{m}.lora_{ab}.weight' for m in modules for ab in ['A','B']}
def adapter_checkpoint_bindings(config):
    result={}
    for state,desc in config['adapter_states'].items():
        best=ROOT/desc['path'];require(sha(best)==desc['sha256'],'Frozen adapter hash differs')
        require(set(tensor_header(best))==expected_adapter_keys(state),'Adapter module-key scope differs')
        folder=best.parent.parent;ts=folder/'trainer_state.json';t=read(ts)
        selected=folder/Path(t['best_model_checkpoint'].replace('\\','/')).name
        require(sha(selected/'adapter_model.safetensors')==desc['sha256'],'Selected checkpoint and best adapter differ')
        files=[best,best.parent/'adapter_config.json',selected/'adapter_model.safetensors',selected/'adapter_config.json',ts]
        cfg=read(best.parent/'adapter_config.json')
        require(cfg['r']==8 and cfg['lora_alpha']==16 and cfg['lora_dropout']==0 and cfg['bias']=='none','Adapter metadata differs')
        result[state]={'checkpoint':selected.relative_to(ROOT).as_posix(),'best_matches_selected_checkpoint':True,
            'actual_saved_tensor_keys_verified':len(expected_adapter_keys(state)),
            'files':{p.relative_to(ROOT).as_posix():sha(p) for p in files}}
    return result
def trace_observation(prompt,sequence,censor_seen):
    require(sequence[:len(prompt)]==prompt,'Decoder prompt changed')
    new=sequence[len(prompt):];require(len(new)<=64,'Trace continuation exceeds64')
    eos=next((i for i,x in enumerate(new) if x==2),None)
    return {'new_token_count':len(new),'selected_EOS_new_token_index':eos,
        'run_reached_diagnostic_censor':bool(censor_seen),
        'selected_hypothesis_termination':'model_EOS' if eos is not None else 'diagnostic_censor_or_incomplete',
        'censored_beam_is_not_guaranteed_full768_prefix':True,'initial_decoder_start_is_not_EOS':True}
def check_config(c):
    old=read(ROOT/'outputs/stage_i/d055_root_cause_v2/PROPOSED_GPU_DIAGNOSTIC_CONFIG.json')
    require(c['cases']==CASES and c['states']==STATES,'Frozen cases/states changed')
    require(c['ceilings']=={'teacher_forced':24,'cache_control':144,'traces':36},'Ceilings changed')
    require(c['trace_new_token_cap']==64,'Trace cap changed')
    require(c['cache_screens']==old['cache_allclose_screen'],'Numerical tolerances changed')
    require(c['cache_prefix_lengths']==[11,12] and c['cache_calls_per_comparison']==3,'Matched-prefix design changed')
    require(4*3*2*2*3==144 and 4*3*2==24 and 4*3*3==36,'Incorrect ceiling arithmetic')
    require(c['source_limit']==5000 and c['label_limit']==768 and c['generation']['max_length']==768,'Frozen lengths changed')
    require(c['generation']==read(PREP/'expected_generation_config.json'),'Baseline effective GenerationConfig changed')
    require(c['D051_strict_replay_status']=='FAIL_UNCHANGED','Replay history changed')
    return True
def approval_gate(approval,config_path,manifest_path):
    a=read(approval);require(a.get('decision')=='D-056' and a.get('scope')=='bounded_read_only_GPU_diagnosis','Wrong approval scope')
    require(a.get('GPU_execution_approved') is True,'GPU execution not authorized')
    require(a.get('researcher_explicit_authorization') and a.get('approval_source')=='researcher_message','Explicit researcher authorization missing')
    require(a.get('config_sha256')==sha(config_path) and a.get('preflight_manifest_sha256')==sha(manifest_path),'Approval binding mismatch')
    require(a.get('training_authorized') is False,'Training outside scope')
    return a
def verify_manifest(manifest):
    d=read(manifest)
    for path,h in d['files'].items():require(sha(Path(path) if Path(path).is_absolute() else ROOT/path)==h,'Frozen file mismatch: '+path)
    return d
