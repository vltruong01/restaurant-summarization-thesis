"""D056 pure offline bindings, alignment, budget and numerical helpers.

No torch/transformers model import, forward, generation, network or CUDA calls.
"""
import ast,hashlib,json,math,types,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/stage_i/d056_diagnostic_preparation_v2'
CONFIG=ROOT/'configs/stage_i_d056_minimal_diagnostic_v2.json'
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
            'expected_id':y[p-1],'cached_prefix_length':p-1,'incremental_decoder_id':z[p-1]} for p in [3,11]]}

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
    require(c['cache_prefix_lengths']==[3,11] and c['cache_calls_per_comparison']==3,'Matched-prefix design changed')
    require(c['amendment']=='D-056-v2','Wrong amendment')
    parent=read(ROOT/'configs/stage_i_d056_minimal_diagnostic_v1.json')
    allowed={'amendment','parent_config_sha256','cache_prefix_lengths','future_output','cache_design_change',
             'fingerprint_convention','trace_candidate_export','beam_finalization'}
    require({k:v for k,v in c.items() if k not in allowed}=={k:v for k,v in parent.items() if k not in allowed},'Unapproved configuration drift')
    require(c['cross_precision_screen']==parent['cross_precision_screen'],'Cross-precision screen changed')
    require(c['trace_candidate_export']=={'raw_top_k':10,'processed_top_k':10,'independent_rankings':True,'tie_break':'ascending token ID','masked_score_encoding':'-inf string'},'Candidate export contract changed')
    require(4*3*2*2*3==144 and 4*3*2==24 and 4*3*3==36,'Incorrect ceiling arithmetic')
    require(c['source_limit']==5000 and c['label_limit']==768 and c['generation']['max_length']==768,'Frozen lengths changed')
    require(c['generation']==read(PREP/'expected_generation_config.json'),'Baseline effective GenerationConfig changed')
    require(c['D051_strict_replay_status']=='FAIL_UNCHANGED','Replay history changed')
    return True
def approval_gate(approval,config_path,manifest_path):
    a=read(approval);require(a.get('decision')=='D-056' and a.get('scope')=='bounded_read_only_GPU_diagnosis','Wrong approval scope')
    require(a.get('amendment')=='D-056-v2','Approval must identify D-056-v2')
    require(a.get('GPU_execution_approved') is True,'GPU execution not authorized')
    require(a.get('researcher_explicit_authorization') and a.get('approval_source')=='researcher_message','Explicit researcher authorization missing')
    require(a.get('config_sha256')==sha(config_path) and a.get('preflight_manifest_sha256')==sha(manifest_path),'Approval binding mismatch')
    require(a.get('training_authorized') is False,'Training outside scope')
    return a
def verify_manifest(manifest):
    d=read(manifest)
    for path,h in d['files'].items():require(sha(Path(path) if Path(path).is_absolute() else ROOT/path)==h,'Frozen file mismatch: '+path)
    return d

def tensor_fingerprint(value):
    """Exact historical fingerprint for a bare Tensor/Parameter (not file SHA)."""
    a=value.detach().cpu().contiguous().numpy()
    d=hashlib.sha256(type(value).__name__.encode()+b':')
    d.update(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes());d.update(b';')
    return d.hexdigest()

def array_binding(value):
    a=np.ascontiguousarray(value)
    return {'shape':list(a.shape),'dtype':str(a.dtype),
            'sha256':hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()}

def separate_candidates(raw,processed,k=10):
    """Independent deterministic rankings; masked -inf is a string in JSON."""
    r=np.asarray(raw,dtype=np.float32);p=np.asarray(processed,dtype=np.float32)
    require(k==10 and r.ndim==p.ndim==2 and r.shape==p.shape and r.shape[1]>=k,'Candidate tensor shape mismatch')
    require(np.isfinite(r).all(),'Nonfinite raw logits')
    require(not np.isnan(p).any() and not np.isposinf(p).any(),'NaN/+Inf processed scores')
    require(np.isfinite(p).any(axis=1).all(),'All processed tokens masked')
    ri=np.argsort(-r,axis=1,kind='stable')[:,:k];pi=np.argsort(-p,axis=1,kind='stable')[:,:k]
    encode=lambda x:[[float(v) if np.isfinite(v) else '-inf' for v in row] for row in x]
    return {'raw_top10_ids':ri.tolist(),'raw_top10_logits':encode(np.take_along_axis(r,ri,axis=1)),
            'processed_top10_ids':pi.tolist(),'processed_top10_scores':encode(np.take_along_axis(p,pi,axis=1)),
            'processed_scores_at_raw_top10':encode(np.take_along_axis(p,ri,axis=1)),
            'rankings':'independently descending; equal values resolved by ascending token ID',
            'values_dtype':'float32 exact values converted losslessly to JSON numbers; masked -inf as string'}

def beam_trace_audit(prompt,sequence,ancestry,transitions,direct_scores,raw_selected_logits,
                     returned_score,length_penalty,score_steps,censor_seen,num_beams=4):
    """Pinned vectorized beam output: ancestry starts at first emitted token."""
    observation=trace_observation(prompt,sequence,censor_seen)
    a=np.asarray(ancestry,dtype=np.int64);t=np.asarray(transitions,dtype=np.float64)
    require(a.ndim==1 and t.ndim==1,'Beam audit requires one selected sequence')
    valid=int((a>=0).sum());require(0<valid<=score_steps<=64,'Invalid beam ancestry length')
    require(np.all(a[:valid]>=0) and np.all(a[valid:]==-1),'Gapped/invalid beam ancestry')
    require(np.all(a[:valid]<num_beams),'Beam row outside single-batch beam scope')
    require(valid==len(sequence)-len(prompt) and len(t)>=valid,'Beam token/transition alignment mismatch')
    direct=np.asarray(direct_scores,dtype=np.float64);raw=np.asarray(raw_selected_logits,dtype=np.float64)
    require(direct.shape==raw.shape==(valid,),'Selected score evidence length mismatch')
    require(np.isfinite(direct).all() and np.isfinite(t[:valid]).all() and np.isfinite(raw).all(),'Nonfinite selected score')
    eos=observation['selected_EOS_new_token_index']
    require(eos is None or eos==valid-1,'EOS before final selected token')
    if eos is None:require(censor_seen and valid==64,'Non-EOS finalization not explained by approved censor')
    require(np.isfinite(returned_score) and length_penalty==1.0,'Unexpected beam scoring configuration')
    reconstructed=float(t[:valid].sum()/(valid**length_penalty))
    return {**observation,'ancestry_valid_new_tokens':valid,'decoder_prompt_length':len(prompt),
            'ancestry_index_origin':'first newly emitted token (not decoder-start/BOS positions)',
            'selected_token_ids':sequence[len(prompt):],'selected_active_beam_rows':a[:valid].tolist(),
            'selected_raw_logits':raw.tolist(),'selected_processed_scores_direct':direct.tolist(),
            'library_transition_scores':t[:valid].tolist(),
            'transition_gather_exact_match':bool(np.array_equal(direct,t[:valid])),
            'transition_scores_direct_max_abs_difference':float(np.max(np.abs(direct-t[:valid]))),
            'sum_transition_scores':float(t[:valid].sum()),'length_penalty':length_penalty,
            'reconstructed_sequence_score':reconstructed,'returned_sequence_score':float(returned_score),
            'absolute_difference':abs(float(returned_score)-reconstructed),
            'selected_finalization_reason':'observed_selected_EOS' if eos is not None else 'custom_64_token_censor',
            'selected_earlier_than_last_global_iteration':valid<score_steps,
            'finalizer_appended_EOS':False,'forced_EOS_at_768_reachable_within_trace':False,
            'interpretation':'Pinned vectorized beam returns scored tokens, including EOS if chosen. Custom stopping can finalize a non-EOS beam. Float32 accumulation differences reported without new pass tolerances; no full-generation-prefix equivalence.'}

def offline_mask_bindings(cases):
    """Execute the real collator body via an explicitly disclosed NumPy torch shim."""
    require('torch' not in sys.modules,'Offline masks must not use a model runtime')
    from stage_hi_led_data import LEDCompleteCollator
    class Array(np.ndarray):
        def bool(self):return self.astype(bool)
    wrap=lambda a:np.asarray(a).view(Array)
    shim=types.ModuleType('torch')
    shim.long=np.int64;shim.full=lambda shape,value,dtype:wrap(np.full(shape,value,dtype=dtype))
    shim.zeros_like=lambda v:wrap(np.zeros_like(v));shim.tensor=lambda v,dtype:wrap(np.array(v,dtype=dtype))
    shim.ones=lambda shape,dtype:wrap(np.ones(shape,dtype=dtype));shim.equal=np.array_equal;shim.any=np.any
    results=[]
    sys.modules['torch']=shim
    try:
        collator=LEDCompleteCollator(5000,768,1024)
        for c in cases:
            inputs=c['input_ids'];labels=c['label_ids'];sl=math.ceil(len(inputs)/1024)*1024;ll=math.ceil(len(labels)/8)*8
            batch=collator([{'input_ids':inputs,'labels':labels}])
            expected={'input_ids':np.array([inputs+[1]*(sl-len(inputs))],dtype=np.int64),
                      'labels':np.array([labels+[-100]*(ll-len(labels))],dtype=np.int64),
                      'attention_mask':np.array([[1]*len(inputs)+[0]*(sl-len(inputs))],dtype=np.int64),
                      'global_attention_mask':np.array([[1]+[0]*(sl-1)],dtype=np.int64)}
            require(all(np.array_equal(batch[k],v) for k,v in expected.items()),'Actual collator body differs from independent expected masks')
            require(hashlib.sha256(c['source'].encode()).hexdigest()==c['source_sha256'],'Source text SHA mismatch')
            results.append({'business_id':c['business_id'],'source_sha256':c['source_sha256'],
                'input_ids_sha256':ids_sha(inputs),'label_ids_sha256':ids_sha(labels),
                'arrays':{k:array_binding(v) for k,v in expected.items()},'masked_label_count':ll-len(labels),
                'decoder_input_shape':[1,len(labels)],'teacher_logits_shape':[1,len(labels),50265],
                'source_mask_rows_identical_for_four_states':True,'global_attention_token0_only':True,
                'labels_not_passed_to_diagnostic_forward_or_generation':True,'no_source_or_target_truncation':True})
    finally:del sys.modules['torch']
    return {'method':'Real LEDCompleteCollator Python body with NumPy-only torch shim, independently compared to expected arrays. Not a Torch/model/GPU feasibility test.',
            'cases':results,'model_instantiations':0,'actual_torch_imported':False}
