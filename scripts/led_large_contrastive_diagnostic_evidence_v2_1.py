"""CPU evidence/reduction utilities. Importing never queries or initializes CUDA.

Analysis uses float64 copies on CPU for difference metrics only. The authoritative
isclose comparison uses original gradient dtypes, native reference as `other`.
No model forward, loss, precision, optimizer, threshold or RNG policy is changed.
"""
from pathlib import Path
import hashlib
import copy
import json
import math

ATOL=1e-5
RTOL=1e-4
REQUIRED_GATES=(
    'lambda0_within_tolerance','lambda0_preconditions','joint_vs_sequential',
    'negative_own_prefix','eight_example_FP16_accumulation',
    'AMP_skip_optimizer_adapter_integrity','RNG_continuity','model_modes_unchanged',
    'memory_headroom',
)
REQUIRED_LAMBDA0_PRECONDITIONS=(
    'rng_start_equal','rng_end_equal','mode_start_equal','mode_native_unchanged',
    'mode_delegated_unchanged','batch_equal','batch_native_unchanged',
    'batch_delegated_unchanged','native_own_prefix','delegated_own_prefix','manual_CE_reduction',
)


def json_safe(value):
    if isinstance(value,float) and not math.isfinite(value):return str(value)
    if isinstance(value,dict):return {str(k):json_safe(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [json_safe(v) for v in value]
    return value


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:
        json.dump(json_safe(value),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def tensor_signature(tensor):
    value=tensor.detach().cpu().contiguous()
    return dict(shape=list(value.shape),dtype=str(value.dtype),
                sha256=hashlib.sha256(value.numpy().tobytes()).hexdigest())


def batch_signature(batch):
    return {name:tensor_signature(t) for name,t in sorted(batch.items())}


def model_context(model):
    import torch
    import led_large_contrastive_sequential_v1 as seq
    config=getattr(model,'config',None)
    signature=seq._mode_signature(model)
    signature['config_dropout']={k:getattr(config,k,None) for k in (
        'dropout','attention_dropout','activation_dropout','encoder_layerdrop','decoder_layerdrop')}
    signature['dropout_modules']={name:dict(p=m.p,inplace=m.inplace) for name,m in model.named_modules() if isinstance(m,torch.nn.Dropout)}
    signature['is_gradient_checkpointing']=getattr(model,'is_gradient_checkpointing',None)
    signature['checkpoint_modules']={}
    for name,module in model.named_modules():
        if hasattr(module,'gradient_checkpointing'):
            fn=getattr(module,'_gradient_checkpointing_func',None)
            signature['checkpoint_modules'][name]=dict(enabled=module.gradient_checkpointing,
                                                       keywords=copy.deepcopy(getattr(fn,'keywords',None)))
    return signature


def compare_tensor(native,delegated):
    import torch
    a=native.detach().cpu().contiguous();b=delegated.detach().cpu().contiguous()
    shape_equal=a.shape==b.shape;dtype_equal=a.dtype==b.dtype
    finite_a=torch.isfinite(a);finite_b=torch.isfinite(b)
    result=dict(native_shape=list(a.shape),delegated_shape=list(b.shape),native_dtype=str(a.dtype),delegated_dtype=str(b.dtype),
        shape_equal=shape_equal,dtype_equal=dtype_equal,native_finite=bool(finite_a.all()),delegated_finite=bool(finite_b.all()),
        native_nonfinite_elements=int((~finite_a).sum()),delegated_nonfinite_elements=int((~finite_b).sum()),
        bit_exact=bool(torch.equal(a,b)),atol=ATOL,rtol=RTOL,reference='native',
        relative_error_definition='abs(delegated-native)/abs(native); zero/zero=0, nonzero/zero=Infinity')
    if not shape_equal or not dtype_equal:
        return dict(result,allclose=False,raw_isclose_all=False,failing_element_count=None,
                    max_absolute_difference=None,max_relative_difference=None,comparable=False)
    finite=finite_a&finite_b
    # Same dtype/torch.isclose semantics as the locked comparator, with reference native.
    close=torch.isclose(b,a,atol=ATOL,rtol=RTOL,equal_nan=False)
    valid=close&finite
    da=a.to(torch.float64);db=b.to(torch.float64)
    difference=(db-da).abs()
    relative=torch.where(da.abs()>0,difference/da.abs(),torch.where(difference==0,torch.zeros_like(difference),torch.full_like(difference,float('inf'))))
    if bool(finite.all()):
        max_abs=float(difference.max()) if a.numel() else 0.
        max_relative=float(relative.max()) if a.numel() else 0.
    else:max_abs=max_relative=None
    return dict(result,comparable=True,raw_isclose_all=bool(close.all()),allclose=bool(valid.all()),
                failing_element_count=int((~valid).sum()),max_absolute_difference=max_abs,
                max_relative_difference=json_safe(max_relative),
                max_absolute_difference_finite_elements=float(difference[finite].max()) if finite.any() else None)


def save_path_snapshot(folder,role,names,gradients,loss,signatures):
    """Persist each path immediately, even if the later path/guard never completes."""
    import torch
    if role not in ('native','delegated'):raise ValueError('Unknown path role')
    if len(names)!=len(gradients) or len(set(names))!=len(names):raise ValueError('Gradient inventory differs')
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    tensors={name:g.detach().cpu().contiguous().clone() for name,g in zip(names,gradients)}
    payload=dict(parameter_names=list(names),gradients=tensors,loss=loss.detach().cpu().clone())
    path=folder/(role+'_gradients.pt')
    with path.open('xb') as f:torch.save(payload,f)
    write_json(folder/(role+'_context.json'),signatures)
    metadata=dict(path=path.name,sha256=sha(path),bytes=path.stat().st_size,tensors=len(tensors),
                  context_path=role+'_context.json',context_sha256=sha(folder/(role+'_context.json')))
    write_json(folder/(role+'_snapshot_receipt.json'),metadata)
    return payload,metadata


def lambda0_comparison(folder,native,delegated,native_receipt,delegated_receipt,preconditions):
    names=native['parameter_names']
    inventory_equal=names==delegated['parameter_names']
    entries=[]
    if inventory_equal:
        for name in names:entries.append(dict(name=name,**compare_tensor(native['gradients'][name],delegated['gradients'][name])))
    loss=compare_tensor(native['loss'],delegated['loss'])
    complete=inventory_equal and len(entries)==len(names)>0
    numeric=complete and loss['allclose'] and all(e['allclose'] for e in entries)
    checks=set(preconditions)==set(REQUIRED_LAMBDA0_PRECONDITIONS) and all(v is True for v in preconditions.values())
    result=dict(status='PASS_WITHIN_TOLERANCE' if numeric and checks else 'FAIL_STOP',
        lambda0_within_tolerance=bool(numeric),lambda0_bit_exact=bool(complete and loss['bit_exact'] and all(e['bit_exact'] for e in entries)),
        thresholds=dict(atol=ATOL,rtol=RTOL,reference='native',equal_nan=False),
        loss=loss,parameter_inventory_equal=inventory_equal,gradient_tensors=len(entries),
        expected_gradient_tensors=len(names),per_tensor=entries,
        preconditions=preconditions,preconditions_pass=checks,
        raw_snapshots=dict(native=native_receipt,delegated=delegated_receipt),
        diagnostic_complete=False,GPU_equivalence_approved=False)
    write_json(Path(folder)/'LAMBDA0_COMPARISON.json',result)
    return result


def diagnostic_receipt(lambda0,gates,memory):
    """No successful final receipt from only the corrected lambda0 comparison."""
    headroom=memory.get('headroom_fraction')
    if (set(gates)!=set(REQUIRED_GATES) or not all(v is True for v in gates.values())
            or lambda0.get('status')!='PASS_WITHIN_TOLERANCE'
            or lambda0.get('lambda0_within_tolerance') is not True
            or lambda0.get('preconditions_pass') is not True
            or lambda0.get('gradient_tensors')!=456 or lambda0.get('expected_gradient_tensors')!=456
            or not isinstance(headroom,(float,int)) or not math.isfinite(headroom) or headroom<.15):
        raise ValueError('Incomplete/failed mandatory diagnostic; no PASS receipt')
    return dict(status='PASS_DIAGNOSTIC_ONLY_REQUIRES_INDEPENDENT_REVIEW',full_pretrained_CUDA_FP16=True,
        lambda0_within_tolerance=True,lambda0_bit_exact=lambda0['lambda0_bit_exact'],
        lambda0_thresholds=lambda0['thresholds'],mandatory_gates=gates,memory=memory,
        successful_optimizer_updates=0,generation_calls=0,training_authorized=False,
        baseline_inference_authorized=False,automatic_continuation=False)
