"""CPU-only localization of the existing LED value-window backward mechanism.

This demonstrates an operation path, not the cause of the historical CUDA error.
"""
from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import collections, hashlib, json, sys
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/stage_i/led_large_contrastive_v2_1_root_cause_review_v1'


def main():
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    from transformers import LEDConfig
    from transformers.models.led.modeling_led import LEDEncoderSelfAttention
    torch.set_num_threads(2)
    def forbidden(*a,**k):raise AssertionError('No CUDA permitted')
    class Trace(TorchDispatchMode):
        def __init__(self):self.operations=collections.Counter();self.views=[]
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            self.operations[str(func)]+=1
            if str(func)=='aten.as_strided.default':
                self.views.append(dict(size=list(args[1]),stride=list(args[2]),dtype=str(args[0].dtype)))
            return func(*args,**(kwargs or {}))
    with patch.object(torch.cuda,'_lazy_init',forbidden),patch.object(torch.Tensor,'cuda',forbidden):
        config=LEDConfig(d_model=16,encoder_attention_heads=1,attention_window=[1024]*12)
        layer=LEDEncoderSelfAttention(config,layer_id=11)
        value=torch.linspace(-1.,1.,4096).reshape(1,4096,1,1).requires_grad_()
        probs=torch.full((1,4096,1,1025),1./1025)
        trace=Trace()
        with trace:
            context=layer._sliding_chunks_matmul_attn_probs_value(probs,value,512)
            gradient,=torch.autograd.grad(context.square().sum(),(value,))
        view=next(v for v in trace.views if v['size']==[1,8,1536,1])
        # Reproduce precisely the value-window storage mapping, using integer IDs.
        indices=torch.arange(5120).reshape(1,5120,1)
        windows=indices.as_strided(view['size'],view['stride'])
        counts=torch.bincount(windows.reshape(-1),minlength=5120)
        real_counts=counts[512:512+4096]
        assert int(real_counts.max())==3
        assert trace.operations['aten.index_add_.default']>0
        assert torch.isfinite(gradient).all()
        raw=json.loads((OUT/'completed_cpu_check/SAVED_GRADIENT_REANALYSIS.json').read_text(encoding='utf-8'))
        final=[x for x in raw['per_tensor'] if '.encoder.layers.11.' in x['name']]
        changed=[x['name'].split('.11.')[1] for x in final if not x['bit_exact']]
        assert len(final)==18
        assert set(changed)=={'self_attn.longformer_self_attn.value.lora_A.default.weight',
                              'self_attn.longformer_self_attn.value.lora_B.default.weight'}
        result=dict(status='CPU_OPERATION_PATH_CONFIRMED_NOT_CUDA_CAUSAL_PROOF',
            real_GPU_final_encoder_layer_tensors=18,real_GPU_final_layer_changed_tensors=changed,
            real_GPU_final_layer_remaining_bit_exact=16,
            real_GPU_final_layer_A_failing_elements=1,real_GPU_final_layer_B_failing_elements=0,
            actual_pinned_LED_method='_sliding_chunks_matmul_attn_probs_value',
            CPU_value_shape=list(value.shape),attention_window=1024,window_overlap=512,
            value_window_view=view,storage_alias_multiplicity_histogram={str(int(k)):int((real_counts==k).sum()) for k in real_counts.unique()},
            index_add_backward_observed=trace.operations['aten.index_add_.default'],
            all_operations=dict(trace.operations),finite_CPU_gradient=True,
            CUDA_calls=0,optimizer_updates=0,generation_calls=0,
            inference='Strong structural support for overlapping-value CUDA accumulation hypothesis; kernel trace and repeat-native CUDA control were not captured historically; cause remains unproven')
    with (OUT/'VALUE_OVERLAP_CPU_EVIDENCE.json').open('x',encoding='utf-8') as f:
        json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='all_operations'},indent=2))


if __name__=='__main__':main()
