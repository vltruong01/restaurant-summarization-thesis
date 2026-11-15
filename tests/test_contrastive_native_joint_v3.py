"""Focused CPU loss/accounting tests; not CUDA numerical qualification."""
from pathlib import Path
import sys,types,unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import led_large_contrastive_native_joint_v3 as objective


class Native(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.tensor([.3,-.2,.1]))
        self.calls=[]
    def forward(self,labels,**kw):
        self.calls.append(labels.clone())
        logits=self.weight.expand(labels.shape[0],labels.shape[1],3)
        return types.SimpleNamespace(loss=torch.nn.functional.cross_entropy(logits.reshape(-1,3),labels.reshape(-1),ignore_index=-100))


def batch(labels):
    return dict(input_ids=torch.tensor([[0,7,2]]),attention_mask=torch.ones(1,3,dtype=torch.long),
                global_attention_mask=torch.tensor([[1,0,0]]),labels=torch.tensor([labels]))


class Tests(unittest.TestCase):
    def test_control_exact_native_loss_and_gradient(self):
        m=Native();p=batch([0,1,2,-100]);a=m(**p).loss;ga=torch.autograd.grad(a,m.weight)[0]
        b,_=objective.objective(m,p,arm='ce');gb=torch.autograd.grad(b,m.weight)[0]
        self.assertTrue(torch.equal(a,b));self.assertTrue(torch.equal(ga,gb))
    def test_pair_formula_and_gradient_joint_reference(self):
        m=Native();p=batch([0,1,2,-100]);n=batch([1,1,2,-100])
        got,_=objective.objective(m,p,n,arm='pair');gg=torch.autograd.grad(got,m.weight)[0]
        ce=m(**p).loss;neg=m(**n).loss;expected=ce+.1*torch.log1p(torch.exp(.1+ce-neg))
        eg=torch.autograd.grad(expected,m.weight)[0]
        self.assertTrue(torch.allclose(got,expected,atol=1e-7,rtol=1e-6))
        self.assertTrue(torch.allclose(gg,eg,atol=1e-7,rtol=1e-6))
    def test_negative_own_labels_and_order(self):
        m=Native();p=batch([0,1,2]);n=batch([1,1,2]);events=[]
        objective.objective(m,p,n,arm='pair',observe=lambda kind,b:events.append((kind,b['labels'].clone())))
        self.assertEqual([k for k,_ in events],['positive','negative'])
        self.assertTrue(torch.equal(m.calls[0],p['labels']));self.assertTrue(torch.equal(m.calls[1],n['labels']))
    def test_padding_excluded_EOS_included_native(self):
        m=Native();p=batch([0,1,2]);q=batch([0,1,2,-100,-100])
        a,_=objective.objective(m,p,arm='ce');b,_=objective.objective(m,q,arm='ce')
        self.assertTrue(torch.equal(a,b));self.assertFalse(torch.equal(a,m(**batch([0,1])).loss))
    def test_accumulation_is_mean_and_no_parameter_update(self):
        m=Native();p=batch([0,1,2]);n=batch([1,1,2]);before=m.weight.detach().clone()
        loss,_=objective.objective(m,p,n,arm='pair');expected=torch.autograd.grad(loss,m.weight)[0]
        for _ in range(8):
            value,_=objective.objective(m,p,n,arm='pair');(value/8).backward()
        self.assertTrue(torch.allclose(m.weight.grad,expected,atol=1e-7,rtol=1e-6))
        self.assertTrue(torch.equal(before,m.weight))
    def test_reject_nonfinite(self):
        m=Native();m.weight.data.fill_(float('nan'))
        with self.assertRaises(ValueError):objective.objective(m,batch([0,1,2]),arm='ce')
    def test_source_mismatch_rejected(self):
        p=batch([0,1,2]);n=batch([1,1,2]);n['input_ids'][0,1]=8
        with self.assertRaises(ValueError):objective.objective(Native(),p,n,arm='pair')
    def test_no_extra_negative_forward_in_control(self):
        m=Native();objective.objective(m,batch([0,1,2]),object(),arm='ce')
        self.assertEqual(len(m.calls),1)


if __name__=='__main__':unittest.main()
