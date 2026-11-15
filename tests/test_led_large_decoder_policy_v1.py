"""Offline tensor/contracts tests. No model construction, forward, generate, backward or optimizer."""
import copy,inspect,json,sys,tempfile,unittest
from pathlib import Path
from datetime import datetime,timedelta,timezone
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from led_large_decoder_label_view_v1 import (remove_initial_bos,validate_view_labels,CorrectedLabelDataset,
    CorrectedLabelCollator,sanity_result,sha,POLICY)
from stage_hi_led_data import LEDCompleteCollator,PackedLEDDataset
from transformers.models.led.modeling_led import shift_tokens_right,LEDForConditionalGeneration
from transformers.generation.utils import GenerationMixin
from transformers import GenerationConfig
import stage_i_led_large_decoder_policy_v1 as helper
import run_stage_i_led_large_decoder_policy_v1 as runner

class LabelPolicy(unittest.TestCase):
    def test_strip_exactly_initial_BOS_and_keep_content_EOS(self):
        old=[0,10975,4629,2];new=remove_initial_bos(old)
        self.assertEqual(old,[0,10975,4629,2]);self.assertEqual(new,[10975,4629,2])
    def test_reject_already_stripped_missing_EOS_duplicate_BOS_and_PAD(self):
        for bad in ([10975,2],[0,10],[0,0,10,2],[0,10,2,2],[0,10,1,2],[0,-100,2],[0,True,2]):
            with self.subTest(bad=bad),self.assertRaises(ValueError):remove_initial_bos(bad)
    def test_collator_keeps_source_global_masks_and_original_padded_shapes(self):
        raw=[dict(input_ids=[0,50,51,2],labels=[0,10,11,2]),dict(input_ids=[0,60,2],labels=[0,12,13,14,2])]
        viewed=[dict(input_ids=f['input_ids'],labels=remove_initial_bos(f['labels'])) for f in raw]
        old=LEDCompleteCollator()(raw);new=CorrectedLabelCollator()(viewed)
        for key in ('input_ids','attention_mask','global_attention_mask'):
            self.assertTrue(torch.equal(old[key],new[key]))
        self.assertEqual(old['labels'].shape,new['labels'].shape)
        self.assertEqual(new['labels'][0,:3].tolist(),[10,11,2]);self.assertTrue((new['labels'][0,3:]==-100).all())
        self.assertFalse((new['labels']==0).any());self.assertEqual(new['global_attention_mask'].sum().item(),2)
    def test_stress_shape_5120_768_preserved(self):
        batch=CorrectedLabelCollator(stress=True)([dict(input_ids=[0,8,2],labels=[10975,4629,2])])
        self.assertEqual(tuple(batch['input_ids'].shape),(1,5120));self.assertEqual(tuple(batch['labels'].shape),(1,768))
    def test_pinned_shift_yields_EOS_first_content_without_BOS(self):
        labels=torch.tensor([[10975,4629,2,-100,-100]])
        shifted=shift_tokens_right(labels,1,2)
        self.assertEqual(shifted.tolist(),[[2,10975,4629,2,1]])
        self.assertEqual(labels.tolist(),[[10975,4629,2,-100,-100]])
    def test_cross_entropy_alignment_and_ignored_padding(self):
        labels=torch.tensor([[10,11,2,-100]])
        logits=torch.full((1,4,20),-10.)
        for i,t in enumerate([10,11,2]):logits[0,i,t]=10.
        loss=torch.nn.CrossEntropyLoss()(logits.view(-1,20),labels.view(-1))
        wrong=torch.nn.CrossEntropyLoss()(logits.view(-1,20),torch.tensor([0,10,11,-100]))
        self.assertLess(loss.item(),1e-5);self.assertGreater(wrong.item(),10.)
    def test_actual_LED_loss_uses_unshifted_labels_once(self):
        src=inspect.getsource(LEDForConditionalGeneration.forward)
        self.assertIn('decoder_input_ids = shift_tokens_right(',src)
        self.assertIn('labels, self.config.pad_token_id, self.config.decoder_start_token_id',src)
        self.assertIn('loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))',src)
    def test_generation_start_helper_creates_only_EOS(self):
        ids,kw=GenerationMixin._prepare_decoder_input_ids_for_generation(SimpleNamespace(device=torch.device('cpu')),
            batch_size=2,model_input_name='input_ids',model_kwargs={},decoder_start_token_id=torch.tensor(2),device=torch.device('cpu'))
        self.assertEqual(ids.tolist(),[[2],[2]]);self.assertEqual(kw,{})
    def test_generation_configuration_has_no_forced_BOS_or_new_constraints(self):
        c=helper.read(helper.CONFIG)
        g=GenerationConfig(**helper.read(ROOT/c['effective_generation_config_file']))
        self.assertIsNone(g.forced_bos_token_id);self.assertEqual(g.decoder_start_token_id,2)
        self.assertTrue(g.use_cache);self.assertEqual(g.num_beams,4);self.assertEqual(g.max_length,768)
        self.assertEqual(g.no_repeat_ngram_size,0);self.assertIsNone(g.suppress_tokens)
    def test_view_dataset_rejects_changed_content_or_business_order(self):
        raw=SimpleNamespace(business_ids=['x'])
        class Base:
            business_ids=['x']
            def __getitem__(self,i):return dict(input_ids=[0,5,2],labels=[0,10,11,2])
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'labels.npz'
            np.savez(p,business_ids=np.asarray(['x']),labels=np.asarray([10,11,2]),label_offsets=np.asarray([0,3]))
            view=CorrectedLabelDataset(Base(),dict(path=p.name,sha256=sha(p)),Path(d))
            self.assertEqual(view[0],dict(input_ids=[0,5,2],labels=[10,11,2]))
            np.savez(p,business_ids=np.asarray(['x']),labels=np.asarray([10,99,2]),label_offsets=np.asarray([0,3]))
            with self.assertRaisesRegex(ValueError,'changed content'):CorrectedLabelDataset(Base(),dict(path=p.name,sha256=sha(p)),Path(d))
    def test_real_F_views_are_complete_and_sources_unchanged(self):
        c=helper.read(helper.CONFIG);train,val=runner.data(c)
        self.assertEqual((len(train),len(val)),(2000,200))
        for data in (train,val):
            for i in (0,len(data)-1):
                self.assertEqual(data[i]['input_ids'],data.original[i]['input_ids'])
                self.assertEqual(data[i]['labels'],data.original[i]['labels'][1:])

class SanityAndScope(unittest.TestCase):
    def test_early_probe_restores_modes_cache_and_rng_on_failure(self):
        import led_large_decoder_early_sanity_v1 as early
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.sub=torch.nn.Identity();self.sub.eval()
                self.config=SimpleNamespace(use_cache=False)
            def generate(self,**kwargs):raise RuntimeError('SYNTHETIC generation fixture failure')
        model=Model();before={n:m.training for n,m in model.named_modules()};restored=[]
        cfg=helper.read(helper.CONFIG)
        bid=cfg['early_sanity']['business_ids'][0]
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);folder=root/'train';checkpoint=folder/'checkpoint-250';checkpoint.mkdir(parents=True)
            (checkpoint/'checkpoint_integrity.json').write_text('{}')
            cases=root/cfg['early_sanity']['cases_file'];cases.parent.mkdir(parents=True,exist_ok=True)
            cases.write_text(json.dumps(dict(business_id=bid,input_ids=[0,9,2],label_ids=[0,10,11,2]))+'\n')
            fixture_rng=dict(cpu=torch.get_rng_state())
            with patch('transformers.LEDTokenizerFast.from_pretrained',return_value=SimpleNamespace()),patch('peft.get_peft_model_state_dict',return_value={'A':torch.ones(1)}),patch('led_large_dynamic_amp_v2.capture_rng',return_value=fixture_rng),patch('led_large_dynamic_amp_v2.restore_rng',side_effect=lambda r:restored.append(r)),patch.object(torch.Tensor,'cuda',lambda t:t):
                # The config file is part of this fixture; no real model forward.
                gen=root/cfg['effective_generation_config_file'];gen.parent.mkdir(parents=True,exist_ok=True)
                gen.write_text(json.dumps(helper.read(ROOT/cfg['effective_generation_config_file'])))
                with self.assertRaisesRegex(RuntimeError,'SYNTHETIC'):
                    early.run_early_sanity(model,folder,dict(early_sanity_generation_approved=True,early_sanity_outputs_max=4),cfg,
                        SimpleNamespace(check=lambda:None),root,{},lambda *a:None,lambda a:None)
        self.assertEqual({n:m.training for n,m in model.named_modules()},before)
        self.assertFalse(model.config.use_cache);self.assertEqual(len(restored),1)

    def test_sanity_rejects_BOS_empty_cap_and_trivial_output(self):
        for ids,text in [([2]+[0]*766+[2],''),([2,0]+[10]*20+[2],'text'),([2]+[10]*766+[2],'text'),([2,2],'')]:
            self.assertEqual(sanity_result(ids,text)['status'],'FAIL_STOP')
    def test_technical_sanity_is_never_quality_GO(self):
        r=sanity_result([2]+[10]*16+[2],'fixture content')
        self.assertEqual(r['status'],'PASS_TECHNICAL_ONLY');self.assertFalse(r['semantic_quality_GO'])
    def test_new_authorization_template_and_old_approval_rejected(self):
        template=helper.PREP/'AUTHORIZATION_TEMPLATE_NOT_APPROVED.json'
        with self.assertRaisesRegex(ValueError,'not approved'):helper.authorize(template,'train',True)
        old=ROOT/'VAST_RESEARCHER_AUTHORIZATION.json'
        if old.is_file():
            with self.assertRaisesRegex(ValueError,'not approved|scope'):helper.authorize(old,'train',True)
    def test_synthetic_bound_authorization_checks_hash_and_amendment(self):
        a=helper.read(helper.PREP/'AUTHORIZATION_TEMPLATE_NOT_APPROVED.json')
        now=datetime.now(timezone.utc)
        a.update(GPU_execution_approved=True,training_authorized=True,scientific_amendment_approved=True,
            researcher_explicit_authorization='SYNTHETIC TEST ONLY',early_sanity_generation_approved=True,
            execution_authorized_at_utc=(now-timedelta(minutes=1)).isoformat(),execution_deadline_utc=(now+timedelta(hours=1)).isoformat(),
            researcher_absolute_stop_utc=(now+timedelta(hours=2)).isoformat(),defense_reserve_at_least_14_days_confirmed=True)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'SYNTHETIC_ONLY.json';p.write_text(json.dumps(a))
            self.assertIn('_approval_sha256',helper.authorize(p,'train',True))
            a['runner_sha256']='wrong';p.write_text(json.dumps(a))
            with self.assertRaisesRegex(ValueError,'hash'):helper.authorize(p,'train',True)
    def test_past_or_extended_deadline_rejected(self):
        a=helper.read(helper.PREP/'AUTHORIZATION_TEMPLATE_NOT_APPROVED.json');now=datetime.now(timezone.utc)
        a.update(execution_authorized_at_utc=(now-timedelta(hours=2)).isoformat(),execution_deadline_utc=(now-timedelta(hours=1)).isoformat(),
            researcher_absolute_stop_utc=(now+timedelta(days=2)).isoformat(),defense_reserve_at_least_14_days_confirmed=True)
        with self.assertRaisesRegex(ValueError,'expired'):helper.check_deadline(a)
    def test_no_resume_or_GPU_call_without_new_approval(self):
        with self.assertRaisesRegex(ValueError,'Fresh corrected'):runner.train({}, {},Path('unused'),resume=Path('old'))
        with patch.object(runner,'preflight',return_value={}),patch.object(sys,'argv',['runner','train']),patch.object(runner,'check_required_gpu',side_effect=AssertionError('GPU MUST NOT BE CALLED')):
            with self.assertRaisesRegex(ValueError,'OFFLINE ONLY'):runner.main()
    def test_epoch1_callback_invokes_sanity_and_propagates_failure(self):
        class Base:
            def __init__(self):self.trainer=SimpleNamespace(model=object());self.folder=Path('fixture');self.approval={};self.memory=object()
            def on_save(self,*a,**kw):pass
        with patch.object(helper,'_core_callback_class',return_value=Base),patch('led_large_decoder_early_sanity_v1.run_early_sanity',side_effect=ValueError('sanity failed')) as screen:
            c=helper.callback_class()()
            with self.assertRaisesRegex(ValueError,'sanity failed'):c.on_save(None,SimpleNamespace(global_step=250),None)
            screen.assert_called_once()

if __name__=='__main__':unittest.main()
