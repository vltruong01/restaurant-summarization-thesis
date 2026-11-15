"""CPU-only mask and routing regressions; no model instantiated or CUDA initialized."""
import copy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from tokenizers import Tokenizer

R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'scripts'))
from led_large_boundary_global_v1 import anchors_from_encoding, MaskIndex, MaskCollator, install_collator
from led_large_decoder_label_view_v1 import CorrectedLabelCollator
import run_led_large_boundary_global_v1 as runner


class Masks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.tok=Tokenizer.from_file(str(R/'assets/tokenizers/led-base-16384/tokenizer.json'))

    def make(self,reviews=2,captions=2,unicode=False):
        source='[BUSINESS_INFORMATION]\nname='+('Café bánh mì' if unicode else 'Cafe')+'\n\n[REVIEWS]\n'
        source+='\n\n'.join(f'[REV_{i}] stars=3 date=2020-01-01\nMixed useful opinion {i}.' for i in range(1,reviews+1))
        source+='\n\n[VISUAL_CONTEXT]'
        if captions:source+='\n'+'\n\n'.join(f'[IMG_CAP_{i}]\nA table with food.' for i in range(1,captions+1))
        enc=self.tok.encode(source)
        row=anchors_from_encoding(source,enc.ids,enc.offsets,reviews,captions)
        return source,enc,row,dict(input_ids=enc.ids,labels=[10,11,12,2])

    def test_multi_token_markers_only_first_BPE_global(self):
        _,_,row,f=self.make()
        self.assertTrue(all(m['token_end_exclusive']-m['token_start']>1 for m in row['markers']))
        b=MaskCollator(CorrectedLabelCollator,MaskIndex([row]),'boundary')([f])
        self.assertEqual(b['global_attention_mask'].nonzero()[:,1].tolist(),row['positions'])
        for m in row['markers']:
            self.assertEqual(b['global_attention_mask'][0,m['anchor']].item(),1)
            self.assertEqual(b['global_attention_mask'][0,m['anchor']+1:m['token_end_exclusive']].sum().item(),0)

    def test_offsets_independent_prefix_decode_with_unicode(self):
        source,enc,row,_=self.make(unicode=True)
        for m in row['markers']:
            self.assertEqual(self.tok.decode(enc.ids[1:m['token_start']],skip_special_tokens=False),source[:m['char_start']])
            self.assertEqual(source[m['char_start']:m['char_end']],m['marker'])

    def test_control_all_tensors_exact(self):
        _,_,row,f=self.make()
        old=CorrectedLabelCollator()([f]);new=MaskCollator(CorrectedLabelCollator,MaskIndex([row]),'bos_only')([f])
        self.assertEqual(set(old),set(new))
        self.assertTrue(all(torch.equal(old[k],new[k]) for k in old))

    def test_boundary_changes_only_global_attention(self):
        _,_,row,f=self.make()
        old=CorrectedLabelCollator()([f]);new=MaskCollator(CorrectedLabelCollator,MaskIndex([row]),'boundary')([f])
        self.assertTrue(all(torch.equal(old[k],new[k]) for k in ('input_ids','attention_mask','labels')))
        self.assertFalse(torch.equal(old['global_attention_mask'],new['global_attention_mask']))

    def test_zero_caption_retains_review_anchors(self):
        _,_,row,f=self.make(reviews=3,captions=0)
        self.assertEqual(len(row['positions']),4)
        self.assertTrue(all(m['kind']=='review' for m in row['markers']))
        self.assertEqual(MaskCollator(CorrectedLabelCollator,MaskIndex([row]),'boundary')([f])['global_attention_mask'].sum().item(),4)

    def test_full_count_envelope_is36(self):
        _,_,row,_=self.make(reviews=30,captions=5)
        self.assertEqual(len(row['positions']),36)

    def test_variable_counts_and_padding(self):
        _,_,a,f=self.make(1,0);_,_,b,g=self.make(5,3)
        batch=MaskCollator(CorrectedLabelCollator,MaskIndex([a,b]),'boundary',stress=True)([f,g])
        self.assertEqual(list(batch['input_ids'].shape),[2,5120])
        self.assertEqual(list(batch['labels'].shape),[2,768])
        self.assertEqual(batch['global_attention_mask'].sum(1).tolist(),[2,9])
        self.assertFalse((batch['global_attention_mask'].bool() & ~batch['attention_mask'].bool()).any().item())

    def test_corrected_EOS_and_ignored_padding_unchanged(self):
        _,_,row,f=self.make()
        batch=MaskCollator(CorrectedLabelCollator,MaskIndex([row]),'boundary')([f])
        labels=batch['labels'][0].tolist()
        self.assertEqual(labels[:4],[10,11,12,2]);self.assertTrue(all(v==-100 for v in labels[4:]))
        self.assertNotIn(0,labels)

    def test_truncated_source_rejected(self):
        _,_,row,f=self.make()
        with self.assertRaisesRegex(ValueError,'truncated'):
            MaskCollator(CorrectedLabelCollator,MaskIndex([row]),'boundary')([dict(f,input_ids=f['input_ids'][:-4]+[2])])

    def test_modified_source_rejected(self):
        _,_,row,f=self.make();f=copy.deepcopy(f);f['input_ids'][2]=123
        with self.assertRaises(ValueError):MaskCollator(CorrectedLabelCollator,MaskIndex([row]),'boundary')([f])

    def test_duplicate_or_missing_marker_rejected(self):
        source,_,_,_=self.make()
        for text in (source.replace('[REV_2]','[REV_1]'),source.replace('[IMG_CAP_2]','[IMG_CAP_3]')):
            enc=self.tok.encode(text)
            with self.assertRaises(ValueError):anchors_from_encoding(text,enc.ids,enc.offsets,2,2)

    def test_wrong_declared_count_rejected(self):
        source,enc,_,_=self.make()
        with self.assertRaises(ValueError):anchors_from_encoding(source,enc.ids,enc.offsets,1,2)

    def test_overflow_rejected_without_truncation(self):
        source,enc,_,_=self.make()
        with self.assertRaises(ValueError):anchors_from_encoding(source,[0]+[10]*5000+[2],[(0,0)]*5002,2,2)

    def test_incomplete_marker_offset_rejected(self):
        source,enc,row,_=self.make();offsets=list(enc.offsets)
        end=row['markers'][-1]['token_end_exclusive']
        offsets[end-1]=(0,0)
        with self.assertRaisesRegex(ValueError,'truncated'):anchors_from_encoding(source,enc.ids,offsets,2,2)

    def test_anchor_on_EOS_rejected(self):
        _,_,row,f=self.make();row=copy.deepcopy(row);row['positions'][-1]=len(f['input_ids'])-1
        with self.assertRaises(ValueError):MaskIndex([row]).lookup(f['input_ids'])

    def test_no_rng_consumption_or_CUDA_initialization(self):
        _,_,row,f=self.make();state=torch.get_rng_state().clone()
        MaskCollator(CorrectedLabelCollator,MaskIndex([row]),'boundary')([f])
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        self.assertFalse(torch.cuda.is_initialized())

    def test_shared_import_route_including_early_sanity_and_restore(self):
        import led_large_decoder_label_view_v1 as labels
        import led_large_decoder_early_sanity_v2 as early
        import run_stage_i_led_large_decoder_policy_v2 as base
        import inspect
        _,_,row,f=self.make();original=labels.CorrectedLabelCollator
        source=inspect.getsource(early.run_early_sanity)
        self.assertIn('from led_large_decoder_label_view_v1 import CorrectedLabelCollator',source)
        observations=[]
        with install_collator(labels,MaskIndex([row]),'boundary',observations.append):
            self.assertEqual(base.collator()([f])['global_attention_mask'].sum().item(),5)
            # Same import statement used inside the unchanged early-sanity function.
            from led_large_decoder_label_view_v1 import CorrectedLabelCollator as imported_for_sanity
            self.assertEqual(imported_for_sanity()([f])['global_attention_mask'].sum().item(),5)
        self.assertIs(labels.CorrectedLabelCollator,original);self.assertEqual(len(observations),2)

    def test_observer_failure_is_not_suppressed(self):
        _,_,row,f=self.make()
        def bad(_):raise RuntimeError('evidence write failed')
        with self.assertRaises(RuntimeError):MaskCollator(CorrectedLabelCollator,MaskIndex([row]),'boundary',observer=bad)([f])


class Contract(unittest.TestCase):
    def test_main_denies_before_parent_preflight(self):
        with patch.object(sys,'argv',['runner','train']),patch.object(runner,'preflight') as parent:
            with self.assertRaises(ValueError):runner.main()
            parent.assert_not_called()

    def test_resource_headroom_below15_blocks_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite=Path(tmp);out=suite/'bos_only';out.mkdir()
            (out/'RESOURCE_GATE.json').write_text(json.dumps(dict(status='PASS',binding={'arm':'bos_only'},
                approval_sha256='approved',headroom_fraction=.149)),encoding='utf-8')
            h=SimpleNamespace(binding=lambda:None,require_gate=lambda out,a:dict(hardware_profile='same'))
            with patch.object(runner,'binding',lambda arm:{'arm':arm}):
                with self.assertRaisesRegex(ValueError,'resource gates'):
                    runner.require_both_gates(suite,dict(_approval_sha256='approved'),h)

    def test_parent_phase_route_restores_objects(self):
        h=SimpleNamespace(CONFIG='old',binding=lambda:'old_h',check_deadline=lambda a:'old')
        b=SimpleNamespace(LARGE_PREP=Path('oldprep'),read=lambda p:{'original':True},
            binding=lambda:'old_b',check_deadline=lambda a:'old')
        old_h=h.binding;old_b=b.binding;old_read=b.read
        with patch.object(runner,'binding',lambda arm:{'arm':arm}):
            with runner.route_parent(h,b,MaskIndex([dict(input_ids_sha256='placeholder')]),'boundary',None):
                self.assertEqual(h.binding(),{'arm':'boundary'})
                self.assertEqual(b.binding(),{'arm':'boundary'})
                self.assertIn('business_ids',b.read(b.LARGE_PREP/'FEASIBILITY_CASES.json'))
                self.assertEqual(h.CONFIG,runner.P/'RUNTIME_CONFIG_boundary.json')
        self.assertIs(h.binding,old_h);self.assertIs(b.binding,old_b);self.assertIs(b.read,old_read)

    def test_no_approval_denied_before_importing_GPU_phase(self):
        with self.assertRaisesRegex(ValueError,'OFFLINE ONLY'):runner.authorize(None,False,'train')

    def test_expired_or_missing_reserve_fails(self):
        now=datetime.now(timezone.utc)
        a=dict(execution_authorized_at_utc=(now-timedelta(hours=2)).isoformat(),
            execution_deadline_utc=(now-timedelta(hours=1)).isoformat(),researcher_absolute_stop_utc=now.isoformat(),
            defense_reserve_at_least_14_days_confirmed=True)
        with self.assertRaises(ValueError):runner.check_time(a,now)
        a.update(execution_deadline_utc=(now+timedelta(hours=1)).isoformat(),researcher_absolute_stop_utc=(now+timedelta(hours=1)).isoformat(),
            defense_reserve_at_least_14_days_confirmed=False)
        with self.assertRaises(ValueError):runner.check_time(a,now)

    def test_config_difference_is_only_mask_and_arm_identity(self):
        parent=R/'configs/stage_i_led_large_decoder_policy_v2.json'
        spec=runner.read(runner.CONFIG)
        configs=runner.validate_runtime_configs(spec,parent)
        changed={k for k in configs['bos_only'] if configs['bos_only'][k]!=configs['boundary'][k]}
        self.assertEqual(changed,{'global_attention','run_root','vast_fresh'})

    def test_tests_did_not_initialize_CUDA(self):self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':unittest.main(verbosity=2)
