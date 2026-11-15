"""Synthetic/TRAIN/validation tests only. No model, CUDA or held-out payloads."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import copy
import json
import os
import sys
import tempfile
import unittest
import subprocess
import numpy as np

R = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'scripts'))
import led_large_rou_gate_only_v2 as g
import led_large_shuffled_production_v2 as s
import led_large_final_analysis_v2 as a
P = R/g.PREP
PARENT = Path(os.environ.get('LED_MATCHED_PARENT_ROOT', R/'outputs/cloud_handoff/LED_LARGE_DECODER_POLICY_V2_WORKSPACE')).resolve()
from tokenizers import Tokenizer


def row(bid, category='Italian', split='train', count=1):
    source = (f'[BUSINESS_INFORMATION]\nname={bid}\ncategories={category}, Restaurants\n\n'
              '[REVIEWS]\n[REV_1] stars=5 date=2020-01-01\nGood food and attentive staff.\n\n[VISUAL_CONTEXT]\n')
    source += '\n\n'.join(f'[IMG_CAP_{i}]\na table with a plate.' for i in range(1,count+1))
    return dict(business_id=bid,split=split,source=source,source_sha256=s.sha_text(source),
                selected_review_ids=['review_'+bid],selected_photo_ids=[f'photo_{bid}_{i}' for i in range(count)])


class GateContracts(unittest.TestCase):
    def synthetic_approval(self, condition='reviews_only'):
        value = g.read(P/'PENDING_APPROVALS'/('RO_GATE_ONLY_PENDING.json' if condition=='reviews_only' else 'U_GATE_ONLY_PENDING.json'))
        now = datetime.now(timezone.utc)
        value.update(GPU_feasibility_approved=True, approval_source='researcher_message',
                     researcher_explicit_authorization='SYNTHETIC UNIT FIXTURE ONLY; no actual researcher approval',
                     authorized_at_utc=now.isoformat(), deadline_utc=(now+timedelta(hours=1)).isoformat(),
                     researcher_absolute_stop_utc=(now+timedelta(hours=2)).isoformat(),
                     defense_reserve_at_least_14_days_confirmed=True)
        return value

    def check(self, value, condition='reviews_only'):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'SYNTHETIC_APPROVAL.json';path.write_text(json.dumps(value))
            return g.authorize(path,condition)

    def test_both_pending_receipts_block_execution(self):
        for name,c in [('RO','reviews_only'),('U','unfiltered')]:
            with self.assertRaises(ValueError):g.authorize(P/'PENDING_APPROVALS'/(name+'_GATE_ONLY_PENDING.json'),c)

    def test_synthetic_valid_gate_receipt_keeps_training_false(self):
        result=self.check(self.synthetic_approval())
        self.assertFalse(result['training_authorized']);self.assertEqual(result['max_training_updates'],0)
        self.assertEqual(result['execution_deadline_utc'],result['deadline_utc'])

    def test_training_or_scope_expansion_is_rejected(self):
        for key,value in [('training_authorized',True),('scopes',['feasibility','train']),
                          ('automatic_train',True),('max_training_updates',1250),('max_generation_calls',6),
                          ('max_actual_microbatches',96),('held_out_access',True)]:
            x=self.synthetic_approval();x[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.check(x)

    def test_expired_or_over48h_or_unconfirmed_reserve_rejected(self):
        for kind in ('expired','over48','reserve','naive'):
            x=self.synthetic_approval();now=datetime.now(timezone.utc)
            if kind=='expired':x['deadline_utc']=(now-timedelta(seconds=1)).isoformat()
            if kind=='over48':
                x['deadline_utc']=(now+timedelta(hours=49)).isoformat();x['researcher_absolute_stop_utc']=(now+timedelta(hours=50)).isoformat()
            if kind=='reserve':x['defense_reserve_at_least_14_days_confirmed']=False
            if kind=='naive':x['authorized_at_utc']='2026-10-24T12:00:00'
            with self.subTest(kind=kind),self.assertRaises(ValueError):self.check(x)

    def test_condition_and_binding_mismatch_rejected(self):
        x=self.synthetic_approval()
        with self.assertRaises(ValueError):self.check(x,'unfiltered')
        x['binding']['runner_sha256']='0'*64
        with self.assertRaises(ValueError):self.check(x)

    def test_train_mode_not_in_CLI(self):
        command=[sys.executable,str(R/'scripts/run_led_large_rou_gate_only_v2.py'),'train',
                 '--condition','reviews_only','--parent-root',str(PARENT)]
        p=subprocess.run(command,capture_output=True,text=True)
        self.assertNotEqual(p.returncode,0);self.assertIn('invalid choice',p.stderr)

    def test_real_native_gate_function_and_condition_routing_unchanged(self):
        # Fresh imports avoid module-global state crossing condition checks.
        code="""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(sys.argv[2])/'scripts'))
import run_stage_i_led_large_decoder_policy_v2 as b
original=b.feasible
import led_large_rou_gate_only_v2 as g
b,c=g.install_gate_context(sys.argv[1],Path(sys.argv[2]).resolve())
assert b.feasible is original
assert b.binding()==g.binding(sys.argv[1])
assert b.check_deadline is g.deadline
train,val=b.data(c)
assert len(train)==2000 and len(val)==200 and c['condition']==sys.argv[1]
expected=g.read(g.ROOT/'outputs/stage_i/led_large_matched_family_v1'/(sys.argv[1]+'_FEASIBILITY_CASES.json'))
assert b.read(b.LARGE_PREP/'FEASIBILITY_CASES.json')==expected
print('NATIVE_GATE_HOOK_PASS')
"""
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(R/'scripts'))
        for condition in g.CONDITIONS:
            p=subprocess.run([sys.executable,'-c',code,condition,str(PARENT)],env=env,cwd=R,
                             capture_output=True,text=True,timeout=180)
            self.assertEqual(p.returncode,0,p.stdout+p.stderr);self.assertIn('NATIVE_GATE_HOOK_PASS',p.stdout)


class ProductionMapper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy=g.read(P/'SHUFFLED_POLICY_PROPOSED.json')
        cls.tok=Tokenizer.from_file(str(PARENT/'assets/tokenizers/led-base-16384/tokenizer.json'))

    def run_map(self,rows):return s.build_mapping(rows,self.tok,self.policy)

    def test_real_train_validation_fixtures_are_reproducible(self):
        rows=g.read(P/'SHUFFLED_SOURCE_FIXTURES.json')['rows']
        self.assertEqual(self.run_map(rows),g.read(P/'SHUFFLED_FIXTURE_RESULTS.json'))

    def test_input_iteration_order_does_not_change_assignments(self):
        rows=[row('A'),row('B'),row('C','Thai')]
        self.assertEqual(self.run_map(rows)['mappings'],self.run_map(list(reversed(rows)))['mappings'])

    def test_same_category_preferred_and_declared_fallback_is_explicit(self):
        out=self.run_map([row('A'),row('B'),row('C','Thai')])['mappings'];d={r['recipient_id']:r for r in out}
        self.assertEqual(d['A']['donor_id'],'B');self.assertEqual(d['B']['donor_id'],'A')
        self.assertEqual(d['C']['tier'],'DECLARED_SAME_COUNT_FALLBACK')

    def test_reuse_allowed_without_silent_one_to_one_constraint(self):
        out=self.run_map([row('A'),row('B'),row('C','Thai')])
        self.assertFalse(out['one_to_one']);self.assertGreaterEqual(max(out['donor_reuse_counts'].values()),2)

    def test_unknown_categories_do_not_expand_taxonomy(self):
        before=copy.deepcopy(self.policy)
        result=self.run_map([row('A','UNSEEN_CATEGORY'),row('B','UNSEEN_CATEGORY')])
        self.assertTrue(all(r['tier']=='DECLARED_SAME_COUNT_FALLBACK' for r in result['mappings']))
        self.assertEqual(before,self.policy)

    def test_zero_caption_source_and_tokens_unchanged(self):
        r=row('ZERO',count=0);out=self.run_map([r])['mappings'][0]
        self.assertIsNone(out['donor_id']);self.assertEqual(out['source'],r['source'])
        self.assertEqual(out['input_ids'],self.tok.encode(r['source']).ids)

    def test_no_self_cross_split_or_count_fallback(self):
        for rows in ([row('A')],[row('A'),row('B',split='validation')],
                     [row('A',count=1),row('B',count=2)]):
            with self.assertRaises(ValueError):self.run_map(rows)

    def test_target_or_reference_or_prediction_columns_rejected(self):
        for name in ('target','reference','prediction'):
            rows=[row('A'),row('B')];rows[0][name]='FORBIDDEN'
            with self.assertRaises(ValueError):self.run_map(rows)

    def test_source_hash_and_marker_integrity_fail_closed(self):
        for kind in ('hash','review_marker','caption_marker'):
            rows=[row('A'),row('B')]
            if kind=='hash':rows[0]['source_sha256']='0'*64
            else:
                rows[0]['source']=rows[0]['source'].replace('[REV_1]','[REV_2]') if kind=='review_marker' else rows[0]['source'].replace('[IMG_CAP_1]','[IMG_CAP_2]')
                rows[0]['source_sha256']=s.sha_text(rows[0]['source'])
            with self.assertRaises(ValueError):self.run_map(rows)

    def test_header_included_in_original_750_caption_allowance(self):
        r=row('A');head,_=s.parts(r);base=head.removesuffix('\n\n[VISUAL_CONTEXT]')
        # Choose a synthetic length that the old header-excluding subtraction
        # would allow, but the exact Stage H accounting rejects.
        found=False
        for words in range(730,755):
            r['source']=head+'\n[IMG_CAP_1]\n'+' table'*words
            exact=len(self.tok.encode(r['source']).ids)-len(self.tok.encode(base).ids)
            old=len(self.tok.encode(r['source']).ids)-len(self.tok.encode(head).ids)
            if old<=750<exact:
                r['source_sha256']=s.sha_text(r['source']);found=True
                with self.assertRaises(ValueError):self.run_map([r,row('B')])
                break
        self.assertTrue(found,'No boundary fixture found')

    def test_review_and_full_source_overflow_stop_without_truncation(self):
        r=row('A');r['source']=r['source'].replace('Good food and attentive staff.',' word'*5100)
        r['source_sha256']=s.sha_text(r['source'])
        with self.assertRaises(ValueError):self.run_map([r,row('B')])

    def test_production_test_scope_is_blocked_without_fresh_receipt(self):
        # Synthetic IDs only; rejection occurs before tokenization/mapping.
        with self.assertRaises(ValueError):self.run_map([row('SYNTHETIC_A',split='test'),row('SYNTHETIC_B',split='test')])

    def test_recipient_review_bytes_and_frozen_review_id_order_unchanged(self):
        rows=[row('A'),row('B')];out=self.run_map(rows)['mappings'];table={r['business_id']:r for r in rows}
        for r in out:
            original=table[r['recipient_id']]
            self.assertEqual(s.parts(r)[0],s.parts(original)[0])
            self.assertEqual(r['selected_review_ids'],original['selected_review_ids'])
            head,_=s.parts(original);base=head.removesuffix('\n\n[VISUAL_CONTEXT]')
            self.assertEqual(r['caption_increment_tokens'],len(r['input_ids'])-len(self.tok.encode(base).ids))


class AnalysisAndBlinding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        lock=g.read(P/'ANALYSIS_LOCK.json');wheel=P/'analysis_source_pin'/lock['agreement']['filename']
        if g.sha(wheel)!=lock['agreement']['sha256']:raise ValueError('Wrong pinned agreement wheel')
        sys.path.insert(0,str(wheel))
        import krippendorff
        cls.alpha=staticmethod(krippendorff.alpha)

    def test_constant_known_paired_delta_and_reproducibility(self):
        values=np.tile([1.,2.,4.,3.],(30,1));x=a.paired_statistics(values)
        self.assertEqual(x,a.paired_statistics(values));self.assertEqual(x['contrasts'][0]['CI95'],[2.,2.])
        self.assertTrue(x['contrasts'][0]['primary']);self.assertFalse(x['p_values_computed'])

    def test_missing_or_tuned_analysis_rejected(self):
        x=np.ones((30,4));x[0,0]=np.nan
        with self.assertRaises(ValueError):a.paired_statistics(x)
        with self.assertRaises(ValueError):a.paired_statistics(np.ones((30,4)),seed=99)

    def test_visual_statistics_never_replace_RO_NA_by_zero(self):
        x=np.tile([np.nan,2.,4.,3.],(30,1));out=a.visual_statistics(x)
        self.assertEqual(len(out['contrasts']),2);self.assertEqual(out['RO'],'N/A')
        x[:,0]=0
        with self.assertRaises(ValueError):a.visual_statistics(x)

    def test_three_rater_business_means_and_RO_visual_NA(self):
        x=np.tile([1.,3.,5.],(30,4,1));self.assertTrue((a.human_means(x)==3).all())
        x[:,0,:]=np.nan;out=a.human_means(x,visual=True);self.assertTrue(np.isnan(out[:,0]).all())
        x[0,1,0]=np.nan
        with self.assertRaises(ValueError):a.human_means(x,visual=True)

    def test_pinned_ordinal_agreement_perfect_nonconstant_and_imperfect(self):
        x=np.tile([1.,2.,3.,4.,5.],(3,1));self.assertEqual(a.ordinal_agreement(x,self.alpha)['alpha'],1.)
        x[0]=[5,4,3,2,1];self.assertLess(a.ordinal_agreement(x,self.alpha)['alpha'],1.)

    def test_constant_agreement_undefined_and_missing_scores_blocked(self):
        self.assertIsNone(a.ordinal_agreement(np.ones((3,10)),self.alpha)['alpha'])
        x=np.ones((3,10));x[0,0]=np.nan
        with self.assertRaises(ValueError):a.ordinal_agreement(x,self.alpha)
        ro=a.ordinal_agreement(np.full((3,30),np.nan),self.alpha)
        self.assertEqual(ro['status'],'NOT_APPLICABLE');self.assertIsNone(ro['alpha'])

    def test_private_key_is_not_in_three_public_120_task_packets(self):
        ids=[f'SYNTHETIC_{i}' for i in range(30)];secret=b'SYNTHETIC_ONLY_NOT_PRODUCTION_KEY'*2
        private,packets=a.separate_annotation_packets(ids,secret)
        self.assertEqual(len(private['condition_key']),360)
        for packet in packets.values():
            self.assertEqual(len(packet['tasks']),120)
            text=json.dumps(packet)
            self.assertNotIn('condition_key',text);self.assertNotIn('business_id',text);self.assertNotIn(secret.decode(),text)
            self.assertTrue(all(t['summary_artifact'] is None and t['source_packet'] is None for t in packet['tasks']))
        self.assertEqual((private,packets),a.separate_annotation_packets(ids,secret))

    def test_changing_private_key_changes_aliases_and_order(self):
        ids=[f'SYNTHETIC_{i}' for i in range(30)]
        one,p1=a.separate_annotation_packets(ids,b'A'*32);two,p2=a.separate_annotation_packets(ids,b'B'*32)
        self.assertNotEqual(set(one['condition_key']),set(two['condition_key']))
        self.assertNotEqual(p1['A01']['tasks'][0]['alias'],p2['A01']['tasks'][0]['alias'])

    def test_metric_constructor_exactly_matches_accepted_lock(self):
        args=a.bertscore_constructor_kwargs('PINNED_LOCAL_MODEL_NOT_LOADED')
        self.assertEqual(args['num_layers'],17);self.assertFalse(args['use_fast_tokenizer'])
        self.assertFalse(args['idf']);self.assertFalse(args['rescale_with_baseline']);self.assertEqual(args['device'],'cpu')


if __name__=='__main__':unittest.main(verbosity=2)
