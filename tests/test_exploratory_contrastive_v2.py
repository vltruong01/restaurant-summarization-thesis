"""Focused locked-amendment/authorization tests. No model load or GPU operations."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
from unittest.mock import patch
import ast
import copy
import sys
import types
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_led_large_exploratory_contrastive_v2 as r

class AmendmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c=r.read(r.CONFIG);cls.a=r.read(r.CONFIG.parent/'RESEARCHER_AUTHORIZATION.json')
        cls.d=r.read(r.CONFIG.parent/'RESEARCHER_ADJUDICATION_AMENDMENT.json')
    def setup_guard(self,mode='diagnose'):
        a=copy.deepcopy(self.a);d=copy.deepcopy(self.d)
        for k in ['config_sha256','runner_sha256','objective_sha256','preflight_sha256','researcher_decisions_sha256']:a[k]='h'
        for group in ['positives','pairs']:
            for item in d[group].values():item['sha256']='h'
        return a,d,types.SimpleNamespace(execute=True,approval=Path('approval'),decisions=Path('decisions'),mode=mode,arm='pair',numerical_receipt=None)
    def test_full_cpu_binding_preflight(self):self.assertEqual(r.preflight()['accepted_pairs'],15)
    def test_quota_and_no_replacements(self):
        self.assertEqual(self.c['pair_ids'],[f'P{i:02d}' for i in range(1,17) if i!=3])
        self.assertEqual(self.d['prior_protocol_outcome'],'NO_GO')
        self.assertTrue(self.c['not_preregistered']);self.assertTrue(self.c['post_adjudication_amendment'])
        self.assertEqual(self.c['total_successful_update_ceiling'],32)
        self.assertEqual(self.c['total_generation_calls_ceiling'],12)
    def test_explicit_amendment_admits_exact15(self):
        a,d,args=self.setup_guard()
        with patch.object(r,'read',side_effect=[a,d]),patch.object(r,'sha',return_value='h'):
            self.assertEqual(r.authorization(args,self.c),a)
    def test_pending_cannot_be_invented_as_accepted(self):
        a,d,args=self.setup_guard();next(iter(d['positives'].values()))['decision']='PENDING'
        with patch.object(r,'read',side_effect=[a,d]),patch.object(r,'sha',return_value='h'):
            with self.assertRaisesRegex(ValueError,'Eight accepted'):r.authorization(args,self.c)
    def test_p03_cannot_return(self):
        a,d,args=self.setup_guard();d['pairs']['P03']=dict(decision='ACCEPT',sha256='h')
        with patch.object(r,'read',side_effect=[a,d]),patch.object(r,'sha',return_value='h'):
            with self.assertRaisesRegex(ValueError,'Exactly15'):r.authorization(args,self.c)
    def test_expired_approval_stops(self):
        a,d,args=self.setup_guard();a['deadline_utc']=(datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat()
        with patch.object(r,'read',return_value=a),patch.object(r,'sha',return_value='h'):
            with self.assertRaisesRegex(ValueError,'Expired'):r.authorization(args,self.c)
    def test_no_execute_precedes_cuda_or_file_read(self):
        a,d,args=self.setup_guard();args.execute=False
        with patch.object(r,'read',side_effect=AssertionError('File read before authorization')):
            with self.assertRaisesRegex(ValueError,'before CUDA'):r.authorization(args,self.c)
    def test_no_training_or_baseline_without_diagnostic(self):
        for mode in ['pilot','baseline']:
            a,d,args=self.setup_guard(mode)
            with patch.object(r,'read',side_effect=[a,d]),patch.object(r,'sha',return_value='h'):
                with self.assertRaisesRegex(ValueError,'Diagnostic receipt'):r.authorization(args,self.c)
    def test_identical_positive_order_and_own_prefixes(self):
        data=r.collated(self.c,'cpu');self.assertEqual(len(data),128)
        import torch
        for i,item in enumerate(data):
            pos=r.read(r.PREP/'positives'/f"{self.c['fit'][i%8]}.json")
            y=item['positive']['labels'];n=item['negative']['labels']
            self.assertEqual(y[y!=-100].tolist(),pos['corrected_labels'])
            self.assertNotEqual(y[y!=-100].tolist(),n[n!=-100].tolist())
            self.assertTrue(torch.equal(item['positive']['input_ids'],item['negative']['input_ids']))
    def test_memory_import_unconditional_and_fixed_optimizer_schedule(self):
        tree=ast.parse(Path(r.__file__).read_text(encoding='utf-8'))
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        attempt=next(n for n in main.body if isinstance(n,ast.Try))
        self.assertTrue(any(isinstance(n,ast.ImportFrom) and any(x.name=='MemoryMeter' for x in n.names) for n in attempt.body))
        src=Path(r.__file__).read_text(encoding='utf-8')
        self.assertIn('maximum=16',src);self.assertIn('num_warmup_steps=1,num_training_steps=16',src)

if __name__=='__main__':unittest.main()
