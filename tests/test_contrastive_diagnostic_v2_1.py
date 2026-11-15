"""CPU-only V2.1 comparator, preservation and diagnostic-only scope regressions."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
from tempfile import TemporaryDirectory
from unittest.mock import patch
import ast
import copy
import json
import sys
import types
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import led_large_contrastive_diagnostic_evidence_v2_1 as evidence
import run_led_large_exploratory_contrastive_diagnostic_v2_1 as runner


class ComparatorTests(unittest.TestCase):
    def comparison(self,a,b,loss_a=1.,loss_b=1.,preconditions=None):
        names=list(a)
        native=dict(parameter_names=names,gradients=a,loss=torch.tensor(loss_a))
        delegated=dict(parameter_names=list(b),gradients=b,loss=torch.tensor(loss_b))
        checks={k:True for k in evidence.REQUIRED_LAMBDA0_PRECONDITIONS} if preconditions is None else preconditions
        with TemporaryDirectory(prefix='cf_v21_test_') as folder:
            result=evidence.lambda0_comparison(folder,native,delegated,{}, {},checks)
            return json.loads((Path(folder)/'LAMBDA0_COMPARISON.json').read_text())
    def test_bit_exact_preserved(self):
        r=self.comparison({'x':torch.tensor([1.,0.])},{'x':torch.tensor([1.,0.])})
        self.assertTrue(r['lambda0_bit_exact']);self.assertTrue(r['lambda0_within_tolerance'])
    def test_non_bit_exact_within_declared_tolerance_passes(self):
        r=self.comparison({'x':torch.tensor([1.,0.])},{'x':torch.tensor([1.+5e-5,5e-6])},loss_b=1.+5e-5)
        self.assertEqual(r['status'],'PASS_WITHIN_TOLERANCE')
        self.assertFalse(r['lambda0_bit_exact']);self.assertNotIn('lambda0_exact',r)
        self.assertEqual(r['per_tensor'][0]['failing_element_count'],0)
    def test_outside_tolerance_has_exact_failure_count(self):
        r=self.comparison({'x':torch.tensor([1.,0.,3.])},{'x':torch.tensor([1.01,2e-5,3.])})
        self.assertEqual(r['status'],'FAIL_STOP');self.assertEqual(r['per_tensor'][0]['failing_element_count'],2)
        self.assertGreater(r['per_tensor'][0]['max_absolute_difference'],.009)
    def test_nonfinite_native_fails_even_matching_inf(self):
        r=self.comparison({'x':torch.tensor([float('inf'),float('nan')])},{'x':torch.tensor([float('inf'),float('nan')])})
        self.assertFalse(r['lambda0_within_tolerance']);self.assertFalse(r['per_tensor'][0]['native_finite'])
        self.assertEqual(r['per_tensor'][0]['failing_element_count'],2)
    def test_nonfinite_delegated_and_loss_fail(self):
        r=self.comparison({'x':torch.tensor([1.])},{'x':torch.tensor([float('inf')])})
        self.assertFalse(r['per_tensor'][0]['delegated_finite']);self.assertEqual(r['status'],'FAIL_STOP')
        r=self.comparison({'x':torch.tensor([1.])},{'x':torch.tensor([1.])},loss_b=float('nan'))
        self.assertFalse(r['loss']['allclose']);self.assertFalse(r['lambda0_within_tolerance'])
    def test_zero_reference_relative_error_is_transparent_json(self):
        r=evidence.compare_tensor(torch.tensor([0.]),torch.tensor([5e-6]))
        self.assertTrue(r['allclose']);self.assertEqual(r['max_relative_difference'],'inf')
        json.dumps(r,allow_nan=False)
        self.assertEqual(evidence.compare_tensor(torch.tensor([0.]),torch.tensor([0.]))['max_relative_difference'],0.)
    def test_reference_orientation_and_metrics(self):
        a=torch.tensor([1.,2.,4.],dtype=torch.float64);b=torch.tensor([1.00001,2.00002,4.00004],dtype=torch.float64)
        r=evidence.compare_tensor(a,b)
        self.assertAlmostEqual(r['max_absolute_difference'],.00004,places=12)
        self.assertAlmostEqual(r['max_relative_difference'],.00001,places=12)
        self.assertEqual(r['reference'],'native');self.assertTrue(r['allclose'])
    def test_shape_dtype_or_inventory_mismatch_fails(self):
        for b in [torch.zeros(3),torch.zeros(2,dtype=torch.float64)]:
            self.assertFalse(evidence.compare_tensor(torch.zeros(2),b)['allclose'])
        r=self.comparison({'a':torch.ones(1)},{'b':torch.ones(1)})
        self.assertFalse(r['parameter_inventory_equal']);self.assertEqual(r['status'],'FAIL_STOP')
    def test_missing_or_false_rng_batch_mode_checks_fail(self):
        for field in evidence.REQUIRED_LAMBDA0_PRECONDITIONS:
            checks={k:True for k in evidence.REQUIRED_LAMBDA0_PRECONDITIONS};checks[field]=False
            r=self.comparison({'x':torch.ones(1)},{'x':torch.ones(1)},preconditions=checks)
            self.assertEqual(r['status'],'FAIL_STOP',field)
        r=self.comparison({'x':torch.ones(1)},{'x':torch.ones(1)},preconditions={'one':True})
        self.assertFalse(r['preconditions_pass'])
    def test_native_snapshot_survives_delegated_failure_and_never_overwrites(self):
        with TemporaryDirectory(prefix='cf_v21_test_') as folder:
            data,receipt=evidence.save_path_snapshot(folder,'native',['p'],[torch.tensor([1.,2.])],torch.tensor(1.),dict(RNG_start={'cpu':'signature'}))
            p=Path(folder)/'native_gradients.pt'
            loaded=torch.load(p,map_location='cpu',weights_only=True)
            self.assertTrue(torch.equal(loaded['gradients']['p'],data['gradients']['p']))
            self.assertEqual(evidence.sha(p),receipt['sha256'])
            self.assertFalse((Path(folder)/'delegated_gradients.pt').exists())
            with self.assertRaises(FileExistsError):evidence.save_path_snapshot(folder,'native',['p'],[torch.ones(2)],torch.tensor(1.),{})
    def test_both_gradient_files_and_contexts_are_bound(self):
        with TemporaryDirectory(prefix='cf_v21_test_') as folder:
            for role in ['native','delegated']:
                _,receipt=evidence.save_path_snapshot(folder,role,['p'],[torch.ones(2)],torch.tensor(1.),dict(RNG_start={'cpu':'same'},RNG_end={'cpu':'next'}))
                self.assertEqual(evidence.sha(Path(folder)/receipt['context_path']),receipt['context_sha256'])
    def test_batch_hash_detects_change_without_touching_rng(self):
        before=torch.get_rng_state().clone();x=dict(labels=torch.tensor([[4,2,-100]]))
        a=evidence.batch_signature(x);x['labels'][0,0]=5;b=evidence.batch_signature(x)
        self.assertNotEqual(a,b);self.assertTrue(torch.equal(before,torch.get_rng_state()))
    def test_model_mode_dropout_and_checkpoint_signature(self):
        from functools import partial
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.dropout=torch.nn.Dropout(.1)
                self.config=types.SimpleNamespace(dropout=.1,use_cache=False)
                self.gradient_checkpointing=True;self._gradient_checkpointing_func=partial(str,base=10)
        m=Model();a=evidence.model_context(m);m.eval();b=evidence.model_context(m)
        self.assertNotEqual(a,b);m.train();m.dropout.p=.2
        self.assertNotEqual(a,evidence.model_context(m))
    def test_checkpoint_signature_is_not_a_mutable_alias(self):
        from functools import partial
        model=torch.nn.Linear(1,1)
        model.gradient_checkpointing=True
        model._gradient_checkpointing_func=partial(str,base=10)
        before=evidence.model_context(model)
        model._gradient_checkpointing_func.keywords['base']=16
        self.assertEqual(before['checkpoint_modules']['']['keywords']['base'],10)
        self.assertNotEqual(before,evidence.model_context(model))


class ReceiptTests(unittest.TestCase):
    def fixture(self):
        return dict(status='PASS_WITHIN_TOLERANCE',lambda0_within_tolerance=True,lambda0_bit_exact=False,
            thresholds=dict(atol=1e-5,rtol=1e-4),preconditions_pass=True,gradient_tensors=456,expected_gradient_tensors=456)
    def test_receipt_never_labels_tolerance_success_as_exact(self):
        r=evidence.diagnostic_receipt(self.fixture(),{k:True for k in evidence.REQUIRED_GATES},dict(headroom_fraction=.15))
        self.assertTrue(r['lambda0_within_tolerance']);self.assertFalse(r['lambda0_bit_exact'])
        self.assertNotIn('lambda0_exact',r);self.assertFalse(r['automatic_continuation']);self.assertFalse(r['training_authorized'])
    def test_any_missing_or_failed_gate_blocks_completion(self):
        for k in evidence.REQUIRED_GATES:
            gates={n:True for n in evidence.REQUIRED_GATES};del gates[k]
            with self.assertRaises(ValueError):evidence.diagnostic_receipt(self.fixture(),gates,dict(headroom_fraction=.2))
            gates[k]=False
            with self.assertRaises(ValueError):evidence.diagnostic_receipt(self.fixture(),gates,dict(headroom_fraction=.2))
    def test_low_nan_or_missing_headroom_blocks_completion(self):
        for value in [.149,float('nan'),None]:
            with self.assertRaises(ValueError):evidence.diagnostic_receipt(self.fixture(),{k:True for k in evidence.REQUIRED_GATES},dict(headroom_fraction=value))
    def test_wrong_gradient_inventory_blocks_completion(self):
        r=self.fixture();r['gradient_tensors']=455
        with self.assertRaises(ValueError):evidence.diagnostic_receipt(r,{k:True for k in evidence.REQUIRED_GATES},dict(headroom_fraction=.2))


class AuthorizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c=runner.read(runner.CONFIG)
        cls.pending=runner.read(runner.CONFIG.parent/'DIAGNOSTIC_AUTHORIZATION_PENDING.json')
    def authorized_fixture(self):
        a=copy.deepcopy(self.pending);now=datetime.now(timezone.utc)
        a.update(GPU_execution_approved=True,approval_source='researcher_message',researcher_explicit_authorization='TEST FIXTURE ONLY',
            authorized_utc=now.isoformat(),deadline_utc=(now+timedelta(hours=1)).isoformat())
        return a
    def args(self,**kw):return types.SimpleNamespace(**dict(dict(mode='diagnose',execute=True,approval=Path('fixture')),**kw))
    def test_offline_preflight_no_cuda_queries(self):
        with patch.object(torch.cuda,'is_available',side_effect=AssertionError('CUDA query forbidden')):
            c=runner.preflight()
        self.assertEqual(c['scopes'],['diagnose']);self.assertEqual(c['pair_ids'],[f'P{i:02d}' for i in range(1,17) if i!=3])
    def test_pending_template_cannot_authorize(self):
        with patch.object(runner,'read',return_value=self.pending):
            with self.assertRaisesRegex(ValueError,'approval absent'):runner.authorization(self.args(),self.c)
    def test_old_failed_v2_approval_rejected(self):
        old=runner.read(runner.CONFIG.parents[1]/'led_large_exploratory_contrastive_v2/RESEARCHER_AUTHORIZATION.json')
        with patch.object(runner,'read',return_value=old):
            with self.assertRaisesRegex(ValueError,'Foreign/failed'):runner.authorization(self.args(),self.c)
    def test_new_contract_synthetic_fixture_only_passes(self):
        a=self.authorized_fixture()
        with patch.object(runner,'read',return_value=a):self.assertEqual(runner.authorization(self.args(),self.c),a)
    def test_train_baseline_scopes_or_budget_cannot_be_added(self):
        for update in [dict(scopes=['diagnose','pilot']),dict(training_authorized=True),dict(baseline_inference_authorized=True),dict(automatic_continuation=True),dict(successful_optimizer_updates_max=1),dict(generation_calls_max=1)]:
            a=self.authorized_fixture();a.update(update)
            with patch.object(runner,'read',return_value=a):
                with self.assertRaises(ValueError):runner.authorization(self.args(),self.c)
    def test_fresh_namespace_deadline_and_bindings_required(self):
        for update in [dict(output_namespace='runs/CF_EXPLORATORY_V2/diagnose_pair'),dict(authorized_utc=None),dict(deadline_utc=None),dict(helper_sha256='wrong'),dict(deadline_utc=(datetime.now(timezone.utc)-timedelta(hours=1)).isoformat())]:
            a=self.authorized_fixture();a.update(update)
            with patch.object(runner,'read',return_value=a):
                with self.assertRaises(ValueError):runner.authorization(self.args(),self.c)
    def test_no_execute_blocks_file_io(self):
        with patch.object(runner,'read',side_effect=AssertionError('Premature file I/O')):
            with self.assertRaises(ValueError):runner.authorization(self.args(execute=False),self.c)
    def test_no_training_or_generation_callable_in_new_runner(self):
        tree=ast.parse(Path(runner.__file__).read_text(encoding='utf-8'))
        names={n.name for n in tree.body if isinstance(n,ast.FunctionDef)}
        self.assertFalse({'train','generate_probes'}&names)
        self.assertNotIn('lambda0_exact=True',Path(runner.__file__).read_text())
        old=ast.parse((Path(runner.__file__).parent/'run_led_large_exploratory_contrastive_v2.py').read_text(encoding='utf-8'))
        original={n.name:n for n in old.body if isinstance(n,ast.FunctionDef)}
        for n in tree.body:
            if isinstance(n,ast.FunctionDef) and n.name in ('collated','load','hardware'):
                self.assertEqual(ast.dump(n),ast.dump(original[n.name]))
    def test_completion_after_memory_context_exit_and_evidence_save(self):
        tree=ast.parse(Path(runner.__file__).read_text(encoding='utf-8'))
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        attempt=next(n for n in fn.body if isinstance(n,ast.Try))
        context=next(n for n in attempt.body if isinstance(n,ast.With))
        self.assertNotIn("COMPLETION.json",ast.unparse(context))
        statements=[ast.unparse(n) for n in attempt.body]
        memory=next(i for i,s in enumerate(statements) if s.startswith('write(') and 'MEMORY_EVIDENCE.json' in s)
        complete=next(i for i,s in enumerate(statements) if s.startswith('write(') and 'COMPLETION.json' in s)
        self.assertLess(memory,complete)
    def test_unchanged_joint_accumulation_and_amp_gates(self):
        new=Path(runner.__file__).read_text(encoding='utf-8')
        old=(Path(runner.__file__).parent/'run_led_large_exploratory_contrastive_v2.py').read_text(encoding='utf-8')
        # The complete block after lambda0 through final memory/check guards stays byte-equivalent.
        start='    check();memory.check();initial=seq.capture_rng(device)'
        end="    event('ALL_MANDATORY_DIAGNOSTICS_PASS'"
        self.assertEqual(new[new.index(start):new.index(end)],old[old.index(start):old.index(end)])

if __name__=='__main__':unittest.main()
