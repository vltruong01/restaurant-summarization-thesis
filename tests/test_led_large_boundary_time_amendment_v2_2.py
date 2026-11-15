"""CPU-only time-amendment regressions; no device or model operations."""
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime, timezone, timedelta
from unittest.mock import patch
import unittest, json, tempfile, shutil, sys, copy, ast

R=Path(__file__).resolve().parents[1]
I=R/'outputs/stage_i/led_large_boundary_dual_v2_2_time_amendment/implementation'
sys.path[:0]=[str(I),str(R/'scripts')]
import run_led_large_boundary_dual_time_amendment_v2_2 as O
import led_large_boundary_dual_contract_v2 as C

class TimeAmendment(unittest.TestCase):
    def setUp(self):
        self.package=R/'outputs/cloud_handoff/LED_LARGE_BOUNDARY_DUAL_V2_1_CPU_CORRECTION_REVIEW'
        self.suite=R/'outputs/stage_i/led_large_boundary_dual_v2_1_gpu_qualification_v1/downloaded/suite'
        self.args=SimpleNamespace(package=self.package,suite=self.suite,phase='train')
        self.a=C.read(self.suite/'GATE_AUTHORIZATION.json')
        self.a.update(scopes=['train'],training_authorized=True,early_sanity_generation_approved=True,
                      validation_payload_access=True,decoding_authorized=False,
                      accepted_gate_approval_sha256=C.sha(self.suite/'GATE_AUTHORIZATION.json'),
                      time_budget_amendment='BOUNDARY_DUAL_V2_2_COST_CAP_REMOVAL_ONLY',
                      effective_cost_caps={'pair_seconds':None,'train_seconds':None},
                      operational_amendment_binding=O.amendment_binding())
        self.gate=C.read(self.suite/'GATE_AUTHORIZATION.json')
        self.hard=datetime.fromisoformat(self.a['researcher_absolute_stop_utc']).timestamp()

    def test_withdrawn_cost_caps_do_not_end_valid_training(self):
        old=C.read(self.suite/'SUITE_START.json')
        after_old_cap=old['pair_deadline']+1
        self.assertEqual(O.validate_time_amendment(self.args,self.a,now=after_old_cap),self.hard)
        self.assertGreater(self.hard-after_old_cap,21600)

    def test_confirmed_absolute_stop_still_fails_closed(self):
        for now in [self.hard,self.hard+1]:
            with self.subTest(now=now),self.assertRaisesRegex(ValueError,'hard stop expired'):
                O.amended_deadline(self.a,self.gate,now=now)

    def test_earlier_gate_expiry_is_never_renewed(self):
        gate=copy.deepcopy(self.gate)
        gate['execution_deadline_utc']='2026-10-25T12:00:00+00:00'
        expected=datetime.fromisoformat(gate['execution_deadline_utc']).timestamp()
        self.assertEqual(O.amended_deadline(self.a,gate,now=expected-1),expected)
        with self.assertRaises(ValueError):O.amended_deadline(self.a,gate,now=expected)

    def test_incomplete_or_modified_amendment_is_rejected(self):
        for field,value in [('effective_cost_caps',{}),('effective_cost_caps',{'pair_seconds':None,'train_seconds':36000}),
                            ('time_budget_amendment','UNAPPROVED'),('operational_amendment_binding',{}),
                            ('execution_deadline_utc','2026-10-28T08:27:23+00:00')]:
            with self.subTest(field=field,value=value):
                a=copy.deepcopy(self.a);a[field]=value
                with self.assertRaises(ValueError):O.validate_time_amendment(self.args,a)

    def test_gate_rerun_and_final_decoding_remain_forbidden(self):
        for phase in ['gates','decode']:
            args=copy.copy(self.args);args.phase=phase
            with self.subTest(phase=phase),self.assertRaisesRegex(ValueError,'TRAIN ONLY'):
                O.validate_time_amendment(args,self.a)
        a=copy.deepcopy(self.a);a['decoding_authorized']=True
        with self.assertRaises(ValueError):O.validate_time_amendment(self.args,a)

    def test_original_48_hour_contract_and_training_scope_still_apply(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'SYNTHETIC_CPU_APPROVAL.json'
            a=copy.deepcopy(self.a)
            a['execution_authorized_at_utc']='2026-10-25T11:42:38+00:00'
            now=datetime.fromisoformat('2026-10-25T12:00:00+00:00')
            p.write_text(json.dumps(a));C.authorize(p,'train',self.package,now=now)
            a['execution_authorized_at_utc']='2026-10-24T08:00:00+00:00'
            p.write_text(json.dumps(a))
            with self.assertRaises(ValueError):C.authorize(p,'train',self.package,now=now)
            a=copy.deepcopy(self.a);a['validation_payload_access']=False;p.write_text(json.dumps(a))
            with self.assertRaises(ValueError):C.authorize(p,'train',self.package,now=now)

    def test_original_suite_and_gate_evidence_are_immutable(self):
        with tempfile.TemporaryDirectory() as d:
            suite=Path(d)
            for name in ['GATE_AUTHORIZATION.json','SUITE_START.json','PAIR_GATES_COMPLETE.json']:
                shutil.copyfile(self.suite/name,suite/name)
            (suite/'SUITE_START.json').write_bytes((suite/'SUITE_START.json').read_bytes()+b'\n')
            args=copy.copy(self.args);args.suite=suite
            with self.assertRaisesRegex(ValueError,'Original suite record changed'):
                O.validate_time_amendment(args,self.a)

    def test_tampered_operational_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for p in I.iterdir():
                if p.is_file():shutil.copyfile(p,root/p.name)
            p=root/'AMENDMENT_SPEC.json';p.write_bytes(p.read_bytes()+b'\n')
            with patch.object(O,'AMENDMENT_ROOT',root),self.assertRaisesRegex(ValueError,'Integrity mismatch'):
                O.verify_amendment_files()

    def test_original_scientific_code_and_gate_comparator_unchanged(self):
        C.verify_manifest(self.package,C.read(self.package/'outputs/stage_i/led_large_boundary_dual_v2/PREFLIGHT_MANIFEST.json')['files'])
        old=ast.parse((self.package/'scripts/run_led_large_boundary_dual_v2.py').read_text())
        new=ast.parse(Path(O.__file__).read_text())
        pick=lambda tree:next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='gate_pair')
        self.assertEqual(ast.dump(pick(old)),ast.dump(pick(new)))
        self.assertIn("C.joint_training(suite,bindings)",Path(O.__file__).read_text())

if __name__=='__main__':unittest.main(verbosity=2)
