"""Reuse the 29 accepted D059 contracts against isolated v2 paths, plus repair constraints."""
import copy,sys,unittest
from pathlib import Path
from datetime import datetime,timedelta,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import stage_i_d059_v2 as d
import test_stage_i_d059 as legacy
legacy.d=d  # Test-module binding only; never alters any runner/helper source.


class D059V2Contracts(legacy.D059Tests):
    def test_deadline_expiry(self):
        start=datetime.fromisoformat('2026-10-18T15:10:48.941817+00:00')
        a={'execution_authorized_at_utc':start.isoformat(),'improvement_deadline_utc':(start+timedelta(days=7)).isoformat()}
        self.assertEqual(d.check_deadline(a,start+timedelta(days=1)),start+timedelta(days=7))
        self.assertRaises(ValueError,d.check_deadline,a,start+timedelta(days=7))
    def test_runner_has_only_import_repair_and_versioned_binding(self):
        original=(ROOT/'scripts/run_stage_i_d059_v1.py').read_text(encoding='utf-8')
        expected=original.replace('import argparse,copy,os,time,traceback','import argparse,copy,json,os,time,traceback',1).replace('from stage_i_d059 import *','from stage_i_d059_v2 import *',1)
        self.assertEqual(d.RUNNER.read_text(encoding='utf-8'),expected)
    def test_published_exact_diffs_match_actual_files(self):
        import difflib
        pairs=[('RUNNER_EXACT_DIFF.patch','scripts/run_stage_i_d059_v1.py','scripts/run_stage_i_d059_v2.py'),
            ('HELPER_BINDING_AND_DEADLINE_DIFF.patch','scripts/stage_i_d059.py','scripts/stage_i_d059_v2.py'),
            ('CONFIG_EXACT_DIFF.patch','configs/stage_i_d059_expanded_lora_v1.json','configs/stage_i_d059_expanded_lora_v2.json')]
        for name,old,new in pairs:
            expected=''.join(difflib.unified_diff((ROOT/old).read_text(encoding='utf-8').splitlines(True),
                (ROOT/new).read_text(encoding='utf-8').splitlines(True),fromfile='v1',tofile='v2'))
            self.assertEqual((d.PREP/name).read_text(encoding='utf-8').rstrip('\n'),expected.rstrip('\n'))
    def test_exact_scientific_equivalence(self):
        old=d.read(ROOT/'configs/stage_i_d059_expanded_lora_v1.json')
        self.assertEqual({k:v for k,v in self.c.items() if k not in ('run_root','technical_restart')},{k:v for k,v in old.items() if k!='run_root'})
        self.assertNotEqual(self.c['run_root'],old['run_root'])
    def test_obsolete_gpu_authorization_denied(self):
        self.assertRaises(ValueError,d.gate,ROOT/'outputs/stage_i/d059_execution_v1/RESEARCHER_EXECUTION_AUTHORIZATION.json','feasibility')
    def test_old_gate_is_historical_not_v2_pass(self):
        old=d.read(ROOT/'outputs/stage_i/d059_expanded_filtered_v1/feasibility/REPORT.json')
        self.assertEqual(old['status'],'PASS');self.assertNotEqual(old['runner_sha256'],d.sha(d.RUNNER))
        self.assertRaises(FileNotFoundError,d.require_feasibility,self.c,{})
    def test_seven_day_clock_reset_denied(self):
        start=datetime.fromisoformat('2026-10-18T15:10:48.941817+00:00')+timedelta(hours=1)
        a={'execution_authorized_at_utc':start.isoformat(),'improvement_deadline_utc':(start+timedelta(days=7)).isoformat()}
        self.assertRaises(ValueError,d.check_deadline,a,start+timedelta(days=1))
    def test_all_original_v1_evidence_unchanged(self):
        m=d.read(d.PREP/'ORIGINAL_V1_ARTIFACT_SNAPSHOT.json')
        for rel,h in m['files'].items():self.assertEqual(d.sha(ROOT/rel),h,rel)


if __name__=='__main__':unittest.main(verbosity=2)
