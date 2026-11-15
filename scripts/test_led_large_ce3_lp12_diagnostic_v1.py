"""Focused CPU contract tests; ownership guard body extracted unchanged from frozen code."""
import ast
import csv
import io
import os
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch
import run_led_large_ce3_lp12_diagnostic_v1 as d


class Contracts(unittest.TestCase):
    def test_only_length_penalty_changes(self):
        original=dict(num_beams=4,length_penalty=1.0,max_length=768,do_sample=False,num_return_sequences=1,use_cache=True)
        new=d.generation_b(original)
        self.assertEqual([k for k in original if original[k]!=new[k]],['length_penalty'])
        self.assertEqual(new['length_penalty'],1.2);self.assertEqual(original['length_penalty'],1.0)

    def test_wrong_decoding_rejected(self):
        original=dict(num_beams=4,length_penalty=1.0,max_length=768,do_sample=False,num_return_sequences=1)
        for k,v in [('num_beams',8),('length_penalty',0.9),('max_length',512),('do_sample',True),('num_return_sequences',4)]:
            with self.assertRaises(ValueError):d.generation_b(dict(original,**{k:v}))

    def guard(self):
        path=Path(__file__).with_name('run_led_large_boundary_global_feasibility_v1.py')
        tree=ast.parse(path.read_text(encoding='utf-8'))
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='gpu_identity')
        env=dict(subprocess=subprocess,csv=csv,io=io,os=os,require=d.require)
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),env)
        return env['gpu_identity']

    def output(self,pids):
        return [subprocess.CompletedProcess([],0,'GPU-abc, NVIDIA GeForce RTX 3090, 0000:01:00.0, 24576, 595.91.07\n',''),
                subprocess.CompletedProcess([],0,pids,'')]

    def test_idle_ownership_passes(self):
        with patch.object(subprocess,'run',side_effect=self.output('')):
            self.assertEqual(self.guard()()['name'],'NVIDIA GeForce RTX 3090')

    def test_self_pid_ownership_passes(self):
        with patch.object(subprocess,'run',side_effect=self.output(str(os.getpid()))):self.guard()()

    def test_foreign_or_namespace_mismatched_pid_stops(self):
        with patch.object(subprocess,'run',side_effect=self.output(str(os.getpid()+9999))):
            with self.assertRaisesRegex(ValueError,'Foreign GPU compute workload'):self.guard()()


if __name__=='__main__':unittest.main(verbosity=2)
