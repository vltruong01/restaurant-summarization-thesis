"""Offline scheduler tests, confined to temporary files; zero teacher API calls."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import run_stage_g_production6_deferred_v1 as runner
import continue_stage_g_production6_deferred_v1 as adapter
import stage_g_production_d042 as frozen


class DeferredTests(unittest.TestCase):
    def test_failed_and_unresolved_never_resent(self):
        jobs=[{'job_id':str(i)} for i in range(5)]
        pending=runner.pending_jobs(jobs,{'0'},{'1':{}},[{'job_id':'2'},{'job_id':'3'}])
        self.assertEqual(pending,[jobs[4]])

    def test_other_conditions_of_failed_business_still_pending(self):
        jobs=[{'job_id':'review'},{'job_id':'unfiltered'},{'job_id':'filtered'}]
        self.assertEqual(runner.pending_jobs(jobs,set(),{},[{'job_id':'review'}]),jobs[1:])

    def test_science_is_frozen(self):
        old=frozen.read(adapter.parent.CONFIG);new=frozen.read(adapter.CONFIG)
        adapter.check_science(new,old)
        new['generation_proposal']['reasoning_effort']='low'
        with self.assertRaises(AssertionError):adapter.check_science(new,old)

    def test_next_batch_runs_after_failure_and_resume_skips_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'execution';out.mkdir()
            config=root/'config.json';schema=root/'schema.json';prompt=root/'prompt.txt'
            prompt.write_text('offline mock',encoding='utf-8')
            cfg={'schema':'schema.json','prompts':{'mock':'prompt.txt'},'requested_model':runner.MODEL}
            frozen.atomic(config,cfg);frozen.atomic(schema,{'mock':{}})
            frozen.atomic(root/'authorization.json',{'approved':True,'paid_calls_authorized':True,'config_sha256':frozen.file_sha(config)})
            frozen.atomic(root/'small_batch_manifest.json',{'business_ids':[]})
            frozen.atomic(root/'small_batch_validation.json',{'status':'prior smoke PASS'})
            jobs=[{'job_id':str(i),'business_id':str(i//3),'task':'mock','images':[]} for i in range(9)]
            businesses=[{'business_id':str(i)} for i in range(3)]
            calls=[]
            def result(j,*args):
                calls.append(j['job_id']);frozen.immutable(out/'request_intents'/(j['job_id']+'.json'),{})
                if j['job_id']=='0':
                    frozen.immutable(out/'checkpoints'/'0.json',{'status':'FAILED_STOP_NO_AUTOMATIC_RETRY','error_type':'ValueError','http_status':None})
                    raise RuntimeError('mock invalid evidence')
                cp={'status':'STRUCTURAL_PASS_AWAITING_HUMAN_REVIEW'}
                frozen.immutable(out/'checkpoints'/(j['job_id']+'.json'),cp);return cp
            def save(b,grouped,imports):
                paths=[out/'checkpoints'/(j['job_id']+'.json') for j in grouped[b['business_id']]]
                return all(p.exists() and frozen.read(p)['status']=='STRUCTURAL_PASS_AWAITING_HUMAN_REVIEW' for p in paths)
            with contextlib.ExitStack() as stack:
                bindings={'ROOT':root,'CONFIG':config,'FOLDER':root,'OUT':out,'STATE':None,
                    'runner_lock':lambda:None,'check_inputs':lambda:(cfg,jobs,businesses,{}),
                    'location':lambda j,i:out,'verify_checkpoint':lambda *a:None,
                    'save_completed_business':save,'job_result':result,'payload':lambda *a:{},
                    'update_docs':lambda state:frozen.atomic(out/'execution_progress.json',state),
                    'credentials':lambda:(runner.BASE_URL,runner.PROVIDER,'offline-test-only')}
                for name,value in bindings.items():stack.enter_context(patch.object(runner,name,value))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                runner.main()
                state=frozen.read(out/'execution_progress.json')
                self.assertEqual(state['status'],'GENERATION_SWEEP_FINISHED_PENDING_RECOVERY')
                self.assertEqual(state['accepted_tasks'],8)
                self.assertEqual(state['businesses_completed'],2)
                self.assertEqual(set(calls),{str(i) for i in range(9)})
                self.assertEqual(len(calls),9)
                self.assertEqual(len(list(frozen.rows(out/'batch_log.jsonl'))),2)
                self.assertFalse((out/'generation_completion.json').exists())
                runner.STATE=None
                runner.main()
                self.assertEqual(len(calls),9,'No failed/unresolved request automatically resent on resume')


if __name__=='__main__':unittest.main()
