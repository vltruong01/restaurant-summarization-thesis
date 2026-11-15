"""Offline retry/resume/integrity checks; only temporary fixture files are written."""
import unittest
import tempfile
from datetime import datetime,timezone
from pathlib import Path
from unittest.mock import patch
import continue_stage_g_production6_retry_v1 as runner
import stage_g_production_d042 as frozen


class RetryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.directory=Path(self.temp.name)
        self.jid='offline_fixture';self.epoch=1000000
    def tearDown(self):self.temp.cleanup()
    def attempt(self,number,status='FAILED_STOP_NO_AUTOMATIC_RETRY',**extra):
        directory=self.directory/f'attempt_{number:02d}'
        cp={'status':status,'failed_utc':datetime.fromtimestamp(self.epoch,timezone.utc).isoformat(),'error_type':'ValueError',**extra}
        frozen.immutable(directory/'checkpoints'/(self.jid+'.json'),cp)
        return {'directory':directory,'config':runner.CONFIG,'historical':number==1}
    def test_fresh_job_ready(self):self.assertEqual(runner.retry_decision(self.jid,[],self.epoch),('ready',0))
    def test_first_retry_waits_thirty_seconds(self):
        attempts=[self.attempt(1)]
        self.assertEqual(runner.retry_decision(self.jid,attempts,self.epoch+10),('cooldown',20))
        self.assertEqual(runner.retry_decision(self.jid,attempts,self.epoch+30),('ready',0))
    def test_second_retry_waits_sixty_seconds(self):
        attempts=[self.attempt(1),self.attempt(2)]
        self.assertEqual(runner.retry_decision(self.jid,attempts,self.epoch+30),('cooldown',30))
        self.assertEqual(runner.retry_decision(self.jid,attempts,self.epoch+60),('ready',0))
    def test_retry_limit_survives_resume(self):
        attempts=[self.attempt(i) for i in (1,2,3)]
        self.assertEqual(runner.retry_decision(self.jid,attempts,self.epoch+10000),('exhausted',0))
    def test_success_is_never_retried(self):
        self.assertEqual(runner.retry_decision(self.jid,[self.attempt(1,runner.PASS)],self.epoch),('accepted',0))
    def test_unknown_intent_not_resent(self):
        directory=self.directory/'unknown'
        frozen.immutable(directory/'request_intents'/(self.jid+'.json'),{'status':'REQUEST_STARTED'})
        self.assertEqual(runner.retry_decision(self.jid,[{'directory':directory}],self.epoch),('unresolved',0))
    def test_auth_or_alias_failure_never_retried(self):
        for i,extra in enumerate(({'http_status':401},{'http_status':402},{'http_status':403},
            {'local_validation_error':'Reported model differs from requested alias; review provider provenance'}),1):
            self.assertEqual(runner.retry_decision(self.jid,[self.attempt(i,**extra)],self.epoch+1000),('fatal',0))
    def test_failed_attempt_bytes_preserved_and_pass_selected(self):
        failed=self.attempt(1);p=failed['directory']/'checkpoints'/(self.jid+'.json');original=p.read_bytes()
        passed=self.attempt(2,runner.PASS)
        with patch.object(runner,'read_index',return_value={}),patch.dict(runner.ATTEMPTS,{self.jid:[failed,passed]},clear=True):
            self.assertEqual(runner.location({'job_id':self.jid},{}),passed['directory'])
        self.assertEqual(p.read_bytes(),original)
    def test_frozen_scientific_config_unchanged(self):
        previous=frozen.read(runner.parent.CONFIG);current=frozen.read(runner.CONFIG)
        for k in frozen.FIXED_KEYS:self.assertEqual(current[k],previous[k],k)
        self.assertEqual(current['identifier_guard'],previous['identifier_guard'])
    def test_attempt_scan_resume_counts_failure_and_success(self):
        a=self.attempt(1);b=self.attempt(2,runner.PASS)
        attempts_out=self.directory/'execution'
        frozen.immutable(self.directory/'historical_failed_tasks.json',{})
        for directory in (a['directory'],b['directory']):
            destination=attempts_out/'attempts'/directory.name/'checkpoints'/(self.jid+'.json')
            destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_bytes((directory/'checkpoints'/(self.jid+'.json')).read_bytes())
        with patch.object(runner,'RUN',self.directory),patch.object(runner,'OUT',attempts_out),patch.dict(runner.ATTEMPTS,{},clear=True):
            runner.load_attempts([{'job_id':self.jid}])
            self.assertEqual(len(runner.ATTEMPTS[self.jid]),2)
            self.assertEqual(runner.retry_decision(self.jid,runner.ATTEMPTS[self.jid],self.epoch),('accepted',0))


if __name__=='__main__':unittest.main()
