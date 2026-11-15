"""Credential setup tests in synthetic temporary workspaces only."""
import contextlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from provision_stage_l_accounts_v1 import provision,user_rows,memorable_passphrase
from stage_l_reference_app_v1 import Server
from test_stage_l_reference_workflow_v1 import fixture,draft_for

class Provisioning(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.store=fixture(Path(self.temp.name)/'SYNTHETIC_ONLY');self.output=[]
    def tearDown(self):self.temp.cleanup()
    def test_new_distinct_memorable_passwords_use_existing_hashing(self):
        phrases=iter(['Blue-River-Maple-Garden-Bread-Moon','Gold-Forest-Apple-Willow-Lake-Cloud'])
        result=provision(self.store,self.output.append,phrase_factory=lambda:next(phrases))
        self.assertEqual(result['status'],'PASS_ACCOUNT_PROVISIONING')
        rows=user_rows(self.store);self.assertEqual(set(rows),{'author01','checker01'})
        self.assertTrue(all(len(r['salt'])==32 and len(r['password'])==64 for r in rows.values()))
        self.assertTrue(rows['author01']['salt']!=rows['checker01']['salt'])
        self.assertTrue('PASSWORD:' in '\n'.join(self.output))
        self.assertTrue('Blue-River-Maple-Garden-Bread-Moon' not in json.dumps(result))
        self.assertTrue('Gold-Forest-Apple-Willow-Lake-Cloud' not in json.dumps(rows))
        self.assertEqual(self.store.audit()['counts']['NOT_STARTED'],1)
    def test_idempotent_preserves_both_credentials_and_metadata(self):
        provision(self.store,self.output.append);before=user_rows(self.store)
        result=provision(self.store,self.output.append,phrase_factory=lambda:(_ for _ in ()).throw(AssertionError('Must not generate a replacement password')))
        self.assertTrue(user_rows(self.store)==before)
        self.assertTrue(all(x['result']=='PRESERVED_EXISTING_ACCOUNT' for x in result['accounts']))
    def test_one_existing_account_is_not_reset_or_duplicated(self):
        phrase='Existing-River-Maple-Garden-Bread-Moon';self.store.add_user('author01','author',phrase);before=user_rows(self.store)['author01']
        phrases=iter([phrase,'Different-Forest-Apple-Willow-Lake-Cloud'])
        result=provision(self.store,self.output.append,phrase_factory=lambda:next(phrases))
        self.assertTrue(user_rows(self.store)['author01']==before)
        self.assertEqual(self.store.login('author01',phrase)['role'],'author')
        self.assertEqual([x['result'] for x in result['accounts']],['PRESERVED_EXISTING_ACCOUNT','CREATED'])
    def test_wrong_existing_role_stops_without_creating_or_overwriting(self):
        self.store.add_user('author01','checker','Existing-Demo-Password');before=user_rows(self.store)
        with self.assertRaises(ValueError):provision(self.store,self.output.append)
        self.assertTrue(user_rows(self.store)==before)
    def test_existing_annotation_and_history_are_unchanged(self):
        b=next(iter(self.store.frame));self.store.apply(b,{'id':'synthetic_writer','role':'author'},'save',{'expected_version':0,'draft':draft_for(self.store,b)})
        before=self.store.get(b);audit=self.store.audit();provision(self.store,self.output.append)
        self.assertTrue(self.store.get(b)==before and self.store.audit()==audit)
    def test_live_http_login_roles_and_permissions_on_demo_only(self):
        server=Server(('127.0.0.1',0),self.store,Path(__file__).parent/'stage_l_reference_web_v1');thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            result=provision(self.store,self.output.append,verify_url=f'http://127.0.0.1:{server.server_port}')
            self.assertEqual([x['http_session_role'] for x in result['accounts']],['author','checker'])
            self.assertTrue(all(x['source_cases_accessible']==1 for x in result['accounts']))
            b=next(iter(self.store.frame));draft=draft_for(self.store,b)
            with self.assertRaises(ValueError):self.store.apply(b,{'id':'checker01','role':'checker'},'save',{'expected_version':0,'draft':draft})
            with self.assertRaises(ValueError):self.store.apply(b,{'id':'author01','role':'author'},'approve',{'expected_version':0})
            self.store.apply(b,{'id':'author01','role':'author'},'save',{'expected_version':0,'draft':draft})
            self.assertEqual(self.store.get(b)[0]['author']['human_id'],'author01')
        finally:server.shutdown();server.server_close();thread.join()
    def test_phrase_generator(self):
        phrases=[memorable_passphrase() for _ in range(20)]
        self.assertTrue(all(len(p)>=12 and len(p.split('-'))==6 and all(w.isalpha() for w in p.split('-')) for p in phrases))
    def test_cli_refuses_nonterminal_before_provisioning(self):
        from provision_stage_l_accounts_v1 import main
        with patch('sys.argv',['provision_stage_l_accounts_v1.py','--workspace',str(self.store.root)]),patch('sys.stdout.isatty',return_value=False):
            with self.assertRaises(ValueError):main()
        self.assertFalse(user_rows(self.store))

if __name__=='__main__':unittest.main(verbosity=2)
