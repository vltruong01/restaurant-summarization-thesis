"""Focused offline D058 contracts, no models, no GPU, no new output annotation."""
import ast,copy,json,pathlib,subprocess,sys,tempfile,unittest
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import stage_i_d058 as d

class D058Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.cfg=d.read(d.CONFIG)
    def test_config(self):self.assertTrue(d.check_config(self.cfg))
    def test_budget(self):
        b=d.budget(self.cfg);self.assertEqual((b['max_optimizer_steps'],b['warmup_steps'],b['generation_outputs']),(1250,63,48))
        self.assertEqual(b['quality_checkpoint_steps'],[750,1250]);self.assertEqual(b['max_new_token_slots'],36816)
    def test_sixth_epoch_denied(self):
        c=copy.deepcopy(self.cfg);c['optimization']['num_train_epochs']=6;self.assertRaises(ValueError,d.check_config,c)
    def test_decoder_change_denied(self):
        c=copy.deepcopy(self.cfg);c['generation']['no_repeat_ngram_size']=4;self.assertRaises(ValueError,d.check_config,c)
    def test_architecture_change_denied(self):
        c=copy.deepcopy(self.cfg);c['arm']='A_decoder_only';self.assertRaises(ValueError,d.check_config,c)
    def test_completed_pilot_resume_denied(self):
        c=copy.deepcopy(self.cfg);c['resume_short_pilot']=True;self.assertRaises(ValueError,d.check_config,c)
    def test_evaluation_scope_change_denied(self):
        c=copy.deepcopy(self.cfg);c['evaluation_epochs']=[2,3,5];self.assertRaises(ValueError,d.check_config,c)
    def test_loss_checkpoint_selection_denied(self):
        c=copy.deepcopy(self.cfg);c['load_best_model_at_end']=True;self.assertRaises(ValueError,d.check_config,c)
    def test_training_arguments(self):
        a=d.training_arguments(self.cfg,ROOT/'unused');self.assertFalse(a['load_best_model_at_end']);self.assertFalse(a['predict_with_generate'])
        self.assertEqual(a['num_train_epochs'],5);self.assertEqual(a['save_total_limit'],5);self.assertEqual(a['gradient_accumulation_steps'],8)
    def test_epoch_permutations(self):
        a=d.read(d.PREP/'EPOCH_ORDERS.json');self.assertEqual(len(a['training_business_ids']),2000)
        self.assertEqual(len(a['indices']),5);self.assertTrue(all(sorted(x)==list(range(2000)) for x in a['indices']))
    def test_fixed_cases(self):
        a=d.read(d.PREP/'EVALUATION_MANIFEST.json');self.assertEqual(len(a['random20']),20);self.assertEqual(len(a['targeted4']),4)
        self.assertFalse(set(a['random20'])&set(a['targeted4']));self.assertEqual(a['business_ids'],a['random20']+a['targeted4'])
    def test_dataset_readiness(self):
        a=d.read(d.PREP/'DATASET_AND_REFERENCE_READINESS.json');self.assertEqual(a['split_overlap'],0);self.assertTrue(a['reference_validation']['valid'])
        self.assertEqual(a['dataset_statistics']['train']['businesses'],2000);self.assertEqual(a['dataset_statistics']['validation']['businesses'],200)
    def test_false_approval_denied(self):self.assertRaises(ValueError,d.gate,d.PREP/'GPU_AUTHORIZATION_TEMPLATE.json','train')
    def test_wrong_hash_denied(self):
        a=d.read(d.PREP/'GPU_AUTHORIZATION_TEMPLATE.json');a.update(GPU_execution_approved=True,training_authorized=True,approval_source='researcher_message',researcher_explicit_authorization='synthetic test only',config_sha256='bad')
        with tempfile.TemporaryDirectory() as temp:
            p=pathlib.Path(temp)/'bad.json';p.write_text(json.dumps(a),encoding='utf-8');self.assertRaises(ValueError,d.gate,p,'train')
    def test_cli_gate_precedes_heavy_imports(self):
        p=subprocess.run([sys.executable,str(ROOT/'scripts/run_stage_i_d058_v1.py'),'train'],cwd=ROOT,capture_output=True,text=True)
        self.assertNotEqual(p.returncode,0);self.assertIn('explicit D058 approval',p.stderr)
        tree=ast.parse((ROOT/'scripts/run_stage_i_d058_v1.py').read_text(encoding='utf-8'))
        gate=next(n.lineno for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='gate')
        imports=[n.lineno for n in ast.walk(tree) if isinstance(n,ast.Import) and any(x.name=='torch' for x in n.names)]
        self.assertTrue(imports and min(imports)>gate)
    def test_no_gpu_or_model_runtime_in_offline_tests(self):self.assertFalse(any(x in sys.modules for x in ['torch','transformers','peft']))
    def test_no_run_or_prediction_namespace(self):self.assertFalse((ROOT/self.cfg['run_root']).exists())
    def test_integrity_and_D051(self):
        _,m=d.preflight();self.assertEqual(self.cfg['D051_strict_replay_status'],'FAIL_UNCHANGED');self.assertFalse(m['GPU_execution'])

if __name__=='__main__':unittest.main(verbosity=2)
