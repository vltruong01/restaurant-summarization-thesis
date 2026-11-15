"""D059 offline contracts/negative controls. No GPU/model execution."""
import ast,copy,json,subprocess,sys,tempfile,unittest
from pathlib import Path
from datetime import datetime,timedelta,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import stage_i_d059 as d


class D059Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.c=d.read(d.CONFIG)
    def test_config(self):self.assertTrue(d.check_config(self.c))
    def test_inventory_exact(self):
        a=d.read(d.PREP/'MODULE_INVENTORY.json');self.assertEqual({x['name'] for x in a['modules']},d.expected_modules())
        self.assertEqual((a['module_count'],a['trainable_parameters'],a['trainable_tensors']),(114,1843200,228))
        self.assertEqual(sum(x['trainable_parameters'] for x in a['modules']),1843200)
        self.assertEqual(a['frozen_backbone_parameters'],161844480)
    def test_heads_embeddings_excluded(self):
        a=d.expected_modules();self.assertFalse(any('lm_head' in n or 'embed' in n for n in a))
        self.assertEqual(sum(n.endswith('.output') for n in a),6)
    def test_keys_ffn_global_present(self):
        a=d.expected_modules();self.assertEqual(sum('key_global' in n for n in a),6)
        self.assertEqual(sum(n.endswith(('.fc1','.fc2')) for n in a),24)
        self.assertEqual(sum('.decoder.' in n and n.endswith('.k_proj') for n in a),12)
    def test_offline_inventory_guards(self):
        a=d.read(d.PREP/'MODULE_INVENTORY.json');self.assertTrue(a['meta_only'] and a['CPU_adapter_serialization_exact'])
        self.assertFalse(a['CUDA_initialized'] or a['backbone_weights_loaded']);self.assertEqual(a['forward_backward_calls'],0)
    def test_rank_change_denied(self):
        c=copy.deepcopy(self.c);c['lora']['r']=16;self.assertRaises(ValueError,d.check_config,c)
    def test_quantization_denied(self):
        c=copy.deepcopy(self.c);c['quantization']=True;self.assertRaises(ValueError,d.check_config,c)
    def test_grid_change_denied(self):
        c=copy.deepcopy(self.c);c['evaluation_epochs']=[1,3,5];self.assertRaises(ValueError,d.check_config,c)
    def test_lr_change_denied(self):
        c=copy.deepcopy(self.c);c['optimization']['learning_rate']=1e-4;self.assertRaises(ValueError,d.check_config,c)
    def test_precision_change_denied(self):
        c=copy.deepcopy(self.c);c['optimization']['bf16']=True;self.assertRaises(ValueError,d.check_config,c)
    def test_sixth_epoch_denied(self):
        c=copy.deepcopy(self.c);c['optimization']['num_train_epochs']=6;self.assertRaises(ValueError,d.check_config,c)
    def test_decoding_change_denied(self):
        c=copy.deepcopy(self.c);c['generation']['no_repeat_ngram_size']=4;self.assertRaises(ValueError,d.check_config,c)
    def test_resume_denied(self):
        c=copy.deepcopy(self.c);c['resume_short_pilot']=True;self.assertRaises(ValueError,d.check_config,c)
    def test_deadline_expiry(self):
        start=datetime(2026,1,1,tzinfo=timezone.utc);a={'execution_authorized_at_utc':start.isoformat(),'improvement_deadline_utc':(start+timedelta(days=7)).isoformat()}
        self.assertEqual(d.check_deadline(a,start+timedelta(days=1)),start+timedelta(days=7))
        self.assertRaises(ValueError,d.check_deadline,a,start+timedelta(days=7))
    def test_deadline_extension_denied(self):
        start=datetime(2026,1,1,tzinfo=timezone.utc);a={'execution_authorized_at_utc':start.isoformat(),'improvement_deadline_utc':(start+timedelta(days=8)).isoformat()}
        self.assertRaises(ValueError,d.check_deadline,a,start)
    def test_false_approval_denied(self):self.assertRaises(ValueError,d.gate,d.PREP/'GPU_AUTHORIZATION_TEMPLATE.json','feasibility')
    def test_fake_binding_denied(self):
        a=d.read(d.PREP/'GPU_AUTHORIZATION_TEMPLATE.json');a.update(GPU_execution_approved=True,approval_source='researcher_message',researcher_explicit_authorization='synthetic test, not real approval',config_sha256='bad')
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'approval.json';p.write_text(json.dumps(a),encoding='utf-8');self.assertRaises(ValueError,d.gate,p,'train')
    def test_RO_U_scope_denied(self):
        a=d.read(d.PREP/'GPU_AUTHORIZATION_TEMPLATE.json')
        a.update(GPU_execution_approved=True,approval_source='researcher_message',researcher_explicit_authorization='SYNTHETIC negative control only',RO_U_training_authorized=True)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'approval.json';p.write_text(json.dumps(a),encoding='utf-8');self.assertRaises(ValueError,d.gate,p,'train')
    def test_feasibility_only_does_not_authorize_train(self):
        a=d.read(d.PREP/'GPU_AUTHORIZATION_TEMPLATE.json')
        a.update(GPU_execution_approved=True,approval_source='researcher_message',researcher_explicit_authorization='SYNTHETIC negative control only',scopes=['feasibility'])
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'approval.json';p.write_text(json.dumps(a),encoding='utf-8');self.assertRaises(ValueError,d.gate,p,'train')
    def test_cli_denies_no_approval_before_torch(self):
        p=subprocess.run([sys.executable,str(d.RUNNER),'feasibility'],cwd=ROOT,capture_output=True,text=True)
        self.assertNotEqual(p.returncode,0);self.assertIn('explicit D059 approval',p.stderr)
        tree=ast.parse(d.RUNNER.read_text(encoding='utf-8'));calls=[n.lineno for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='gate']
        imports=[n.lineno for n in ast.walk(tree) if isinstance(n,ast.Import) and any(x.name=='torch' for x in n.names)]
        self.assertLess(min(calls),min(imports))
    def test_data_and_zero_scope(self):
        r=d.read(d.PREP/'DATA_AND_MASK_READINESS.json');self.assertEqual(r['exact_sources_targets_checked'],2200)
        self.assertEqual(r['stats']['train']['zero_filtered_caption_businesses'],9)
        self.assertEqual(r['stats']['validation']['zero_filtered_caption_businesses'],1)
        self.assertFalse(r['held_out_contents_accessed'])
    def test_cases_and_orders_unchanged(self):
        for name in ('EPOCH_ORDERS.json','EVALUATION_MANIFEST.json'):
            self.assertEqual(d.sha(d.PREP/name),d.sha(ROOT/'outputs/stage_i/d058_preparation_v1'/name))
    def test_masks(self):
        r=d.read(d.PREP/'FEASIBILITY_CASES.json');self.assertGreaterEqual(len(r['business_ids']),2)
        for c in r['cases']:
            self.assertEqual((c['source_shape'],c['label_shape'],c['global_tokens']),([1,5120],[1,768],1))
    def test_benchmarks_reused(self):
        r=d.read(d.PREP/'BENCHMARKS_UNCHANGED.json');self.assertEqual(len(r),48)
        self.assertEqual({x['arm'] for x in r},{'historical_full_F','B_full_epoch5'})
    def test_execution_budget(self):
        b=d.read(d.PREP/'EXECUTION_BUDGET.json');self.assertEqual((b['max_optimizer_steps'],b['warmup_steps'],b['generation_outputs']),(1250,63,48))
        self.assertEqual((b['feasibility_optimizer_updates'],b['feasibility_microbatches'],b['RO_U_runs_authorized']),(3,24,0))
    def test_no_gpu_run_namespace(self):self.assertFalse((ROOT/self.c['run_root']).exists())
    def test_integrity(self):d.preflight()
    def test_no_heavy_imports_in_tests(self):self.assertFalse(any(x in sys.modules for x in ('torch','transformers','peft')))
    def test_runtime_deadline_checks(self):
        s=d.RUNNER.read_text(encoding='utf-8');self.assertIn('check_deadline(approval)',s)
        self.assertIn('feasibility_report=require_feasibility(cfg,approval)',s)
        self.assertIn("'Expanded_full_epoch",s)


if __name__=='__main__':unittest.main(verbosity=2)
