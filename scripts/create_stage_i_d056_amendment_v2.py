"""One-time versioned preparation. File copying/AST/NumPy only; no model load."""
import copy,json,shutil,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1]
O=R/'outputs/stage_i/d056_diagnostic_preparation_v2'

def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    O.mkdir(exist_ok=False)
    prior=R/'outputs/stage_i/d056_diagnostic_preparation_v1'
    protected={p.relative_to(R).as_posix():digest(p) for p in prior.rglob('*') if p.is_file()}
    names=['stage_i_d056.py','run_stage_i_d056_gpu_diagnostic.py','check_stage_i_d056_offline.py',
           'verify_stage_i_d056_review_bundle.py','package_stage_i_d056_review.py']
    for name in names:
        p=R/'scripts'/name;protected[p.relative_to(R).as_posix()]=digest(p)
        text=p.read_text(encoding='utf-8').replace('from stage_i_d056 import','from stage_i_d056_v2 import').replace('import stage_i_d056 as','import stage_i_d056_v2 as')
        text=text.replace('d056_diagnostic_preparation_v1','d056_diagnostic_preparation_v2').replace('stage_i_d056_minimal_diagnostic_v1','stage_i_d056_minimal_diagnostic_v2')
        text=text.replace('d056_gpu_diagnostic_v1','d056_gpu_diagnostic_v2').replace('stage_i_d056_review_bundle_v1','stage_i_d056_review_bundle_v2')
        dest=R/'scripts'/(p.stem+'_v2.py')
        dest.write_text(text,encoding='utf-8',newline='\n')
    for rel in ['configs/stage_i_d056_minimal_diagnostic_v1.json','tests/test_stage_i_d056_offline.py',
                'outputs/stage_i/stage_i_d056_review_bundle_v1.zip','outputs/stage_i/stage_i_d056_review_bundle_v1.zip.sha256',
                'scripts/prepare_stage_i_d056_offline.py','specs/009k_d056_literature_informed_diagnostic_preparation.md']:
        protected[rel]=digest(R/rel)
    (O/'PARENT_V1_PRESERVATION_MANIFEST.json').write_text(json.dumps(protected,indent=2)+'\n',encoding='utf-8')
    for name in ['cases.json','ADAPTER_GEOMETRY.json','IMPLEMENTATION_BINDINGS.json',
                 'LITERATURE_INFORMED_RATIONALE.md','INTERPRETATION_AND_NEXT_EXPERIMENT_TREE.md']:
        shutil.copyfile(prior/name,O/name)
    shutil.copytree(prior/'implementation',O/'implementation')
    for rel in ['DECISIONS.md','thesis_sdd_docs/THESIS_PROGRESS.md','thesis_sdd_docs/CHATGPT_NEW_CONVERSATION_HANDOFF.md']:
        p=O/'documentation_before'/Path(rel).name;p.parent.mkdir(exist_ok=True);shutil.copyfile(R/rel,p)
    cfg=json.loads((R/'configs/stage_i_d056_minimal_diagnostic_v1.json').read_text(encoding='utf-8'))
    cfg.update(amendment='D-056-v2',parent_config_sha256=digest(R/'configs/stage_i_d056_minimal_diagnostic_v1.json'),
               cache_prefix_lengths=[3,11],future_output='outputs/stage_i/d056_gpu_diagnostic_v2',
               cache_design_change='D-056-v2 prioritizes observed marker failure p=3 and newline failure p=11. Three exact-prefix calls each. p=12 and visual boundaries remain in the existing TF measurements; no added calls.',
               fingerprint_convention='D050 recursive bare-Parameter convention, reused in D052/D054; type-name:dtype+shape+C-order bytes; final semicolon',
               trace_candidate_export={'raw_top_k':10,'processed_top_k':10,'independent_rankings':True,'tie_break':'ascending token ID','masked_score_encoding':'-inf string'},
               beam_finalization='Pinned Transformers4.51.3 vectorized _beam_search; EOS and custom censor separately recorded; no appended EOS inferred from finalization')
    (R/'configs/stage_i_d056_minimal_diagnostic_v2.json').write_text(json.dumps(cfg,indent=2)+'\n',encoding='utf-8')
    tests=(R/'tests/test_stage_i_d056_offline.py').read_text(encoding='utf-8')
    tests=tests.replace('import stage_i_d056 as d','import stage_i_d056_v2 as d').replace('run_stage_i_d056_gpu_diagnostic.py','run_stage_i_d056_gpu_diagnostic_v2.py')
    tests=tests.replace("['cache_probes'][0];self.assertEqual(p['expected_label_index'],10)","['cache_probes'][1];self.assertEqual(p['expected_label_index'],10)")
    start=tests.index('    def test_cache_prefix12_predicts_first_body')
    end=tests.index('    def test_initial_decoder_eos_not_label_eos',start)
    tests=tests[:start]+'''    def test_cache_prefix3_predicts_RE(self):
        for c in self.cases:
            p=d.alignment(c,self.tok)['cache_probes'][0];self.assertEqual(p['reference_logit_row'],2);self.assertEqual(p['expected_id'],4629)
    def test_teacher_forced_retains_p12_first_body(self):
        for c in self.cases:
            a=d.alignment(c,self.tok);self.assertIn(11,a['critical_logit_rows']);self.assertEqual(a['token_positions'][11]['expected_id'],c['label_ids'][11])
''' +tests[end:]
    (R/'tests/test_stage_i_d056_offline_v2.py').write_text(tests,encoding='utf-8',newline='\n')
    print(json.dumps({'v2_created':str(O),'v1_files_preserved':len(protected)}))

if __name__=='__main__':main()
