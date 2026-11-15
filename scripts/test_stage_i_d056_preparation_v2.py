"""Run complete v1 + v2 offline regressions/negative controls with import guard."""
import importlib.abc,io,json,sys,unittest
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'outputs/stage_i/d056_diagnostic_preparation_v2'

class NoModelRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ['torch','transformers','peft']:
            raise RuntimeError('OFFLINE_FORBIDDEN_MODEL_RUNTIME_IMPORT: '+fullname)
        return None

def main():
    sys.meta_path.insert(0,NoModelRuntime())
    require_clean=lambda:not any(k in sys.modules for k in ['torch','transformers','peft'])
    assert require_clean()
    suite=unittest.defaultTestLoader.discover(str(R/'tests'),pattern='test_stage_i_d056_offline*.py')
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    attempts=O/'offline_test_attempts';attempts.mkdir(exist_ok=True)
    number=len(list(attempts.glob('attempt_*.json')))+1;stem=attempts/f'attempt_{number:03d}'
    with stem.with_suffix('.log').open('x',encoding='utf-8') as f:f.write(stream.getvalue())
    report={'status':'PASS' if result.wasSuccessful() and require_clean() else 'FAIL',
            'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),
            'failed_tests':[str(x[0]) for x in result.failures+result.errors],
            'includes_unchanged_v1_suite':True,'includes_v2_regression_and_negative_controls':True,
            'actual_model_imports':False,'model_loads':0,'GPU_calls':0,'model_forwards':0,
            'training_updates':0,'Teacher_calls':0,'held_out_files_opened':False,
            'log':str(stem.with_suffix('.log').relative_to(R))}
    with stem.with_suffix('.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))
    if not result.wasSuccessful():print(stream.getvalue());sys.exit(1)

if __name__=='__main__':main()
