"""CPU-only test harness with denied CUDA initialization and real payload reads."""
from pathlib import Path
import os,sys,unittest,json,io,builtins,time
from unittest.mock import patch
R=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(R/'tests'),str(R/'scripts')]
os.environ.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
import torch
real_open=builtins.open;real_io_open=io.open;blocked=[]
def forbidden(path):
    if isinstance(path,(str,os.PathLike)):
        name=str(path).replace('\\','/').lower()
        if str(R).replace('\\','/').lower() in name:
            if any(x in name for x in ['validation_filtered.npz','validation_filtered_content_eos_labels.npz','evaluation_inputs.jsonl','final_researcher_approved_references.json','/test_filtered','/test_reviews','/test_unfiltered']):return True
            if ('/assets/models/' in name or '/checkpoint-' in name) and name.endswith(('.bin','.pt','.safetensors')):return True
    return False
def guard(fn):
    def opened(path,*args,**kwargs):
        if forbidden(path):blocked.append(str(path));raise AssertionError('Unauthorized real payload access during CPU tests: '+str(path))
        return fn(path,*args,**kwargs)
    return opened
class Result(unittest.TextTestResult):
    ids=[]
    def startTest(self,test):self.ids.append(test.id());super().startTest(test)
def main():
    out=R/'outputs/stage_i/led_large_boundary_dual_v2_2_time_amendment';out.mkdir(parents=True,exist_ok=True)
    name=sys.argv[1] if len(sys.argv)>1 else 'FINAL_CPU_TESTS'
    start=time.monotonic()
    with (out/(name+'.log')).open('x',encoding='utf-8') as log:
        with patch.object(torch.cuda,'init',side_effect=AssertionError('CUDA initialization forbidden')),patch.object(torch.cuda,'is_available',return_value=False),patch.object(builtins,'open',guard(real_open)),patch.object(io,'open',guard(real_io_open)):
            loader=unittest.TestLoader();suite=unittest.TestSuite()
            for module in (sys.argv[2:] or ['test_led_large_boundary_dual_v2_1','test_led_large_boundary_global_v1','test_led_large_boundary_time_amendment_v2_2']):suite.addTests(loader.loadTestsFromName(module))
            result=unittest.TextTestRunner(stream=log,verbosity=2,resultclass=Result).run(suite)
    report=dict(status='PASS' if result.wasSuccessful() and not blocked and not torch.cuda.is_initialized() else 'FAIL',
        tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),skips=result.skipped,
        seconds=time.monotonic()-start,test_ids=result.ids,blocked_payload_attempts=blocked,
        CUDA_initialized=torch.cuda.is_initialized(),real_GPU_work=False,real_LED_forward_backward=False,
        native_FP16_scaler_verification='UNVERIFIED; synthetic backoff and real CPU tensors only',
        real_process_tests='Two local CPU subprocesses; no SSH/network/NVIDIA query',
        results=[dict(test=str(t),traceback=e) for t,e in result.failures+result.errors],
        environment={k:__import__('importlib.metadata',fromlist=['version']).version(k) for k in ['torch','transformers','peft','accelerate','numpy']})
    (out/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['test_ids','results']},indent=2))
    for t,e in result.failures+result.errors:print(str(t)+'\n'+e)
    raise SystemExit(0 if report['status']=='PASS' else 1)
if __name__=='__main__':main()
