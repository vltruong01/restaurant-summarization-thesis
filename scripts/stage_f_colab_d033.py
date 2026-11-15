"""Colab notebook-facing adapter: same frozen core, independent runtime namespace, no remote shell."""
from pathlib import Path
import argparse,platform
from importlib.metadata import version
import stage_f_cloud_d033 as cloud

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/stage_f/production_caption_construction_d033_colab_v1'
GEN=OUT/'generation_remaining'
CONFIG=ROOT/'configs/stage_f_colab_d033.yaml'


def preflight():
    cfg=cloud.check_execution_seal();assert platform.system()=='Linux','Colab runtime only; local job stays stopped'
    import torch
    assert torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory/2**30>=14,'At least14GiB CUDA GPU required for this Colab smoke'
    expected=cloud.legacy.read(cloud.LOCAL/'generation_train_test/environment_report.json')
    for name,value in expected['versions'].items():assert version(name)==value,f'Pinned library mismatch: {name}'
    if (GEN/'environment_report.json').exists():
        r=cloud.legacy.read(GEN/'environment_report.json')
        assert r['status']=='preflight_passed_not_model_load_passed'
        assert r['gpu_name']==torch.cuda.get_device_name(0) and r['versions']==expected['versions'], 'Different GPU/runtime: stop; do not mix resume identities'
    else:cloud.legacy.preflight(cfg,GEN)


def bind():
    cloud.OUT=OUT;cloud.GEN=GEN;cloud.CONFIG=CONFIG;cloud.preflight=preflight


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['check','download','preflight','smoke','chunk','score','reviews','construct'])
    p.add_argument('--max-new-photos',type=int,default=500);a=p.parse_args();bind()
    if a.action=='check':cloud.check_execution_seal();print('PASS_COLAB_EXECUTION_BINDINGS_UNCHANGED_SCIENCE')
    elif a.action=='download':cloud.download()
    elif a.action=='preflight':preflight()
    elif a.action=='smoke':cloud.smoke()
    elif a.action=='chunk':
        assert a.max_new_photos>0
        cfg=cloud.check_execution_seal();preflight()
        assert cloud.legacy.read(OUT/'benchmark_report.json')['status'].startswith('PASS_CLOUD_TRAIN_ONLY_SMOKE')
        from filelock import FileLock
        with FileLock(GEN/'caption_model_execution.lock',timeout=0):
            cloud.generate_locked(cfg,GEN,max_new_photos=a.max_new_photos)
    else:cloud.process_step(a.action)


if __name__=='__main__':main()
