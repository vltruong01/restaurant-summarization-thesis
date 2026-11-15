"""Operational adapter only: invoke the unchanged, pinned D-026 generator."""
from pathlib import Path
import argparse
import yaml
import blip2_caption_pilot as legacy
from generate_stage_f_validation_captions import generate_locked
from stage_f_d033_protocol import freeze_method

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/stage_f_production_d033.yaml'
legacy.ORIGIN='agent'


def configuration():
    freeze_method()
    cfg=yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    prior=yaml.safe_load((ROOT/'configs/stage_f_validation_caption_pool.yaml').read_text(encoding='utf-8'))
    for key in ['seed','model_id','requested_revision','generation','device','basic_quality_policy']:
        assert cfg[key]==prior[key],f'D026 scientific configuration changed: {key}'
    out=ROOT/cfg['output_directory']
    identity=legacy.read(out/'pilot_identity.json');legacy.check_sources(identity)
    plan=legacy.read(out/'pilot_photo_plan.json')
    assert len(plan)==27511 and len({r['photo_id'] for r in plan})==len(plan)
    assert {r['split'] for r in plan}=={'train','test'}
    assert legacy.read(out/'model_revision.json')['resolved_commit']==cfg['requested_revision']
    return cfg,out


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['preflight','generate'])
    p.add_argument('--max-new-photos',type=int);a=p.parse_args();cfg,out=configuration()
    if a.action=='preflight':
        if (out/'environment_report.json').exists():
            # Stable device-map profile across resumes; never recalculate on a partial run.
            assert legacy.read(out/'environment_report.json')['status']=='preflight_passed_not_model_load_passed'
            print('Existing immutable preflight profile retained')
        else:legacy.preflight(cfg,out)
    else:
        from filelock import FileLock
        with FileLock(out/'caption_model_execution.lock',timeout=0):
            generate_locked(cfg,out,a.max_new_photos)


if __name__=='__main__':main()
