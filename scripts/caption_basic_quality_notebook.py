"""Explicit E1 CLI launcher; default cached notebook never invokes it."""
from pathlib import Path
import json
import os
import subprocess
import yaml

ROOT = Path(__file__).resolve().parents[1]


def configuration():
    return yaml.safe_load((ROOT/'configs/blip2_caption_basic_quality_pilot.yaml').read_text())


def output_directory():
    return ROOT/configuration()['output_directory']


def ensure_environment(allow_install=False):
    directory = ROOT/configuration()['isolated_environment']
    python = directory/('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if not python.is_file():
        raise RuntimeError('Create the isolated caption environment first; E1 launcher does not silently install software.')
    return python


def run_step(python, step, allow_download=False, execution_origin='user'):
    out = output_directory(); (out/'logs').mkdir(parents=True, exist_ok=True)
    env = os.environ.copy(); env.update(PYTHONUTF8='1', TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1')
    command = ([str(python), str(ROOT/'scripts/validate_basic_quality_caption_outputs.py')]
               if step == 'validate' else
               [str(python), str(ROOT/'scripts/blip2_caption_basic_quality_pilot.py'), step,
                '--execution-origin', execution_origin])
    with (out/'logs'/f'{step}.log').open('w', encoding='utf-8') as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
        for line in process.stdout:
            stream.write(line); stream.flush(); print(line, end='', flush=True)
        code = process.wait()
    if code:
        raise RuntimeError(f'E1 {step} failed; inspect {out}/logs/{step}.log. No automatic research change.')
    filename = {'prepare':'pilot_preparation_report.json', 'preflight':'environment_report.json',
                'generate':'generation_report.json', 'report':'pilot_report.json', 'validate':'validation_report.json'}[step]
    return json.loads((out/filename).read_text(encoding='utf-8'))
