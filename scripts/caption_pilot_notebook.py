"""Notebook launcher for the isolated E0 environment; no work at import time."""
from pathlib import Path
import json,subprocess,sys,os,time
import yaml

ROOT=Path(__file__).resolve().parents[1]


def configuration():
    return yaml.safe_load((ROOT/'configs/blip2_caption_feasibility_pilot.yaml').read_text(encoding='utf-8'))


def output_directory():return ROOT/configuration()['output_directory']


def run_logged(command,log_name,allow_nonzero=False):
    out=output_directory();(out/'logs').mkdir(parents=True,exist_ok=True)
    log=out/'logs'/log_name;started=time.perf_counter()
    env=os.environ.copy();env['PYTHONUTF8']='1';env['TOKENIZERS_PARALLELISM']='false';env['RAYON_NUM_THREADS']='4'
    env['PIP_CONFIG_FILE']=os.devnull
    for key in ['PIP_EXTRA_INDEX_URL','PIP_INDEX_URL','PIP_TRUSTED_HOST','PIP_FIND_LINKS']:env.pop(key,None)
    print('Running:',subprocess.list2cmdline(list(map(str,command))),flush=True)
    with log.open('w',encoding='utf-8') as stream:
        with subprocess.Popen(list(map(str,command)),cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',bufsize=1) as process:
            for line in process.stdout:
                stream.write(line);stream.flush();print(line,end='',flush=True)
            code=process.wait()
    if code and not allow_nonzero:
        report={'status':'blocked','command':list(map(str,command)),'exit_code':code,'log':log.relative_to(ROOT).as_posix(),'elapsed_seconds':round(time.perf_counter()-started,3),'agent_executed':False,'message':'User-run step failed. Inspect log/report; no automatic checkpoint/model/precision/source-budget change.'}
        (out/'last_step_failure.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        raise RuntimeError(f'E0 step failed; log: {log}. Send this log/report for review.')
    return code


def ensure_environment(allow_install=True):
    cfg=configuration();directory=(ROOT/cfg['isolated_environment']).resolve()
    if not directory.is_relative_to(ROOT.resolve()) or directory.name=='.venv-preprocess':raise ValueError('Caption environment must be separate and inside project.')
    python=directory/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    if not python.is_file():
        if not allow_install:raise RuntimeError('Caption environment absent; enable installation or create it separately.')
        run_logged([sys.executable,'-m','venv',str(directory)],'environment_create.log')
    marker=directory/'caption_pilot_setup.json'
    requirements=ROOT/'requirements/caption_pilot.in'
    import hashlib
    signature=hashlib.sha256(requirements.read_bytes()).hexdigest()
    if not marker.is_file() or json.loads(marker.read_text()).get('requirements_sha256')!=signature:
        if not allow_install:raise RuntimeError('Caption dependencies not installed for this profile.')
        # Reuse the confirmed installed CUDA runtime read-only; install caption deps
        # locally so their pinned versions take precedence. No global package edits.
        probe=subprocess.run([sys.executable,'-c','import json,torch,torchvision; print(json.dumps({"torch":torch.__version__,"torchvision":torchvision.__version__,"cuda":torch.cuda.is_available()}))'],capture_output=True,text=True,encoding='utf-8',errors='replace')
        available=json.loads(probe.stdout.strip().splitlines()[-1]) if probe.returncode==0 else {}
        reuse=available.get('cuda') and available.get('torch')=='2.1.0+cu118' and available.get('torchvision')=='0.16.0+cu118'
        if reuse:
            venv_cfg=directory/'pyvenv.cfg';text=venv_cfg.read_text(encoding='utf-8');text=text.replace('include-system-site-packages = false','include-system-site-packages = true');venv_cfg.write_text(text,encoding='utf-8')
            # Exact direct caption versions already exist in preprocessing; import
            # them read-only ahead of the older global hub, without copies/downloads.
            substrate=ROOT/'.venv-preprocess'/('Lib/site-packages' if os.name=='nt' else 'lib/python3.11/site-packages')
            site=directory/('Lib/site-packages' if os.name=='nt' else 'lib/python3.11/site-packages')
            (site/'caption_preprocessing_readonly.pth').write_text('import sys; sys.path.insert(0, '+repr(str(substrate))+')\n',encoding='utf-8')
            (output_directory()/'environment_readonly_bindings.json').write_text(json.dumps({'preprocessing_site':str(substrate),'global_CUDA_runtime':available,'bound_read_only':True,'global_and_preprocessing_packages_modified':False},indent=2)+'\n',encoding='utf-8')
        else:
            run_logged([python,'-m','pip','install','--no-cache-dir','torch==2.7.1','torchvision==0.22.1','--index-url','https://download.pytorch.org/whl/cu128'],'environment_torch_install.log')
            run_logged([python,'-m','pip','install','--no-cache-dir','--index-url','https://pypi.org/simple','-r',requirements],'environment_caption_install.log')
        check_code=run_logged([python,'-m','pip','check'],'environment_pip_check.log',allow_nonzero=bool(reuse))
        run_logged([python,'-c','import torch,torchvision,transformers,accelerate,tokenizers,huggingface_hub; from transformers import Blip2Processor,Blip2ForConditionalGeneration; print({"torch":torch.__version__,"torchvision":torchvision.__version__,"transformers":transformers.__version__,"hub":huggingface_hub.__version__,"cuda":torch.cuda.is_available()}); assert torch.cuda.is_available(); assert transformers.__version__=="4.51.3"; assert huggingface_hub.__version__=="0.36.2"'],'environment_runtime_imports.log')
        run_logged([python,'-c','from pathlib import Path; from importlib.metadata import version; lines=Path("requirements/caption_pilot.in").read_text().splitlines(); deps=[line.split("==") for line in lines if line and not line.startswith("#")]; actual={name:version(name) for name,expected in deps}; print(actual); assert all(actual[name]==expected for name,expected in deps)'],'environment_pinned_versions.log')
        marker.write_text(json.dumps({'requirements_sha256':signature,'CUDA_runtime_reused_read_only':bool(reuse),'existing_runtime':available if reuse else None,'global_packages_modified':False,'pip_check_exit_code':check_code,'runtime_imports_passed':True,'status':'caption_dependencies_ready'},indent=2)+'\n',encoding='utf-8')
    return python


def run_step(python,step,allow_download=False,execution_origin='user'):
    command=[python,ROOT/'scripts/blip2_caption_pilot.py',step,'--execution-origin',execution_origin]
    if allow_download:command.append('--allow-model-download')
    run_logged(command,step+'.log')
    path=output_directory()/({'prepare':'pilot_preparation_report.json','preflight':'environment_report.json','generate':'generation_report.json','report':'pilot_report.json'}[step])
    if not path.is_file():raise RuntimeError(f'Missing report from completed {step}: {path}')
    return json.loads(path.read_text(encoding='utf-8'))
