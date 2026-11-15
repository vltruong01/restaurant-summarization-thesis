"""Colab execution harness; primary dataset/history notebook remains00."""
from pathlib import Path
import json,textwrap

ROOT=Path(__file__).resolve().parents[1]
report=json.loads((ROOT/'outputs/cloud_handoff/stage_f_colab_d033_v1/addon_report.json').read_text(encoding='utf-8'))
patch_report=json.loads((ROOT/'outputs/cloud_handoff/stage_f_colab_d033_v2/patch_report.json').read_text(encoding='utf-8'))
cells=[]
def add(kind,text):
    text=textwrap.dedent(text).strip()+'\n'
    if kind=='code':compile(text,'ColabCell','exec')
    c={'cell_type':kind,'id':f'colab-stage-f-{len(cells):03d}','metadata':{},'source':text.splitlines(keepends=True)}
    if kind=='code':c.update(outputs=[],execution_count=None)
    cells.append(c)

add('markdown','''
# Stage F D-033 — Colab T4 smoke + checkpoint resume

Đây là notebook **execution trên Colab**, không thay notebook00 quản lý dataset/method/history. K5,5000/4000/750, COCOcommit/beam3/batch1/prompts/QC và filtering giữ nguyên. Tái sử dụng2709caption, sinh27318ảnh còn thiếu từ toàn bộ eligible pool.

Upload vào folder GoogleDrive `thesis_stage_f`: baseZIP2.91GB đã tạo choVast và ColabaddonZIP nhỏ. TrênColab chọn Runtime → Change runtime type → GPU. GPU miễn phí không được bảo đảm; VM có thể ngắt. Notebook chỉ chạy qua cell UI, khôngSSH/remote-shell/keepalive/distributed workers.

**Production continuation v2:** Researcher approved continuation after the100TRAIN-photo smoke passed. Mean0.80614s/photo,100Basicpass, reportedGPUallocated7.44GiB. RunAll now restores completedcaptions, reuses a passing smoke and continues the remainingpool in one model load, then runs scoring/review-input preparation/finalmanifests only after all27318cloudplan photos exist. StopbeforeGPTteacher/LEDtraining.

Upload the small `stage_f_colab_runtime_patch_v2.zip` to the sameDrivefolder as the existingtwoZIPs. This operationalpatch snapshots identicalhash/archive bytes while progressreports change; no generator/model/labels/filter/K/budget changes. Python3.11/pinnedlibraries remain separate. Photos run from `/content`; checkpointZIPs saved toDrive every120seconds andoncellfinish. FreeColabcaninterrupt; restore nextsession. NoSSH/keepaliveworkaround.
''')
add('code','''
from pathlib import Path
import os,sys,subprocess,hashlib,json,zipfile,shutil
from google.colab import drive
drive.mount('/content/drive')
DRIVE_DIR=Path('/content/drive/MyDrive/thesis_stage_f')
BASE_ZIP=DRIVE_DIR/'stage_f_d033_vast_bundle.zip'
ADDON_ZIP=DRIVE_DIR/'stage_f_colab_addon.zip'
PATCH_ZIP=DRIVE_DIR/'stage_f_colab_runtime_patch_v2.zip'
BANK=DRIVE_DIR/'colab_checkpoints'
ROOT=Path('/content/yelp_dataset')
RUN_MORE_CAPTIONS=True
RUN_FINAL_PROCESSING=True # Runs only after the complete caption pool exists.
NEW_PHOTOS_PER_CELL=27318 # Operational cap covering allremainingphotos; modelbatch stays1.
''')
add('code',f'''
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()
expected_base='7075b60da3495510d9ccfdd50deaf9d639923d04cee0bfcb608b2b9b7b0bfa0b'
expected_addon='{report['sha256']}'
expected_patch='{patch_report['sha256']}'
for source,digest in [(BASE_ZIP,expected_base),(ADDON_ZIP,expected_addon),(PATCH_ZIP,expected_patch)]:
    assert source.is_file(),f'Upload {{source.name}} to {{DRIVE_DIR}} first'
    local=Path('/content')/source.name
    if not local.exists() or sha(local)!=digest:shutil.copyfile(source,local)
    assert sha(local)==digest,'Wrong/corrupt upload archive'
    with zipfile.ZipFile(local) as z:
        for name in z.namelist():
            p=Path(name);assert not p.is_absolute() and '..' not in p.parts and name.startswith('yelp_dataset/')
        if source==PATCH_ZIP:
            receipt=json.loads(z.read('yelp_dataset/colab_runtime_patch_manifest.json'))
            assert set(receipt['files'])=={{'scripts/colab_stage_f_checkpoints.py'}}
            for name,entry in receipt['files'].items():
                assert sha(ROOT/name) in [entry['previous_sha256'],entry['sha256']]
                assert hashlib.sha256(z.read('yelp_dataset/'+name)).hexdigest()==entry['sha256']
        z.extractall('/content')
subprocess.run([sys.executable,str(ROOT/'scripts/verify_stage_f_cloud_package_d033.py')],check=True,cwd=ROOT)
print('Base bytes verified; Colab additions retain original scientific files.')
''')
add('code','''
# Bootstrap a Python3.11 interpreter even if Colab's notebook kernel is newer.
tools=Path('/content/thesis_uv_bootstrap')
subprocess.run([sys.executable,'-m','pip','install','--quiet','--target',str(tools),'uv'],check=True)
UV=next(p for p in tools.rglob('uv') if p.is_file() and os.access(p,os.X_OK))
venv=ROOT/'.venv-stage-f-cloud'
if not (venv/'bin/python').exists():subprocess.run([str(UV),'venv','--python','3.11','--seed',str(venv)],check=True)
PY=str(venv/'bin/python')
for args in [
    ['--index-url','https://pypi.org/simple','numpy==1.26.4'],
    ['--index-url','https://download.pytorch.org/whl/cu118','torch==2.1.0+cu118','torchvision==0.16.0+cu118'],
    ['--index-url','https://pypi.org/simple','-r','requirements/stage_f_cloud_d033.in']]:
    subprocess.run([PY,'-m','pip','install','--no-cache-dir',*args],cwd=ROOT,check=True)
subprocess.run([PY,'-m','pip','check'],check=True)
''')
add('code','''
sys.path.insert(0,str(ROOT/'scripts'))
import importlib,colab_stage_f_checkpoints
importlib.reload(colab_stage_f_checkpoints) # Use patch even in the existing Colabkernel.
from colab_stage_f_checkpoints import restore,backup,run_cell
KNOWN=restore(ROOT,BANK)
def run_stage(action,*args):
    run_cell([PY,'-u','scripts/stage_f_colab_d033.py',action,*args],ROOT,BANK,KNOWN)
run_stage('check')
# Uses pinnedHFrevision + originalfileSHA; no inference yet.
for key in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE']:os.environ.pop(key,None)
run_stage('download')
run_stage('preflight')
''')
add('code','''
run_stage('smoke')
OUT=ROOT/'outputs/stage_f/production_caption_construction_d033_colab_v1'
benchmark=json.loads((OUT/'benchmark_report.json').read_text())
display({k:v for k,v in benchmark.items() if k not in ['sample_photo_ids','caption_checkpoint_sha256']})
print('Passing smoke is reused; already completed100captions are not regenerated.')
''')
add('markdown','''
## Tiếp tục từng phiên sau khi smoke pass

Celltiếp theo chạy toàn bộremainingpool với mộtmodelload; model vẫnbatch1 và dùngfullpendingplan, khôngchọnảnh theoK/chấtlượng. Muốn giới hạn ngắn hơn, đổi NEW_PHOTOS_PER_CELL (engineering invocationlimit only). VMngắt thì kết nốiGPUlại và chạy setup/restore/smoke; checkpointđã lưu được khôi phục. NếuGPU/runtime đổi và guardfails, gửioutput; không bỏguard/giảmbeam/quantize/đổi method.

Checkpoint có thể mất phần chưabackup (tốiđa khoảng120giây khi VMbịngắt độtngột); originals và các ZIPđãverify còn trênDrive. Không dùng mẹo giữphiên/SSH. Colabfree không bảođảm12giờliêntiếp hay GPUngaykhi cần.
''')
add('code','''
if RUN_MORE_CAPTIONS:
    run_stage('chunk','--max-new-photos',str(NEW_PHOTOS_PER_CELL))
else:
    print('Captioncontinuation disabled by RUN_MORE_CAPTIONS.')
completed=len(list((OUT/'generation_remaining/caption_checkpoints').glob('*.json')))
print({'new_colab_captions':completed,'total_including_reused':2709+completed,'remaining':27318-completed})
backup(ROOT,BANK,KNOWN)
''')
add('code','''
if RUN_FINAL_PROCESSING and completed==27318:
    for action in ['score','reviews','construct']:run_stage(action)
    display(json.loads((OUT/'completion_report.json').read_text()))
else:
    print('Finalprocessing waits for the complete27318caption pool; currentcount:',completed,'. No teacher/LEDtraining.')
''')
add('markdown','''
Giữ `colab_checkpoints/` và ZIPnguồn trênDrive. Sau khi hoàn tất, checkpointbundles chứa scores/manifests/completionreport; có thể download đểCodexlocalkiểmtra. Notebook00 và THESIS_PROGRESS.md vẫn là entrypointmethod/history chính.

Sources: [ColabFAQ](https://research.google.com/colaboratory/faq.html), [NVIDIAT4](https://www.nvidia.com/en-us/data-center/tesla-t4/), [uvPython](https://docs.astral.sh/uv/guides/install-python/).
''')
nb={'nbformat':4,'nbformat_minor':5,'metadata':{'colab':{'name':'09_stage_f_colab_smoke_resume.ipynb'},'kernelspec':{'name':'python3','display_name':'Python3'},'language_info':{'name':'python'}},'cells':cells}
path=ROOT/'notebooks/09_stage_f_colab_smoke_resume.ipynb';path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8');print(path)
