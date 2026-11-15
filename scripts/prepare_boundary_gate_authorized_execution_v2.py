"""Fresh operational retry after explicit researcher gate-only approval; no science edits."""
from pathlib import Path
from datetime import datetime, timedelta, timezone
import difflib
import hashlib
import json
import zipfile

R = Path(__file__).resolve().parents[1]
OLD = R/'outputs/stage_i/led_large_boundary_global_feasibility_v1_execution_v1'
OUT = R/'outputs/stage_i/led_large_boundary_global_feasibility_v1_execution_v2'
STAMP = '20261024T184247Z'
REMOTE = '/workspace/BOUNDARY_GATE_'+STAMP


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(name, value):
    with (OUT/name).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2); f.write('\n')


def main():
    if OUT.exists():
        assert {p.name for p in OUT.iterdir()} == {'LOCAL_PACKAGING_CHECK_FAILURE.json'}, 'Fresh local execution namespace required'
    else:
        OUT.mkdir()
    package=R/'outputs/cloud_handoff/LED_LARGE_BOUNDARY_GLOBAL_FEASIBILITY_V1_OFFLINE_REVIEW.zip'
    assert sha(package)=='06f71d2944147f761463f01be26f13b82aa1264a6240619f559e64e7eea7bf12'
    with zipfile.ZipFile(package) as z:
        assert z.testzip() is None
        files=json.loads(z.read('SHA256_MANIFEST.json'))['files']
        assert len(files)==79
        for name, digest in files.items():
            assert hashlib.sha256(z.read(name)).hexdigest()==digest, name
            if (R/name).is_file():
                assert sha(R/name)==digest, name
            else:
                assert name=='READ_THIS_FIRST.md', name
    write('LOCAL_79_HASH_VERIFICATION.json',dict(status='PASS',files=79,original_package_sha256=sha(package),scientific_files_modified=False))
    start=datetime(2026,10,24,18,42,47,tzinfo=timezone.utc)
    end=start+timedelta(hours=48)
    approval=json.loads((R/'outputs/stage_i/led_large_boundary_global_feasibility_v1/AUTHORIZATION_TEMPLATE_NOT_APPROVED.json').read_text(encoding='utf-8'))
    approval.update(GPU_execution_approved=True,
        researcher_explicit_authorization='Current researcher message: "ok hãy làm". In context this accepts the immediately proposed Boundary feasibility gate first, with STOP for review; it does not authorize full training. The frozen gate-only scope remains BOS-only then Boundary,3discardedupdates+2TRAINtraces+full-cache stress per arm.',
        execution_authorized_at_utc=start.isoformat(),execution_deadline_utc=end.isoformat(),
        researcher_absolute_stop_utc=end.isoformat(),defense_reserve_at_least_14_days_confirmed=True)
    write('RESEARCHER_AUTHORIZATION_'+STAMP+'.json',approval)
    write('AUTHORIZATION_PROVENANCE.json',dict(current_user_message='ok hãy làm',
        scope_interpretation='Accepts the preceding gate-first proposal only. Full training requires another researcher decision.',
        observed_start_of_authorized_turn_UTC=start.isoformat(),actual_user_message_UTC_unavailable=True,
        timestamp_source='clock tool at first authorized-turn observation, not a fabricated chat metadata timestamp',
        maximum_window_hours=48,deadline_utc=end.isoformat(),old_approvals_reused=False,
        reserve_basis='Retained explicit researcher confirmation: Có, tôi xác nhận còn ít nhất14ngày. Same provenance accepted by the previous gate contract; no defense date inferred or new confirmation invented.',
        researcher_manages_calendar=True,old_reserve_provenance_sha256=sha(OLD/'AUTHORIZATION_PROVENANCE.json'),
        previous_failure_preserved=True,authorized_operational_fix='PYTHONPATH includes the complete unchanged parent/scripts',
        fresh_remote_root=REMOTE,instance_id=57481997,full_training_authorized=False))
    original=(OLD/'HOST_DRIVER.py').read_text(encoding='utf-8')
    driver=original.replace('/workspace/BOUNDARY_GATE_20261024T105400Z',REMOTE).replace("STAMP = '20261024T105400Z'",f"STAMP = '{STAMP}'")
    needle="OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')"
    assert driver.count(needle)==1
    driver=driver.replace(needle,"OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1',\n        PYTHONPATH=str(PARENT/'scripts'))")
    (OUT/'HOST_DRIVER.py').write_text(driver,encoding='utf-8')
    (OUT/'OPERATIONAL_DIFF.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),driver.splitlines(True),fromfile='preserved_V1/HOST_DRIVER.py',tofile='fresh_execution_V2/HOST_DRIVER.py')),encoding='utf-8')
    deploy=r'''"""Copy immutable original material into a fresh root; no model calls."""
from pathlib import Path
import hashlib,json,shutil,subprocess,zipfile
ROOT=Path(REMOTE_ROOT)
OLD=Path('/workspace/BOUNDARY_GATE_20261024T105400Z')
OPS=ROOT/'ops';P=ROOT/'parent';V=ROOT/'overlay'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(n,d):
    with (OPS/n).open('x') as f:json.dump(d,f,indent=2);f.write('\n')
assert ROOT.parent==Path('/workspace') and not P.exists() and not V.exists()
expected=read(OPS/'OPS_MANIFEST.json')['files']
for name,digest in expected.items():assert sha(OPS/name)==digest,name
manifest=read(OLD/'ops/PARENT_TRANSFER_MANIFEST.json')
assert len(manifest['files'])==10025
assert {p.relative_to(OLD/'parent').as_posix() for p in (OLD/'parent').rglob('*') if p.is_file()}==set(manifest['files'])
P.mkdir()
for name,digest in manifest['files'].items():
    source=(OLD/'parent'/name).resolve();dest=(P/name).resolve()
    assert source.is_relative_to((OLD/'parent').resolve()) and dest.is_relative_to(P.resolve()) and not source.is_symlink()
    assert sha(source)==digest,name
    dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
with (OPS/'PARENT_TRANSFER_MANIFEST.json').open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')
archive=OLD/'ops/LED_LARGE_BOUNDARY_GLOBAL_FEASIBILITY_V1_OFFLINE_REVIEW.zip'
assert sha(archive)=='06f71d2944147f761463f01be26f13b82aa1264a6240619f559e64e7eea7bf12'
shutil.copyfile(archive,OPS/archive.name)
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for n in z.namelist():assert not Path(n).is_absolute() and '..' not in Path(n).parts and '\\' not in n
    V.mkdir();z.extractall(V)
    files=json.loads(z.read('SHA256_MANIFEST.json'))['files']
    assert len(files)==79
    for name,digest in files.items():assert sha(V/name)==digest,name
program=PROGRAM_NAME
config=Path('/etc/supervisor/conf.d')/(program+'.conf')
assert not config.exists()
content=f"""[program:{program}]
command=/workspace/cad_runtime/bin/python -u {OPS}/HOST_DRIVER.py
directory={OPS}
autostart=false
autorestart=false
startretries=0
startsecs=0
stopasgroup=true
killasgroup=true
stdout_logfile={OPS}/HOST_DRIVER.log
redirect_stderr=true
stdout_logfile_maxbytes=0
environment=CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",TOKENIZERS_PARALLELISM="false"
"""
with config.open('x') as f:f.write(content)
with (OPS/'SUPERVISOR_PROGRAM.conf').open('x') as f:f.write(content)
write('DEPLOYMENT_PASS.json',dict(parent_files_copied=10025,overlay_hashes=79,
    scientific_files_modified=False,operational_import_path_fix_only=True,old_failure_preserved=True,
    program=program,full_training_authorized=False,automatic_retry=False))
for command in [['supervisorctl','reread'],['supervisorctl','update',program],['supervisorctl','start',program]]:
    p=subprocess.run(command,capture_output=True,text=True,check=True,timeout=30)
    print(p.stdout.strip(),flush=True)
print('FRESH_ONE_SHOT_GATE_SERVICE_STARTED',program,flush=True)
'''.replace('REMOTE_ROOT',repr(REMOTE)).replace('PROGRAM_NAME',repr('boundary_gate_20261024_184247'))
    (OUT/'DEPLOY_HOST.py').write_text(deploy,encoding='utf-8')
    write('OPS_MANIFEST.json',dict(files={p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file()},
        purpose='Operational files/approval only; original79scientific payload files unchanged'))
    print(str(OUT))


if __name__=='__main__':main()
