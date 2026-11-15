"""Package the completed offline investigation; performs no model operations."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, shutil, zipfile
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/stage_i/led_large_contrastive_v2_1_root_cause_review_v1'
EXEC=ROOT/'outputs/stage_i/led_large_exploratory_contrastive_v2_1_execution_v1'
RUN=EXEC/'downloaded/runs/CF_EXPLORATORY_V2_1_DIAGNOSTIC_ONLY/diagnose_pair'


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')


def main():
    cpu=read(OUT/'completed_cpu_check/CPU_PATH_TEST_RESULTS.json')
    raw=read(OUT/'completed_cpu_check/SAVED_GRADIENT_REANALYSIS.json')
    overlap=read(OUT/'VALUE_OVERLAP_CPU_EVIDENCE.json')
    preserved=read(OUT/'completed_cpu_check/PRESERVATION_BEFORE.json')['files']
    assert all(sha(ROOT/r)==d for r,d in preserved.items())
    binary=ROOT/'outputs/cloud_handoff/LED_LARGE_DECODER_POLICY_V2_WORKSPACE/assets/models/led-large-16384/pytorch_model.bin'
    assert sha(binary)=='b31752450099a5eb3fcd5199001c5a6831e4a32e13f17a287a5aa6e10801d081'
    checks={
        'preserved_153_files_unchanged':len(preserved)==153,
        'original_backbone_binary_hash_unchanged':True,
        'saved_failure_counts_recomputed':raw['summary']['failing_tensors']==106 and raw['summary']['failing_elements']==63611,
        'saved_native_delegated_all_finite':raw['summary']['all_finite'],
        'saved_all_240_decoder_gradients_bit_exact':all(t['bit_exact'] for t in raw['per_tensor'] if '.decoder.' in t['name']),
        'CPU_extra_CE_vs_delegated_456_exact':cpu['comparisons'][0]['bit_exact_tensors']==456,
        'CPU_native_repeat_456_exact':cpu['comparisons'][1]['bit_exact_tensors']==456,
        'CPU_loss_reachable_graphs_equal':all(c['graph_equal'] for c in cpu['comparisons']),
        'CPU_dropout_and_RNG_traces_equal':all(c['dropout_execution_trace_equal'] and c['RNG_end_equal'] for c in cpu['comparisons']),
        'CPU_all_24_layers_forward_plus_recompute':all(len(c['layer_forward_and_recompute_counts'])==24 and set(c['layer_forward_and_recompute_counts'].values())=={2} for c in cpu['cases']),
        'CPU_parameters_unchanged_no_updates':cpu['model_weights_unchanged'] and cpu['successful_optimizer_updates']==0,
        'actual_CPU_value_backward_index_add_observed':overlap['index_add_backward_observed']==1,
        'value_three_way_overlap_and_final_encoder_localization':overlap['storage_alias_multiplicity_histogram']=={'2':1024,'3':3072} and overlap['real_GPU_final_layer_remaining_bit_exact']==16,
    }
    assert all(checks.values())
    write(OUT/'EVIDENCE_CHECK_RESULTS.json',dict(checks=checks,passed=len(checks),total=len(checks),
        description='Evidence and CPU regression checks, not 13 independent full-model unit tests',
        no_CUDA_qualification=True))
    # Preserve exact, numbered local source excerpts rather than changing dependencies.
    installed=read(OUT/'completed_cpu_check/LOCAL_RUNTIME_SOURCE_RECEIPT.json')['files']
    source_specs=[
        (ROOT/'scripts/run_led_large_exploratory_contrastive_diagnostic_v2_1.py',[(115,185)]),
        (ROOT/'scripts/led_large_contrastive_sequential_v1.py',[(140,156)]),
    ]
    for file in installed:
        p=Path(file)
        source_specs.append((p,[(372,390),(496,538),(1845,1875),(2107,2130),(2388,2435)] if p.name=='modeling_led.py' else [(980,990),(1158,1205)]))
    excerpts=[]
    for p,ranges in source_specs:
        lines=p.read_text(encoding='utf-8').splitlines()
        excerpts.append(dict(path=str(p),sha256=sha(p),excerpts=[dict(first_line=a,last_line=min(b,len(lines)),
            text='\n'.join(f'{i+1}: {lines[i]}' for i in range(a-1,min(b,len(lines))))) for a,b in ranges]))
    write(OUT/'STATIC_CODE_EVIDENCE.json',dict(excerpts=excerpts,
        primary_public_sources=[
            'https://github.com/pytorch/pytorch/blob/v2.1.0/torch/csrc/autograd/FunctionsManual.cpp#L2956-L2975',
            'https://github.com/pytorch/pytorch/blob/v2.1.0/aten/src/ATen/native/cuda/Indexing.cu#L272-L279',
            'https://github.com/pytorch/pytorch/blob/v2.1.0/aten/src/ATen/native/cuda/Indexing.cu#L792-L845',
            'https://github.com/pytorch/pytorch/blob/v2.1.0/docs/source/notes/randomness.rst'],
        public_source_use='Read-only technical source inspection, no new model data or code execution from the internet'))
    write(OUT/'FINAL_PRESERVATION_RECEIPT.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        original_files_checked=len(preserved),all_unchanged=True,files=preserved,
        original_backbone=dict(path=str(binary.relative_to(ROOT)),sha256=sha(binary)),
        historical_GPU_review_zip_sha256='2fc650edf9720b46f8fd696981c93b690746a7a271cae616180c08ec8070c053',
        conclusion='UNRESOLVED',recommendation='STOP',scientific_patch_proposed=False,
        GPU_authorization_granted=False,remote_connections=0,new_CUDA_forwards=0,new_CUDA_backwards=0,
        successful_optimizer_updates=0,generation_calls=0))
    packet=OUT/'portable_review';packet.mkdir(exist_ok=False)
    def copy(src,rel):
        dst=packet/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
        assert sha(src)==sha(dst)
    for p in OUT.iterdir():
        if p.is_file():copy(p,p.name)
    for p in (OUT/'completed_cpu_check').rglob('*'):
        if p.is_file():copy(p,p.relative_to(OUT))
    for p in RUN.rglob('*'):
        if p.is_file():copy(p,Path('evidence/failed_v2_1')/p.relative_to(RUN))
    for name in ('HOST_PREFLIGHT_PASS.json','DIAGNOSTIC_INVOCATION.json','STATUS.json','FAILURE_CLOSEOUT.json','DIAGNOSTIC.log'):
        copy(EXEC/'downloaded/operations'/name,Path('evidence/operations')/name)
    for p in (EXEC/'downloaded/operations/locked_payload').rglob('*'):
        if p.is_file():copy(p,Path('evidence/locked_payload')/p.relative_to(EXEC/'downloaded/operations/locked_payload'))
    for name in ('run_led_large_exploratory_contrastive_diagnostic_v2_1.py',
        'led_large_contrastive_diagnostic_evidence_v2_1.py','led_large_contrastive_sequential_v1.py',
        'led_large_decoder_label_view_v1.py','stage_hi_led_data.py','stage_i_paired_pilot.py',
        'investigate_contrastive_v2_1_cpu_v1.py',
        'investigate_led_value_overlap_cpu_v1.py','package_contrastive_root_cause_review_v1.py'):
        copy(ROOT/'scripts'/name,Path('scripts')/name)
    q=Path('outputs/stage_i/led_large_contrastive_pair_qualification_v1')
    for rel in ('positives/EzJI5x25U5GgbYBWhu5ZQQ.json','pairs/P02.json'):
        copy(ROOT/q/rel,q/rel)
    write(packet/'REPRODUCE_CPU_CHECKS.json',dict(
        requirements='torch 2.1.0, transformers 4.51.3, peft 0.15.2; CPU only. No pretrained model download.',
        full_workspace_command='.venv-caption-pilot/Scripts/python.exe scripts/investigate_contrastive_v2_1_cpu_v1.py --output <NEW_EMPTY_OUTPUT_DIRECTORY>',
        portable_note='Run REPRODUCE_PORTABLE_CPU.py from the extracted packet. Does not run the frozen GPU runner.',
        expected_raw_tensors=456,expected_failed_tensors=106,expected_failed_elements=63611,
        expected_CPU_fixture_bit_exact_tensors=456,
        limitation='Random reduced-width CPU model cannot establish full pretrained CUDA FP16 equivalence.'))
    portable_script='''"""Optional CPU-only reproduction from this extracted review packet."""
from pathlib import Path
import os,sys,json
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'scripts'))
import torch
from unittest.mock import patch
import investigate_contrastive_v2_1_cpu_v1 as audit
audit.ROOT=root;audit.RUN=root/'evidence/failed_v2_1'
torch.set_num_threads(2)
def forbidden(*a,**k):raise AssertionError('CUDA prohibited')
with patch.object(torch.cuda,'_lazy_init',forbidden),patch.object(torch.Tensor,'cuda',forbidden):
    raw=audit.raw_gradient_analysis(torch)
    fixture=audit.cpu_paths(torch)
print(json.dumps(dict(raw=raw['summary'],CPU_fixture=fixture['comparisons']),indent=2))
'''
    with (packet/'REPRODUCE_PORTABLE_CPU.py').open('x',encoding='utf-8') as f:f.write(portable_script)
    entries={p.relative_to(packet).as_posix():sha(p) for p in sorted(packet.rglob('*')) if p.is_file()}
    write(packet/'SHA256_MANIFEST.json',dict(files=entries))
    archive=ROOT/'outputs/cloud_handoff/LED_LARGE_CONTRASTIVE_V2_1_ROOT_CAUSE_V1_OFFLINE_REVIEW.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(packet.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(packet).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert all(hashlib.sha256(z.read(rel)).hexdigest()==digest for rel,digest in entries.items())
    with archive.with_suffix('.zip.sha256').open('x',encoding='utf-8') as f:f.write(sha(archive)+'  '+archive.name+'\n')
    receipt=dict(archive=str(archive.relative_to(ROOT)),sha256=sha(archive),bytes=archive.stat().st_size,
        portable_manifest_entries=len(entries),portable_manifest_pass=True,evidence_checks_passed=len(checks),
        result='UNRESOLVED / STOP',GPU_execution=False)
    write(OUT/'DELIVERY_RECEIPT.json',receipt)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
