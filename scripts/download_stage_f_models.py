"""Download pinned approved scorers; verify original LFS bytes and exact format conversion."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import time
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/stage_f/validation_calibration_v1'


def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def main():
    os.environ['HF_HUB_DISABLE_XET']='1';os.environ['HF_HUB_ENABLE_HF_TRANSFER']='0'
    from huggingface_hub import snapshot_download
    cfg=yaml.safe_load((ROOT/'configs/stage_f_validation_protocol.yaml').read_text())
    started=time.perf_counter(); snapshots={}
    for role,m in cfg['models'].items():
        print('Downloading pinned',role,m['revision'],flush=True)
        local=ROOT/'assets/model_cache/stage_f_native_models'/m['id'].replace('/','--')/m['revision']
        local.mkdir(parents=True,exist_ok=True)
        blob=ROOT/'assets/model_cache'/('models--'+m['id'].replace('/','--'))/'blobs'/m['weight_sha256']
        target=local/m['weight_file']
        if blob.is_file() and not target.exists() and digest(blob)==m['weight_sha256']:
            os.link(blob,target)  # NTFS hardlink needs no symlink privilege; immutable weight bytes reused.
        path=Path(snapshot_download(m['id'],revision=m['revision'],local_dir=local,
                      allow_patterns=['*.json','*.txt',m['weight_file']],max_workers=2))
        assert digest(path/m['weight_file'])==m['weight_sha256']
        snapshots[role]=path
        print(role,'original LFS SHA verified',flush=True)
    # Official publisher's pinned checksum-verified tensor-only state dict; no custom classes/code.
    import torch
    from safetensors.torch import save_file,load_file
    clip=snapshots['CLIP'];converted=ROOT/'assets/model_cache/verified_conversions/clip-vit-base-patch32'/cfg['models']['CLIP']['revision'];converted.mkdir(parents=True,exist_ok=True)
    for p in clip.iterdir():
        if p.suffix in ['.json','.txt']:shutil.copy2(p,converted/p.name)
    state=torch.load(clip/'pytorch_model.bin',map_location='cpu',weights_only=True)
    assert isinstance(state,dict) and all(isinstance(v,torch.Tensor) for v in state.values())
    if not (converted/'model.safetensors').exists():
        save_file({k:v.contiguous() for k,v in state.items()},str(converted/'model.safetensors'),metadata={'format':'pt'})
    safe=load_file(str(converted/'model.safetensors'))
    assert set(safe)==set(state) and all(safe[k].dtype==v.dtype and safe[k].shape==v.shape and torch.equal(safe[k],v) for k,v in state.items())
    del state,safe
    pooling=json.loads((snapshots['SBERT']/'1_Pooling/config.json').read_text())
    assert pooling['pooling_mode_mean_tokens'] and not pooling['pooling_mode_cls_token'] and not pooling['pooling_mode_max_tokens']
    result={'status':'PINNED_SCORER_FILES_AND_EXACT_CONVERSION_VERIFIED_NOT_INFERENCE',
            'models':cfg['models'],'CLIP_local_safetensors':converted.relative_to(ROOT).as_posix(),
            'CLIP_converted_safetensors_sha256':digest(converted/'model.safetensors'),
            'CLIP_all_tensor_keys_shapes_dtypes_values_equal_original':True,
            'SBERT_local_snapshot':snapshots['SBERT'].relative_to(ROOT).as_posix(),
            'SBERT_pooling_config':pooling,'model_inference_started':False,
            'elapsed_seconds':round(time.perf_counter()-started,3)}
    (OUT/'model_files_report.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
