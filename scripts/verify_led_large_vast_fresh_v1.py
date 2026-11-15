"""Verify the entire Vast handoff, optionally all2200 source/label tokenizations. CPU only."""
import argparse
import json
from pathlib import Path
from verify_led_large_colab_bundle_v1 import sha, read, require, safe_path, verify


def main():
    root=Path(__file__).resolve().parents[1]
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tokenize',action='store_true')
    ap.add_argument('--report',type=Path)
    args=ap.parse_args()
    manifest=read(root/'VAST_SHA256_MANIFEST.json')['files']
    for rel,h in manifest.items():
        require(sha(safe_path(root,rel))==h,'New package hash differs: '+rel)
    result=dict(status='PASS_PACKAGE_INTEGRITY_CPU_ONLY',files_verified=len(manifest),GPU_operations=0)
    if args.tokenize:
        result['data_integrity']=verify(root,True)
    if args.report:
        with args.report.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
