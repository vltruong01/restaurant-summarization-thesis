"""Optional CPU-only download. Hash-check binary; never instantiate a model or GPU."""
import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

REVISION = 'cd59d11c3528415b7dda4dfc95cc8f138aceda2e'
DIGEST = 'b31752450099a5eb3fcd5199001c5a6831e4a32e13f17a287a5aa6e10801d081'
SIZE = 1839633783


def verify_file(p, expected=DIGEST, size=SIZE):
    if p.stat().st_size != size:
        raise ValueError('Checkpoint byte count mismatch; stop')
    with p.open('rb') as f:
        actual = hashlib.file_digest(f, 'sha256').hexdigest()
    if actual != expected:
        raise ValueError('Checkpoint checksum mismatch; stop')
    return actual


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--download', action='store_true', help='Explicitly fetch the one pinned public checkpoint')
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    dest = root / 'assets/models/led-large-16384'
    dest.mkdir(parents=True, exist_ok=True)
    weight = dest / 'pytorch_model.bin'
    if not weight.exists():
        if not a.download:
            raise SystemExit('Checkpoint not bundled. Pass --download for CPU-only public download; no GPU authorization implied.')
        partial = dest / 'pytorch_model.bin.partial'
        if partial.exists():
            raise SystemExit('Prior partial download exists; preserve and inspect it. No automatic retry.')
        url = f'https://huggingface.co/allenai/led-large-16384/resolve/{REVISION}/pytorch_model.bin'
        with urllib.request.urlopen(url, timeout=120) as response, partial.open('xb') as output:
            shutil.copyfileobj(response, output, length=1024*1024)
        verify_file(partial)
        partial.rename(weight)
    digest = verify_file(weight)
    metadata = root / 'outputs/stage_i/led_large_preparation_v1/remote_metadata/official'
    for name in ('config.json', 'generation_config.json'):
        source = metadata / name
        target = dest / name
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise ValueError('Existing model metadata differs; stop')
        if not target.exists():
            shutil.copy2(source, target)
    print(json.dumps(dict(binary_sha256=digest, bytes=SIZE, revision=REVISION,
                         binary_bytes_verified=True, tensor_values_verified=False,
                         GPU_operations=0, model_loaded=False, training_authorized=False), indent=2))


if __name__ == '__main__':
    main()
