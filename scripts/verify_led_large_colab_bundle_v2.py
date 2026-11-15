"""Portable v2 full-notebook preflight. No model load, forward or GPU use."""
import argparse
import importlib.metadata
from pathlib import Path
import sys
from verify_led_large_colab_bundle_v1 import verify
from stage_i_led_large_runtime_v2 import read, write, require, check_config, CONFIG, PREP, ROOT, sha


def main():
    p = argparse.ArgumentParser(description=__doc__);p.add_argument('--tokenize',action='store_true')
    p.add_argument('--runtime',action='store_true');p.add_argument('--report',type=Path)
    a = p.parse_args();c = check_config(read(CONFIG))
    if a.runtime:
        require(sys.version_info[:2] == (3,11), 'Python3.11 required')
        require({k:importlib.metadata.version(k) for k in c['runtime_pins']} == c['runtime_pins'], 'Pinned runtime differs')
    for rel,h in read(PREP/'PREFLIGHT_MANIFEST.json')['files'].items():
        require(sha(ROOT/rel) == h, 'Execution dependency changed: '+rel)
    result = verify(ROOT,a.tokenize)
    result.update(training_runner_status='PREPARED_COMPLETE_LARGE_F_RUNNER_GPU_NOT_AUTHORIZED',runtime_pins_checked=a.runtime,
                  GPU_execution_tested=False,large_checkpoint_values_verified=False)
    if a.report:write(a.report,result)
    print(__import__('json').dumps(result,indent=2))


if __name__ == '__main__':
    main()
