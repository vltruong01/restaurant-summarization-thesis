"""Teacher-only launcher: offline check by default, explicit bounded pilot only.

Historical pilot producers stay byte-identical. This launcher replaces their
credential resolver in memory only; prompts, payloads and validators stay intact.
Production teacher generation is not authorized by this launcher.
"""
import argparse
import json

from stage_g_teacher_credentials import MODEL, credentials, status


def execute_d039():
    import run_stage_g_d039_focused_pilot as pilot
    cfg = pilot.read(pilot.CONFIG)
    if cfg['requested_model'] != MODEL:
        raise ValueError('Teacher model differs from the authorized isolated runtime.')
    folder = pilot.ROOT / cfg['output_directory']
    if (folder / 'execution_v1/completion.json').exists():
        raise ValueError('D-039 already completed. Review its results; this launcher will not regenerate it or start production.')
    # Read-only local integrity checks before asking for a credential.
    pre = pilot.read(folder / 'preflight.json')
    for relative, expected in {**pre['input_sha256'], **pre['preserved_original_sha256']}.items():
        if pilot.file_sha(pilot.ROOT / relative) != expected:
            raise ValueError('Protected input changed: ' + relative)
    teacher_credentials = credentials(prompt=True)
    original_resolver = pilot.credentials
    try:
        pilot.credentials = lambda: teacher_credentials
        pilot.run()
    finally:
        pilot.credentials = original_resolver


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--check-config', action='store_true', help='Offline status only; no API calls.')
    modes.add_argument('--execute-d039-pilot', action='store_true', help='Resume only the existing approved bounded pilot, if incomplete.')
    args = parser.parse_args()
    if args.execute_d039_pilot:
        execute_d039()
    else:
        print(json.dumps(status(), indent=2))


if __name__ == '__main__':
    try:
        main()
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
