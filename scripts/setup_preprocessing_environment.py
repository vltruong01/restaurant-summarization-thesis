"""Create the A1 environment without modifying global Python packages."""

from pathlib import Path
import os
import platform
import subprocess
import sys
import venv

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    if platform.python_version() != "3.11.7":
        print("This A1 spec uses Python 3.11.7. Review the spec before changing runtime.")
        return 1
    destination = REPO / ".venv-preprocess"
    runtime = destination / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not runtime.exists():
        venv.EnvBuilder(with_pip=True).create(destination)
    lock = REPO / "requirements/preprocessing.lock.txt"
    source = lock if lock.exists() else REPO / "requirements/preprocessing.in"
    environment = os.environ.copy()
    # Skip inherited package indexes unrelated to this CPU preprocessing environment.
    environment["PIP_CONFIG_FILE"] = os.devnull
    environment["PIP_INDEX_URL"] = "https://pypi.org/simple"
    environment["PIP_EXTRA_INDEX_URL"] = "https://pypi.org/simple"
    environment["PIP_RETRIES"] = "1"
    environment["PIP_TIMEOUT"] = "20"
    log_path = REPO / "logs/a1_setup.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            [str(runtime), "-m", "pip", "install", "-r", str(source)],
            env=environment, stdout=stream, stderr=subprocess.STDOUT,
        )
    print(f"Installation log: {log_path}")
    if result.returncode:
        print("Installation failed; the environment is not validated.")
        return result.returncode
    result = subprocess.run([str(runtime), "-m", "pip", "check"], env=environment)
    if result.returncode:
        return result.returncode
    resolved = subprocess.run(
        [str(runtime), "-m", "pip", "freeze"], env=environment,
        capture_output=True, text=True, check=True,
    ).stdout
    if not lock.exists():
        lock.write_text(resolved, encoding="utf-8")
    else:
        expected = set(lock.read_text(encoding="utf-8").splitlines())
        if set(resolved.splitlines()) != expected:
            print("Runtime differs from the existing lock; review before proceeding.")
            return 1
    print("Dependency installation passed. Run validate_local_environment.py for A1 checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
