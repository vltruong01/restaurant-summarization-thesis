"""Teacher-only API credentials. Never read Codex configuration or login files."""
import os
from getpass import getpass
from pathlib import Path

BASE_URL = 'https://www.coletoken.de/v1'
PROVIDER = 'coletoken'
MODEL = 'gpt-6.1-sol'
KEY_VARIABLE = 'STAGE_G_TEACHER_API_KEY'


def key_file():
    """Dedicated teacher secret, outside both the repository and Codex home."""
    default_root = Path(os.environ.get('LOCALAPPDATA') or Path.home() / '.local/share')
    path = Path(os.environ.get('STAGE_G_TEACHER_API_KEY_FILE') or default_root / 'thesis_teacher/api_key.txt').expanduser().resolve()
    repo = Path(__file__).resolve().parents[1]
    forbidden = [repo, Path.home() / '.codex']
    if os.environ.get('CODEX_HOME'):
        forbidden.append(Path(os.environ['CODEX_HOME']).expanduser())
    if any(path.is_relative_to(root.resolve()) for root in forbidden):
        raise ValueError('Teacher key file must be outside the repository and Codex home.')
    return path


def credentials(*, prompt=False):
    """Use a teacher-specific environment variable or a hidden terminal prompt.

    The key stays in process memory. There is intentionally no OPENAI_API_KEY,
    CODEX_HOME, auth.json, or config.toml fallback. Endpoint/model remain those
    authorized for the existing study; authentication does not select a model.
    """
    key = os.environ.get(KEY_VARIABLE, '').strip()
    if not key:
        path = key_file()
        if path.is_file():
            key = path.read_text(encoding='utf-8-sig').strip()
    if not key and prompt:
        key = getpass('Teacher API key (hidden; not saved): ').strip()
    if not key:
        raise ValueError('Teacher key missing: set STAGE_G_TEACHER_API_KEY for this process or use the hidden prompt.')
    if any(c.isspace() for c in key):
        raise ValueError('Teacher key contains whitespace.')
    return BASE_URL, PROVIDER, key


def status():
    """Non-secret, offline readiness information; no key value or fingerprint."""
    return {
        'base_url': BASE_URL,
        'provider': PROVIDER,
        'requested_model': MODEL,
        'key_environment_variable': KEY_VARIABLE,
        'key_present': bool(os.environ.get(KEY_VARIABLE, '').strip()),
        'dedicated_teacher_key_file': str(key_file()),
        'dedicated_teacher_key_file_present': key_file().is_file(),
        'hidden_prompt_available_on_execution': True,
        'reads_or_writes_codex_configuration': False,
        'reads_or_writes_codex_authentication': False,
        'network_calls': 0,
    }
