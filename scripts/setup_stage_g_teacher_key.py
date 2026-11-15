"""Explicit hidden-input setup for a teacher-only key; no network/Codex access."""
from getpass import getpass
from stage_g_teacher_credentials import key_file


def main():
    path = key_file()
    key = getpass('Paste teacher API key (hidden): ').strip()
    if not key or any(c.isspace() for c in key):
        raise SystemExit('Empty key or whitespace in key; nothing saved.')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(key + '\n', encoding='utf-8')
    path.chmod(0o600)
    print('Teacher-only key saved outside Codex and the repository: ' + str(path))
    print('No API request was made. Do not include this secret file in exports.')


if __name__ == '__main__':
    main()
