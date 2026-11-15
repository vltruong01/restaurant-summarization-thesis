"""Check export secrets without misclassifying opaque API ciphertext as a key.

Only the secret-screening view is normalized; actual export bytes stay unchanged.
No network calls and no global Codex configuration/authentication access.
"""
import base64
import json
import re
from functools import lru_cache
from stage_g_teacher_credentials import credentials


@lru_cache(maxsize=1)
def isolated_key_bytes():
    # Read only the already authorized isolated teacher credential, never print it.
    return credentials()[2].encode('utf-8')


def check_export_bytes(data, name):
    key = isolated_key_bytes()
    assert key and key not in data, 'Actual teacher credential in export: ' + name
    pattern = rb'sk-[A-Za-z0-9_-]{20,}'
    if not re.search(pattern, data):
        return
    # JSON API reasoning can contain URL-safe base64 ciphertext with random sk-.
    assert name.startswith('raw_api_responses/') and name.endswith('.json'), 'Possible secret in export: ' + name
    obj = json.loads(data)
    count = 0
    for item in obj.get('output', []):
        if item.get('type') != 'reasoning':
            continue
        encrypted = item.get('encrypted_content')
        if not isinstance(encrypted, str) or not re.search(pattern.decode(), encrypted):
            continue
        assert len(encrypted) >= 256 and re.fullmatch(r'[A-Za-z0-9_+/=-]+', encrypted), 'Unexpected encrypted-content encoding'
        decoded = base64.b64decode(encrypted + '=' * (-len(encrypted) % 4), altchars=b'-_', validate=True)
        assert len(decoded) >= 128 and key not in decoded, 'Invalid encrypted-content export'
        # Screening copy only. The ZIP still receives the original API bytes.
        item['encrypted_content'] = '[opaque validated ciphertext; original export bytes unchanged]'
        count += 1
    assert count > 0, 'Possible secret in export: ' + name
    assert not re.search(pattern, json.dumps(obj, ensure_ascii=False).encode('utf-8')), 'Possible secret in export: ' + name
