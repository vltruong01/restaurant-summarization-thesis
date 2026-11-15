"""Portable stdlib-only canonical JSON and hash helpers."""
from pathlib import Path
import json,hashlib
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def file_hash(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
