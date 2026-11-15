"""Private, resumable backup of data/; verify remote archive/file bytes; never delete sources."""
from pathlib import Path
import argparse
import ctypes
import datetime
import gzip
import hashlib
import io
import json
import os
import shutil
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/hf_dataset_backup_20261008'
STAGE = OUT / 'upload'
REPO = 'vltruong01/multimodal_summarization_thesis'
MANIFEST = STAGE / 'backup_manifest.json'


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.writing')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def status(phase, **values):
    write_json(OUT / 'progress.json', {'phase': phase, 'updated_at_utc': now(),
               'pid': os.getpid(), 'repo_id': REPO, 'sources_deleted': False, **values})


def sha_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class HashReader:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()
        self.bytes = 0

    def read(self, size=-1):
        block = self.stream.read(size)
        self.digest.update(block)
        self.bytes += len(block)
        return block


class RemoteReader(io.RawIOBase):
    """Sequential remote bytes, with no downloaded archive/cache copy on disk."""
    def __init__(self, iterator):
        self.iterator = iter(iterator)
        self.buffer = b''
        self.digest = hashlib.sha256()
        self.bytes = 0

    def readable(self):
        return True

    def read(self, size=-1):
        if size < 0:
            raise ValueError('Unbounded remote reads are disabled')
        result = bytearray()
        while len(result) < size:
            if not self.buffer:
                try:
                    self.buffer = next(self.iterator)
                except StopIteration:
                    break
                self.digest.update(self.buffer)
                self.bytes += len(self.buffer)
            take = min(size - len(result), len(self.buffer))
            result.extend(self.buffer[:take])
            self.buffer = self.buffer[take:]
        return bytes(result)


def inventory():
    status('inventory')
    files = []
    for base, dirs, names in os.walk(ROOT / 'data', followlinks=False):
        for name in dirs + names:
            p = Path(base) / name
            if p.is_symlink() or (getattr(p.lstat(), 'st_file_attributes', 0) & 0x400):
                raise RuntimeError('Reparse point inside source data; stopped')
        for name in names:
            p = Path(base) / name
            s = p.stat()
            files.append({'path': p.relative_to(ROOT).as_posix(),
                          'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns})
    return sorted(files, key=lambda r: r['path'])


def make_groups(files):
    groups = []
    raw = [r for r in files if len(Path(r['path']).parts) == 2]
    for r in raw:
        groups.append(([r], True))
    photos = [r for r in files if r['path'].startswith('data/yelp_photos/photos/')]
    for i in range(0, len(photos), 10000):
        groups.append((photos[i:i + 10000], False))
    rest = [r for r in files if r not in raw and not r['path'].startswith('data/yelp_photos/photos/')]
    group = []
    size = 0
    for r in rest:
        if group and size + r['bytes'] > 1024 ** 3:
            groups.append((group, True))
            group, size = [], 0
        group.append(r)
        size += r['bytes']
    if group:
        groups.append((group, True))
    return groups


def pack():
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
        for archive in manifest['archives']:
            p = STAGE / archive['name']
            if p.stat().st_size != archive['bytes'] or sha_file(p) != archive['sha256']:
                raise RuntimeError('Existing backup archive changed')
        return manifest
    files = inventory()
    logical = sum(r['bytes'] for r in files)
    if shutil.disk_usage(ROOT).free < logical + 3 * 1024 ** 3:
        raise RuntimeError('Not enough temporary space; no source deleted')
    groups = make_groups(files)
    archives = []
    for index, (rows, compressed) in enumerate(groups, 1):
        name = f'data-{index:03d}.tar' + ('.gz' if compressed else '')
        target = STAGE / 'archives' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        receipt = OUT / 'packed' / (name + '.json')
        if receipt.exists():
            cached = json.loads(receipt.read_text(encoding='utf-8'))
            if ([{k: r[k] for k in ('path', 'bytes', 'mtime_ns')} for r in cached['files']] != rows
                    or sha_file(target) != cached['sha256']):
                raise RuntimeError('Source or packed shard changed; stopped')
            rows = cached['files']
            archive = {k: cached[k] for k in ('name', 'bytes', 'sha256')}
        else:
            status('packing', shard=index, total_shards=len(groups),
                   total_source_files=len(files), total_source_bytes=logical)
            with target.open('wb') as raw_stream:
                stream = gzip.GzipFile(filename='', mode='wb', compresslevel=1,
                                       fileobj=raw_stream, mtime=0) if compressed else raw_stream
                with tarfile.open(fileobj=stream, mode='w|', format=tarfile.GNU_FORMAT) as tar:
                    for r in rows:
                        source = ROOT / r['path']
                        before = source.stat()
                        if before.st_size != r['bytes'] or before.st_mtime_ns != r['mtime_ns']:
                            raise RuntimeError('Source changed before snapshot')
                        info = tarfile.TarInfo(r['path'])
                        info.size, info.mtime, info.mode = r['bytes'], int(before.st_mtime), 0o644
                        with source.open('rb') as inp:
                            reader = HashReader(inp)
                            tar.addfile(info, reader)
                        after = source.stat()
                        if reader.bytes != r['bytes'] or before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
                            raise RuntimeError('Source changed during snapshot')
                        r['sha256'] = reader.digest.hexdigest()
                if compressed:
                    stream.close()
            archive = {'name': 'archives/' + name, 'bytes': target.stat().st_size,
                       'sha256': sha_file(target)}
            write_json(receipt, {**archive, 'files': rows})
        archives.append(archive)
        # Resume receipts own the hashes; refresh the inventory entries too.
        by_path = {r['path']: r for r in rows}
        for r in files:
            if r['path'] in by_path:
                r.update(by_path[r['path']])
    for r in files:
        s = (ROOT / r['path']).stat()
        if s.st_size != r['bytes'] or s.st_mtime_ns != r['mtime_ns']:
            raise RuntimeError('Source inventory changed before publication')
    manifest = {'format': 'private-yelp-data-backup-v1', 'created_at_utc': now(),
                'repository': REPO, 'source_root': 'data/', 'source_files': len(files),
                'source_bytes': logical, 'files': files, 'archives': archives,
                'excludes': ['outputs/', 'assets/', 'environments/'],
                'private_only': True, 'restore_verified': False,
                'original_yelp_terms_included': True}
    write_json(MANIFEST, manifest)
    return manifest


def publish_and_verify(manifest):
    from huggingface_hub import HfApi, hf_hub_url
    from huggingface_hub.utils import build_hf_headers, get_session
    api = HfApi()
    assert api.endpoint.rstrip('/') == 'https://huggingface.co'
    assert api.whoami()['name'] == REPO.split('/')[0]
    api.create_repo(REPO, repo_type='dataset', private=True, exist_ok=True)
    assert api.repo_info(REPO, repo_type='dataset').private is True
    card = '''---
license: other
language:
- en
task_categories:
- summarization
---
# Multimodal summarization thesis: private research dataset

Private research copy of the local `data/` directory: original Yelp JSONs and
photos, processed 2,300-business datasets, English reviews, split files and
preprocessing reports. Original Yelp Dataset Terms of Use are included inside
the archives and apply to all Yelp content and derivatives. This card does not
grant a redistribution license. Do not make this repository public.

Archives preserve original file paths and contents. `backup_manifest.json`
records SHA-256 for every archive and every original file. Verification reads
fresh remote bytes, checks every member without creating a second extracted
dataset, and records the result in `remote_verification.json`.

Until `remote_verification.json` reports PASS, this backup is incomplete.
No model weights, scientific outputs, caption records, credentials or Python
environments are included. Local sources are never deleted by the uploader.
Restore archives into the project root, checking manifest paths/hashes first.
'''
    (STAGE / 'README.md').write_text(card, encoding='utf-8')
    uploaded_path = OUT / 'uploaded.json'
    uploaded = json.loads(uploaded_path.read_text(encoding='utf-8')) if uploaded_path.exists() else {}
    targets = [r['name'] for r in manifest['archives']] + ['backup_manifest.json', 'README.md']
    for i, name in enumerate(targets, 1):
        digest = sha_file(STAGE / name)
        if uploaded.get(name, {}).get('sha256') == digest:
            continue
        assert api.repo_info(REPO, repo_type='dataset').private is True
        status('uploading', file=name, file_number=i, total_upload_files=len(targets),
               uploaded_files=len(uploaded), source_files=manifest['source_files'])
        commit = api.upload_file(path_or_fileobj=STAGE / name, path_in_repo=name,
                                 repo_id=REPO, repo_type='dataset',
                                 commit_message='Private thesis dataset backup: ' + name)
        uploaded[name] = {'sha256': digest, 'commit': commit.oid, 'uploaded_at_utc': now()}
        write_json(uploaded_path, uploaded)
    info = api.repo_info(REPO, repo_type='dataset')
    assert info.private is True
    revision = info.sha
    expected = {r['path']: r for r in manifest['files']}
    verified_path = OUT / 'verified_archives.json'
    verified = json.loads(verified_path.read_text(encoding='utf-8')) if verified_path.exists() else {}
    seen = set()
    for i, archive in enumerate(manifest['archives'], 1):
        name = archive['name']
        cached = verified.get(name)
        if cached and cached['revision'] == revision and cached['sha256'] == archive['sha256']:
            seen.update(cached['files'])
            continue
        status('remote_restore_verification', archive=name, archive_number=i,
               total_archives=len(manifest['archives']), source_files_verified=len(seen),
               revision=revision)
        url = hf_hub_url(REPO, name, repo_type='dataset', revision=revision)
        # httpx strips Authorization when following redirects to a different origin.
        with get_session().stream('GET', url, headers=build_hf_headers(token=True), timeout=120) as response:
            response.raise_for_status()
            stream = RemoteReader(response.iter_bytes(chunk_size=1024 * 1024))
            archive_members = []
            with tarfile.open(fileobj=stream, mode='r|gz' if name.endswith('.gz') else 'r|') as tar:
                for member in tar:
                    if not member.isfile() or member.name not in expected or member.name in seen:
                        raise RuntimeError('Unexpected or duplicate remote archive member')
                    r = expected[member.name]
                    if member.size != r['bytes']:
                        raise RuntimeError('Remote member size mismatch')
                    h = hashlib.sha256()
                    with tar.extractfile(member) as inp:
                        for block in iter(lambda: inp.read(4 * 1024 * 1024), b''):
                            h.update(block)
                    if h.hexdigest() != r['sha256']:
                        raise RuntimeError('Remote member SHA-256 mismatch')
                    seen.add(member.name)
                    archive_members.append(member.name)
            while stream.read(1024 * 1024):
                pass
            if stream.bytes != archive['bytes'] or stream.digest.hexdigest() != archive['sha256']:
                raise RuntimeError('Fresh remote archive SHA-256 mismatch')
        verified[name] = {'revision': revision, 'sha256': archive['sha256'], 'files': archive_members}
        write_json(verified_path, verified)
    if seen != set(expected):
        raise RuntimeError('Incomplete remote restore file set')
    # Verify remote manifest too; do not accept a local-only inventory as a backup.
    with get_session().stream('GET', hf_hub_url(REPO, 'backup_manifest.json', repo_type='dataset', revision=revision),
                              headers=build_hf_headers(token=True), timeout=120) as response:
        response.raise_for_status()
        h = hashlib.sha256()
        for b in response.iter_bytes():
            h.update(b)
    assert h.hexdigest() == sha_file(MANIFEST)
    result = {'status': 'PASS_FULL_DATA_REMOTE_RESTORE_HASH_VERIFIED', 'verified_at_utc': now(),
              'repo_id': REPO, 'url': 'https://huggingface.co/datasets/' + REPO,
              'data_revision': revision, 'private': True, 'source_files_verified': len(seen),
              'source_bytes': manifest['source_bytes'], 'archives_verified': len(manifest['archives']),
              'archive_bytes': sum(r['bytes'] for r in manifest['archives']),
              'raw_and_processed_datasets_and_photos_uploaded': True,
              'sources_deleted': False, 'source_data_still_needed_for_active_thesis': True}
    write_json(OUT / 'remote_verification.json', result)
    api.upload_file(path_or_fileobj=OUT / 'remote_verification.json', path_in_repo='remote_verification.json',
                    repo_id=REPO, repo_type='dataset', commit_message='Record full fresh remote restore verification')
    assert api.repo_info(REPO, repo_type='dataset').private is True
    status('complete', **result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--publish', action='store_true')
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    STAGE.mkdir(parents=True, exist_ok=True)
    os.environ['HF_XET_HIGH_PERFORMANCE'] = '0'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    if os.name == 'nt':
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    try:
        manifest = pack()
        if args.publish:
            publish_and_verify(manifest)
        else:
            status('packed_not_uploaded', source_files=manifest['source_files'])
    except Exception as e:
        # Never log request URLs, signed URLs, tokens or authorization headers.
        status('stopped_with_error', error_type=type(e).__name__,
               http_status=getattr(getattr(e, 'response', None), 'status_code', None),
               local_dataset_preserved=True)
        print('BACKUP_STOPPED', type(e).__name__, flush=True)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
