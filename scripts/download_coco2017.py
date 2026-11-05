#!/usr/bin/env python3
"""Download official COCO2017 objects with resumable ranges and ZIP checks."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.request
import zipfile

OBJECTS = {
    'annotations_trainval2017.zip': ('annotations/annotations_trainval2017.zip', 252907541),
    'val2017.zip': ('zips/val2017.zip', 815585330),
    'train2017.zip': ('zips/train2017.zip', 19336861798)}
BASE = 'https://s3.amazonaws.com/images.cocodataset.org/'
CHECKSUMS = {'train2017.zip':'cced6f7f71b7629ddf16f17bbcfab6b2',
             'val2017.zip':'442b8da7639aecaf257c1dceb8ba8c80',
             'annotations_trainval2017.zip':'f4bbac642086de4f52a3fdda2de5fa2c'}


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.writing')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def safe_members(archive, names):
    selected = [i for i in archive.infolist() if names(i.filename)]
    for info in selected:
        path = Path(info.filename)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Unsafe ZIP path')
    return selected


def download(root, name, relative, size, workers):
    url = BASE + relative
    with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=30) as response:
        assert int(response.headers['Content-Length']) == size
        etag = response.headers['ETag']
    parts = root / '.downloads' / (name + '.parts'); parts.mkdir(parents=True, exist_ok=True)
    identity = dict(url=url, bytes=size, etag=etag, chunk_bytes=64 << 20)
    identity_file = parts / 'identity.json'
    if identity_file.exists():
        assert json.loads(identity_file.read_text()) == identity
    else:
        write_json(identity_file, identity)
    archive_path = root / '.downloads' / name
    chunk_bytes = identity['chunk_bytes']

    def part(index):
        start, end = index * chunk_bytes, min(size, (index + 1) * chunk_bytes) - 1
        target = parts / f'{index:05d}.part'
        if target.exists() and target.stat().st_size == end - start + 1:
            return target
        temporary = target.with_suffix('.partial')
        for attempt in range(6):
            have = temporary.stat().st_size if temporary.exists() else 0
            if have == end - start + 1:
                temporary.replace(target); return target
            assert 0 <= have < end - start + 1
            request = urllib.request.Request(url, headers={'Range': f'bytes={start + have}-{end}', 'If-Match': etag})
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start + have}-{end}/{size}':
                        raise ValueError('Server did not honor the requested range')
                    with temporary.open('ab') as out:
                        shutil.copyfileobj(response, out, 1 << 20)
                assert temporary.stat().st_size == end - start + 1
                temporary.replace(target); return target
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(min(2 ** attempt, 15))
        raise RuntimeError('Unreachable retry state')

    count = (size + chunk_bytes - 1) // chunk_bytes
    if not archive_path.exists():
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            pending = {pool.submit(part, i) for i in range(count)}
            while pending:
                done, pending = concurrent.futures.wait(pending, timeout=5, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    future.result()
                received = sum(p.stat().st_size for p in parts.glob('*.part')) + sum(p.stat().st_size for p in parts.glob('*.partial'))
                write_json(root / '.downloads' / f'{name}.progress.json', dict(received_bytes=received, total_bytes=size))
        digest, md5 = hashlib.sha256(), hashlib.md5()
        temporary = archive_path.with_suffix('.assembling')
        with temporary.open('wb') as out:
            for i in range(count):
                with (parts / f'{i:05d}.part').open('rb') as source:
                    for data in iter(lambda: source.read(8 << 20), b''):
                        digest.update(data); md5.update(data); out.write(data)
        temporary.replace(archive_path)
    else:
        digest, md5 = hashlib.sha256(), hashlib.md5()
        with archive_path.open('rb') as source:
            for data in iter(lambda: source.read(8 << 20), b''): digest.update(data); md5.update(data)
    assert archive_path.stat().st_size == size
    assert md5.hexdigest() == CHECKSUMS[name], 'Archive differs from the published COCO checksum'
    with zipfile.ZipFile(archive_path) as archive:
        bad = archive.testzip()
        assert bad is None, bad
        folder = 'annotations' if name.startswith('annotations') else name.removesuffix('.zip')
        if not (root / folder).exists():
            destination = root / '.extracting' / name.removesuffix('.zip')
            if destination.exists():
                shutil.rmtree(destination)
            destination.mkdir(parents=True)
            names = (lambda p: p in {'annotations/instances_train2017.json', 'annotations/instances_val2017.json'}) if folder == 'annotations' else (lambda p: p.startswith(folder + '/'))
            selected = safe_members(archive, names)
            archive.extractall(destination, members=selected)
            (destination / folder).replace(root / folder)
        count_files = sum(1 for p in (root / folder).iterdir() if p.is_file())
        assert count_files == (2 if folder == 'annotations' else 5000 if folder == 'val2017' else 118287)
    result = dict(identity, sha256=digest.hexdigest(), md5=md5.hexdigest(), published_md5_verified=True,
        checksum_source='https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.9/ppdet/utils/download.py',zip_crc_passed=True, extracted_files=count_files,
        source_page='https://cocodataset.org/#download', transport_note='Verified TLS S3 path for the same official images.cocodataset.org bucket.')
    write_json(root / '.downloads' / f'{name}.complete.json', result)
    for path in parts.glob('*.part'): path.unlink()
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    assert 1 <= a.workers <= 16
    a.out.mkdir(parents=True, exist_ok=True)
    assert shutil.disk_usage(a.out).free > 65 * (1 << 30), 'At least 65 GiB free required'
    started = time.perf_counter()
    # Make validation images and annotations usable while the larger train archive downloads.
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        pending = {pool.submit(download, a.out, n, relative, size, a.workers if n == 'train2017.zip' else 2): n
                   for n, (relative, size) in OBJECTS.items()}
        results = {}
        for future in concurrent.futures.as_completed(pending):
            name = pending[future]; results[name] = future.result()
            print(name, 'downloaded, CRC checked and extracted', flush=True)
            write_json(a.out / 'download_status.json', dict(completed=results, elapsed_seconds=time.perf_counter()-started))
    write_json(a.out / 'download_complete.json', dict(passed=True, archives=results, elapsed_seconds=time.perf_counter()-started))


if __name__ == '__main__': main()
