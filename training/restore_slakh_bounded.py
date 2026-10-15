"""Restore the original BabySlakh fitting files, never reserved audio or scores."""

import hashlib
import json
import shutil
import tarfile
from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen

SIZE = 882818115
MD5 = '311096dc2bde7d61c97e930edbfc7f78'
URL = 'https://zenodo.org/api/records/4603870/files/babyslakh_16k.tar.gz/content'
BLOCK = 1024 * 1024
MAX_MEMBER = 256 * BLOCK
MAX_EXTRACTED = 3_000_000_000


def extract(archive, destination, expected):
    """Stream only the original fitting whitelist; verify each raw file hash."""
    if destination.exists() or destination.is_symlink():
        raise ValueError('Preserve existing extraction')
    destination.mkdir()
    seen, total, records = set(), 0, []
    with tarfile.open(archive, 'r|gz') as source:
        for member in source:
            path = PurePosixPath(member.name)
            if path.is_absolute() or '..' in path.parts or member.issym() or member.islnk():
                raise ValueError('Unsafe archive member')
            name = str(path)
            if name not in expected:
                continue
            if name in seen or not member.isfile() or not 0 < member.size <= MAX_MEMBER:
                raise ValueError('Duplicate or invalid fitting member')
            seen.add(name)
            total += member.size
            if total > MAX_EXTRACTED:
                raise ValueError('Extraction exceeds reserved storage')
            target = destination / path
            target.parent.mkdir(parents=True, exist_ok=True)
            digest, count = hashlib.sha256(), 0
            stream = source.extractfile(member)
            with target.open('xb') as output:
                while block := stream.read(BLOCK):
                    count += len(block)
                    if count > member.size:
                        raise ValueError('Oversized member')
                    output.write(block)
                    digest.update(block)
            if count != member.size or digest.hexdigest() != expected[name]:
                raise ValueError('Changed original fitting bytes')
            records.append({'path': name, 'bytes': count, 'sha256': digest.hexdigest()})
    if seen != set(expected):
        raise ValueError('Missing original fitting files')
    return records


def restore(production, root):
    from approved_model_finetune import storage_preflight
    production, root = Path(production), Path(root)
    evidence = production / 'training/results/slakh-training-v1'
    plan = json.loads((evidence / 'plan.json').read_text())
    expected = {'babyslakh_16k/' + identity + '/' + relative: digest
                for identity, files in plan['source_hashes'].items()
                for relative, digest in files.items()}
    if (len(expected) != 389 or len(plan['source_hashes']) != 16
            or any(plan['project_partition'][identity] not in ('train', 'validation')
                   for identity in plan['source_hashes'])):
        raise ValueError('Changed original fitting source inventory')
    target = root / 'babyslakh'
    if target.exists():
        raise ValueError('Preserve existing source restoration')
    links = {'left-precision-v43-replay/training/results': production / 'training/results'}
    print(json.dumps({'storage': storage_preflight(root, 5_000_000_000, audited_links=links)}), flush=True)
    target.mkdir()
    with urlopen('https://zenodo.org/api/records/4603870', timeout=60) as response:
        publisher = json.loads(response.read(2_000_001))
    if (publisher['id'] != 4603870 or publisher['metadata']['license']['id'] != 'cc-by-4.0'
            or not any(r['key'] == 'babyslakh_16k.tar.gz' and r['size'] == SIZE
                       and r['checksum'] == 'md5:' + MD5 for r in publisher['files'])):
        raise ValueError('Changed original publisher grant or object')
    (target / 'record.json').write_text(json.dumps(publisher, indent=2) + '\n')
    archive = target / 'original.tar.gz.partial'
    count, md5, sha = 0, hashlib.md5(), hashlib.sha256()
    with urlopen(Request(URL, headers={'Accept-Encoding': 'identity'}), timeout=60) as response:
        length = response.headers.get('Content-Length')
        if response.status != 200 or (length is not None and int(length) != SIZE):
            raise ValueError('Changed source transfer identity')
        with archive.open('xb') as output:
            while block := response.read(BLOCK):
                count += len(block)
                if count > SIZE:
                    raise ValueError('Oversized source transfer')
                output.write(block)
                md5.update(block)
                sha.update(block)
                if count % (128 * BLOCK) < len(block):
                    print(json.dumps({'downloaded_bytes': count, 'total': SIZE}), flush=True)
    if count != SIZE or md5.hexdigest() != MD5:
        raise ValueError('Original archive checksum mismatch')
    rows = extract(archive, target / 'data', expected)
    # Empty reserved directories preserve the publisher identity inventory only;
    # no reserved waveform, MIDI or label is extracted or inferred.
    for identity in plan['project_partition']:
        (target / 'data/babyslakh_16k' / identity).mkdir(exist_ok=True)
    for name in ('duplicates.json', 'redux.json', 'checksum.json', 'rights.json'):
        shutil.copy2(evidence / name, target / name)
    result = {'passes': True, 'record': 4603870, 'license': 'cc-by-4.0',
              'archive': {'bytes': count, 'md5': md5.hexdigest(), 'sha256': sha.hexdigest()},
              'original_fitting_files': rows, 'reserved_source_files_extracted': 0,
              'no_model_inference': True, 'no_user_audio_or_scores': True,
              'archive_removed_after_verification': True}
    (target / 'restoration.json').write_text(json.dumps(result, indent=2) + '\n')
    archive.unlink()
    print(json.dumps({k: v for k, v in result.items() if k != 'original_fitting_files'}), flush=True)


if __name__ == '__main__':
    restore(Path('/src'), Path('/training'))
