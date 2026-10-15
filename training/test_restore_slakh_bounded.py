import hashlib
import io
import tarfile

import pytest

import restore_slakh_bounded as subject


def archive(path, entries):
    with tarfile.open(path, 'w:gz') as stream:
        for name, data, kind in entries:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.type = kind
            stream.addfile(info, io.BytesIO(data))


def test_only_hash_verified_fitting_files_are_extracted(tmp_path):
    source = tmp_path / 'source.tar.gz'
    archive(source, [('babyslakh_16k/Track00001/a', b'original', tarfile.REGTYPE),
                     ('babyslakh_16k/Track00015/a', b'reserved', tarfile.REGTYPE)])
    target = tmp_path / 'data'
    name = 'babyslakh_16k/Track00001/a'
    result = subject.extract(source, target, {name: hashlib.sha256(b'original').hexdigest()})
    assert len(result) == 1 and (target / name).read_bytes() == b'original'
    assert not (target / 'babyslakh_16k/Track00015').exists()


@pytest.mark.parametrize('name,kind', [('../escape', tarfile.REGTYPE),
                                      ('/absolute', tarfile.REGTYPE),
                                      ('link', tarfile.SYMTYPE), ('hard', tarfile.LNKTYPE)])
def test_unsafe_members_rejected_even_outside_whitelist(tmp_path, name, kind):
    source = tmp_path / 'source.tar.gz'
    archive(source, [(name, b'', kind)])
    with pytest.raises(ValueError, match='Unsafe'):
        subject.extract(source, tmp_path / 'data', {})


@pytest.mark.parametrize('case', ['changed', 'missing', 'duplicate', 'oversized', 'total'])
def test_incomplete_or_changed_bytes_fail_closed(tmp_path, monkeypatch, case):
    source = tmp_path / 'source.tar.gz'
    rows = [('file', b'abc', tarfile.REGTYPE)]
    expected = {'file': hashlib.sha256(b'abc').hexdigest()}
    if case == 'changed': expected['file'] = '0' * 64
    if case == 'missing': expected['other'] = '0' * 64
    if case == 'duplicate': rows *= 2
    if case == 'oversized': monkeypatch.setattr(subject, 'MAX_MEMBER', 2)
    if case == 'total': monkeypatch.setattr(subject, 'MAX_EXTRACTED', 2)
    archive(source, rows)
    with pytest.raises(ValueError):
        subject.extract(source, tmp_path / 'data', expected)


def test_existing_target_preserved(tmp_path):
    target = tmp_path / 'data'
    target.mkdir()
    with pytest.raises(ValueError, match='Preserve'):
        subject.extract(tmp_path / 'missing', target, {})
