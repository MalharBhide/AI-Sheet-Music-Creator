"""Large source files cannot be promoted with wrong bytes or unbounded writes."""

import hashlib
import io

import pytest

import restore_tinysol_bounded as source


class Response:
    def __init__(self, data, status=200, length=None):
        self.status = status
        self.headers = {} if length is None else {"Content-Length": str(length)}
        self.stream = io.BytesIO(data)

    def read(self, size):
        assert size == source.BLOCK
        return self.stream.read(size)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stream.close()


def identity(data):
    return {"bytes": len(data), "md5": hashlib.md5(data).hexdigest(),
            "sha256": hashlib.sha256(data).hexdigest()}


def publisher(monkeypatch, data, **kwargs):
    def fetch(request, timeout):
        assert request.get_header("Accept-encoding") == "identity" and timeout == 60
        return Response(data, **kwargs)
    monkeypatch.setattr(source.urllib.request, "urlopen", fetch)


def test_exact_bytes_promote_with_or_without_content_length(tmp_path, monkeypatch):
    data = b"original bytes" * 100
    for index, length in enumerate((None, len(data))):
        publisher(monkeypatch, data, length=length)
        path = tmp_path / str(index)
        assert source.download("https://publisher.invalid/file", path, identity(data)) == identity(data)
        assert path.read_bytes() == data and not path.with_name(path.name + ".partial").exists()


@pytest.mark.parametrize("mutation", ["truncated", "oversized", "checksum", "status", "length"])
def test_failed_transfer_never_promotes_a_source(tmp_path, monkeypatch, mutation):
    data = b"original"
    delivered = data[:-1] if mutation == "truncated" else data + b"x" if mutation == "oversized" else data
    expected = identity(data)
    if mutation == "checksum":
        expected["sha256"] = "0" * 64
    publisher(monkeypatch, delivered, status=404 if mutation == "status" else 200,
              length=1 if mutation == "length" else None)
    path = tmp_path / "archive"
    with pytest.raises(ValueError):
        source.download("https://publisher.invalid/file", path, expected)
    assert not path.exists()


@pytest.mark.parametrize("suffix", ["", ".partial"])
def test_existing_evidence_cannot_be_overwritten(tmp_path, suffix):
    path = tmp_path / "archive"
    existing = tmp_path / ("archive" + suffix)
    existing.write_bytes(b"preserve")
    with pytest.raises(ValueError, match="Preserve existing"):
        source.download("https://publisher.invalid/file", path, identity(b"a"))
    assert existing.read_bytes() == b"preserve"


def test_bad_identity_rejected_without_network_or_files(tmp_path):
    expected = identity(b"a")
    expected["bytes"] = True
    with pytest.raises(ValueError, match="identity"):
        source.download("https://publisher.invalid/file", tmp_path / "archive", expected)
    assert not list(tmp_path.iterdir())
