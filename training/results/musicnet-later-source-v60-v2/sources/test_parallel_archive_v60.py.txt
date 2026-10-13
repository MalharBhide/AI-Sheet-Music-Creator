import io
import re

import parallel_archive_v60 as transport
import pytest


class Response:
    def __init__(self, data, first, last, total, mutation=None):
        self.status = 200 if mutation == "status" else 206
        self.headers = {"Content-Range": f"bytes {first}-{last}/{total}",
                        "Content-Length": str(last-first+1)}
        if mutation == "offset":
            self.headers["Content-Range"] = f"bytes {first+1}-{last}/{total}"
        elif mutation == "total":
            self.headers["Content-Range"] = f"bytes {first}-{last}/{total+1}"
        elif mutation == "length":
            self.headers["Content-Length"] = str(last-first)
        self.stream = io.BytesIO(data[first:last+1] if mutation != "truncated" else data[first:last])

    def read(self, size):
        return self.stream.read(size)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stream.close()


def fake_publisher(monkeypatch, data, mutation=None):
    requested = []

    def fetch(request, timeout):
        assert timeout == 60 and request.get_header("Accept-encoding") == "identity"
        first, last = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", request.get_header("Range")).groups())
        requested.append((first, last))
        return Response(data, first, last, len(data), mutation)

    monkeypatch.setattr(transport.urllib.request, "urlopen", fetch)
    return requested


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_ranges_reassemble_exact_original_bytes_in_order_across_read_boundaries(monkeypatch, workers):
    data = bytes(range(137))
    requested = fake_publisher(monkeypatch, data)
    with transport.ParallelArchive("https://publisher.example/archive", len(data), workers, 16) as source:
        output = [source.read(3), source.read(31), source.read(1), source.read(1000), source.read(1)]
        assert b"".join(output) == data and output[-1] == b""
        assert source.position == len(data)
    assert sorted(requested) == [(start, min(start+15, len(data)-1)) for start in range(0, len(data), 16)]


@pytest.mark.parametrize("mutation", ["status", "offset", "total", "length", "truncated"])
def test_wrong_partial_response_cannot_enter_the_verified_stream(monkeypatch, mutation):
    data = bytes(range(128))
    fake_publisher(monkeypatch, data, mutation)
    with transport.ParallelArchive("https://publisher.example/archive", len(data), 4, 16) as source:
        with pytest.raises(ValueError, match="range"):
            source.read(10)


@pytest.mark.parametrize("changes", [{"total_bytes": 0}, {"workers": 0}, {"workers": 5},
    {"chunk_bytes": 0}, {"chunk_bytes": transport.MAX_CHUNK+1}])
def test_unbounded_transport_configuration_rejected(changes):
    options = {"total_bytes": 10, "workers": 4, "chunk_bytes": 16}
    options.update(changes)
    with pytest.raises(ValueError, match="configuration"):
        transport.ParallelArchive("https://publisher.example/archive", **options)


def test_unbounded_reads_and_reads_after_close_cannot_allocate_or_request_data(monkeypatch):
    requested = fake_publisher(monkeypatch, b"abc")
    source = transport.ParallelArchive("https://publisher.example/archive", 3, 4, 16)
    with pytest.raises(ValueError, match="bounded"):
        source.read()
    with pytest.raises(ValueError, match="bounded"):
        source.read(transport.MAX_READ+1)
    assert requested == []
    source.close()
    with pytest.raises(ValueError, match="open"):
        source.read(1)


def test_pending_chunk_buffers_remain_bounded_to_the_prefetch_window(monkeypatch):
    data = bytes(range(128))
    fake_publisher(monkeypatch, data)
    with transport.ParallelArchive("https://publisher.example/archive", len(data), 4, 16) as source:
        while source.read(1):
            assert len(source.pending)+(bool(source.block)) <= 4
