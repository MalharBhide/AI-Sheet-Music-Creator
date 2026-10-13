"""Bounded ordered byte-range transport; whole-object verification stays external."""

import urllib.request
from concurrent.futures import ThreadPoolExecutor

MAX_CHUNK = 16*1024*1024
MAX_READ = 1024*1024


class ParallelArchive:
    def __init__(self, url, total_bytes, workers=4, chunk_bytes=MAX_CHUNK):
        if (type(total_bytes) is not int or total_bytes <= 0
                or type(workers) is not int or not 1 <= workers <= 4
                or type(chunk_bytes) is not int or not 1 <= chunk_bytes <= MAX_CHUNK):
            raise ValueError("Invalid bounded archive transport configuration")
        self.url, self.total, self.workers, self.chunk_bytes = url, total_bytes, workers, chunk_bytes
        self.chunks = (total_bytes+chunk_bytes-1)//chunk_bytes
        self.executor, self.pending = None, {}
        self.submitted, self.consumed, self.position = 0, 0, 0
        self.block, self.offset, self.closed = b"", 0, False

    def _fetch(self, index):
        first = index*self.chunk_bytes
        last = min(self.total-1, first+self.chunk_bytes-1)
        amount = last-first+1
        request = urllib.request.Request(self.url, headers={
            "Range": f"bytes={first}-{last}", "Accept-Encoding": "identity"})
        with urllib.request.urlopen(request, timeout=60) as response:
            if (response.status != 206
                    or response.headers.get("Content-Range") != f"bytes {first}-{last}/{self.total}"
                    or int(response.headers.get("Content-Length", -1)) != amount):
                raise ValueError("Publisher did not return the exact requested archive range")
            data = response.read(amount+1)
        if len(data) != amount:
            raise ValueError("Truncated or oversized archive range")
        return data

    def _advance(self):
        # Release the consumed chunk before replenishing the window. Pending
        # results plus the current chunk occupy at most four chunk buffers.
        self.block, self.offset = b"", 0
        if self.executor is None:
            self.executor = ThreadPoolExecutor(max_workers=self.workers)
        while len(self.pending) < self.workers and self.submitted < self.chunks:
            self.pending[self.submitted] = self.executor.submit(self._fetch, self.submitted)
            self.submitted += 1
        if self.consumed >= self.chunks:
            return False
        future = self.pending.pop(self.consumed)
        self.block = future.result()
        self.consumed += 1
        return True

    def read(self, size=-1):
        if self.closed or type(size) is not int or not 0 <= size <= MAX_READ:
            raise ValueError("Archive reads must be bounded and transport must be open")
        remaining, parts = min(size, self.total-self.position), []
        while remaining:
            if self.offset == len(self.block) and not self._advance():
                break
            amount = min(remaining, len(self.block)-self.offset)
            parts.append(self.block[self.offset:self.offset+amount])
            self.offset += amount
            self.position += amount
            remaining -= amount
        return b"".join(parts)

    def close(self):
        if not self.closed:
            self.closed = True
            if self.executor is not None:
                self.executor.shutdown(wait=True, cancel_futures=True)
            self.pending.clear()
            self.block = b""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
