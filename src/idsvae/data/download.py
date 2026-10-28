"""Streamed reading of the official CSE-CIC-IDS2018 CSV files from the public dataset bucket ("Processed Traffic Data for
ML Algorithms"), without a copy on disk."""
import http.client
import io
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from idsvae import config

BUCKET_URL = "https://cse-cic-ids2018.s3.ca-central-1.amazonaws.com/Processed%20Traffic%20Data%20for%20ML%20Algorithms/"
# One connection to the bucket gives about 0.25 MB/s and 8 parallel ranges about 1.4 MB/s (measured), so more parallel
# ranges raise the throughput. The stream keeps at most 2 x RANGE_WORKERS ranges in flight: 2 x 24 x 32 MB, about 1.5 GB of RAM.
RANGE_BYTES = 32 * 1024 * 1024
RANGE_WORKERS = 24
RANGE_ATTEMPTS = 5
PARTIAL_CONTENT = 206


def require_size(source, size, expected_size):
    if size != expected_size:
        raise OSError(f"{source} has {size} bytes, expected {expected_size}")


def fetch_range(url, start, end):
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    for attempt in range(RANGE_ATTEMPTS):
        if attempt > 0:
            # Exponential backoff gives a throttled or overloaded server time to recover.
            time.sleep(2 ** attempt)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                # A server that ignores the Range header sends the whole file with status 200; a retry gets the same answer.
                if response.status != PARTIAL_CONTENT:
                    raise RuntimeError(f"HTTP status {response.status} for range {start}-{end} of {url}, expected {PARTIAL_CONTENT}")
                body = response.read()
            if len(body) == end - start + 1:
                return body
        except (OSError, http.client.HTTPException):
            if attempt == RANGE_ATTEMPTS - 1:
                raise
    raise OSError(f"incomplete range {start}-{end} of {url}")


def remote_size(url):
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as response:
        return int(response.headers["Content-Length"])


class ParallelRangeStream(io.RawIOBase):
    """A readable stream of one remote file: byte ranges are fetched in parallel, a window ahead, and returned in order."""

    def __init__(self, url, expected_size):
        self.size = remote_size(url)
        require_size(url, self.size, expected_size)
        self.url = url
        self.starts = list(range(0, self.size, RANGE_BYTES))
        self.pool = ThreadPoolExecutor(RANGE_WORKERS)
        self.pending = {}
        self.next_index, self.buffer, self.offset = 0, b"", 0
        self.submit_window()

    def submit_window(self):
        for index in range(self.next_index, min(self.next_index + 2 * RANGE_WORKERS, len(self.starts))):
            if index not in self.pending:
                start = self.starts[index]
                self.pending[index] = self.pool.submit(fetch_range, self.url, start, min(start + RANGE_BYTES, self.size) - 1)

    def readable(self):
        return True

    def readinto(self, target):
        if self.offset == len(self.buffer):
            if self.next_index == len(self.starts):
                return 0
            self.buffer, self.offset = self.pending.pop(self.next_index).result(), 0
            self.next_index += 1
            self.submit_window()
        count = min(len(target), len(self.buffer) - self.offset)
        target[:count] = self.buffer[self.offset:self.offset + count]
        self.offset += count
        return count

    def close(self):
        self.pool.shutdown(cancel_futures=True)
        super().close()


def stream(file_name):
    """A buffered readable stream of one official CSV file of the bucket."""
    url = BUCKET_URL + urllib.parse.quote(file_name)
    return io.BufferedReader(ParallelRangeStream(url, config.SOURCE_FILE_BYTES[file_name]), buffer_size=8 * 1024 * 1024)
