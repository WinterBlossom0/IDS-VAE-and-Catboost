"""File writes that a stopped process cannot leave half done, and the shared GPU lock."""

import contextlib
import fcntl
import json
import os
import tempfile
from pathlib import Path

# mkstemp creates files readable by the owner only; outputs get the usual mode of a new file under the process umask.
UMASK = os.umask(0)
os.umask(UMASK)


@contextlib.contextmanager
def atomic_path(path):
    """Yields a temporary path next to path; when the block ends without an error, the file replaces path in one step."""
    # A restart treats an output file as the marker of a finished job, so a kill during the write must not leave a truncated
    # file. A unique temporary name keeps two writers of the same path apart; fsync makes the rename safe after a power loss.
    path = Path(path)
    handle, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(handle)
    os.chmod(name, 0o666 & ~UMASK)
    temporary = Path(name)
    try:
        yield temporary
        with open(temporary, "rb") as written:
            os.fsync(written.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_atomic(path, payload, **dump_options):
    with atomic_path(path) as temporary:
        temporary.write_text(json.dumps(payload, indent=2, **dump_options))


def write_parquet_atomic(frame, path, **parquet_options):
    with atomic_path(path) as temporary:
        frame.to_parquet(temporary, **parquet_options)


# Environment variable that names the shared GPU lock file; the run scripts set the same default.
GPU_LOCK_VARIABLE = "IDSVAE_GPU_LOCK"
DEFAULT_GPU_LOCK = "/tmp/idsvae-gpu.lock"


@contextlib.contextmanager
def gpu_lock():
    """Holds the shared GPU lock file, so that other programs on the same GPU wait."""
    with open(os.environ.get(GPU_LOCK_VARIABLE) or DEFAULT_GPU_LOCK, "a") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
