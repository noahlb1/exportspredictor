"""Downloading raw sources and writing derived files so that no reader sees half a file.

Every build step writes to `<name>.part` and renames it when complete (`writing`), because
several processes share one data directory. `download` fetches a URL that way, once.
"""

import hashlib
import shutil
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import polars as pl

USER_AGENT = "exportspredictor-data-build"


@contextmanager
def writing(target: Path) -> Iterator[Path]:
    """Yield `<target>.part` to write to; rename it to `target` only if the block completes."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    try:
        yield part
        part.replace(target)
    except BaseException:
        part.unlink(missing_ok=True)
        raise


def write_parquet(frame: pl.DataFrame, target: Path, **options) -> Path:
    """Write a table to `target` through a `.part` file; `options` go to `write_parquet`."""
    with writing(target) as part:
        frame.write_parquet(part, **options)
    return Path(target)


def sha256(path: Path) -> str:
    """The SHA-256 hex digest of the file at `path`, read in 1 MiB blocks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(url: str, *, timeout: float = 300) -> bytes:
    """The body of `url`."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def download(url: str, target: Path, *, timeout: float = 600, replace: bool = False) -> bool:
    """Stream `url` to `target` unless it exists (or `replace`). Returns whether it downloaded."""
    target = Path(target)
    if target.exists() and not replace:
        return False
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with writing(target) as part, urllib.request.urlopen(request, timeout=timeout) as response, part.open("wb") as out:
        shutil.copyfileobj(response, out, length=1 << 20)
    return True
