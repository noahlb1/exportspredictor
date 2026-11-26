"""The artefact release: one artefact version packed as a GitHub Release asset, and fetched back.

The released version is `RELEASED_VERSION`, attached to the release tagged `TAG` as two assets:

    <version>.zip         the version folder (tables, metadata.json, model_card.md, models/),
                          every entry under `<version>/`, so unzipping it into the artefacts
                          directory gives the folder the API serves
    <version>.zip.sha256  its SHA-256, as `sha256sum` writes it (`<hex>  <version>.zip`)

Command line:

    python -m exportspredictor.release pack [--version V] [--out DIR]
        packs the newest version under the artefacts directory (or V) into DIR
        (default `<data>/release`)
    python -m exportspredictor.release fetch [--url URL]
        downloads the zip at URL (default `RELEASE_URL`) and its `.sha256` beside it, checks it
        and unpacks it into the artefacts directory; a version already there is skipped

Fetch reads the public download URL over HTTPS: no `gh`, no token.
"""

import argparse
import re
import shutil
import tempfile
import zipfile
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlparse

from exportspredictor import downloads
from exportspredictor.artefacts import METADATA_FILE, latest_folder
from exportspredictor.paths import artefacts_dir, data_dir

TAG = "v1.0"
REPOSITORY = "noahlb1/exportspredictor"
RELEASED_VERSION = "20261125T171812Z"
ASSET_SUFFIX = ".zip"
CHECKSUM_SUFFIX = ".sha256"
# GitHub refuses a release asset of 2 GiB or more.
MAX_ASSET_BYTES = 2 * 1024**3
# A fetch's temporary folder under the artefacts directory, named `.fetch-<version>-<random>`.
_STAGING_PREFIX = ".fetch-"
_SHA256 = re.compile(r"[0-9a-f]{64}")


def asset_name(version: str) -> str:
    """The release asset holding artefact `version`."""
    return version + ASSET_SUFFIX


def checksum_name(version: str) -> str:
    """The release asset holding the SHA-256 of `asset_name(version)`."""
    return asset_name(version) + CHECKSUM_SUFFIX


RELEASE_URL = f"https://github.com/{REPOSITORY}/releases/download/{TAG}/{asset_name(RELEASED_VERSION)}"


class ChecksumMismatch(Exception):
    """A downloaded asset whose SHA-256 is not the published one."""


def pack(
    artefacts_root: Path, out_dir: Path, *, version: str | None = None, max_bytes: int = MAX_ASSET_BYTES
) -> Path:
    """Zip artefact `version` (default: the newest) under `artefacts_root` into `out_dir`, with its checksum file.

    Refuses an archive of `max_bytes` or more, writing nothing. Returns the zip's path."""
    folder = latest_folder(artefacts_root) if version is None else Path(artefacts_root) / version
    if not (folder / METADATA_FILE).exists():
        raise FileNotFoundError(f"No artefact version {folder.name} with {METADATA_FILE} under {artefacts_root}")
    target = Path(out_dir) / asset_name(folder.name)
    with downloads.writing(target) as part:
        with zipfile.ZipFile(part, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(folder.rglob("*")):
                if path.is_file():
                    archive.write(path, PurePosixPath(folder.name, *path.relative_to(folder).parts))
        size = part.stat().st_size
        if size >= max_bytes:
            raise ValueError(
                f"{target.name} would be {size:,} bytes, over the {max_bytes:,}-byte limit on a release asset"
            )
    checksum = target.with_name(checksum_name(folder.name))
    try:
        with downloads.writing(checksum) as part:
            # Bytes, so Windows writes no `\r`, which `sha256sum -c` would read as part of the name.
            part.write_bytes(f"{downloads.sha256(target)}  {target.name}\n".encode("ascii"))
    except BaseException:
        target.unlink(missing_ok=True)  # a zip without its checksum file is no release
        raise
    return target


def fetch(url: str, artefacts_root: Path) -> Path:
    """Download the release asset at `url`, check it against `<url>.sha256` and unpack it into `artefacts_root`.

    A version already under `artefacts_root` is skipped without downloading; no other version is touched.
    The version appears only once complete: the zip downloads to a temporary folder (through a `.part`
    file), unpacks there, and is renamed into place. Temporary folders left by an interrupted fetch of the
    same version are removed first. Returns the version folder."""
    root = Path(artefacts_root)
    name = PurePosixPath(unquote(urlparse(url).path)).name
    if not name.endswith(ASSET_SUFFIX) or name == ASSET_SUFFIX:
        raise ValueError(f"{url} is not a release asset: its name must be <version>{ASSET_SUFFIX}")
    version = name.removesuffix(ASSET_SUFFIX)
    target = root / version
    for stale in root.glob(f"{_STAGING_PREFIX}{version}-*") if root.is_dir() else []:
        shutil.rmtree(stale, ignore_errors=True)
    if (target / METADATA_FILE).exists():
        return target
    if target.exists():
        raise FileExistsError(
            f"{target} exists without {METADATA_FILE}, so it is incomplete. Move or delete it, then fetch again."
        )
    published = downloads.fetch(url + CHECKSUM_SUFFIX).decode("utf-8").split()  # any line ending
    expected = published[0] if published else ""
    if not _SHA256.fullmatch(expected):
        raise ValueError(f"{checksum_name(version)} holds no SHA-256 (got {expected[:20]!r}). Nothing was fetched.")
    root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f"{_STAGING_PREFIX}{version}-", dir=root))
    try:
        archive = staging / name
        downloads.download(url, archive)
        found = downloads.sha256(archive)
        if found != expected:
            raise ChecksumMismatch(
                f"{name} has SHA-256 {found}, not the published {expected}: the download is incomplete or "
                "altered. Nothing was unpacked; fetch again."
            )
        unpacked = staging / "unpacked"
        with zipfile.ZipFile(archive) as opened:
            _check_entries(opened.namelist(), name, version)
            opened.extractall(unpacked)
        (unpacked / version).rename(target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return target


def _check_entries(entries: list[str], name: str, version: str) -> None:
    """Refuse archive `name` unless every entry is under `<version>/` and it holds the version's metadata."""
    outside = [entry for entry in entries if (parts := PurePosixPath(entry).parts)[0] != version or ".." in parts]
    if outside:
        raise ValueError(f"{name} holds entries outside {version}/: {outside[:3]}")
    if f"{version}/{METADATA_FILE}" not in entries:
        raise ValueError(f"{name} holds no {version}/{METADATA_FILE}, so it is not an artefact version")


def main(argv: Sequence[str] | None = None) -> None:
    """`python -m exportspredictor.release pack|fetch`: see the module docstring."""
    parser = argparse.ArgumentParser(prog="python -m exportspredictor.release", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    pack_command = commands.add_parser("pack", help="zip an artefact version and write its checksum file")
    pack_command.add_argument("--version", help="the version to pack (default: the newest)")
    pack_command.add_argument("--out", type=Path, help="where to write the assets (default: <data>/release)")
    fetch_command = commands.add_parser("fetch", help="download, check and unpack a released artefact version")
    fetch_command.add_argument("--url", default=RELEASE_URL, help=f"the zip's URL (default: {RELEASE_URL})")
    args = parser.parse_args(argv)
    if args.command == "pack":
        zip_path = pack(artefacts_dir(), args.out or data_dir() / "release", version=args.version)
        print(f"Wrote {zip_path} and {checksum_name(zip_path.stem)}")
    else:
        fetched = fetch(args.url, artefacts_dir())
        print(f"Artefact version ready: {fetched}")
        served = latest_folder(artefacts_dir())
        if served != fetched:
            print(f"Note: the API will serve the newer {served.name} instead.")


if __name__ == "__main__":
    main()
