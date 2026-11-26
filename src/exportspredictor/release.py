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

import shutil
import tempfile
import zipfile
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlparse

from exportspredictor import downloads
from exportspredictor.artefacts import METADATA_FILE, latest_folder

TAG = "v1.0"
REPOSITORY = "noahlb1/exportspredictor"
RELEASED_VERSION = "20261125T171812Z"
CHECKSUM_SUFFIX = ".sha256"
# GitHub refuses a release asset of 2 GiB or more.
MAX_ASSET_BYTES = 2 * 1024**3


def asset_name(version: str) -> str:
    """The release asset holding artefact `version`."""
    return f"{version}.zip"


def download_url(version: str = RELEASED_VERSION, tag: str = TAG) -> str:
    """The public download URL of `version`'s asset on the release tagged `tag`."""
    return f"https://github.com/{REPOSITORY}/releases/download/{tag}/{asset_name(version)}"


RELEASE_URL = download_url()


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
    checksum = target.with_name(target.name + CHECKSUM_SUFFIX)
    with downloads.writing(checksum) as part:
        part.write_text(f"{downloads.sha256(target)}  {target.name}\n", encoding="utf-8")
    return target


def fetch(url: str, artefacts_root: Path) -> Path:
    """Download the release asset at `url`, check it against `<url>.sha256` and unpack it into `artefacts_root`.

    A version already under `artefacts_root` is skipped without downloading; no other version is touched.
    The version appears only once complete: the zip downloads to a temporary folder (through a `.part`
    file), unpacks there, and is renamed into place. Returns the version folder."""
    root = Path(artefacts_root)
    name = PurePosixPath(unquote(urlparse(url).path)).name
    version = name.removesuffix(".zip")
    target = root / version
    if target.exists():
        return target
    expected = downloads.fetch(url + CHECKSUM_SUFFIX).decode("utf-8").split()[0]
    root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".fetch-{version}-", dir=root))
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
            outside = [entry for entry in opened.namelist() if PurePosixPath(entry).parts[0] != version
                       or ".." in PurePosixPath(entry).parts]
            if outside:
                raise ValueError(f"{name} holds entries outside {version}/: {outside[:3]}")
            opened.extractall(unpacked)
        (unpacked / version).rename(target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return target


def main(argv: Sequence[str] | None = None) -> None:
    """`python -m exportspredictor.release pack|fetch`: see the module docstring."""
    import argparse

    from exportspredictor.paths import artefacts_dir, data_dir

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
        print(f"Wrote {zip_path} and {zip_path.name}{CHECKSUM_SUFFIX}")
    else:
        print(f"Artefact version ready: {fetch(args.url, artefacts_dir())}")


if __name__ == "__main__":
    main()
