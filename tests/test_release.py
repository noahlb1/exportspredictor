"""The release module at its boundary: pack an artefact version, fetch it back from a `file://` URL."""

import hashlib
import io
import zipfile

import pytest
from conftest import FIXTURE_VERSION
from fastapi.testclient import TestClient

from api.main import create_app
from exportspredictor import release
from exportspredictor.artefacts import METADATA_FILE
from exportspredictor.downloads import sha256
from exportspredictor.release import CHECKSUM_SUFFIX

SCENARIO = {"add_agreements": ["IND"]}


@pytest.fixture(scope="module")
def packed(fixture_artefacts_dir, tmp_path_factory):
    """The fixture artefact packed as a release asset: the zip's path."""
    return release.pack(fixture_artefacts_dir, tmp_path_factory.mktemp("release"))


def health_and_scenario_responses(root):
    """The API's /health and Scenario responses when serving the artefacts under `root`."""
    with TestClient(create_app(root)) as client:
        return client.get("/health").json(), client.post("/scenarios", json=SCENARIO).json()


def serve(folder, name, archive, checksum):
    """A release asset `name` holding `archive` (bytes) beside a `.sha256` holding `checksum`: its URL."""
    folder.mkdir()
    (folder / name).write_bytes(archive)
    (folder / (name + CHECKSUM_SUFFIX)).write_bytes(checksum.encode("utf-8"))
    return (folder / name).as_uri()


def zipped(entries):
    """A zip archive holding `entries` ({name: text}), as bytes."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, text in entries.items():
            archive.writestr(name, text)
    return buffer.getvalue()


def test_a_fetched_release_serves_as_the_original_does(fixture_artefacts_dir, packed, tmp_path):
    fetched = release.fetch(packed.as_uri(), tmp_path)

    assert fetched == tmp_path / FIXTURE_VERSION
    assert health_and_scenario_responses(tmp_path) == health_and_scenario_responses(fixture_artefacts_dir)


@pytest.mark.parametrize("checksum", ["", "not-a-hash  x.zip\n"])
def test_a_missing_or_malformed_checksum_fails_clearly_and_unpacks_nothing(packed, tmp_path, checksum):
    url = serve(tmp_path / "served", packed.name, packed.read_bytes(), checksum)
    root = tmp_path / "artefacts"

    with pytest.raises(ValueError, match="SHA-256"):
        release.fetch(url, root)

    assert not root.exists() or list(root.iterdir()) == []


def test_an_archive_without_the_version_folder_fails_clearly_and_unpacks_nothing(tmp_path):
    archive = zipped({f"{FIXTURE_VERSION}/notes.txt": "no metadata"})
    url = serve(tmp_path / "served", f"{FIXTURE_VERSION}.zip", archive, hashlib.sha256(archive).hexdigest())
    root = tmp_path / "artefacts"

    with pytest.raises(ValueError, match=f"no {FIXTURE_VERSION}/{METADATA_FILE}"):
        release.fetch(url, root)

    assert list(root.iterdir()) == []


def test_a_url_that_is_not_a_release_asset_fails_clearly(tmp_path):
    with pytest.raises(ValueError, match=r"\.zip"):
        release.fetch((tmp_path / "20261125T171812Z.tar").as_uri(), tmp_path / "artefacts")


def test_stale_staging_folders_from_an_interrupted_fetch_are_removed(packed, tmp_path):
    stale = tmp_path / f".fetch-{FIXTURE_VERSION}-abc123"
    stale.mkdir()
    (stale / f"{FIXTURE_VERSION}.zip.part").write_bytes(b"half")

    release.fetch(packed.as_uri(), tmp_path)

    assert sorted(p.name for p in tmp_path.iterdir()) == [FIXTURE_VERSION]


def test_a_version_folder_without_metadata_is_not_reported_ready_and_is_kept(packed, tmp_path):
    incomplete = tmp_path / FIXTURE_VERSION
    incomplete.mkdir()
    (incomplete / "baseline.parquet").write_bytes(b"partial")

    with pytest.raises(FileExistsError, match=METADATA_FILE):
        release.fetch(packed.as_uri(), tmp_path)

    assert (incomplete / "baseline.parquet").read_bytes() == b"partial"


def test_a_checksum_mismatch_fails_and_leaves_nothing_behind(packed, tmp_path):
    served = tmp_path / "served"
    served.mkdir()
    altered = served / packed.name
    altered.write_bytes(packed.read_bytes() + b"tail")
    checksum = packed.name + CHECKSUM_SUFFIX
    (served / checksum).write_bytes((packed.parent / checksum).read_bytes())
    root = tmp_path / "artefacts"

    with pytest.raises(release.ChecksumMismatch, match="not the published"):
        release.fetch(altered.as_uri(), root)

    assert list(root.iterdir()) == []


def test_a_version_already_present_is_skipped_without_downloading(packed, tmp_path):
    present = tmp_path / FIXTURE_VERSION
    present.mkdir()
    (present / METADATA_FILE).write_text("kept", encoding="utf-8")
    # A URL with nothing behind it: fetching would fail, so passing shows nothing was downloaded.
    absent = (tmp_path / "nowhere" / packed.name).as_uri()

    assert release.fetch(absent, tmp_path) == present
    assert (present / METADATA_FILE).read_text(encoding="utf-8") == "kept"


@pytest.mark.parametrize("already_present", [False, True])
def test_fetch_leaves_other_versions_untouched(packed, tmp_path, already_present):
    other = tmp_path / "20200101T000000Z"
    other.mkdir()
    (other / METADATA_FILE).write_text("{}", encoding="utf-8")
    if already_present:
        release.fetch(packed.as_uri(), tmp_path)

    release.fetch(packed.as_uri(), tmp_path)

    assert sorted(p.name for p in tmp_path.iterdir()) == ["20200101T000000Z", FIXTURE_VERSION]
    assert (other / METADATA_FILE).read_text(encoding="utf-8") == "{}"


def test_pack_leaves_no_zip_when_the_checksum_file_cannot_be_written(fixture_artefacts_dir, tmp_path):
    # A folder where the checksum file goes: writing it fails.
    (tmp_path / release.checksum_name(FIXTURE_VERSION)).mkdir()

    with pytest.raises(OSError):
        release.pack(fixture_artefacts_dir, tmp_path)

    assert sorted(p.name for p in tmp_path.iterdir()) == [release.checksum_name(FIXTURE_VERSION)]


def test_pack_refuses_an_archive_at_the_size_limit_and_writes_nothing(fixture_artefacts_dir, packed, tmp_path):
    with pytest.raises(ValueError, match="limit"):
        release.pack(fixture_artefacts_dir, tmp_path, max_bytes=packed.stat().st_size)

    assert list(tmp_path.iterdir()) == []


def test_the_checksum_file_names_the_zip_as_sha256sum_writes_it_with_a_bare_newline(packed):
    # Raw bytes: a text read turns `\r\n` into `\n` and hides it, and `sha256sum -c` rejects the `\r`.
    written = (packed.parent / release.checksum_name(FIXTURE_VERSION)).read_bytes()

    assert written == f"{sha256(packed)}  {FIXTURE_VERSION}.zip\n".encode("ascii")


def test_a_checksum_file_with_windows_line_endings_is_accepted(packed, tmp_path):
    root = tmp_path / "artefacts"
    url = serve(tmp_path / "served", packed.name, packed.read_bytes(), f"{sha256(packed)}  {packed.name}\r\n")

    assert release.fetch(url, root) == root / FIXTURE_VERSION


def test_stale_staging_folders_of_another_version_are_left_alone(packed, tmp_path):
    # A concurrent fetch of another version may be using it.
    other = tmp_path / ".fetch-20200101T000000Z-abc123"
    other.mkdir()

    release.fetch(packed.as_uri(), tmp_path)

    assert other.is_dir()


def test_pack_names_a_version(fixture_artefacts, tmp_path):
    from dataclasses import replace

    from exportspredictor import artefacts as store

    root = tmp_path / "artefacts"
    store.save(fixture_artefacts, root)
    store.save(replace(fixture_artefacts, metadata={**fixture_artefacts.metadata, "version": "zz-newer"}), root)

    assert release.pack(root, tmp_path / "out").name == "zz-newer.zip"
    assert release.pack(root, tmp_path / "out", version=FIXTURE_VERSION).name == f"{FIXTURE_VERSION}.zip"


def test_the_command_line_packs_and_fetches(fixture_artefacts_dir, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("EXPORTSPREDICTOR_ARTEFACTS_DIR", str(fixture_artefacts_dir))
    release.main(["pack", "--out", str(tmp_path / "out")])
    zip_path = tmp_path / "out" / f"{FIXTURE_VERSION}.zip"
    monkeypatch.setenv("EXPORTSPREDICTOR_ARTEFACTS_DIR", str(tmp_path / "artefacts"))
    release.main(["fetch", "--url", zip_path.as_uri()])

    assert (tmp_path / "artefacts" / FIXTURE_VERSION / METADATA_FILE).exists()
    assert "Artefact version ready" in capsys.readouterr().out


def test_the_command_line_says_when_a_newer_version_will_be_served_instead(packed, tmp_path, monkeypatch, capsys):
    newer = tmp_path / "zz-newer"
    newer.mkdir()
    (newer / METADATA_FILE).write_text("{}", encoding="utf-8")
    monkeypatch.setenv("EXPORTSPREDICTOR_ARTEFACTS_DIR", str(tmp_path))

    release.main(["fetch", "--url", packed.as_uri()])

    assert "the API will serve the newer zz-newer instead" in capsys.readouterr().out


def test_the_default_url_is_the_pinned_release():
    assert release.RELEASE_URL == (
        "https://github.com/noahlb1/exportspredictor/releases/download/v1.1/20261126T232453Z.zip"
    )
