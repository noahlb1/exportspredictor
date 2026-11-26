"""The release module at its boundary: pack an artefact version, fetch it back from a `file://` URL."""

import pytest
from conftest import FIXTURE_VERSION
from fastapi.testclient import TestClient

from api.main import create_app
from exportspredictor import release

SCENARIO = {"add_agreements": ["IND"]}


@pytest.fixture(scope="module")
def packed(fixture_artefacts_dir, tmp_path_factory):
    """The fixture artefact packed as a release asset: the zip's path."""
    return release.pack(fixture_artefacts_dir, tmp_path_factory.mktemp("release"))


def answers(root):
    with TestClient(create_app(root)) as client:
        return client.get("/health").json(), client.post("/scenarios", json=SCENARIO).json()


def test_a_fetched_release_serves_as_the_original_does(fixture_artefacts_dir, packed, tmp_path):
    fetched = release.fetch(packed.as_uri(), tmp_path)

    assert fetched == tmp_path / FIXTURE_VERSION
    assert answers(tmp_path) == answers(fixture_artefacts_dir)


def test_a_checksum_mismatch_fails_and_leaves_nothing_behind(packed, tmp_path):
    served = tmp_path / "served"
    served.mkdir()
    altered = served / packed.name
    altered.write_bytes(packed.read_bytes() + b"tail")
    (served / f"{packed.name}.sha256").write_bytes((packed.parent / f"{packed.name}.sha256").read_bytes())
    root = tmp_path / "artefacts"

    with pytest.raises(release.ChecksumMismatch, match="not the published"):
        release.fetch(altered.as_uri(), root)

    assert list(root.iterdir()) == []


def test_a_version_already_present_is_skipped_and_others_are_untouched(packed, tmp_path):
    other = tmp_path / "20200101T000000Z"
    other.mkdir()
    (other / "metadata.json").write_text("{}", encoding="utf-8")
    present = tmp_path / FIXTURE_VERSION
    present.mkdir()
    (present / "marker").write_text("kept", encoding="utf-8")
    # A URL with nothing behind it: fetching would fail, so passing shows nothing was downloaded.
    absent = (tmp_path / "nowhere" / packed.name).as_uri()

    assert release.fetch(absent, tmp_path) == present
    assert (present / "marker").read_text(encoding="utf-8") == "kept"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["20200101T000000Z", FIXTURE_VERSION]
    assert (other / "metadata.json").read_text(encoding="utf-8") == "{}"


def test_fetch_leaves_other_versions_untouched(packed, tmp_path):
    other = tmp_path / "20200101T000000Z"
    other.mkdir()
    (other / "metadata.json").write_text("{}", encoding="utf-8")

    release.fetch(packed.as_uri(), tmp_path)

    assert sorted(p.name for p in tmp_path.iterdir()) == ["20200101T000000Z", FIXTURE_VERSION]
    assert (other / "metadata.json").read_text(encoding="utf-8") == "{}"


def test_pack_refuses_an_archive_at_the_size_limit_and_writes_nothing(fixture_artefacts_dir, packed, tmp_path):
    with pytest.raises(ValueError, match="limit"):
        release.pack(fixture_artefacts_dir, tmp_path, max_bytes=packed.stat().st_size)

    assert list(tmp_path.iterdir()) == []


def test_the_checksum_file_names_the_zip_as_sha256sum_writes_it(packed):
    from exportspredictor.downloads import sha256

    line = (packed.parent / f"{packed.name}.sha256").read_text(encoding="utf-8")

    assert line == f"{sha256(packed)}  {FIXTURE_VERSION}.zip\n"


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

    assert (tmp_path / "artefacts" / FIXTURE_VERSION / "metadata.json").exists()
    assert "Artefact version ready" in capsys.readouterr().out


def test_the_default_url_is_the_pinned_release():
    assert release.RELEASE_URL == (
        f"https://github.com/noahlb1/exportspredictor/releases/download/v1.0/{release.RELEASED_VERSION}.zip"
    )
