"""The shared download and write-then-rename helpers every data build step uses."""

import pytest

from exportspredictor import downloads


def test_download_writes_the_file_once_and_leaves_no_part_file(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"flows")
    target = tmp_path / "raw" / "flows.bin"

    first = downloads.download(source.as_uri(), target)
    source.write_bytes(b"changed")
    second = downloads.download(source.as_uri(), target)

    assert (first, second) == (True, False)
    assert target.read_bytes() == b"flows"
    assert not target.with_name("flows.bin.part").exists()


def test_download_can_replace_an_existing_file(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"new")
    target = tmp_path / "flows.bin"
    target.write_bytes(b"old")

    assert downloads.download(source.as_uri(), target, replace=True) is True
    assert target.read_bytes() == b"new"


def test_a_failed_write_leaves_no_target(tmp_path):
    target = tmp_path / "derived" / "table.parquet"

    with pytest.raises(RuntimeError), downloads.writing(target) as part:
        part.write_bytes(b"half")
        raise RuntimeError("interrupted")

    assert not target.exists()
