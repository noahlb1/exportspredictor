"""The test-year rule (ADR 0009): the test year is read once per frozen configuration, and every read is in the ledger."""

import json

import polars as pl
import pytest

from exportspredictor import ledger, reporting
from exportspredictor.ledger import Ledger, TestYearAlreadyRead
from exportspredictor.paths import artefacts_dir
from exportspredictor.pipeline import build_artefacts

METADATA = {
    "version": "v-a",
    "source": "fixture",
    "test_year": 2023,
    "features": ["a", "b"],
    "params": {"max_depth": 8},
    "rounds": 100,
    "model_form": {"form": "blend", "naive_weight": 0.3},
}


def _book(tmp_path, *, override: bool = False, command: str = "python -m exportspredictor.pipeline") -> Ledger:
    return Ledger(path=tmp_path / "ledger.csv", command=command, override=override)


def test_a_first_read_appends_one_entry(tmp_path):
    book = _book(tmp_path)
    ledger.record_read(book, METADATA, read="test-year evaluation")
    rows = ledger.read_ledger(book.path)
    assert rows.height == 1
    row = rows.row(0, named=True)
    assert row["version"] == "v-a"
    assert row["fingerprint"] == ledger.fingerprint(METADATA)
    assert row["read"] == "test-year evaluation"
    assert row["command"] == "python -m exportspredictor.pipeline"
    assert row["override"] is False


def test_a_second_read_of_the_same_configuration_is_refused(tmp_path):
    book = _book(tmp_path)
    ledger.record_read(book, METADATA, read="test-year evaluation")
    rebuilt = METADATA | {"version": "v-b", "created_at": "later", "training_seconds": 12.0}
    with pytest.raises(TestYearAlreadyRead, match="v-a"):
        ledger.record_read(book, rebuilt, read="test-year evaluation")
    assert ledger.read_ledger(book.path).height == 1


def test_an_override_reads_again_and_records_itself(tmp_path):
    ledger.record_read(_book(tmp_path), METADATA, read="test-year evaluation")
    again = _book(tmp_path, override=True, command="python -m exportspredictor.pipeline --read-test-year-again")
    ledger.record_read(again, METADATA | {"version": "v-b"}, read="test-year evaluation")
    rows = ledger.read_ledger(again.path)
    assert rows["override"].to_list() == [False, True]
    assert rows["version"].to_list() == ["v-a", "v-b"]
    assert rows["command"][1].endswith("--read-test-year-again")


def test_a_changed_configuration_is_a_new_version(tmp_path):
    book = _book(tmp_path)
    ledger.record_read(book, METADATA, read="test-year evaluation")
    ledger.record_read(book, METADATA | {"features": ["a", "b", "c"], "version": "v-c"}, read="test-year evaluation")
    assert ledger.read_ledger(book.path).height == 2


def test_the_override_flag_comes_from_the_command_line():
    book = Ledger.for_command("python -m exportspredictor.pipeline", ["--event-backtests", ledger.OVERRIDE_FLAG])
    assert book.override
    assert book.command == "python -m exportspredictor.pipeline --event-backtests --read-test-year-again"
    assert not Ledger.for_command("python -m exportspredictor.pipeline", []).override


def _build(world, tuned, book: Ledger):
    return build_artefacts(
        world, version="ledger", search_results=tuned["search"], ranking_results=tuned["ranking_experiments"],
        test_year_ledger=book,
    )


def test_the_fixture_pipeline_reads_the_test_year_once_per_configuration(fixture_world, fixture_artefacts, tmp_path):
    tuned = fixture_artefacts.metadata
    bundle = _build(fixture_world, tuned, _book(tmp_path, command="build_artefacts"))
    rows = ledger.read_ledger(tmp_path / "ledger.csv")
    assert rows.height == 1
    assert rows["fingerprint"][0] == ledger.fingerprint(bundle.metadata)
    assert "test-year evaluation" in rows["read"][0]

    with pytest.raises(TestYearAlreadyRead):
        _build(fixture_world, tuned, _book(tmp_path, command="build_artefacts"))
    assert ledger.read_ledger(tmp_path / "ledger.csv").height == 1

    again = _build(fixture_world, tuned, _book(tmp_path, override=True, command="build_artefacts --read-test-year-again"))
    assert again.evaluation is not None
    rows = ledger.read_ledger(tmp_path / "ledger.csv")
    assert rows["override"].to_list() == [False, True]
    assert rows["fingerprint"].n_unique() == 1

def test_the_ledger_starts_with_the_three_earlier_reads_each_citing_its_record():
    rows = ledger.read_ledger()
    seeded = rows.head(3)
    assert [v.split(" ")[0] for v in seeded["version"]] == ["v1", "v2", "the"]
    assert seeded["version"][2].startswith("the Blend")
    assert all(record.strip() for record in seeded["record"])
    assert not any(seeded["override"])
    assert seeded["fingerprint"].n_unique() == 3


@pytest.mark.parametrize(
    ("name", "served"), [("v1", "20261117T022550Z"), ("v2", "20261118T042154Z"), ("the Blend", "20261118T185910Z")]
)
def test_each_earlier_read_carries_its_version_s_fingerprint(name, served):
    folder = artefacts_dir() / served
    if not folder.exists():
        pytest.skip(f"No saved version {served} in {artefacts_dir()}")
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    row = ledger.read_ledger().filter(pl.col("version").str.starts_with(name)).row(0, named=True)
    assert row["fingerprint"] == ledger.fingerprint(metadata)

def test_the_notebook_shows_the_ledger_from_the_ledger_file_alone(tmp_path):
    book = _book(tmp_path)
    ledger.record_read(book, METADATA, read="test-year evaluation")
    shown = reporting.test_year_ledger(book.path)
    assert shown.columns == ["date", "version", "fingerprint", "read", "command", "override"]
    assert shown.row(0, named=True)["version"] == "v-a"
    assert reporting.test_year_ledger().height == ledger.read_ledger().height

REACH = {"form": "share", "beta": 4.0, "pseudo_count": 5.0, "lag_years": 1}
FITTED = {**REACH, "reference_year": 2023, "constant": 0.81}


def test_market_reach_settings_are_part_of_the_configuration_fingerprint():
    reached = {**METADATA, "market_reach": REACH, "model_form": {**METADATA["model_form"], "market_reach": FITTED}}

    assert "market_reach" in ledger.FINGERPRINT_KEYS
    assert ledger.fingerprint(reached) != ledger.fingerprint(METADATA)
    for setting, other in (("form", "rank"), ("beta", 2.0), ("pseudo_count", 1.0), ("lag_years", 2)):
        changed = {**reached, "market_reach": {**REACH, setting: other}}
        assert ledger.fingerprint(changed) != ledger.fingerprint(reached), setting


def test_market_reachs_fitted_constant_does_not_change_the_fingerprint():
    reached = {**METADATA, "market_reach": REACH, "model_form": {**METADATA["model_form"], "market_reach": FITTED}}
    refitted = {**FITTED, "constant": 0.8100000001}
    rebuilt = {**reached, "model_form": {**METADATA["model_form"], "market_reach": refitted}}

    assert ledger.fingerprint(rebuilt) == ledger.fingerprint(reached)


def test_a_configuration_without_market_reach_keeps_its_fingerprint():
    # The fingerprint of METADATA as the ledger computed it before Market Reach existed (ticket #122's form).
    assert ledger.fingerprint({**METADATA, "market_reach": None}) == ledger.fingerprint(METADATA)


def test_rebuilt_feature_files_are_a_new_version_and_the_original_files_keep_their_fingerprint():
    """#127: the gravity fill changes the features' values, not their names, so the feature set is fingerprinted."""
    metadata = {"source": "BACI", "test_year": 2023, "features": ["a"], "model_params": {"seed": 0}}

    original = ledger.fingerprint({**metadata, "feature_set": ""})

    assert original == ledger.fingerprint(metadata)  # versions recorded before the key existed
    assert ledger.fingerprint({**metadata, "feature_set": "exporter_count"}) != original
