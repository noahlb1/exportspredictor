"""The test-year rule (ADR 0009): the test year is read once per frozen configuration, and every read is in the ledger."""

import json

import polars as pl
import pytest

from exportspredictor import ledger, reporting
from exportspredictor.ledger import Ledger, TestYearAlreadyRead
from exportspredictor.paths import artefacts_dir
from exportspredictor.pipeline import build_artefacts, save_version

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


def _completed_read(book: Ledger, metadata: dict) -> None:
    """A read whose figures were saved: recorded, then completed."""
    ledger.record_read(book, metadata, read="test-year evaluation")
    ledger.complete_read(book, ledger.fingerprint(metadata))


def test_a_second_read_of_the_same_configuration_is_refused(tmp_path):
    book = _book(tmp_path)
    _completed_read(book, METADATA)
    rebuilt = METADATA | {"version": "v-b", "created_at": "later", "training_seconds": 12.0}
    with pytest.raises(TestYearAlreadyRead, match="v-a"):
        ledger.record_read(book, rebuilt, read="test-year evaluation")
    assert ledger.read_ledger(book.path).height == 1


def test_an_override_reads_again_and_records_itself(tmp_path):
    _completed_read(_book(tmp_path), METADATA)
    again =_book(tmp_path, override=True, command="python -m exportspredictor.pipeline --read-test-year-again")
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
    book = _book(tmp_path, command="build_artefacts")
    bundle = _build(fixture_world, tuned, book)
    rows = ledger.read_ledger(tmp_path / "ledger.csv")
    assert rows.height == 1
    assert rows["fingerprint"][0] == ledger.fingerprint(bundle.metadata)
    assert "test-year evaluation" in rows["read"][0]
    assert rows["status"][0] == "started"

    save_version(bundle, tmp_path / "artefacts", test_year_ledger=book)
    assert ledger.read_ledger(tmp_path / "ledger.csv")["status"].to_list() == ["completed"]

    with pytest.raises(TestYearAlreadyRead):
        _build(fixture_world, tuned, book)
    assert ledger.read_ledger(tmp_path / "ledger.csv").height == 1

    override = _book(tmp_path, override=True, command="build_artefacts --read-test-year-again")
    again = _build(fixture_world, tuned, override)
    assert again.evaluation is not None
    save_version(again, tmp_path / "again", test_year_ledger=override)
    rows = ledger.read_ledger(tmp_path / "ledger.csv")
    assert rows["override"].to_list() == [False, True]
    assert rows["status"].to_list() == ["completed", "completed"]
    assert rows["fingerprint"].n_unique() == 1


def test_the_fixture_pipeline_re_run_after_a_crash_completes_the_same_read(fixture_world, fixture_artefacts, tmp_path):
    tuned, book = fixture_artefacts.metadata, _book(tmp_path, command="build_artefacts")
    _build(fixture_world, tuned, book)  # the run crashed before saving: its read stays started, with no figures

    rerun = _build(fixture_world, tuned, book)
    folder = save_version(rerun, tmp_path / "artefacts", test_year_ledger=book)

    rows = ledger.read_ledger(tmp_path / "ledger.csv")
    assert rows.height == 1
    row = rows.row(0, named=True)
    assert row["fingerprint"] == ledger.fingerprint(rerun.metadata)
    assert (row["override"], row["status"]) == (False, "completed")
    assert (folder / "evaluation.json").exists()
    with pytest.raises(TestYearAlreadyRead):
        _build(fixture_world, tuned, book)

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
    assert shown.columns == ["date", "version", "fingerprint", "read", "command", "override", "kind", "status"]
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


# --- A read is started, then completed once its figures are saved (review before #128) ---------------------------


def test_a_read_is_started_when_recorded_and_completed_in_the_same_row(tmp_path):
    book = _book(tmp_path)
    ledger.record_read(book, METADATA, read="test-year evaluation")
    assert ledger.read_ledger(book.path)["status"].to_list() == ["started"]

    ledger.complete_read(book, ledger.fingerprint(METADATA))

    rows = ledger.read_ledger(book.path)
    assert rows.height == 1
    assert rows["status"].to_list() == ["completed"]


def test_a_re_run_after_a_crash_completes_the_same_row_without_an_override(tmp_path):
    book = _book(tmp_path)
    ledger.record_read(book, METADATA, read="test-year evaluation")  # crashed: nothing saved, never completed

    ledger.record_read(book, METADATA | {"version": "v-b"}, read="test-year evaluation")
    ledger.complete_read(book, ledger.fingerprint(METADATA))

    rows = ledger.read_ledger(book.path)
    assert rows.height == 1
    row = rows.row(0, named=True)
    assert (row["version"], row["override"], row["status"]) == ("v-b", False, "completed")
    with pytest.raises(TestYearAlreadyRead):
        ledger.record_read(book, METADATA | {"version": "v-c"}, read="test-year evaluation")


def test_completing_a_read_that_was_never_started_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no started read"):
        ledger.complete_read(_book(tmp_path), ledger.fingerprint(METADATA))


def test_a_ledger_written_before_reads_had_a_status_reads_as_completed_evaluations(tmp_path):
    path = tmp_path / "ledger.csv"
    path.write_text(
        "date,version,fingerprint,read,command,override,record\n"
        f"2026-11-18,v2,{ledger.fingerprint(METADATA)},test-year evaluation,python -m x,false,notes\n",
        encoding="utf-8",
    )
    rows = ledger.read_ledger(path)
    assert (rows["kind"][0], rows["status"][0]) == ("evaluation", "completed")
    with pytest.raises(TestYearAlreadyRead):
        ledger.record_read(Ledger(path=path), METADATA, read="test-year evaluation")

    ledger.record_read(Ledger(path=path), METADATA | {"features": ["c"]}, read="test-year evaluation")

    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "date,version,fingerprint,read,command,override,record,kind,status"
    assert lines[1].endswith(",notes,evaluation,completed")
    assert ledger.read_ledger(path).height == 2


def test_completing_a_read_leaves_the_earlier_rows_as_they_were(tmp_path):
    path = tmp_path / "ledger.csv"
    path.write_text(ledger.LEDGER_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    before = ledger.read_ledger(path)
    book = Ledger(path=path)
    ledger.record_read(book, METADATA, read="test-year evaluation")
    ledger.complete_read(book, ledger.fingerprint(METADATA))

    after = ledger.read_ledger(path)
    assert after.head(before.height).equals(before)
    assert after.height == before.height + 1


# --- Re-scoring earlier versions' saved test-year scores is its own kind of read -----------------------------------

EARLIER = {"v1": "a77726cf17b0", "v2": "f7cdd149f731", "the Blend": "f4d0be3058fd"}


def test_re_scoring_earlier_versions_saved_scores_is_recorded_once_as_its_own_read(tmp_path):
    book = _book(tmp_path)
    for rounds in (100, 200, 300):  # three earlier versions' reads
        _completed_read(book, METADATA | {"rounds": rounds})

    row = ledger.record_rescore(book, measure="tie-proof", fingerprints=EARLIER.values(), version="v-new")
    ledger.complete_read(book, row["fingerprint"])

    rows = ledger.read_ledger(book.path)
    assert rows.height == 4
    rescore = rows.row(3, named=True)
    assert (rescore["kind"], rescore["override"], rescore["status"]) == ("re-score", False, "completed")
    assert rescore["fingerprint"] == ledger.rescore_fingerprint("tie-proof", EARLIER.values())
    assert rescore["fingerprint"] not in EARLIER.values()
    with pytest.raises(TestYearAlreadyRead):
        ledger.record_rescore(book, measure="tie-proof", fingerprints=reversed(list(EARLIER.values())), version="v-x")
    assert ledger.read_ledger(book.path).height == 4


def test_a_re_score_under_another_measure_or_of_other_versions_is_another_read():
    keys = list(EARLIER.values())
    first = ledger.rescore_fingerprint("tie-proof", keys)
    assert ledger.rescore_fingerprint("tie-proof", reversed(keys)) == first
    assert ledger.rescore_fingerprint("other", keys) != first
    assert ledger.rescore_fingerprint("tie-proof", keys[:2]) != first
