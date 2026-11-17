"""Freezing v1: v1's predictions saved as fixed tables that v2 joins to (spec #39, *Freezing v1*)."""

import hashlib
import json
import subprocess
from dataclasses import replace

import numpy as np
import polars as pl
import pytest

from exportspredictor import backtests, evaluation, freeze, pipeline
from exportspredictor.features import FEATURES, build_features, features_dir, uk_grid
from exportspredictor.fixture import UK
from exportspredictor.folds import assign_folds

PLAN = pipeline.FIXTURE_BACKTESTS
TEST_YEAR = 2023
LIFT_YEARS = (2021, 2022)  # the fixture Lift model's own years; real v1 freezes 2015–2017
PROVENANCE = {"tag": "v1", "commit": "0123abcd"}
PARAMS = {"max_depth": 2}
ROUNDS = 20


@pytest.fixture(scope="module")
def v1(fixture_world):
    """The fixture's stand-in for v1's saved models and feature rows."""
    training = pipeline.fixture_training(fixture_world)
    flows = fixture_world.flows.filter(pl.col("year") == TEST_YEAR)
    exporters = training[max(PLAN.lift.gap_years)].filter(pl.col("exporter") != UK)["exporter"].unique()
    held = backtests.held_out_exporters(exporters, share=backtests.HELD_OUT_SHARE)
    common = {"uk": UK, "params": PARAMS, "rounds": ROUNDS, "device": "cpu"}
    return freeze.V1(
        rows=training,
        test_grid=build_features(uk_grid(flows, UK), flows, fixture_world.sources),
        test_models=evaluation.train_test_models(training, test_year=TEST_YEAR, **common),
        lift_model=backtests.train_backtest_model(
            training, name="lift", training_years=PLAN.lift.training_years, before=PLAN.lift.before,
            held_out=held, **common,
        ),
        event_model=backtests.train_backtest_model(
            training, name=PLAN.event.name, training_years=PLAN.event.training_years, before=PLAN.event.event_year,
            **common,
        ),
        event=PLAN.event,
        lift_years=LIFT_YEARS,
        model_ids={"artefact": "fixture", "test_models": "t", "lift": "l", "event_backtest": "e"},
    )


@pytest.fixture(scope="module")
def frozen(v1, tmp_path_factory):
    return freeze.freeze(tmp_path_factory.mktemp("artefact"), v1, uk=UK, provenance=PROVENANCE)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_manifest_records_provenance_and_matches_every_file(frozen):
    manifest = json.loads((frozen / freeze.MANIFEST_FILE).read_text(encoding="utf-8"))

    assert frozen.name == "v1_predictions"
    assert manifest["tag"] == "v1" and manifest["commit"] == "0123abcd"
    assert manifest["source_models"]["test_models"] == "t"
    assert set(manifest["tables"]) == set(freeze.TABLES)
    for name, entry in manifest["tables"].items():
        path = frozen / entry["file"]
        assert entry["rows"] == pl.read_parquet(path).height > 0, name
        assert entry["sha256"] == _sha256(path), name


def _table(frozen, name):
    return pl.read_parquet(frozen / freeze.TABLES[name])


def test_the_uk_grid_covers_every_pair_each_scored_by_the_fold_model_that_held_it_out(frozen, v1):
    grid = _table(frozen, "uk_grid")
    pairs = v1.test_grid.select(pl.col("hs6").cast(pl.String), iso3=pl.col("importer").cast(pl.String))

    assert grid.height == pairs.height == 5 * 4
    assert grid.select("hs6", "iso3").sort("hs6", "iso3").equals(pairs.sort("hs6", "iso3"))
    expected_folds = assign_folds(grid.select("hs6", "iso3"))["fold"]
    assert (grid["fold"] == expected_folds).all()
    features = v1.test_models.transform.apply(v1.test_grid).with_columns(
        hs6=pl.col("hs6").cast(pl.String), iso3=pl.col("importer").cast(pl.String)
    )
    for row in grid.iter_rows(named=True):
        model = v1.test_models.models[row["fold"]]
        assert model.attr("held_out_fold") == str(row["fold"])
        data = features.filter(hs6=row["hs6"], iso3=row["iso3"]).select(FEATURES).to_numpy().astype(np.float32)
        assert row["export_potential_usd"] == pytest.approx(float(model.inplace_predict(data)[0]))


def test_the_test_rows_keep_v1s_own_keys_and_weights(frozen, v1):
    rows = _table(frozen, "test_rows")
    keys = ["exporter", "importer", "hs6", "year", "value_usd", "weight"]
    source = v1.rows[TEST_YEAR].select(pl.col("exporter", "importer", "hs6").cast(pl.String), *keys[3:])

    assert rows.select(keys).sort(keys).equals(source.sort(keys))
    assert rows["export_potential_usd"].is_not_null().all()


def test_lift_covers_every_held_out_exporter_pair_in_each_lift_year(frozen, v1):
    lift = _table(frozen, "lift")
    held = _table(frozen, "lift_held_out_exporters")["exporter"].to_list()
    expected = pl.concat(
        [v1.rows[y].filter(pl.col("exporter").is_in(held)).select(pl.col("exporter", "importer", "hs6").cast(pl.String), "year")
         for y in LIFT_YEARS]
    )

    assert held == sorted(v1.lift_model.held_out_exporters) and held
    assert set(lift["year"]) == set(LIFT_YEARS)
    assert lift.select(expected.columns).sort(expected.columns).equals(expected.sort(expected.columns))


def test_the_event_backtest_covers_each_direction_and_post_year_with_the_agreement_on_and_off(frozen, v1):
    event = _table(frozen, "event_backtest")
    (direction,) = PLAN.event.directions  # the fixture's UK exports to Japan
    treated = pl.concat([v1.rows[y] for y in PLAN.event.post_years]).filter(
        pl.col("exporter").is_in(list(direction.exporters)), pl.col("importer").is_in(list(direction.importers))
    )

    assert event.select("direction", "year").unique().height == len(PLAN.event.post_years)
    assert event.height == treated.height
    assert set(event["direction"]) == {direction.label}
    # The Trade Agreement feature is monotone-constrained (ADR 0002): switching it on never lowers Export Potential.
    assert (event["export_potential_on_usd"] >= event["export_potential_off_usd"]).all()
    on = v1.event_model.predict(treated.sort("year", "hs6").with_columns(trade_agreement=pl.lit(1, pl.Int8)))
    assert event.sort("year", "hs6")["export_potential_on_usd"].to_numpy() == pytest.approx(on)


def test_v2_reads_the_frozen_held_out_exporters_restricted_to_its_scope(frozen, v1):
    (held,) = v1.lift_model.held_out_exporters
    scope = [UK, held, "ZZZ"]

    assert freeze.lift_held_out_exporters(frozen) == [held]
    assert backtests.lift_exporters(freeze.lift_held_out_exporters(frozen), scope=scope, uk=UK) == [held]


def test_the_frozen_held_out_list_is_refused_if_it_no_longer_matches_its_manifest(v1, tmp_path):
    folder = freeze.freeze(tmp_path, v1, uk=UK, provenance=PROVENANCE)
    pl.DataFrame({"exporter": ["XXX"]}).write_parquet(folder / freeze.TABLES["lift_held_out_exporters"])

    with pytest.raises(ValueError, match="lift_held_out_exporters"):
        freeze.lift_held_out_exporters(folder)


def test_re_running_the_freeze_leaves_it_as_it_was(frozen, v1):
    before = {path.name: _sha256(path) for path in frozen.iterdir()}

    again = freeze.freeze(frozen.parent, v1, uk=UK, provenance={"tag": "v1", "commit": "different"})

    assert again == frozen
    assert {path.name: _sha256(path) for path in frozen.iterdir()} == before


def test_a_fresh_freeze_reproduces_identical_checksums(frozen, v1, tmp_path):
    other = freeze.freeze(tmp_path, v1, uk=UK, provenance=PROVENANCE)

    def checksums(folder):
        return {name: entry["sha256"] for name, entry in freeze.verify(folder)["tables"].items()}

    assert checksums(other) == checksums(frozen)


def test_verify_refuses_a_table_that_no_longer_matches_its_manifest(v1, tmp_path):
    folder = freeze.freeze(tmp_path, v1, uk=UK, provenance=PROVENANCE)
    pl.DataFrame({"exporter": ["XXX"]}).write_parquet(folder / freeze.TABLES["lift_held_out_exporters"])

    with pytest.raises(ValueError, match="lift_held_out_exporters"):
        freeze.verify(folder)


def test_a_freeze_that_fails_part_way_writes_nothing(v1, tmp_path, monkeypatch):
    written = []
    real = freeze.downloads.write_parquet

    def failing(frame, target, **options):
        if len(written) == 2:
            raise OSError("disk full")
        written.append(target)
        return real(frame, target, **options)

    monkeypatch.setattr(freeze.downloads, "write_parquet", failing)

    with pytest.raises(OSError):
        freeze.freeze(tmp_path, v1, uk=UK, provenance=PROVENANCE)
    assert written
    assert list(tmp_path.iterdir()) == []


def test_a_leaking_event_backtest_model_is_refused_before_anything_is_written(v1, tmp_path):
    late = backtests.train_backtest_model(
        v1.rows, name="late", training_years=[2021, 2022], before=2023, uk=UK, params=PARAMS, rounds=5, device="cpu"
    )

    with pytest.raises(backtests.LeakageError):
        freeze.freeze(tmp_path, replace(v1, event_model=late), uk=UK, provenance=PROVENANCE)
    assert list(tmp_path.iterdir()) == []


# --- Reading v1 from disk -----------------------------------------------------------------------


def test_v1_loaded_from_its_data_root_freezes_the_same_tables(v1, frozen, tmp_path):
    root = tmp_path / "data"
    folder = features_dir(root)
    folder.mkdir(parents=True)
    for year, rows in v1.rows.items():
        rows.write_parquet(folder / f"training_{year}.parquet")
    v1.test_grid.write_parquet(folder / f"uk_grid_{TEST_YEAR}.parquet")
    (folder / "features.json").write_text(json.dumps({"years": sorted(v1.rows)}), encoding="utf-8")
    ids = {"artefact": "fixture", "test_models": "2023_abc", "lift": "lift_def", "event_backtest": "uk_japan_ghi"}
    v1.test_models.save(folder.parent / "test_models" / ids["test_models"])
    v1.lift_model.save(backtests.cache_dir(root) / ids["lift"])
    v1.event_model.save(backtests.cache_dir(root) / ids["event_backtest"])

    loaded = freeze.load_v1(root, ids=ids, event=PLAN.event, lift_years=LIFT_YEARS)
    refrozen = freeze.freeze(tmp_path / "artefact", loaded, uk=UK, provenance=PROVENANCE)

    tables = {name: entry["sha256"] for name, entry in freeze.verify(refrozen)["tables"].items()}
    assert tables == {name: entry["sha256"] for name, entry in freeze.verify(frozen)["tables"].items()}
    assert freeze.verify(refrozen)["source_models"] == ids


def _git(repo, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=repo, check=True, capture_output=True)


def test_provenance_names_the_v1_tag_only_when_it_points_at_the_commit(tmp_path):
    _git(tmp_path, "init", "-q")
    (tmp_path / "a.txt").write_text("a")
    _git(tmp_path, "add", "a.txt")
    _git(tmp_path, "commit", "-q", "-m", "first")

    untagged = freeze.git_provenance(tmp_path)
    _git(tmp_path, "tag", "v1")
    tagged = freeze.git_provenance(tmp_path)
    (tmp_path / "a.txt").write_text("changed")

    assert untagged["tag"] is None
    assert tagged == {"tag": "v1", "commit": untagged["commit"], "dirty": False}
    assert len(tagged["commit"]) == 40
    assert freeze.git_provenance(tmp_path)["dirty"] is True
