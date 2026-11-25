"""The feature ablation and the learning curve (ticket #53): measured on the validation year, never the test year."""

import json
from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest

from exportspredictor import experiments, features
from exportspredictor.features import ALL_FEATURES, training_rows
from exportspredictor.fixture import UK


def _keys(table: pl.DataFrame) -> pl.DataFrame:
    return table.select(
        pl.col("exporter", "importer", "hs6").cast(pl.String), pl.col("year").cast(pl.Int64), "value_usd", "weight"
    ).sort("exporter", "importer", "hs6", "year")


def test_subsampling_zeros_gives_the_rows_a_lower_rate_would_have_built(fixture_world):
    flows = fixture_world.flows.filter(pl.col("year") == 2022)
    built = training_rows(flows, rate=0.8, seed=3, uk=UK)

    subsampled = features.subsample_zeros(built, built_rate=0.8, rate=0.4, seed=3, uk=UK)

    expected = training_rows(flows, rate=0.4, seed=3, uk=UK)
    assert expected.filter(pl.col("value_usd") == 0).height < built.filter(pl.col("value_usd") == 0).height
    assert _keys(subsampled).equals(_keys(expected))


def test_subsampling_zeros_keeps_every_flow_and_every_uk_zero_cell(fixture_world):
    built = training_rows(fixture_world.flows.filter(pl.col("year") == 2022), rate=1.0, uk=UK)

    subsampled = features.subsample_zeros(built, built_rate=1.0, rate=0.02, uk=UK)

    uk_or_flow = (pl.col("exporter") == UK) | (pl.col("value_usd") > 0)
    assert _keys(subsampled.filter(uk_or_flow)).equals(_keys(built.filter(uk_or_flow)))
    assert set(subsampled.filter(~uk_or_flow)["weight"]) <= {50.0}


def test_uk_zero_cells_can_be_sampled_and_weighted_as_other_exporters_zero_cells_are(fixture_world):
    built = training_rows(fixture_world.flows, rate=0.8, seed=3, uk=UK)  # every year: a few UK Zero Cells
    uk_zero = (pl.col("exporter") == UK) & (pl.col("value_usd") == 0)

    sampled = features.subsample_zeros(built, built_rate=0.8, rate=0.4, seed=3, uk=UK, uk_zeros_sampled=True)

    # The UK treated as any other Exporter: the rows of a subsample that knows no UK at all.
    as_any_exporter = features.subsample_zeros(built, built_rate=0.8, rate=0.4, seed=3, uk="no UK")
    assert _keys(sampled).equals(_keys(as_any_exporter))
    assert 0 < sampled.filter(uk_zero).height < built.filter(uk_zero).height
    assert set(sampled.filter(uk_zero)["weight"]) == {2.5}


def test_uk_zero_cells_are_sampled_at_the_rate_the_rows_were_built_at_too(fixture_world):
    built = training_rows(fixture_world.flows, rate=0.8, seed=3, uk=UK)  # every year: a few UK Zero Cells
    uk_zero = (pl.col("exporter") == UK) & (pl.col("value_usd") == 0)

    at_built = features.subsample_zeros(built, built_rate=0.8, rate=0.8, seed=3, uk=UK, uk_zeros_sampled=True)
    lower = features.subsample_zeros(built, built_rate=0.8, rate=0.4, seed=3, uk=UK, uk_zeros_sampled=True)

    kept = _keys(at_built.filter(uk_zero))
    assert lower.filter(uk_zero).height < kept.height < built.filter(uk_zero).height
    assert set(kept["weight"]) == {1.25}
    assert _keys(at_built.filter(~uk_zero)).equals(_keys(built.filter(~uk_zero)))


def test_subsampling_cannot_raise_the_zero_rate(fixture_world):
    built = training_rows(fixture_world.flows.filter(pl.col("year") == 2022), rate=0.05, uk=UK)

    with pytest.raises(ValueError, match="0.05"):
        features.subsample_zeros(built, built_rate=0.05, rate=0.1, uk=UK)


@pytest.fixture(scope="module")
def fixture_results(fixture_world):
    data = experiments.fixture_data(fixture_world)
    return experiments.run(data, experiments.FIXTURE_PLAN)


def test_the_ablation_drops_each_new_feature_in_turn(fixture_results):
    ablation = pl.DataFrame(fixture_results["ablation"])

    assert ablation["dropped"].to_list() == [None, *experiments.CANDIDATE_FEATURES]
    assert ablation["features"].to_list() == [len(ALL_FEATURES)] + [len(ALL_FEATURES) - 1] * len(
        experiments.CANDIDATE_FEATURES
    )
    assert ablation["validation_deviance"].is_finite().all()
    assert ablation["validation_year"].unique().to_list() == [2022]


def test_a_feature_is_dropped_only_when_it_hurts_both_measures_and_the_applied_tariff_never_is():
    ablation = [
        {"dropped": None, "validation_deviance": 10.0, "uk_spearman": 0.50},
        # without it, deviance and Spearman are both better: it hurts
        {"dropped": "reexport_ratio", "validation_deviance": 9.0, "uk_spearman": 0.52},
        # better deviance, worse Spearman: kept
        {"dropped": "relatedness", "validation_deviance": 9.5, "uk_spearman": 0.45},
        # worse on both: kept
        {"dropped": "market_import_growth", "validation_deviance": 11.0, "uk_spearman": 0.40},
        # hurts, but Scenarios need it
        {"dropped": "log_applied_tariff", "validation_deviance": 9.0, "uk_spearman": 0.55},
    ]

    kept = experiments.kept_features(ablation)

    assert "reexport_ratio" not in kept
    assert {"relatedness", "market_import_growth", "log_applied_tariff"} <= set(kept)
    assert kept == [f for f in ALL_FEATURES if f != "reexport_ratio"]


def test_the_learning_curve_covers_every_rate_pair(fixture_results):
    curve = pl.DataFrame(fixture_results["learning_curve"])
    plan = experiments.FIXTURE_PLAN

    assert curve.height == len(plan.non_uk_rates) * len(plan.zero_rates)
    assert set(zip(curve["non_uk_rate"], curve["zero_rate"])) == {
        (r, z) for r in plan.non_uk_rates for z in plan.zero_rates
    }
    assert curve["validation_deviance"].is_finite().all()
    # more sampled rows, never fewer, as either rate rises
    for z in plan.zero_rates:
        rows = curve.filter(pl.col("zero_rate") == z).sort("non_uk_rate")["training_rows"]
        assert rows.is_sorted()
    assert fixture_results["kept_features"] == experiments.kept_features(fixture_results["ablation"])


def test_the_runs_refuse_the_test_year(fixture_world):
    data = experiments.fixture_data(fixture_world)
    plan = experiments.FIXTURE_PLAN

    with pytest.raises(experiments.TestYearRead):
        experiments.run(data, replace(plan, validation_year=2023))
    with pytest.raises(experiments.TestYearRead):
        experiments.fixture_data(fixture_world, validation_year=2023)


def test_the_real_path_never_reads_test_year_files(derived_data_root, monkeypatch):
    read = []
    for name in ("scan_parquet", "read_parquet"):
        original = getattr(pl, name)

        def spy(source, *args, _original=original, **kwargs):
            read.append(str(source))
            return _original(source, *args, **kwargs)

        monkeypatch.setattr(pl, name, spy)

    folder = experiments.build(derived_data_root, validation_year=2022, years=[2021, 2022], sectors=None, rate=1.0)
    results = experiments.run(experiments.real_data(derived_data_root), experiments.FIXTURE_PLAN)
    saved = experiments.save(results, derived_data_root)

    assert read and not [path for path in read if "2023" in Path(path).name or "2024" in Path(path).name]
    assert any("flows_2022" in path for path in read)
    assert json.loads((folder / experiments.SETTINGS_FILE).read_text())["validation_year"] == 2022
    assert experiments.load(derived_data_root) == json.loads(saved.read_text())


# --- Saved plans, progress and results --------------------------------------------------


def test_a_saved_plan_is_a_dict_with_its_sectors_as_a_list():
    plan = experiments.Plan(
        params={"eta": 0.1}, max_rounds=10, sectors=("22", "87"), built_zero_rate=0.1,
        ablation_rates=(0.1, 0.1), non_uk_rates=(0.1,), zero_rates=(0.1,),
    )

    saved = experiments.plan_dict(plan)

    assert saved["sectors"] == ["22", "87"]
    assert saved["max_rounds"] == 10


def test_a_plan_over_every_sector_saves_none():
    plan = experiments.Plan(
        params={}, max_rounds=1, sectors=None, built_zero_rate=0.1,
        ablation_rates=(0.1, 0.1), non_uk_rates=(0.1,), zero_rates=(0.1,),
    )

    assert experiments.plan_dict(plan)["sectors"] is None


def test_progress_is_printed_as_it_happens(capsys):
    experiments.log_progress("trial 3 done")

    assert capsys.readouterr().out == "trial 3 done\n"


def test_saved_results_are_read_back_from_their_folder(tmp_path):
    (tmp_path / "results.json").write_text(json.dumps({"trials": [1, 2]}), encoding="utf-8")

    assert experiments.read_results(tmp_path) == {"trials": [1, 2]}


def test_there_are_no_saved_results_without_a_folder_or_a_file(tmp_path):
    assert experiments.read_results(None) is None
    assert experiments.read_results(tmp_path) is None
