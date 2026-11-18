"""The experiment loop (ticket #67): arms chosen on the selection measure, on the validation year only."""

import json
from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest
import xgboost as xgb

from exportspredictor import features, loop

MEASURES = {
    "selection_measure",
    "non_zero_spearman",
    "zero_vs_non_zero_spearman",
    "value_weighted_spearman",
    "top_ten_share",
    "tweedie_deviance",
}


@pytest.fixture(scope="module")
def data(fixture_world):
    return loop.fixture_data(fixture_world)


def test_a_run_returns_one_row_per_arm_with_every_measure_for_the_model_and_the_naive_baseline(data):
    arms = [loop.Arm("shallow", {"params": {"max_depth": 1}}), loop.Arm("deeper", {"params": {"max_depth": 3}})]

    results = loop.Loop(data, loop.FIXTURE_LOOP).run_experiment("depth", arms)

    rows = results["experiments"]["depth"]["arms"]
    assert [row["arm"] for row in rows] == ["shallow", "deeper"]
    for row in rows:
        assert MEASURES | {"rounds", "seconds", "trained"} <= set(row)
        assert -1 <= row["selection_measure"] <= 1
        assert 0 <= row["top_ten_share"] <= 1
        assert row["rounds"] >= 1
    naive = results["naive"]
    assert MEASURES <= set(naive)
    assert naive["validation_year"] == 2022


def test_the_arm_with_the_highest_selection_measure_is_best_even_when_another_has_the_lower_deviance():
    rows = [
        {"arm": "low deviance", "selection_measure": 0.52, "tweedie_deviance": 300.0},
        {"arm": "ranks best", "selection_measure": 0.56, "tweedie_deviance": 400.0},
        {"arm": "unmeasured", "selection_measure": None, "tweedie_deviance": 100.0},
    ]

    marked, kept = loop.decide(rows, best_so_far=0.53, threshold=0.01)

    assert kept == "ranks best"
    assert [(r["arm"], r["best"], r["status"]) for r in marked] == [
        ("low deviance", False, "dropped"),
        ("ranks best", True, "kept"),
        ("unmeasured", False, "dropped"),
    ]


def test_an_arm_that_does_not_beat_the_best_so_far_by_more_than_the_threshold_is_dropped():
    rows = [{"arm": "a", "selection_measure": 0.540}, {"arm": "b", "selection_measure": 0.535}]

    marked, kept = loop.decide(rows, best_so_far=0.530, threshold=0.010)

    assert kept is None
    assert [r["status"] for r in marked] == ["dropped", "dropped"]
    assert marked[0]["best"] is True


def test_the_win_threshold_is_twice_the_seed_sd_and_at_least_0_005():
    assert loop.win_threshold([0.50, 0.52, 0.54]) == pytest.approx(0.04)
    assert loop.win_threshold([0.530, 0.531, 0.532]) == 0.005


def test_a_kept_arm_becomes_the_best_so_far_and_the_next_experiment_starts_from_it(data):
    # The fixture's seeds give the same model, so with no minimum any better arm wins.
    lp = loop.Loop(data, replace(loop.FIXTURE_LOOP, min_win_threshold=0.0))
    lp.run_experiment("blend", [loop.Arm("three quarters naive", {"naive_weight": 0.75})])

    results = lp.run_experiment("depth", [loop.Arm("deeper", {"params": {"max_depth": 3}})])

    assert results["experiments"]["blend"]["kept"] == "three quarters naive"
    assert results["experiments"]["depth"]["base"]["arm"] == "three quarters naive"
    deeper = results["experiments"]["depth"]["arms"][0]["settings"]
    assert deeper["params"] == {"max_depth": 3, "eta": 0.3} and deeper["naive_weight"] == 0.75


def test_an_arm_inside_the_noise_is_recorded_as_dropped(data):
    lp = loop.Loop(data, replace(loop.FIXTURE_LOOP, min_win_threshold=2.0))  # nothing can clear it

    results = lp.run_experiment("depth", [loop.Arm("deeper", {"params": {"max_depth": 3}})])

    depth = results["experiments"]["depth"]
    assert depth["kept"] is None and depth["arms"][0]["status"] == "dropped"
    assert results["best"]["experiment"] == "noise_floor"
    floor = results["noise_floor"]
    assert [row["arm"] for row in floor["arms"]] == ["seed 0", "seed 1", "seed 2"]
    assert floor["win_threshold"] == 2.0


def _no_training(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a model was trained")

    monkeypatch.setattr(xgb, "train", refuse)


def test_a_scoring_only_arm_trains_no_model(data, tmp_path, monkeypatch):
    loop.Loop(data, loop.FIXTURE_LOOP, tmp_path).noise_floor()
    _no_training(monkeypatch)

    # A new loop on the same folder: the noise floor's model is read from the cache.
    results = loop.Loop(data, loop.FIXTURE_LOOP, tmp_path).run_experiment(
        "blend", [loop.Arm("half naive", {"naive_weight": 0.5})]
    )

    row = results["experiments"]["blend"]["arms"][0]
    assert row["trained"] is False
    assert row["training_key"] == results["noise_floor"]["arms"][0]["training_key"]
    assert row["selection_measure"] is not None


def test_a_rerun_skips_finished_arms(data, tmp_path, monkeypatch):
    arms = [loop.Arm("deeper", {"params": {"max_depth": 3}}), loop.Arm("shallow", {"params": {"max_depth": 1}})]
    first = loop.Loop(data, loop.FIXTURE_LOOP, tmp_path).run_experiment("depth", arms)
    _no_training(monkeypatch)

    again = loop.Loop(data, loop.FIXTURE_LOOP, tmp_path).run_experiment("depth", arms)

    assert again["experiments"]["depth"] == first["experiments"]["depth"]
    assert again["noise_floor"] == first["noise_floor"]
    assert json.loads((tmp_path / loop.RESULTS_FILE).read_text())["experiments"] == first["experiments"]


def test_the_loop_refuses_the_test_year_on_the_fixture(fixture_world):
    with pytest.raises(loop.TestYearRead):
        loop.fixture_data(fixture_world, validation_year=2023)


def test_the_real_path_refuses_the_test_year_and_never_reads_its_files(derived_data_root, monkeypatch):
    features.build(derived_data_root, years=[2021, 2022, 2023], rate=1.0)
    with pytest.raises(loop.TestYearRead):
        loop.real_data(derived_data_root, validation_year=2023)
    read = []
    for name in ("scan_parquet", "read_parquet"):
        original = getattr(pl, name)

        def spy(source, *args, _original=original, **kwargs):
            read.append(str(source))
            return _original(source, *args, **kwargs)

        monkeypatch.setattr(pl, name, spy)

    folder = loop.loop_dir(derived_data_root)
    results = loop.Loop(loop.real_data(derived_data_root), loop.FIXTURE_LOOP, folder).noise_floor()

    assert results["win_threshold"] >= loop.MIN_WIN_THRESHOLD
    assert loop.load(derived_data_root)["noise_floor"] == results
    assert read and not [path for path in read if "2023" in Path(path).name or "2024" in Path(path).name]
    assert any("flows_2022" in path for path in read)


# --- #70: the naive baseline as a blend and as the starting point -------------------------------


def test_the_blend_arm_chooses_its_weight_from_the_fixed_grid_by_the_selection_measure(data):
    lp = loop.Loop(data, loop.FIXTURE_LOOP)

    results = lp.run_experiment("naive", [loop.blend_arm()])

    row = results["experiments"]["naive"]["arms"][0]
    candidates = row["grid"]
    assert [c["naive_weight"] for c in candidates] == list(loop.BLEND_WEIGHTS)
    best = max(candidates, key=lambda c: c["selection_measure"])
    assert row["settings"]["naive_weight"] == best["naive_weight"]
    assert row["selection_measure"] == pytest.approx(best["selection_measure"])
    assert row["trained"] is False  # the blend scores the best so far's model


def test_blended_export_potential_is_in_dollars_and_positive_for_every_pair():
    model = pl.Series([0.0, 100.0, 400.0])
    naive = pl.Series([10_000.0, 0.0, 100.0])

    blended = loop.blend(model, naive, 0.5)

    # Geometric means in dollars, each side floored at $1: sqrt(1 × 10,000), sqrt(100 × 1), sqrt(400 × 100).
    assert blended.to_list() == pytest.approx([100.0, 10.0, 200.0])
    assert (blended > 0).all()


def test_the_blend_records_the_rank_average_of_model_and_naive_as_a_reference_without_dollars(data):
    results = loop.Loop(data, loop.FIXTURE_LOOP).run_experiment("naive", [loop.blend_arm()])

    reference = results["experiments"]["naive"]["arms"][0]["rank_average"]
    assert -1 <= reference["selection_measure"] <= 1
    assert reference["tweedie_deviance"] is None


def test_a_starting_point_model_that_learns_nothing_ranks_as_the_naive_baseline(data):
    # With a learning rate of 0 the trees add nothing, so Export Potential is the naive baseline.
    arm = replace(loop.starting_point_arm(), changes={**loop.starting_point_arm().changes, "params": {"eta": 0.0}})

    results = loop.Loop(data, loop.FIXTURE_LOOP).run_experiment("naive", [arm])

    row = results["experiments"]["naive"]["arms"][0]
    assert row["trained"] is True
    assert row["settings"]["base_margin"] == "naive"
    assert row["selection_measure"] == pytest.approx(results["naive"]["selection_measure"])


def test_a_starting_point_model_learns_a_correction_to_the_naive_baseline(data):
    results = loop.Loop(data, loop.FIXTURE_LOOP).run_experiment("naive", [loop.starting_point_arm()])

    row = results["experiments"]["naive"]["arms"][0]
    assert row["trained"] is True and row["rounds"] >= 1
    assert MEASURES <= set(row)


def test_the_training_rows_naive_baseline_from_their_features_is_the_naive_baseline_from_flows(fixture_world, data):
    from exportspredictor.evaluation import naive_potential

    rows = data.data.validation
    flows = fixture_world.flows.filter(pl.col("year") == 2022)

    from_features = loop.naive_from_features(rows, loop.world_products(fixture_world.flows))

    expected = naive_potential(rows, flows)["naive_usd"]
    assert (expected > 0).any()
    assert from_features.to_list() == pytest.approx(expected.to_list(), rel=1e-5, abs=1e-3)


# --- #71: select on ranking ----------------------------------------------------------------------

# The shape of the saved search results (`search.load()`): three confirmed configurations.
SAVED_SEARCH = {
    "plan": {"eta": 0.1},
    "confirmation": [
        {"trial": 15, "params": {"max_depth": 12, "min_child_weight": 3.0, "subsample": 0.9,
                                 "colsample_bytree": 0.9, "reg_lambda": 5.0, "tweedie_variance_power": 1.27}},
        {"trial": 23, "params": {"max_depth": 11, "min_child_weight": 6.8, "subsample": 0.86,
                                 "colsample_bytree": 0.99, "reg_lambda": 23.8, "tweedie_variance_power": 1.27}},
        {"trial": 19, "params": {"max_depth": 11, "min_child_weight": 1.4, "subsample": 0.88,
                                 "colsample_bytree": 0.95, "reg_lambda": 0.96, "tweedie_variance_power": 1.28}},
    ],
}


def test_select_on_ranking_has_v1_the_three_confirmed_configurations_and_a_depth_6_one():
    arms = loop.select_arms(SAVED_SEARCH)

    assert [arm.name for arm in arms] == ["v1", "search trial 15", "search trial 23", "search trial 19", "depth 6"]
    v1 = arms[0].changes["params"]
    assert v1 == {"max_depth": 8, "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.8,
                  "reg_lambda": 1.0, "tweedie_variance_power": 1.3, "eta": 0.1}
    assert arms[2].changes["params"] == {**SAVED_SEARCH["confirmation"][1]["params"], "eta": 0.1}
    assert arms[4].changes["params"] == {"max_depth": 6}
    for arm in arms:
        assert arm.grid == {"naive_weight": loop.BLEND_WEIGHTS}


def test_select_on_ranking_trains_at_most_one_model_per_new_configuration_and_reuses_the_best_so_fars(data):
    # The fixture's "search" confirmed its own start configuration and a deeper one.
    saved = {"plan": {"eta": 0.3}, "confirmation": [{"trial": 0, "params": {"max_depth": 2}},
                                                   {"trial": 1, "params": {"max_depth": 3}}]}
    lp = loop.Loop(data, loop.FIXTURE_LOOP)

    results = lp.run_experiment(loop.SELECT_EXPERIMENT, loop.select_arms(saved))

    section = results["experiments"][loop.SELECT_EXPERIMENT]
    rows = {row["arm"]: row for row in section["arms"]}
    assert list(rows) == ["v1", "search trial 0", "search trial 1", "depth 6"]
    assert rows["search trial 0"]["trained"] is False
    assert rows["search trial 0"]["training_key"] == results["noise_floor"]["arms"][0]["training_key"]
    assert sum(row["trained"] for row in rows.values()) == 3
    for row in rows.values():
        assert row["rounds"] >= 1
        assert row["settings"]["naive_weight"] in loop.BLEND_WEIGHTS
        assert row["status"] in {"kept", "dropped"}
    assert rows["v1"]["settings"]["params"]["max_depth"] == 8
    assert rows["depth 6"]["settings"]["params"] == {"max_depth": 6, "eta": 0.3}


# --- #72: the two-part model ---------------------------------------------------------------------


def test_the_two_part_arm_runs_through_the_loop_and_returns_a_result_row(data):
    results = loop.Loop(data, loop.FIXTURE_LOOP).run_experiment(loop.TWO_PART_EXPERIMENT, [loop.two_part_arm()])

    row = results["experiments"][loop.TWO_PART_EXPERIMENT]["arms"][0]
    assert row["trained"] is True and row["settings"]["two_part"] is True
    assert MEASURES <= set(row)
    assert -1 <= row["selection_measure"] <= 1
    assert row["status"] in {"kept", "dropped"}
    assert row["settings"]["naive_weight"] in loop.BLEND_WEIGHTS
    for part in ("any_flow", "size"):
        assert row["parts"][part]["rounds"] >= 1


def test_no_uk_pair_is_scored_by_a_part_that_trained_on_it(data, monkeypatch):
    # Two folds: fold 0's UK pairs are scored, so neither part may see them; fold 1's train both.
    plan = replace(loop.FIXTURE_LOOP, n_folds=2)
    trained_on = []
    original = loop.training_matrix

    def spy(chunks, prepare=None, **kwargs):
        if prepare is not None:
            trained_on.append(pl.concat([prepare(features_of(c)) for c in chunks]))
        return original(chunks, prepare, **kwargs)

    def features_of(chunk):
        return chunk if isinstance(chunk, pl.DataFrame) else pl.read_parquet(chunk)

    monkeypatch.setattr(loop, "training_matrix", spy)
    lp = loop.Loop(data, plan)
    lp.run_arm(loop.two_part_arm(), plan.start)  # one arm, no noise floor: two folds are too few to rank

    scored = set(lp.grid.select("importer", "hs6").iter_rows())
    assert scored
    any_flow, size = trained_on  # the binary part, then the size part
    assert set(any_flow["value_usd"].unique()) <= {0.0, 1.0}
    assert (size["value_usd"] > 0).all()
    for rows in (any_flow, size):
        uk = rows.filter(pl.col("exporter").cast(pl.String) == plan.uk)
        assert uk.height  # the other fold's UK pairs do train
        assert not scored & set(uk.select("importer", "hs6").iter_rows())


def test_adding_a_trade_agreement_never_lowers_the_two_part_export_potential(data):
    lp = loop.Loop(data, loop.FIXTURE_LOOP)
    results = lp.run_experiment(loop.TWO_PART_EXPERIMENT, [loop.two_part_arm()])
    settings = loop.Settings.from_dict(results["experiments"][loop.TWO_PART_EXPERIMENT]["arms"][0]["settings"])
    agreement = lp.grid["trade_agreement"].dtype

    for weight in (0.0, settings.naive_weight):  # the two-part model alone and as blended
        scored = replace(settings, naive_weight=weight)
        off = lp.export_potential(scored, lp.grid.with_columns(trade_agreement=pl.lit(0, agreement)))
        on = lp.export_potential(scored, lp.grid.with_columns(trade_agreement=pl.lit(1, agreement)))
        assert (off > 0).all()
        assert (on >= off * (1 - 1e-6)).all()
