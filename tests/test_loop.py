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


def test_after_the_recheck_the_loop_decides_on_the_recomputed_win_threshold(data, tmp_path, monkeypatch):
    # Without the recheck nothing is too small to win (minimum 0): the same arm is kept in
    # `test_a_kept_arm_becomes_the_best_so_far_and_the_next_experiment_starts_from_it`.
    plan = replace(loop.FIXTURE_LOOP, min_win_threshold=0.0)
    lp = loop.Loop(data, plan, tmp_path)
    lp.noise_floor()
    rescored_best = {**lp.results["best"], "selection_measure": 0.25}
    # #116: the noise floor's seeds re-scored under the tie-proof measure. Their SD is 1, so the
    # recomputed threshold is 2: no change of a correlation can clear it.
    loop.save_recheck(tmp_path, loop.recheck_results(lp.results, [-0.5, 0.5, 1.5], best=rescored_best))
    _no_training(monkeypatch)

    results = loop.Loop(data, plan, tmp_path).run_experiment(
        "blend", [loop.Arm("three quarters naive", {"naive_weight": 0.75})]
    )

    section = results["experiments"]["blend"]
    assert section["win_threshold"] == pytest.approx(2.0)
    assert section["kept"] is None
    assert section["base"]["selection_measure"] == 0.25  # the best so far, re-scored
    assert results["noise_floor"]["win_threshold"] == 0.0  # the original record stays as it was


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


def test_a_model_cached_on_one_feature_set_is_never_reused_on_another(data, tmp_path):
    """Rebuilt feature files (#126) share the loop's folder: an arm on them trains its own model."""
    original = loop.Loop(data, loop.FIXTURE_LOOP, tmp_path).noise_floor()["arms"][0]
    rebuilt = replace(data, feature_set="exporter_count")

    results = loop.Loop(rebuilt, loop.FIXTURE_LOOP, tmp_path).run_experiment(
        "rebuilt base", [loop.Arm("base on rebuilt files", {})]
    )

    row = results["experiments"]["rebuilt base"]["arms"][0]
    assert row["trained"] is True
    assert row["training_key"] != original["training_key"]


def test_the_real_loop_reads_the_feature_set_the_environment_names(derived_data_root, monkeypatch):
    features.build(derived_data_root, years=[2021, 2022], rate=1.0)
    monkeypatch.setenv(features.FEATURE_SET_ENV, "exporter_count")
    (derived_data_root / "derived" / "country_year_filled.parquet").write_bytes(
        (derived_data_root / "derived" / "country_year.parquet").read_bytes()
    )
    folder = features.build(derived_data_root, years=[2021, 2022], rate=1.0)

    real = loop.real_data(derived_data_root)

    assert real.feature_set == "exporter_count"
    assert real.data.grid == folder / "uk_grid_2022.parquet"


def test_an_experiment_that_would_train_more_than_six_models_is_refused_before_training_any(data, monkeypatch):
    lp = loop.Loop(data, loop.FIXTURE_LOOP)
    lp.noise_floor()
    _no_training(monkeypatch)
    # Seven new seeds: seven models. Scoring-only arms reuse the cached model and do not count.
    arms = [loop.Arm(f"seed {seed}", {"seed": seed}) for seed in range(10, 17)]
    arms.append(loop.Arm("half naive", {"naive_weight": 0.5}))

    with pytest.raises(ValueError, match="at most 6 trained models"):
        lp.run_experiment("too many", arms)


def test_a_two_part_arm_counts_as_two_trained_models(data, monkeypatch):
    lp = loop.Loop(data, loop.FIXTURE_LOOP)
    lp.noise_floor()
    _no_training(monkeypatch)
    arms = [loop.Arm(f"seed {seed}", {"seed": seed}) for seed in range(10, 15)]  # five models
    arms.append(loop.Arm("two-part", {"two_part": True}))  # and two more

    with pytest.raises(ValueError, match="at most 6 trained models"):
        lp.run_experiment("too many", arms)


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
    original = loop.trainers.training_matrix

    def spy(chunks, prepare=None, **kwargs):
        if prepare is not None:
            trained_on.append(pl.concat([prepare(features_of(c)) for c in chunks]))
        return original(chunks, prepare, **kwargs)

    def features_of(chunk):
        return chunk if isinstance(chunk, pl.DataFrame) else pl.read_parquet(chunk)

    monkeypatch.setattr(loop.trainers, "training_matrix", spy)
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


def test_the_scope_split_has_uk_zero_cells_sampled_and_the_lower_zero_rate_as_two_experiments():
    uk_zeros, zero_rate = loop.SCOPE_EXPERIMENTS

    assert uk_zeros == (loop.UK_ZEROS_EXPERIMENT, [loop.Arm(loop.UK_ZEROS_SAMPLED, {"uk_zeros_sampled": True}, loop.BLEND_GRID)])
    assert zero_rate == (loop.ZERO_RATE_EXPERIMENT, [loop.Arm("zero rate 2%", {"zero_rate": 0.02}, loop.BLEND_GRID)])


def _with_more_uk_zero_cells(data: loop.LoopData, uk: str) -> loop.LoopData:
    """The fixture's training rows plus UK Zero Cells in made-up Products: it has only one of its own.

    Each UK row is copied as a Zero Cell into five Products "9<copy>…", which no other row uses.
    """

    def more(rows: pl.DataFrame) -> pl.DataFrame:
        uk_rows = rows.filter(pl.col("exporter").cast(pl.String) == uk)
        copies = [
            uk_rows.with_columns(
                (pl.lit(f"9{copy}") + pl.col("hs6").cast(pl.String).str.slice(2)).cast(rows["hs6"].dtype).alias("hs6"),
                value_usd=pl.lit(0.0),
                weight=pl.lit(1.0),
            )
            for copy in range(5)
        ]
        return pl.concat([rows, *copies])

    training = {year: more(rows) for year, rows in data.data.training.items()}
    return replace(data, data=replace(data.data, training=training))


def test_the_scope_arms_subsample_the_built_rows_and_weight_them_by_the_inverse_rate(data, monkeypatch):
    # Two folds, so the other fold's UK pairs train; the fixture's rows are built with every Zero Cell.
    plan = replace(loop.FIXTURE_LOOP, n_folds=2, start=replace(loop.FIXTURE_LOOP.start, zero_rate=0.5))
    data = _with_more_uk_zero_cells(data, plan.uk)
    trained_on = []
    original = loop.trainers.training_matrix

    def spy(chunks, prepare=None, **kwargs):
        if prepare is not None:
            trained_on.append(pl.concat([prepare(c if isinstance(c, pl.DataFrame) else pl.read_parquet(c)) for c in chunks]))
        return original(chunks, prepare, **kwargs)

    monkeypatch.setattr(loop.trainers, "training_matrix", spy)
    lp = loop.Loop(data, plan)
    for arm in (loop.Arm("as built"), loop.Arm("UK sampled", {"uk_zeros_sampled": True}), loop.Arm("lower", {"zero_rate": 0.25})):
        lp.run_arm(arm, plan.start)

    def zero_weights(rows, uk):
        is_uk = pl.col("exporter").cast(pl.String) == plan.uk
        return rows.filter((pl.col("value_usd") == 0) & (is_uk if uk else ~is_uk))["weight"]

    as_built, uk_sampled, lower = trained_on
    assert set(zero_weights(as_built, uk=True)) == {1.0}
    assert set(zero_weights(as_built, uk=False)) == {2.0}
    assert 0 < len(zero_weights(uk_sampled, uk=True)) < len(zero_weights(as_built, uk=True))
    assert set(zero_weights(uk_sampled, uk=True)) == {2.0}
    assert set(zero_weights(lower, uk=True)) == {1.0}
    assert len(zero_weights(lower, uk=True)) == len(zero_weights(as_built, uk=True))
    assert set(zero_weights(lower, uk=False)) == {4.0}
    flows = pl.col("value_usd") > 0
    for rows in (uk_sampled, lower):
        assert rows.filter(flows).height == as_built.filter(flows).height


def test_the_two_scope_experiments_run_in_order_each_from_the_best_so_far(data):
    lp = loop.Loop(data, loop.FIXTURE_LOOP)

    for name, arms in loop.SCOPE_EXPERIMENTS:
        best_before = lp.results.get("best")
        section = lp.run_experiment(name, arms)["experiments"][name]

        assert section["decided"]
        if best_before is not None:
            assert section["base"] == best_before
        (row,) = section["arms"]
        assert row["status"] in {"kept", "dropped"}
        assert row["settings"]["naive_weight"] in loop.BLEND_WEIGHTS
    zero_rate_row = lp.results["experiments"][loop.ZERO_RATE_EXPERIMENT]["arms"][0]
    assert zero_rate_row["settings"]["zero_rate"] == 0.02


def test_an_arm_adding_a_product_detail_feature_trains_and_scores_with_it(data, monkeypatch):
    trained_on = []
    original = loop.trainers.training_matrix

    def spy(chunks, prepare=None, **kwargs):
        if prepare is not None:
            trained_on.append(pl.concat([prepare(c if isinstance(c, pl.DataFrame) else pl.read_parquet(c)) for c in chunks]))
        return original(chunks, prepare, **kwargs)

    monkeypatch.setattr(loop.trainers, "training_matrix", spy)
    lp = loop.Loop(data, loop.FIXTURE_LOOP)
    start = lp.plan.start
    row = lp.run_arm(loop.Arm("+ hs4", {"features": (*start.features, "hs4")}), start)

    (rows,) = trained_on
    assert rows["hs4"].to_list() == rows["hs6"].cast(pl.String).str.slice(0, 4).cast(pl.Int16).to_list()
    assert row["settings"]["features"][-1] == "hs4"
    assert -1 <= row["selection_measure"] <= 1


def test_the_product_detail_experiments_add_one_feature_each_in_order_from_the_best_so_far(data):
    lp = loop.Loop(data, loop.FIXTURE_LOOP)

    results = loop.run_product_detail(lp)

    names = [loop.product_detail_experiment(f) for f in features.PRODUCT_DETAIL_FEATURES]
    assert [n for n in results["experiments"] if n in names] == names
    for name, feature in zip(names, features.PRODUCT_DETAIL_FEATURES):
        section = results["experiments"][name]
        (row,) = section["arms"]
        base_features = section["base"]["settings"]["features"]
        assert row["settings"]["features"] == [*base_features, feature]
        assert row["settings"]["naive_weight"] in loop.BLEND_WEIGHTS
        assert row["status"] in {"kept", "dropped"}


# --- #75: the monotone constraints on and off ----------------------------------------------------


def test_the_constraints_experiment_reuses_the_best_so_fars_model_and_trains_one_without_constraints(data, monkeypatch):
    trained_with = []
    original = xgb.train

    def spy(params, *args, **kwargs):
        trained_with.append(params)
        return original(params, *args, **kwargs)

    lp = loop.Loop(data, loop.FIXTURE_LOOP)
    lp.noise_floor()
    monkeypatch.setattr(xgb, "train", spy)

    results = lp.run_experiment(loop.CONSTRAINTS_EXPERIMENT, loop.constraint_arms())

    section = results["experiments"][loop.CONSTRAINTS_EXPERIMENT]
    rows = {row["arm"]: row for row in section["arms"]}
    assert list(rows) == ["constraints on", "constraints off"]
    assert rows["constraints on"]["trained"] is False
    assert rows["constraints on"]["training_key"] == results["noise_floor"]["arms"][0]["training_key"]
    assert rows["constraints off"]["trained"] is True
    assert rows["constraints off"]["settings"]["monotone"] is False
    (params,) = trained_with
    assert "monotone_constraints" not in params
    for row in rows.values():
        assert row["settings"]["naive_weight"] in loop.BLEND_WEIGHTS
        assert row["status"] in {"kept", "dropped"}


def test_the_constrained_model_never_lowers_export_potential_for_an_added_trade_agreement(data):
    lp = loop.Loop(data, loop.FIXTURE_LOOP)
    start = lp.plan.start
    agreement = lp.grid["trade_agreement"].dtype

    off = lp.export_potential(start, lp.grid.with_columns(trade_agreement=pl.lit(0, agreement)))
    on = lp.export_potential(start, lp.grid.with_columns(trade_agreement=pl.lit(1, agreement)))

    assert (on >= off * (1 - 1e-6)).all()


# --- #76: the five-fold confirmation -------------------------------------------------------------

# Two folds on the fixture: each fold's grid is too small to rank, but pooled they hold every UK pair.
TWO_FOLDS = replace(loop.FIXTURE_LOOP, n_folds=2)


def _uk_grid_pairs(data) -> int:
    return loop.Loop(data, loop.FIXTURE_LOOP).grid.height  # one fixture "fold" holds every UK pair out


def test_the_confirmation_pools_every_held_out_uk_pair_and_compares_the_winner_with_v2_by_the_threshold(data):
    v2 = TWO_FOLDS.start
    winner = replace(v2, naive_weight=0.5)

    section = loop.confirm(data, TWO_FOLDS, winner, v2, threshold=0.01)

    assert [row["fold"] for row in section["folds"]] == [0, 1]
    assert sum(row["uk_pairs"] for row in section["folds"]) == _uk_grid_pairs(data)
    for name in ("winner", "v2", "naive"):
        assert section[name]["uk_pairs"] == _uk_grid_pairs(data)
        assert -1 <= section[name]["selection_measure"] <= 1
    assert section["winner"]["settings"]["naive_weight"] == 0.5
    assert section["v2"]["settings"]["naive_weight"] == 0.0
    difference = section["winner"]["selection_measure"] - section["v2"]["selection_measure"]
    assert section["difference"] == pytest.approx(difference)
    assert section["win_threshold"] == 0.01
    assert section["beats_v2"] is (difference > 0.01)


def test_the_confirmation_reuses_the_loops_fold_0_model_shares_v2s_model_and_a_rerun_trains_nothing(
    data, tmp_path, monkeypatch
):
    v2 = TWO_FOLDS.start
    winner = replace(v2, naive_weight=0.3)  # differs from v2 only in how the model is scored
    fold_0 = loop.Loop(data, TWO_FOLDS, tmp_path)
    fold_0.export_potential(v2, fold_0.grid)  # the experiments' fold-0 model, now in the cache
    trained = []
    original = xgb.train

    def spy(*args, **kwargs):
        trained.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(xgb, "train", spy)

    first = loop.confirm(data, TWO_FOLDS, winner, v2, threshold=0.01, folder=tmp_path)

    assert len(trained) == 1  # fold 1 only, shared by the winner and v2
    by_fold = {row["fold"]: row for row in first["folds"]}
    assert by_fold[0]["v2"]["trained"] is False
    assert by_fold[1]["v2"]["trained"] is True
    for row in first["folds"]:
        assert row["winner"]["training_key"] == row["v2"]["training_key"]
        assert row["winner"]["trained"] is False
    _no_training(monkeypatch)

    again = loop.confirm(data, TWO_FOLDS, winner, v2, threshold=0.01, folder=tmp_path)

    assert again["folds"] == first["folds"]
    for name in ("winner", "v2", "naive"):  # pooled means over Products: equal up to summation order
        assert again[name]["selection_measure"] == pytest.approx(first[name]["selection_measure"])
    assert json.loads((tmp_path / loop.RESULTS_FILE).read_text())[loop.CONFIRMATION]["folds"] == first["folds"]


# --- What the pipeline serves: the configuration and rounds the selection measure chose -------------

V2_PARAMS = {"max_depth": 12, "eta": 0.1}
SELECTED = {
    "params": V2_PARAMS, "features": list(features.FEATURES), "non_uk_rate": 0.6157, "zero_rate": 0.05, "seed": 0,
    "max_rounds": 5000, "patience": 100, "base_margin": None, "naive_weight": 0.3,
}


def _results(best_settings=SELECTED, *, confirmed=None, best_rounds=951):
    """Loop results shaped like the real `results.json`: the noise floor, the blend kept, maybe a confirmation."""
    start = {**SELECTED, "naive_weight": 0.0}
    results = {
        "validation_year": 2022,
        "noise_floor": {"arms": [{"arm": "seed 0", "settings": start, "rounds": 951}]},
        "experiments": {"naive baseline": {"arms": [{"arm": "blend", "settings": best_settings, "rounds": best_rounds}]}},
        "best": {"experiment": "naive baseline", "arm": "blend", "settings": best_settings},
    }
    if confirmed is not None:
        settings, rounds = confirmed
        results["confirmation"] = {"settings": {"v2": start, "winner": settings}, "winner": {"rounds": rounds}}
    return results


def test_the_served_rounds_are_the_median_of_the_five_fold_confirmations_rounds():
    # Each fold's rounds were chosen by the selection measure on that fold's held-out UK pairs.
    served = loop.served_configuration(_results(confirmed=(SELECTED, [951, 1075, 1097, 1303, 1120])))

    assert served.rounds == 1097
    assert served.params == V2_PARAMS
    assert served.non_uk_rate == 0.6157
    assert served.form.naive_weight == 0.3
    assert served.validation_year == 2022


def test_without_a_confirmation_the_served_rounds_are_the_best_arms_own():
    served = loop.served_configuration(_results(best_rounds=42))

    assert served.rounds == 42
    assert served.params == V2_PARAMS


def test_a_confirmation_of_other_settings_than_the_best_so_far_is_refused():
    other = {**SELECTED, "params": {"max_depth": 6, "eta": 0.1}}

    with pytest.raises(ValueError, match="confirmation"):
        loop.served_configuration(_results(confirmed=(other, [951, 1075, 1097, 1303, 1120])))


def test_a_best_so_far_that_changed_how_zero_cells_are_sampled_is_refused():
    with pytest.raises(ValueError, match="Zero Cells"):
        loop.served_configuration(_results({**SELECTED, "zero_rate": 0.02}))
    with pytest.raises(ValueError, match="Zero Cells"):
        loop.served_configuration(_results({**SELECTED, "uk_zeros_sampled": True}))


def test_the_exporter_count_experiment_is_shown_in_the_glossary_s_words():
    # Not one of the research note's seven numbered experiments (its own note: exporter-count-2022.md).
    assert loop.experiment_label(loop.EXPORTER_COUNT_EXPERIMENT, ["base on rebuilt files", "+ log_exporters_loo"]) == (
        "Exporter count"
    )
