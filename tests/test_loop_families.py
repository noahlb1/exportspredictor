"""Other model families in the loop (ticket #167): LightGBM and CatBoost Arms behind the trainer seam."""

from dataclasses import replace

import polars as pl
import pytest

from exportspredictor import loop, market_reach

# Each family's own parameters for the fixture's tiny world (the fixture XGBoost start has depth 2, eta 0.3).
FAMILY_ARMS = [
    loop.Arm("lightgbm", {"family": loop.LIGHTGBM, "params": {"num_leaves": 4, "learning_rate": 0.3}}),
    loop.Arm("catboost", {"family": loop.CATBOOST, "params": {"depth": 2, "learning_rate": 0.3}}),
]
XGBOOST_ARM = loop.Arm("xgboost deeper", {"params": {"max_depth": 3}})


@pytest.fixture(scope="module")
def data(fixture_world):
    return loop.fixture_data(fixture_world)


@pytest.mark.parametrize("arm", FAMILY_ARMS, ids=lambda arm: arm.name)
def test_a_family_arm_returns_the_same_result_row_as_an_xgboost_arm_and_is_decided_by_the_win_rule(data, arm):
    lp = loop.Loop(data, loop.FIXTURE_LOOP)
    results = lp.run_experiment("family", [XGBOOST_ARM, arm])

    section = results["experiments"]["family"]
    xgboost_row, family_row = section["arms"]
    assert set(family_row) == set(xgboost_row)
    assert family_row["trained"] is True and family_row["rounds"] >= 1
    assert -1 <= family_row["selection_measure"] <= 1
    # The win rule: the higher Selection Measure is best, kept only if it beats the base by the threshold.
    top = max(section["arms"], key=lambda row: row["selection_measure"])
    assert [row["best"] for row in section["arms"]] == [row is top for row in section["arms"]]
    wins = top["selection_measure"] - section["base"]["selection_measure"] > section["win_threshold"]
    assert section["kept"] == (top["arm"] if wins else None)


@pytest.mark.parametrize("arm", [XGBOOST_ARM, *FAMILY_ARMS], ids=lambda arm: arm.name)
def test_every_family_trains_until_patience_rounds_pass_without_a_better_measure(data, arm):
    # With a patience the fixture's round cap allows, training ends `patience` rounds after the best one.
    plan = replace(loop.FIXTURE_LOOP, start=replace(loop.FIXTURE_LOOP.start, max_rounds=500, patience=5))
    row = loop.Loop(data, plan).run_arm(arm, plan.start)

    assert not row["hit_round_cap"]
    assert row["trained_rounds"] == row["rounds"] + 5


@pytest.mark.parametrize("arm", FAMILY_ARMS, ids=lambda arm: arm.name)
def test_a_family_arm_trains_on_its_own_parameters_not_the_xgboost_best_so_fars(data, arm):
    row = loop.Loop(data, loop.FIXTURE_LOOP).run_arm(arm, loop.FIXTURE_LOOP.start)

    assert row["settings"]["family"] == arm.changes["family"]
    assert row["settings"]["params"] == arm.changes["params"]


@pytest.mark.parametrize("arm", FAMILY_ARMS, ids=lambda arm: arm.name)
def test_adding_a_trade_agreement_never_lowers_the_familys_export_potential(data, arm):
    lp = loop.Loop(data, loop.FIXTURE_LOOP)
    settings = arm.apply(loop.FIXTURE_LOOP.start)
    agreement = lp.grid["trade_agreement"].dtype

    for weight in (0.0, 0.3):  # the model alone and blended
        scored = replace(settings, naive_weight=weight)
        off = lp.export_potential(scored, lp.grid.with_columns(trade_agreement=pl.lit(0, agreement)))
        on = lp.export_potential(scored, lp.grid.with_columns(trade_agreement=pl.lit(1, agreement)))
        assert (off > 0).all()
        assert (on >= off * (1 - 1e-6)).all()


@pytest.mark.parametrize("arm", FAMILY_ARMS, ids=lambda arm: arm.name)
def test_no_uk_pair_is_scored_by_a_family_model_that_trained_on_it(data, monkeypatch, arm):
    # Two folds: fold 0's UK pairs are scored, so the model may not see them; fold 1's train it.
    plan = replace(loop.FIXTURE_LOOP, n_folds=2)
    trained_on = []
    original = loop.runner.preparer

    def spy(**kwargs):
        prepare = original(**kwargs)

        def recorded(table):
            rows = prepare(table)
            trained_on.append(rows)
            return rows

        return recorded

    monkeypatch.setattr(loop.runner, "preparer", spy)
    lp = loop.Loop(data, plan)
    lp.run_arm(arm, plan.start)  # one arm, no noise floor: two folds are too few to rank

    scored = set(lp.grid.select("importer", "hs6").iter_rows())
    assert scored and trained_on
    uk = pl.concat(trained_on).filter(pl.col("exporter").cast(pl.String) == plan.uk)
    assert uk.height  # the other fold's UK pairs do train
    assert not scored & set(uk.select("importer", "hs6").iter_rows())


@pytest.mark.parametrize("arm", FAMILY_ARMS, ids=lambda arm: arm.name)
def test_a_family_model_is_cached_and_a_new_loop_scores_it_without_training(data, tmp_path, arm):
    first = loop.Loop(data, loop.FIXTURE_LOOP, tmp_path).run_arm(arm, loop.FIXTURE_LOOP.start)
    again = loop.Loop(data, loop.FIXTURE_LOOP, tmp_path).run_arm(arm, loop.FIXTURE_LOOP.start)

    assert first["trained"] is True and again["trained"] is False
    assert again["training_key"] == first["training_key"]
    assert again["selection_measure"] == pytest.approx(first["selection_measure"])


def test_every_pre_registered_family_arm_names_its_family_and_rechooses_its_blend_weight():
    assert list(loop.MODEL_FAMILY_ARMS) == [loop.LIGHTGBM, loop.CATBOOST]
    for family, arm in loop.MODEL_FAMILY_ARMS.items():
        assert arm.changes["family"] == family and arm.grid == loop.BLEND_GRID


@pytest.mark.parametrize("arm", FAMILY_ARMS, ids=lambda arm: arm.name)
def test_a_kept_family_becomes_the_best_so_far_with_market_reach_and_the_next_idea_starts_from_it(data, tmp_path, arm):
    family = arm.changes["family"]
    loop.Loop(data, loop.FIXTURE_LOOP, tmp_path).noise_floor()
    saved = loop.read_results(tmp_path)
    saved["noise_floor"]["win_threshold"] = -1.0  # any Arm wins, so the test can see a kept family
    loop.save_results(tmp_path, saved)
    lp = loop.Loop(data, loop.FIXTURE_LOOP, tmp_path)

    loop.run_model_family(lp, family, {family: arm})

    section = lp.results["experiments"][loop.model_family_experiment(family)]
    assert section["base_arm"]["settings"]["market_reach_beta"] == market_reach.CHOSEN_BETA
    assert section["kept"] == arm.name
    best = loop.Settings.from_dict(lp.results["best"]["settings"])
    assert best.family == family and best.market_reach_beta == market_reach.CHOSEN_BETA
    assert loop.served_base_arm(best).changes == {}  # the next idea runs against the kept family as it is
