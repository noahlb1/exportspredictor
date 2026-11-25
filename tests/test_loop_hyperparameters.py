"""The final round's hyperparameter search, staged experiments of the loop."""

import pytest

from exportspredictor import loop, market_reach


@pytest.fixture(scope="module")
def data(fixture_world):
    return loop.fixture_data(fixture_world)


def _models(folder) -> set[str]:
    return {p.name for p in (folder / loop.MODELS_DIR).glob("*.ubj")}


def test_a_stage_that_would_train_more_than_the_budget_is_refused_before_training_anything(data, tmp_path):
    lp = loop.Loop(data, loop.FIXTURE_LOOP, tmp_path)
    lp.noise_floor()
    before = _models(tmp_path)
    too_many = {"depth": [loop.Arm(f"max_depth {d}", {"params": {"max_depth": d}}) for d in range(3, 3 + loop.MAX_TRAINED_MODELS + 1)]}

    with pytest.raises(ValueError, match="budget"):
        loop.run_hyperparameter_stage(lp, "depth", too_many)

    assert _models(tmp_path) == before


def test_every_pre_registered_stage_fits_the_budget():
    # The search's order: depth, learning rate, min child weight, subsampling, regularisation.
    assert list(loop.HYPERPARAMETER_STAGES) == ["depth", "learning rate", "min child weight", "subsampling", "regularisation"]
    for arms in loop.HYPERPARAMETER_STAGES.values():
        assert 1 <= len(arms) <= loop.MAX_TRAINED_MODELS  # the served base is cached, so each Arm is one model
        assert all(set(arm.changes) == {"params"} and arm.grid == loop.BLEND_GRID for arm in arms)


STAGES = {
    "first": [loop.Arm("max_depth 3", {"params": {"max_depth": 3}}, loop.BLEND_GRID)],
    "second": [loop.Arm("eta 0.2", {"params": {"eta": 0.2}}, loop.BLEND_GRID)],
}


def _any_arm_wins(tmp_path):
    saved = loop.read_results(tmp_path)
    saved["noise_floor"]["win_threshold"] = -1.0  # any Arm wins, so the test can see a kept stage
    loop.save_results(tmp_path, saved)


def test_a_kept_stage_becomes_the_best_so_far_and_the_next_stage_stacks_on_it(data, tmp_path):
    loop.Loop(data, loop.FIXTURE_LOOP, tmp_path).noise_floor()
    _any_arm_wins(tmp_path)
    lp = loop.Loop(data, loop.FIXTURE_LOOP, tmp_path)

    loop.run_hyperparameter_stage(lp, "first", STAGES)
    first = lp.results["experiments"][loop.hyperparameter_experiment("first")]
    assert first["kept"] == "max_depth 3"
    assert first["base_arm"]["settings"]["market_reach_beta"] == market_reach.CHOSEN_BETA  # Reach added once
    best = lp.results["best"]["settings"]
    assert best["params"]["max_depth"] == 3 and best["market_reach_beta"] == market_reach.CHOSEN_BETA

    results = loop.run_hyperparameter_stage(lp, "second", STAGES)

    second = results["experiments"][loop.hyperparameter_experiment("second")]
    assert second["base_arm"]["settings"] == best  # the served base adds nothing to a kept stage
    assert second["base_arm"]["trained"] is False  # the kept Arm's cached model
    (arm,) = second["arms"]
    assert arm["settings"]["params"] == {**best["params"], "eta": 0.2}
    assert arm["settings"]["market_reach_beta"] == market_reach.CHOSEN_BETA


def test_a_stage_is_refused_until_the_stages_before_it_are_decided(data, tmp_path):
    lp = loop.Loop(data, loop.FIXTURE_LOOP, tmp_path)

    with pytest.raises(ValueError, match="first"):
        loop.run_hyperparameter_stage(lp, "second", STAGES)
    with pytest.raises(ValueError, match="unknown"):
        loop.run_hyperparameter_stage(lp, "third", STAGES)


def test_arms_whose_best_round_is_below_fifty_are_flagged_as_decided_by_the_stopping_rule():
    rows = [
        {"arm": "a", "rounds": 12},
        {"arm": "b", "rounds": 50},
        {"arm": "c", "rounds": 790},
        {"arm": "two-part", "rounds": 400, "parts": {"any_flow": {"rounds": 14}, "size": {"rounds": 400}}},
    ]

    assert loop.decided_by_stopping_rule(rows) == ["a", "two-part"]
