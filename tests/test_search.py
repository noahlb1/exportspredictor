"""The hyperparameter search: a tiny search on the fixture, end to end and resumable."""

from dataclasses import replace

import pytest

from exportspredictor import experiments, search


@pytest.fixture(scope="module")
def data(fixture_world):
    return experiments.fixture_data(fixture_world)


@pytest.fixture(scope="module")
def results(data):
    return search.run(data, data, search.FIXTURE_SEARCH)


def test_trial_0_is_v1s_configuration(results):
    first = results["trials"][0]

    assert first["number"] == 0
    assert first["v1_config"] is True
    assert first["params"] == {
        "max_depth": 8,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "tweedie_variance_power": 1.3,
    }
    assert sum(trial["v1_config"] for trial in results["trials"]) == 1


def test_the_trial_table_logs_each_trials_deviance_rounds_and_uk_spearman(results):
    trials = results["trials"]

    assert len(trials) == search.FIXTURE_SEARCH.n_trials
    for trial in trials:
        assert trial["state"] in {"COMPLETE", "PRUNED"}
        assert set(trial["params"]) == set(search.SPACE)
        assert {"validation_deviance", "rounds", "hit_round_cap", "uk_spearman", "seconds"} <= set(trial)
    assert results["capped_trials"] == [t["number"] for t in trials if t["hit_round_cap"]]


def test_the_winner_is_the_confirmed_configuration_with_the_lowest_deviance(results):
    confirmed = results["confirmation"]
    stage1 = sorted(
        (t for t in results["trials"] if t["state"] == "COMPLETE"), key=lambda t: t["validation_deviance"]
    )

    assert [row["trial"] for row in confirmed] == [t["number"] for t in stage1[: search.FIXTURE_SEARCH.top]]
    best = min(confirmed, key=lambda row: row["validation_deviance"])
    assert results["winner"]["trial"] == best["trial"]
    assert results["winner"]["params"] == {**best["params"], "eta": search.FIXTURE_SEARCH.eta}
    assert results["winner"]["rounds"] == best["rounds"]
    assert results["v1"]["trial"] == 0


def test_killing_and_rerunning_the_search_resumes_the_study(data, tmp_path):
    import optuna

    short = replace(search.FIXTURE_SEARCH, n_trials=3)
    search.run(data, data, short, tmp_path)
    # A run killed mid-trial leaves that trial running in the study.
    study = optuna.load_study(study_name=search.STUDY_NAME, storage=f"sqlite:///{(tmp_path / search.STUDY_FILE).as_posix()}")
    killed = study.ask()
    killed_params = search.suggest(killed, search.FIXTURE_SPACE)

    resumed = search.run(data, data, replace(short, n_trials=5), tmp_path)

    trials = resumed["trials"]
    assert trials[0]["v1_config"] and sum(t["v1_config"] for t in trials) == 1
    assert trials[killed.number]["state"] == "FAIL"
    retry = next(t for t in trials if t["retry_of"] == killed.number)
    assert retry["params"] == killed_params
    assert sum(t["state"] in {"COMPLETE", "PRUNED"} for t in trials) == 5
    assert experiments.read_results(tmp_path)["trials"] == trials


def test_the_search_never_reads_the_test_year(fixture_world):
    with pytest.raises(experiments.TestYearRead):
        experiments.fixture_data(fixture_world, validation_year=2023)


def test_the_round_count_scales_with_the_training_rows():
    assert search.scaled_rounds(1000, training_rows=120_000_000, tuning_rows=100_000_000) == 1200
    assert search.scaled_rounds(7, training_rows=110, tuning_rows=100) == 8
