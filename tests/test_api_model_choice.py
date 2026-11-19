"""How the model was chosen (spec #66 stories 46–49), served by `/evaluation` from what the artefact records."""

from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from exportspredictor import artefacts as store


@pytest.fixture(scope="module")
def client(fixture_artefacts_dir):
    with TestClient(create_app(fixture_artefacts_dir)) as client:
        yield client


def _served(artefacts, tmp_path) -> dict:
    store.save(artefacts, tmp_path)
    with TestClient(create_app(tmp_path)) as client:
        return client.get("/evaluation").json()


def test_evaluation_serves_the_selection_measure_the_model_form_and_the_win_threshold(client, fixture_artefacts):
    recorded = fixture_artefacts.metadata["ranking_experiments"]

    choice = client.get("/evaluation").json()["model_choice"]

    assert choice["recorded"] is True
    assert choice["selection_measure"] == recorded["selection_measure"]
    assert choice["validation_year"] == recorded["validation_year"]
    assert choice["win_threshold"] == pytest.approx(recorded["noise_floor"]["win_threshold"])
    assert choice["model_form"]["form"] == "blend"
    assert choice["model_form"]["naive_weight"] == pytest.approx(fixture_artefacts.metadata["model_form"]["naive_weight"])
    assert "Naive Baseline" in choice["model_form"]["text"]


def test_evaluation_serves_every_experiment_and_whether_it_was_kept(client, fixture_artefacts):
    recorded = fixture_artefacts.metadata["ranking_experiments"]["experiments"]

    experiments = client.get("/evaluation").json()["model_choice"]["experiments"]

    assert [e["name"] for e in experiments] == list(recorded)
    for served, section in zip(experiments, recorded.values(), strict=True):
        assert served["kept"] == section.get("kept")
        best = next(row for row in section["arms"] if row.get("best"))
        assert served["best_arm"] == best["arm"]
        assert served["selection_measure"] == pytest.approx(best["selection_measure"])


CONFIRMATION = {
    "n_folds": 2, "validation_year": 2022, "difference": 0.0429, "win_threshold": 0.0089, "beats_v2": True,
    "winner": {"selection_measure": 0.5635}, "v2": {"selection_measure": 0.5206},
    "naive": {"selection_measure": 0.5332},
    "folds": [
        {"fold": 0, "v2": {"selection_measure": 0.5334}, "winner": {"selection_measure": 0.5771}},
        {"fold": 1, "v2": {"selection_measure": 0.5295}, "winner": {"selection_measure": 0.5745}},
    ],
}


def test_evaluation_serves_the_five_fold_confirmation_fold_by_fold(fixture_artefacts, tmp_path):
    ranking = {**fixture_artefacts.metadata["ranking_experiments"], "confirmation": CONFIRMATION}
    confirmed = replace(fixture_artefacts, metadata={**fixture_artefacts.metadata, "ranking_experiments": ranking})

    confirmation = _served(confirmed, tmp_path)["model_choice"]["confirmation"]

    assert confirmation["n_folds"] == 2 and confirmation["beats_v2"] is True
    assert confirmation["winner"] == 0.5635 and confirmation["v2"] == 0.5206 and confirmation["naive"] == 0.5332
    assert confirmation["difference"] == 0.0429 and confirmation["win_threshold"] == 0.0089
    assert confirmation["folds"] == [{"fold": 0, "winner": 0.5771, "v2": 0.5334}, {"fold": 1, "winner": 0.5745, "v2": 0.5295}]


def test_the_fixture_records_no_confirmation_and_serves_none(client):
    assert client.get("/evaluation").json()["model_choice"]["confirmation"] is None


def test_an_artefact_without_experiment_results_says_so(fixture_artefacts, tmp_path):
    metadata = {key: value for key, value in fixture_artefacts.metadata.items() if key != "ranking_experiments"}

    choice = _served(replace(fixture_artefacts, metadata=metadata), tmp_path)["model_choice"]

    assert choice["recorded"] is False
    assert choice["experiments"] == [] and choice["confirmation"] is None and choice["win_threshold"] is None
    assert "No experiment results" in choice["text"]
    assert choice["model_form"]["form"] == "blend"  # the served form is still named


V2 = {"version": "v2-version", "uk_ranking": 0.5115, "naive": 0.5233, "lift": 5.23, "size_matched_lift": 0.788,
      "tweedie_deviance_all_exporters": 355.5, "tweedie_deviance_uk": 568.6}


def _with_v2(fixture_artefacts, v2: dict = V2):
    from exportspredictor import evaluation

    report = {**fixture_artefacts.evaluation}
    report["earlier_versions"] = evaluation.earlier_versions(report, v2=v2)
    return replace(fixture_artefacts, evaluation=report)


def test_evaluation_serves_the_test_year_ranking_beside_the_naive_baseline_and_earlier_versions(
    fixture_artefacts, tmp_path
):
    artefacts = _with_v2(fixture_artefacts)
    block = artefacts.evaluation["earlier_versions"]

    test_year = _served(artefacts, tmp_path)["model_choice"]["test_year"]

    assert test_year["test_year"] == block["test_year"]
    assert [(r["name"], r["uk_ranking"]) for r in test_year["rows"]] == [(r["name"], r["uk_ranking"]) for r in block["rows"]]
    assert [r["uk_ranking"] for r in test_year["naive"]] == [r["uk_ranking"] for r in block["naive"]]
    assert test_year["beats_naive"] == block["beats_naive"] and test_year["text"] == block["text"]


def test_evaluation_says_what_the_blend_traded_away_with_the_deviance_read_from_the_artefact(
    fixture_artefacts, tmp_path
):
    this = fixture_artefacts.evaluation["accuracy"]["all_exporters"]["overall"]["model_tweedie_deviance"]
    v2 = {**V2, "tweedie_deviance_all_exporters": round(this / 2, 1)}  # v2 estimated size better
    artefacts = _with_v2(fixture_artefacts, v2)

    traded = _served(artefacts, tmp_path)["model_choice"]["traded_away"]

    assert traded["deviance"] == pytest.approx(this) and traded["earlier_deviance"] == round(this / 2, 1)
    assert traded["earlier_name"] == "v2"
    assert f"{this:,.1f}" in traded["text"] and f"{round(this / 2, 1):,.1f}" in traded["text"]
    assert "ranking" in traded["text"] and "dollar" in traded["text"]


def test_without_an_earlier_version_to_compare_nothing_is_said_to_be_traded_away(client):
    assert client.get("/evaluation").json()["model_choice"]["traded_away"] is None  # the fixture has no v2


def test_nothing_is_said_to_be_traded_away_when_the_deviance_is_no_worse(fixture_artefacts, tmp_path):
    this = fixture_artefacts.evaluation["accuracy"]["all_exporters"]["overall"]["model_tweedie_deviance"]
    artefacts = _with_v2(fixture_artefacts, {**V2, "tweedie_deviance_all_exporters": this * 2})

    assert _served(artefacts, tmp_path)["model_choice"]["traded_away"] is None


def test_the_event_backtest_note_says_it_measured_the_model_alone_and_served_effects_are_smaller(
    client, fixture_artefacts
):
    weight = 1 - fixture_artefacts.metadata["model_form"]["naive_weight"]

    note = client.get("/evaluation").json()["event_backtest_note"]

    assert "model alone" in note["text"]
    assert note["model_weight"] == pytest.approx(weight)
    predicted = fixture_artefacts.evaluation["event_backtest"]["events"][0]["directions"][0]["pooled"]["predicted"]
    assert note["model_effect"] == pytest.approx(predicted)
    assert note["served_effect"] == pytest.approx((1 + predicted) ** weight - 1)  # scaled on the log scale
    assert abs(note["served_effect"]) <= abs(note["model_effect"])


def test_no_event_backtest_note_when_the_model_alone_is_served(fixture_artefacts, tmp_path):
    metadata = {**fixture_artefacts.metadata, "model_form": {"form": "model alone", "naive_weight": 0.0}}

    assert _served(replace(fixture_artefacts, metadata=metadata), tmp_path)["event_backtest_note"] is None


def test_the_v1_comparisons_detail_is_still_not_served(client, fixture_artefacts):
    assert "v1_comparison" in fixture_artefacts.evaluation

    body = client.get("/evaluation").json()

    assert "v1_comparison" not in body and "v1_comparison" not in body["model_choice"]
    assert "ladder" not in str(body["model_choice"]) and "verdict" not in str(body["model_choice"])
