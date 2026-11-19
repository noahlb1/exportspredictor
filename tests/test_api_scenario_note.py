"""The Scenario note (spec #66, ticket #84): every Scenario Effect is served as Illustrative, beside
the PPML benchmark, the model's own Trade Agreement association and the primary Event Backtest."""

from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from exportspredictor import artefacts as store

SCENARIO = {"add_agreements": ["IND"]}


@pytest.fixture(scope="module")
def client(fixture_artefacts_dir):
    with TestClient(create_app(fixture_artefacts_dir)) as client:
        yield client


def _note(artefacts, tmp_path) -> dict:
    store.save(artefacts, tmp_path)
    with TestClient(create_app(tmp_path)) as client:
        return client.post("/scenarios", json=SCENARIO).json()["note"]


def test_the_scenario_response_labels_its_effects_illustrative(client):
    note = client.post("/scenarios", json=SCENARIO).json()["note"]

    assert note["label"] == "Illustrative"
    assert "forecast" in note["text"]


def test_the_note_carries_the_saved_ppml_benchmark_with_its_interval(client, fixture_artefacts):
    saved = fixture_artefacts.ppml_benchmark["ppml"]

    benchmark = client.post("/scenarios", json=SCENARIO).json()["note"]["benchmark"]

    assert benchmark["run"] is True
    assert benchmark["effect_pct"] == pytest.approx(saved["effect_pct"])
    assert benchmark["low_pct"] == pytest.approx(saved["effect_low_pct"])
    assert benchmark["high_pct"] == pytest.approx(saved["effect_high_pct"])
    assert f"{saved['effect_pct']:+.1f}%" in benchmark["text"]


def test_the_note_carries_the_models_own_trade_agreement_association_on_the_served_form(client, fixture_artefacts):
    saved = fixture_artefacts.ppml_benchmark["model"]

    association = client.post("/scenarios", json=SCENARIO).json()["note"]["model_association"]

    assert association["run"] is True
    assert association["effect_pct"] == pytest.approx(saved["effect_pct"])
    assert association["pairs"] == saved["pairs"]
    assert association["model_form"] == saved["model_form"]["form"]
    assert "associates" in association["text"]


def test_the_note_carries_the_primary_event_backtests_predicted_and_observed_change(client, fixture_artefacts):
    primary = next(e for e in fixture_artefacts.evaluation["event_backtest"]["events"] if e["role"] == "primary")
    pooled = primary["directions"][0]["pooled"]

    backtest = client.post("/scenarios", json=SCENARIO).json()["note"]["event_backtest"]

    assert backtest["run"] is True
    assert backtest["title"] == primary["title"]
    assert backtest["direction"] == primary["directions"][0]["label"]
    assert backtest["predicted"] == pytest.approx(pooled["predicted"])
    assert backtest["observed"] == pytest.approx(pooled["observed"])
    assert [backtest["observed_low"], backtest["observed_high"]] == pytest.approx(pooled["observed_ci"])


def test_the_note_says_a_scenario_moves_only_the_models_share_of_the_blend(client, fixture_artefacts):
    weight = 1 - fixture_artefacts.metadata["model_form"]["naive_weight"]
    predicted = fixture_artefacts.evaluation["event_backtest"]["events"][0]["directions"][0]["pooled"]["predicted"]

    blend = client.post("/scenarios", json=SCENARIO).json()["note"]["blend"]

    assert blend["model_weight"] == pytest.approx(weight)
    # The served effect is the model's scaled by its weight on the log scale: (1 + e) ** w - 1
    assert blend["served_effect"] == pytest.approx((1 + predicted) ** weight - 1)
    assert "model alone" in blend["text"]


def test_with_no_saved_benchmark_the_note_keeps_the_label_and_says_the_benchmark_was_not_run(fixture_artefacts, tmp_path):
    note = _note(replace(fixture_artefacts, ppml_benchmark=None), tmp_path)

    assert note["label"] == "Illustrative"
    for part in ("benchmark", "model_association"):
        assert note[part]["run"] is False
        assert "not run for this version" in note[part]["text"]
    assert note["event_backtest"]["run"] is True


def test_with_no_event_backtests_the_note_keeps_the_label_and_says_they_were_not_run(fixture_artefacts, tmp_path):
    report = {key: value for key, value in fixture_artefacts.evaluation.items() if key != "event_backtest"}

    note = _note(replace(fixture_artefacts, evaluation=report), tmp_path)

    assert note["label"] == "Illustrative"
    assert note["event_backtest"]["run"] is False
    assert "not run for this version" in note["event_backtest"]["text"]
    assert note["benchmark"]["run"] is True


def test_the_labels_endpoint_carries_the_illustrative_label(client):
    label = client.get("/labels").json()["illustrative"]

    assert label["badge"] == "Illustrative"
    assert "forecast" in label["text"]


def test_with_no_evaluation_the_note_still_carries_the_label(fixture_artefacts, tmp_path):
    note = _note(replace(fixture_artefacts, evaluation=None), tmp_path)

    assert note["label"] == "Illustrative"
    assert note["event_backtest"]["run"] is False
    assert note["benchmark"]["run"] is True


def test_the_blocked_opportunities_response_carries_the_same_scenario_note(client):
    scenario_note = client.post("/scenarios", json=SCENARIO).json()["note"]

    blocked = client.get("/opportunities/blocked").json()

    assert blocked["note"] == scenario_note


def test_with_no_saved_benchmark_the_blocked_note_says_the_benchmark_was_not_run(fixture_artefacts, tmp_path):
    store.save(replace(fixture_artefacts, ppml_benchmark=None), tmp_path)
    with TestClient(create_app(tmp_path)) as client:
        note = client.get("/opportunities/blocked").json()["note"]

    assert note["label"] == "Illustrative"
    assert note["benchmark"]["run"] is False and "not run for this version" in note["benchmark"]["text"]
