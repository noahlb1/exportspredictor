"""Governance: the model card, data lineage and drift checks of the write-up."""

import polars as pl
import pytest

from exportspredictor import evaluation, governance

# --- Model card --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def card(fixture_artefacts):
    return governance.model_card(fixture_artefacts)


def test_model_card_covers_the_sections_a_reader_needs(card):
    for section in ("intended_use", "out_of_scope", "metrics", "limitations", "data_sources", "decisions"):
        assert card[section], section


def test_model_card_names_no_owner_rather_than_omitting_one(card):
    # Course: print UNASSIGNED rather than omitting a section.
    assert card["owner"] == "UNASSIGNED"


def test_model_card_metrics_are_the_evaluated_numbers(card, fixture_artefacts):
    headline = fixture_artefacts.evaluation["headline"]
    metrics = {row["name"]: row for row in card["metrics"]}

    assert metrics["UK ranking (headline)"]["model"] == pytest.approx(headline["model"])
    assert metrics["UK ranking (headline)"]["naive"] == pytest.approx(headline["naive"])
    assert "Lift" in metrics and "Lift against pairs of similar size" in metrics


def test_model_card_carries_every_known_limitation(card):
    titles = {row["title"] for row in card["limitations"]}

    assert {row["title"] for row in evaluation.LIMITATIONS} <= titles
    assert "Trained on a sample of other Exporters' Flows" in titles


def test_model_card_renders_as_markdown_with_every_section(card):
    text = governance.render_model_card(card)

    for heading in ("## Intended use", "## Out-of-scope use", "## Metrics", "## Limitations", "## Data sources"):
        assert heading in text
    assert "UNASSIGNED" in text


def test_model_card_summarises_the_v1_comparison_and_the_gain_ladder(card, fixture_artefacts):
    block = fixture_artefacts.evaluation["v1_comparison"]
    summary = card["v1_comparison"]

    assert summary["verdict"] == block["verdict"]["text"]
    assert summary["v2_beats_v1"] == block["verdict"]["v2_beats_v1"]
    assert [step["to"] for step in summary["ladder"]] == ["A", "B", "C"]
    text = governance.render_model_card(card)
    assert "## Comparison with v1" in text
    assert block["verdict"]["text"] in text
    assert "decides nothing" in text


def test_a_model_card_without_a_v1_comparison_says_so(fixture_artefacts):
    from dataclasses import replace

    report = {k: v for k, v in fixture_artefacts.evaluation.items() if k != "v1_comparison"}
    card = governance.model_card(replace(fixture_artefacts, evaluation=report))
    assert card["v1_comparison"] is None
    assert "not been run" in governance.render_model_card(card)


def test_writing_the_model_card_puts_it_next_to_the_models(fixture_artefacts, tmp_path):
    path = governance.write_model_card(fixture_artefacts, tmp_path)

    assert path == tmp_path / "model_card.md"
    assert fixture_artefacts.version in path.read_text(encoding="utf-8")


# --- Data lineage --------------------------------------------------------------------------


def test_lineage_fingerprints_each_raw_file_that_exists(tmp_path):
    raw = tmp_path / "raw" / "hmrc"
    raw.mkdir(parents=True)
    (raw / "bdsexp_23archive.zip").write_bytes(b"abc")

    lineage = governance.data_lineage(tmp_path)

    row = lineage.filter(pl.col("file") == "raw/hmrc/bdsexp_23archive.zip").row(0, named=True)
    assert row["bytes"] == 3
    assert row["sha256"] == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"  # SHA-256 of "abc"
    assert row["source"].startswith("HMRC")
    assert lineage["file"].str.contains("baci").sum() == 0  # files not downloaded are not listed


# --- Drift --------------------------------------------------------------------------------


def test_psi_is_zero_for_identical_distributions():
    values = pl.Series(range(100), dtype=pl.Float64)

    assert governance.psi(values, values) == pytest.approx(0.0, abs=1e-12)


def test_psi_matches_a_worked_example():
    # Reference deciles each hold 10%. The current data sit half in the first decile and half in
    # the last; the eight empty bins are floored at 0.5%:
    # 2 x (0.5 - 0.1) ln(0.5 / 0.1) + 8 x (0.005 - 0.1) ln(0.005 / 0.1) = 3.5643
    reference = pl.Series(range(100), dtype=pl.Float64)
    current = pl.Series([0.0] * 50 + [99.0] * 50)

    assert governance.psi(reference, current) == pytest.approx(3.5643, abs=1e-3)


def test_feature_drift_rates_each_feature_with_the_course_thresholds():
    reference = pl.DataFrame({"steady": [float(i) for i in range(100)], "moved": [float(i) for i in range(100)]})
    current = pl.DataFrame({"steady": [float(i) for i in range(100)], "moved": [0.0] * 50 + [99.0] * 50})

    drift = governance.feature_drift(reference, current, ["steady", "moved"])

    assert drift["feature"].to_list() == ["moved", "steady"]  # largest first
    assert drift["status"].to_list() == ["act", "stable"]
