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


def _with_metadata(artefacts, **changes):
    from dataclasses import replace

    return replace(artefacts, metadata={**artefacts.metadata, **changes})


def test_model_card_reads_the_sampling_rates_and_exporter_scope_from_the_metadata(fixture_artefacts):
    scope = [f"X{i:02d}" for i in range(99)] + ["GBR"]
    artefacts = _with_metadata(
        fixture_artefacts, exporter_scope=scope, non_uk_sampling_rate=0.5894, zero_sampling_rate=0.05,
        uk_zero_sampling_rate=1.0,
    )

    card = governance.model_card(artefacts)

    assert card["training_scope"]["exporters"] == 100
    assert card["training_scope"]["uk_in_scope"] is True
    assert card["training_scope"]["non_uk_sampling_rate"] == 0.5894
    assert card["training_scope"]["zero_sampling_rate"] == 0.05
    text = governance.render_model_card(card)
    assert "58.9%" in text and "5%" in text and "100 Exporters" in text
    # v1's rates were hard-coded once; nothing in the card may still quote them.
    assert "30%" not in text and " 2%" not in text


def test_model_card_does_not_call_the_applied_tariff_a_missing_input(card):
    # v2 has the Applied Tariff as a policy feature (ADR 0006); only tariff changes after 2021 are unseen.
    out_of_scope = " ".join(card["out_of_scope"])
    sentences = out_of_scope.split(". ")
    assert not any("tariff" in s.lower() and "none of them is a model input" in s for s in sentences)
    assert "carried forward from 2021" in out_of_scope


def test_model_card_names_the_features_the_ablation_dropped(fixture_artefacts):
    experiments = {
        "ablation": [
            {"dropped": None}, {"dropped": "relatedness"}, {"dropped": "market_import_growth"},
        ],
        "kept_features": [f for f in fixture_artefacts.metadata["features"]],
    }
    card = governance.model_card(_with_metadata(fixture_artefacts, experiments=experiments))

    assert card["dropped_features"] == ["market_import_growth"]
    assert "Market import growth" in governance.render_model_card(card)


def test_model_card_states_that_the_potential_range_is_a_fixed_dollar_width_per_band(fixture_artefacts):
    from dataclasses import replace

    bands = pl.DataFrame({
        "band": [0, 1], "min_export_potential_usd": [0.0, 2_290_000.0], "rows": [1000, 1000],
        "low_offset_usd": [-500.0, -10_350_000.0], "high_offset_usd": [800.0, 12_530_000.0],
    })
    card = governance.model_card(replace(fixture_artefacts, potential_range_bands=bands))

    limit = next(x for x in card["limitations"] if x["title"] == "The Potential Range is a fixed width in dollars")
    # Every pair with Export Potential of $2.3m or more gets the same $22.9m-wide range, so a large
    # Opportunity is Uncertain only when its Export Gap is under the band's $10.3m lower offset.
    assert "$2.3m" in limit["text"] and "$22.9m" in limit["text"] and "$10.3m" in limit["text"]


def test_model_card_says_plainly_when_the_model_does_not_beat_the_naive_ranking(fixture_artefacts):
    from dataclasses import replace

    headline = {**fixture_artefacts.evaluation["headline"], "model": 0.511, "naive": 0.523, "beats_naive": False}
    card = governance.model_card(replace(fixture_artefacts, evaluation={**fixture_artefacts.evaluation, "headline": headline}))

    titles = {x["title"]: x["text"] for x in card["limitations"]}
    assert "Does not beat a naive ranking for UK pairs" in titles
    assert "0.51" in titles["Does not beat a naive ranking for UK pairs"]
    assert "0.52" in titles["Does not beat a naive ranking for UK pairs"]
    assert "Only slightly better than a naive ranking for UK pairs" not in titles


def test_model_card_compares_scenario_effects_with_the_gravity_benchmark_when_given(card, fixture_artefacts):
    benchmark = {"ppml": {"effect_pct": 5.1, "effect_low_pct": -0.4, "effect_high_pct": 11.0},
                 "model": {"effect_pct": 58.1, "pairs": 678_216}}

    with_benchmark = governance.model_card(fixture_artefacts, gravity_benchmark=benchmark)

    limit = next(x for x in with_benchmark["limitations"] if x["title"] == "Scenario Effects are far above a gravity benchmark")
    assert "+5.1%" in limit["text"] and "+58.1%" in limit["text"] and "−0.4% to +11.0%" in limit["text"]
    assert "associates" in limit["text"]
    assert "gravity benchmark" not in " ".join(x["title"] for x in card["limitations"])


_RANGE_REPORT = {
    "target": 0.8,
    "coverage": 0.86,
    "by_exporter": [
        {"exporter": "IRQ", "rows": 9_365, "coverage": 0.938},
        {"exporter": "CHN", "rows": 285_016, "coverage": 0.786},
        {"exporter": "DEU", "rows": 206_511, "coverage": 0.810},
        {"exporter": "AGO", "rows": 20_735, "coverage": 0.903},
    ],
}


def test_bias_review_rates_potential_range_coverage_per_exporter_against_its_target():
    table = governance.range_coverage_by_exporter(_RANGE_REPORT)

    assert table["exporter"].to_list() == ["CHN", "DEU", "AGO", "IRQ"]  # lowest coverage first
    assert table["status"].to_list() == ["near target", "near target", "too wide", "too wide"]
    assert table.filter(pl.col("exporter") == "AGO")["difference"].item() == pytest.approx(0.103)


def test_model_card_bias_review_reports_potential_range_coverage_by_exporter(fixture_artefacts):
    from dataclasses import replace

    report = {**fixture_artefacts.evaluation, "potential_range": _RANGE_REPORT}
    card = governance.model_card(replace(fixture_artefacts, evaluation=report))

    review = card["range_coverage_by_exporter"]
    assert review["exporters"] == 4
    assert review["lowest"] == {"exporter": "CHN", "coverage": pytest.approx(0.786)}
    assert review["too_wide"] == ["IRQ", "AGO"]  # largest excess first
    text = governance.render_model_card(card)
    assert "## Bias review: Potential Range coverage by Exporter" in text
    assert "CHN" in text and "79%" in text


def test_model_card_data_sources_include_trains_and_the_unsd_concordance(card):
    names = " ".join(source["name"] for source in card["data_sources"])

    assert "UNCTAD TRAINS" in names
    assert "HS 2017" in names and "HS 2012" in names


def test_model_card_decisions_cite_every_v2_adr_and_the_map_decisions(card):
    decisions = " ".join(f"{d['title']} {d['text']}" for d in card["decisions"])

    for adr in ("ADR 0001", "ADR 0002", "ADR 0003", "ADR 0004", "ADR 0005", "ADR 0006", "ADR 0007"):
        assert adr in decisions, adr
    for topic in ("Gap Period", "Tree-structured Parzen Estimator", "pre-registered", "ablation", "PPML"):
        assert topic in decisions, topic


def test_model_card_lists_the_excluded_products_by_reason(card, fixture_artefacts):
    reasons = fixture_artefacts.excluded_products["reason"].unique().to_list()

    assert {group["reason"] for group in card["excluded_products"]} == set(reasons)
    text = governance.render_model_card(card)
    assert "## Excluded Products" in text
    assert "270900" in text


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


def test_lineage_lists_the_trains_files_and_the_unsd_concordances_with_checksums(tmp_path):
    trains = tmp_path / "raw" / "trains" / "2021"
    trains.mkdir(parents=True)
    (trains / "AUS.parquet").write_bytes(b"abc")
    (tmp_path / "raw" / "trains" / "manifest.json").write_bytes(b"{}")
    (trains / "JPN.parquet.part").write_bytes(b"half")  # an interrupted download is not an input
    unsd = tmp_path / "raw" / "unsd"
    unsd.mkdir(parents=True)
    (unsd / "HS2017toHS2012ConversionAndCorrelationTables.xlsx").write_bytes(b"abc")

    lineage = governance.data_lineage(tmp_path)

    rows = {row["file"]: row for row in lineage.iter_rows(named=True)}
    assert set(rows) == {
        "raw/trains/2021/AUS.parquet", "raw/trains/manifest.json",
        "raw/unsd/HS2017toHS2012ConversionAndCorrelationTables.xlsx",
    }
    assert rows["raw/trains/2021/AUS.parquet"]["source"] == "UNCTAD TRAINS"
    assert rows["raw/trains/2021/AUS.parquet"]["sha256"] == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
    assert rows["raw/unsd/HS2017toHS2012ConversionAndCorrelationTables.xlsx"]["source"].startswith("UN HS")


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


def test_the_sampling_decision_notes_the_measured_row_budget_against_the_specs_figure(fixture_artefacts):
    from dataclasses import replace

    measured = replace(fixture_artefacts, metadata={**fixture_artefacts.metadata, "row_budget": 100_032_198})

    decisions = {row["title"]: row["text"] for row in governance.model_card(measured)["decisions"]}
    sampling = next(text for title, text in decisions.items() if "keeping every UK row" in title)

    assert "100,032,198" in sampling and "110 M" in sampling
