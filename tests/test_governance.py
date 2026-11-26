"""Governance: the model card, data lineage and drift checks of the write-up."""

import polars as pl
import pytest

from exportspredictor import evaluation, governance, tariffs
from exportspredictor.reporting import history

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




def test_writing_the_model_card_puts_it_next_to_the_models(fixture_artefacts, tmp_path):
    path = governance.write_model_card(fixture_artefacts, tmp_path)

    assert path == tmp_path / "model_card.md"
    assert tmp_path.name in path.read_text(encoding="utf-8")


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


def test_model_card_names_the_year_the_applied_tariff_is_carried_forward_from(card):
    # v2 has the Applied Tariff as a policy feature (ADR 0006); only tariff changes after TRAINS ends are unseen.
    out_of_scope = " ".join(card["out_of_scope"])
    assert "Applied Tariff" in out_of_scope and str(tariffs.LAST_TRAINS_YEAR) in out_of_scope


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


def test_model_card_compares_scenario_effects_with_the_saved_gravity_benchmark(fixture_default_artefacts, fixture_artefacts):
    from dataclasses import replace

    saved = fixture_artefacts.ppml_benchmark
    benchmark = {**saved, "ppml": {**saved["ppml"], "effect_pct": 5.1, "effect_low_pct": -0.4, "effect_high_pct": 11.0},
                 "published_range": {"low_pct": 21.0, "high_pct": 100.0},
                 "model": {**saved["model"], "effect_pct": 58.1, "pairs": 678_216}}

    card = governance.model_card(replace(fixture_default_artefacts, ppml_benchmark=benchmark))

    limit = next(x for x in card["limitations"] if x["title"] == "Scenario Effects are far above a gravity benchmark")
    assert "+5.1%" in limit["text"] and "+58.1%" in limit["text"] and "−0.4% to +11.0%" in limit["text"]
    assert "+21%" in limit["text"] and "+100%" in limit["text"]
    assert "associates" in limit["text"]
    assert "PPML benchmark" not in [part["title"] for part in card["not_run"]]


def test_model_card_says_the_gravity_benchmark_was_not_run_for_a_version_without_it(fixture_default_artefacts):
    card = governance.model_card(fixture_default_artefacts)

    assert "gravity benchmark" not in " ".join(x["title"] for x in card["limitations"])
    assert "PPML benchmark: not run for this version" in governance.render_model_card(card)


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


def test_model_card_decisions_cite_every_adr_behind_the_model_and_the_map_decisions(card):
    decisions = " ".join(f"{d['title']} {d['text']}" for d in card["decisions"])

    for adr in ("ADR 0001", "ADR 0002", "ADR 0003", "ADR 0004", "ADR 0005", "ADR 0006", "ADR 0007"):
        assert adr in decisions, adr
    for topic in ("Gap Period", "Tree-structured Parzen Estimator", "ablation", "PPML"):
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


def test_the_sampling_decision_notes_the_measured_row_budget_against_the_planned_110_million(fixture_artefacts):
    from dataclasses import replace

    measured = replace(fixture_artefacts, metadata={**fixture_artefacts.metadata, "row_budget": 100_032_198})

    decisions = {row["title"]: row["text"] for row in governance.model_card(measured)["decisions"]}
    sampling = next(text for title, text in decisions.items() if "keeping every UK row" in title)

    assert "100,032,198" in sampling and "110 M" in sampling



def test_model_card_names_the_form_the_selection_measure_and_the_experiments(card):
    assert card["model_form"]["form"] == "blend" and card["model_form"]["naive_weight"]
    ranking = card["ranking_experiments"]
    assert ranking["selection_measure"] and ranking["experiments"]


def test_model_card_says_the_event_backtests_measure_the_model_alone_when_a_blend_is_served(card):
    events = [m for m in card["metrics"] if m["name"].startswith("Event Backtest")]
    assert events and all("model alone" in m["text"] for m in events)


# --- The card's lead ------------------------------------------------------


def _headings(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith("## ")]


def test_model_card_opens_with_the_headline_the_model_form_and_the_illustrative_label(card):
    headings = _headings(governance.render_model_card(card))

    assert headings[:3] == ["## Headline", "## Model Form", "## Illustrative Scenario Effects"]



def test_the_model_form_section_gives_the_blend_weight_from_the_artefact(card, fixture_artefacts):
    weight = fixture_artefacts.metadata["model_form"]["naive_weight"]
    assert card["model_form"]["naive_weight"] == weight

    section = _section(governance.render_model_card(card), "## Model Form")
    assert f"{weight:g}" in section and f"{1 - weight:g}" in section


def _section(text: str, heading: str) -> str:
    return text.split(f"\n{heading}\n", 1)[1].split("\n## ", 1)[0]


def test_the_illustrative_section_shows_the_saved_benchmark_figures(card, fixture_artefacts):
    saved = fixture_artefacts.ppml_benchmark
    section = _section(governance.render_model_card(card), "## Illustrative Scenario Effects")

    assert "Illustrative" in section
    assert f"{saved['ppml']['effect_pct']:+.1f}%" in section
    assert f"{saved['model']['effect_pct']:+.1f}%" in section


def test_the_illustrative_section_says_which_parts_were_not_run(fixture_default_artefacts):
    card = governance.model_card(fixture_default_artefacts)
    section = _section(governance.render_model_card(card), "## Illustrative Scenario Effects")

    assert "Illustrative" in section
    assert section.count("not run for this version") == 3  # benchmark, model association, Event Backtest


def test_model_card_reports_the_share_of_opportunities_inside_their_potential_range(card, fixture_artefacts):
    saved = fixture_artefacts.evaluation["potential_range"]
    assert card["potential_range"]["uncertain_opportunity_share"] == saved["uncertain_opportunity_share"]

    section = _section(governance.render_model_card(card), "## Potential Range")
    assert f"{saved['uncertain_opportunities']:,}" in section and f"{saved['opportunities']:,}" in section
    assert f"{saved['uncertain_opportunity_share']:.0%}" in section


def test_model_card_says_the_rounds_were_chosen_by_the_selection_measure(card, fixture_artefacts):
    rounds = fixture_artefacts.metadata["rounds"]
    assert card["round_count"]["chosen_by"] == "Selection Measure"
    assert card["round_count"]["rounds"] == rounds
    assert f"{rounds:,} rounds" in _section(governance.render_model_card(card), "## Model Form")


def test_model_card_discloses_rounds_chosen_on_deviance_and_the_next_rebuilds_count(fixture_artefacts):
    # The served real artefact (20261118T185910Z and its derived versions) kept v2's search rounds.
    card = governance.model_card(_deviance_rounds(fixture_artefacts))

    assert card["round_count"]["chosen_by"].startswith("Tweedie deviance")
    assert card["round_count"]["next_rounds"] == 1097  # the median of the confirmation's five folds
    section = _section(governance.render_model_card(card), "## Model Form")
    assert "1,453" in section and "1,198" in section and "1,097" in section


def _deviance_rounds(fixture_artefacts):
    """The served real artefact's shape: the search's 1,198 rounds scaled to 1,453, and a five-fold confirmation."""
    record = fixture_artefacts.metadata["ranking_experiments"]
    confirmed = {
        "n_folds": 5, "difference": 0.0429, "win_threshold": 0.0089, "beats_v2": True, "folds": [],
        "settings": {"winner": record["best"]["settings"]},
        "winner": {"selection_measure": 0.5635, "rounds": [951, 1075, 1097, 1303, 1120]},
        "v2": {"selection_measure": 0.5206}, "naive": {"selection_measure": 0.5472},
    }
    return _with_metadata(
        fixture_artefacts, rounds=1453, round_scaling={"tuned_rounds": 1198, "ratio": 1.2127, "rounds": 1453},
        ranking_experiments={**record, "confirmation": confirmed},
    )


@pytest.mark.parametrize("shape", ["selection measure", "deviance"])
def test_the_model_card_and_the_notebook_state_the_same_round_counts(fixture_artefacts, shape):
    artefacts = fixture_artefacts if shape == "selection measure" else _deviance_rounds(fixture_artefacts)

    stated = governance.model_card(artefacts)["round_count"]
    served, next_rebuild = history.round_count(artefacts).to_dicts()

    assert (stated["rounds_chosen"], stated["rounds"]) == (served["rounds_chosen"], served["rounds_trained"])
    assert stated["next_rounds"] == next_rebuild["rounds_chosen"]





def test_model_card_reports_potential_range_coverage_by_size(card, fixture_artefacts):
    saved = fixture_artefacts.evaluation["potential_range"]["by_size"]
    assert card["potential_range"]["by_size"] == saved

    section = _section(governance.render_model_card(card), "## Potential Range")
    assert "| Size of Export Potential | Held-out rows | Coverage |" in section
    for row in saved:
        assert f"| {row['size']} | {row['rows']:,} |" in section


def test_model_card_reports_what_market_reach_changes(card, fixture_artefacts):
    section = _section(governance.render_model_card(card), "## What Market Reach changes")
    # The card has no table, so it states every figure in the text (`figures=True`).
    assert card["market_reach_effect"]["text"] in section
    assert "Lift is" in section and "The Potential Range holds" in section and "Scenario Effects" in section



def test_model_card_states_low_uk_pair_coverage_as_the_api_does(fixture_artefacts):
    from dataclasses import replace

    report = fixture_artefacts.evaluation
    low = {"with_market_reach": 0.735, "without_market_reach": 0.807}
    artefacts = replace(fixture_artefacts, evaluation={
        **report, "potential_range": {**report["potential_range"], "uk_pair_coverage": low}})

    titles = {row["title"] for row in governance.model_card(artefacts)["limitations"]}

    assert "The Potential Range holds fewer UK pairs than its target" in titles
    assert {row["title"] for row in evaluation.known_limitations(artefacts)} <= titles




def test_the_scenario_note_states_the_models_own_trade_agreement_association(card, fixture_artefacts):
    assert fixture_artefacts.ppml_benchmark["model"]["model_form"].get("market_reach")

    section = _section(governance.render_model_card(card), "## Illustrative Scenario Effects")
    assert "This app's model associates adding a Trade Agreement with" in section


def test_model_card_states_the_blocked_opportunities_and_the_split_of_their_value(blocked_artefacts):
    # The Blocked world: 3 Blocked Opportunities in India and Hong Kong, $30.7m, of which whisky in India's Tariff
    # part is $7.4m (24%).
    card = governance.model_card(blocked_artefacts)
    text = governance.render_model_card(card)

    assert card["blocked_opportunities"]["blocked_opportunities"] == 3
    assert card["blocked_opportunities"]["markets"] == 2
    section = text.split("## Blocked Opportunities\n", 1)[1].split("\n## ", 1)[0]
    for words in ("3 Blocked Opportunities in 2 Markets", "$30.7m", "Tariff part", "$7.4m (24%)", "room to grow",
                  "Illustrative", "none of them"):
        assert words in section, words


def test_the_model_cards_blocked_opportunity_rule_includes_room_to_grow(card):
    text = governance.render_model_card(card)

    assert "A Blocked Opportunity gains at least 20% and $1M." not in text
    assert "room to grow" in text.split("## Decisions behind the model", 1)[1]


def test_the_card_states_every_served_limitation_in_order_then_its_own_extras(fixture_artefacts):
    # One source: the card builds on `known_limitations`, so the two lists cannot drift.
    served = evaluation.known_limitations(fixture_artefacts)

    card = governance.model_card(fixture_artefacts)["limitations"]

    assert card[: len(served)] == served
    assert [row["title"] for row in card[len(served):]] == ["Scenario Effects are far above a gravity benchmark"]


# --- One model (ADR 0013) -------------------------------------------------


def _sized(fixture_artefacts, model_deviance=2610.84, naive_deviance=2094.91):
    """The fixture with the served artefact's sizing: UK deviance above the Naive Baseline's."""
    from dataclasses import replace

    report = fixture_artefacts.evaluation
    uk = report["accuracy"]["uk_out_of_fold"]
    overall = {**uk["overall"], "model_tweedie_deviance": model_deviance, "naive_tweedie_deviance": naive_deviance}
    accuracy = {**report["accuracy"], "uk_out_of_fold": {**uk, "overall": overall}}
    return replace(fixture_artefacts, evaluation={**report, "accuracy": accuracy})


def test_the_headline_is_one_test_year_ranking_beside_the_naive_baseline(card, fixture_artefacts):
    headline = fixture_artefacts.evaluation["headline"]
    section = _section(governance.render_model_card(card), "## Headline")

    assert headline["text"] in section
    assert "| Version |" not in section  # no table of versions


def test_the_card_has_no_history_sections(card):
    text = governance.render_model_card(card)

    for removed in ("## Frozen results", "### Comparison with v1", "Gain ladder"):
        assert removed not in text, removed
    for key in ("v1_comparison", "earlier_versions", "large_markets", "traded_away"):
        assert key not in card, key


def test_the_alternatives_considered_on_2022_are_their_own_section(card):
    section = _section(governance.render_model_card(card), "## Alternatives considered on 2022")

    assert "| Experiment | Best Arm | Selection Measure |" in section


def test_the_confirmation_is_against_the_model_alone_in_either_saved_shape():
    confirmation = {"n_folds": 5, "difference": 0.0429, "win_threshold": 0.0089, "winner": 0.5635, "naive": 0.5332}
    for alone in ({"v2": 0.5206, "beats_v2": True}, {"model_alone": 0.5206}, {}):
        line = governance.confirmation_text({**confirmation, **alone}, 2022)
        assert "Confirmed on all 5 folds of 2022" in line and "+0.0429" in line and "(confirmed)" in line
        assert ("the model alone 0.5206" in line) == bool(alone)
        assert "v2" not in line


def test_the_limitations_lead_with_sizing_when_the_model_sizes_uk_flows_worse(fixture_artefacts):
    card = governance.model_card(_sized(fixture_artefacts))

    first = card["limitations"][0]
    assert first == evaluation.sizing_limitation(_sized(fixture_artefacts).evaluation)
    assert "2,611" in first["text"] and "2,095" in first["text"]


def test_the_title_carries_the_folder_s_own_artefact_id(fixture_artefacts, tmp_path):
    folder = tmp_path / "20991231T235959Z"
    folder.mkdir()

    text = governance.write_model_card(fixture_artefacts, folder).read_text(encoding="utf-8")

    assert text.splitlines()[0] == "# Model card: UK Exports Predictor, Artefact ID 20991231T235959Z"
    assert fixture_artefacts.version not in text.splitlines()[0]
