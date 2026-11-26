"""The reporting module: one chart or table per main result, read from a loaded artefact.

Tested on the fixture artefact for what each function returns (the rows behind a chart, the
table's rows, "not run" where a part is absent), not for how a chart looks.
"""

from dataclasses import replace
from html import escape

import polars as pl
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from exportspredictor import artefacts as store
from exportspredictor import reporting, scenarios
from exportspredictor.caveats import CaveatContext, with_caveat_columns
from exportspredictor.exclusions import SECTOR_71_REASON
from exportspredictor.volatility import ONE_OFF_REASON, VOLATILE_REASON


def test_a_chart_is_a_static_matplotlib_svg_image_naming_each_row(fixture_artefacts):
    chart = reporting.lift_by_tenth(fixture_artefacts)

    svg = chart._repr_svg_()
    assert svg.lstrip().startswith(("<?xml", "<svg"))
    assert "Matplotlib" in svg  # drawn by matplotlib, not by hand
    assert "<script" not in svg  # static: nothing GitHub would refuse to render
    for tenth in chart.data["tenth"]:
        assert escape(f"tenth {tenth}", quote=False) in svg  # text stays text, so labels are searchable


def test_a_chart_is_small_enough_for_a_committed_notebook_output(fixture_artefacts):
    chart = reporting.event_backtests(fixture_artefacts)

    assert len(chart._repr_svg_().encode()) < reporting.CHART_LIMIT_BYTES


def test_the_experiments_table_has_every_arm_and_whether_it_was_kept(fixture_artefacts):
    record = fixture_artefacts.metadata["ranking_experiments"]

    table = reporting.experiments(fixture_artefacts)

    arms = [row["arm"] for section in record["experiments"].values() for row in section["arms"]]
    assert list(table["arm"]) == arms
    # The fixture's loop ran the blend alone: the research note's experiment 1, in the Glossary's words.
    assert set(table["experiment"]) == {"1. Naive Baseline"}
    kept = [section["kept"] for section in record["experiments"].values()]
    for row in table.iter_rows(named=True):
        assert row["kept"] == (row["arm"] in kept)
        assert row["stopped_early"] == (row["rounds"] < reporting.EARLY_STOP_ROUNDS)


def test_without_experiment_results_the_experiments_are_not_run(fixture_artefacts):
    bare = replace(fixture_artefacts, metadata={**fixture_artefacts.metadata, "ranking_experiments": None})

    assert isinstance(reporting.experiments(bare), reporting.NotRun)
    assert "not run" in repr(reporting.experiments(bare))


def test_lift_by_tenth_reads_each_tenths_growth_rate_beside_the_base_rate(fixture_artefacts):
    lift = fixture_artefacts.evaluation["lift"]

    chart = reporting.lift_by_tenth(fixture_artefacts)

    assert chart.data["tenth"].to_list() == [row["decile"] for row in lift["by_decile"]]
    assert chart.data["rate"].to_list() == [row["rate"] for row in lift["by_decile"]]
    assert set(chart.data["base_rate"]) == {lift["base_rate"]}


def test_the_event_backtests_set_each_predicted_change_beside_the_observed_change(fixture_artefacts):
    events = fixture_artefacts.evaluation["event_backtest"]["events"]

    chart = reporting.event_backtests(fixture_artefacts)

    pooled = chart.data.filter(year="pooled")
    expected = [direction["pooled"] for event in events for direction in event["directions"]]
    assert pooled["predicted"].to_list() == [row["predicted"] for row in expected]
    assert pooled["observed"].to_list() == [row["observed"] for row in expected]
    assert pooled["observed_low"].to_list() == [row["observed_ci"][0] for row in expected]
    by_year = sum(len(direction["by_year"]) for event in events for direction in event["directions"])
    assert chart.data.height == by_year + len(expected)


def test_a_report_without_lift_or_event_backtests_says_they_were_not_run(fixture_artefacts):
    report = {k: v for k, v in fixture_artefacts.evaluation.items() if k not in ("lift", "event_backtest")}
    bare = replace(fixture_artefacts, evaluation=report)

    assert isinstance(reporting.lift_by_tenth(bare), reporting.NotRun)
    assert isinstance(reporting.event_backtests(bare), reporting.NotRun)


def test_the_ppml_table_reads_the_saved_benchmark(fixture_artefacts):
    saved = fixture_artefacts.ppml_benchmark

    table = reporting.ppml_benchmark(fixture_artefacts)

    assert table["effect_pct"][0] == pytest.approx(saved["ppml"]["effect_pct"])
    assert table["low_pct"][0] == pytest.approx(saved["ppml"]["effect_low_pct"])
    assert table["effect_pct"][1] == pytest.approx(saved["model"]["effect_pct"])
    assert isinstance(reporting.ppml_benchmark(replace(fixture_artefacts, ppml_benchmark=None)), reporting.NotRun)


def test_the_provenance_names_the_version_and_its_source(fixture_artefacts):
    text = reporting.provenance(fixture_artefacts)

    assert fixture_artefacts.version in text
    assert "source: fixture" in text


def test_the_provenance_names_the_version_a_blocked_value_rebuild_derived_from(fixture_artefacts):
    metadata = {**fixture_artefacts.metadata, "blocked_values_from": "20260101T000000Z"}

    assert "derived from version 20260101T000000Z" in reporting.provenance(replace(fixture_artefacts, metadata=metadata))


def test_the_provenance_names_the_latest_copy_step_and_what_it_added_beside_the_earlier_derivation(
    fixture_artefacts, tmp_path
):
    # 20261125T171812Z's shape: a copy of a version whose Blocked Opportunity values were rebuilt from a third, with
    # one evaluation block added.
    source = {**fixture_artefacts.metadata, "version": "20260201T000000Z", "blocked_values_from": "20260101T000000Z"}
    store.save(replace(fixture_artefacts, metadata=source), tmp_path)
    copy = replace(
        fixture_artefacts,
        metadata={**source, "version": "20260301T000000Z", "copied_from": "20260201T000000Z"},
        evaluation={**fixture_artefacts.evaluation, "new_block": {"rows": []}},
    )
    store.save(copy, tmp_path)

    text = reporting.provenance(copy, artefacts_root=tmp_path)

    assert "a copy of version 20260201T000000Z with new_block added to its evaluation" in text
    assert "derived from version 20260101T000000Z" in text
    # Where the source's folder is gone, the step alone.
    assert "a copy of version 20260201T000000Z; derived" in reporting.provenance(copy, artefacts_root=tmp_path / "elsewhere")
    assert reporting.provenance_versions(text) == [("20260301T000000Z", fixture_artefacts.metadata["source"])]
    # Without lineage (the Model History reads no older folder), the step alone, though the folder is there.
    assert "a copy of version 20260201T000000Z; derived" in reporting.provenance(
        copy, artefacts_root=tmp_path, lineage=False)


def test_provenance_lines_are_read_back_from_printed_outputs(fixture_artefacts):
    printed = f"other output\n{reporting.provenance(fixture_artefacts)}\nmore output"

    assert reporting.provenance_versions(printed) == [(fixture_artefacts.version, fixture_artefacts.metadata["source"])]
    assert reporting.provenance_versions("no provenance here") == []


def test_the_model_card_excerpt_is_its_headline_and_its_contents_with_a_link_to_the_whole(fixture_artefacts, tmp_path):
    from exportspredictor import governance

    path = governance.write_model_card(fixture_artefacts, tmp_path)
    card = path.read_text(encoding="utf-8")

    excerpt = reporting.model_card_excerpt(path)

    headings = [line for line in card.splitlines() if line.startswith("## ")]
    assert len(excerpt) < 4_000 < len(card)  # "Outputs stay small"
    assert "## Headline" in excerpt and card.split("## Headline", 1)[1].split("\n## ", 1)[0].strip() in excerpt
    for heading in headings:
        assert heading[3:] in excerpt, heading
    assert str(path.name) in excerpt
    assert "## Limitations" not in excerpt  # later sections are listed, not shown


def test_each_caveat_tier_s_reach_counts_its_opportunities_overall_and_in_the_top_of_the_ranking(fixture_artefacts):
    baseline = pl.DataFrame({
        "hs6": ["220830", "220830", "870323", "711319", "300490"],
        "iso3": ["SAU", "RUS", "RUS", "HKG", "JPN"],
        "export_gap_usd": [500.0, 400.0, 300.0, -50.0, 100.0],
    })
    bundle = replace(fixture_artefacts, baseline=with_caveat_columns(baseline, CaveatContext(baseline_year=2024)))

    reach = reporting.caveat_tier_reach(bundle, tops=(1, 3))

    assert reach.columns == ["Caveat Tier", "Opportunities", "in top 1", "in top 3"]
    assert reach.rows() == [
        ("Flagged Product", 0, 0, 0),  # the jewellery to Hong Kong is not an Opportunity
        ("Re-export Hub", 0, 0, 0),
        ("Sanctioned Market", 2, 0, 2),
        ("Restricted Pair", 1, 1, 1),
        ("Offset Pair", 0, 0, 0),  # the context holds no Offset Pair
    ]


def test_flagged_products_are_counted_by_reason(fixture_artefacts):
    baseline = pl.DataFrame({
        "hs6": ["711319", "711319", "300490", "300490", "880230", "870323"],
        "iso3": ["HKG", "JPN", "HKG", "JPN", "QAT", "JPN"],
        "export_gap_usd": [-50.0, 10.0, 100.0, 200.0, 300.0, 400.0],
    })
    reasons = {"300490": VOLATILE_REASON, "880230": ONE_OFF_REASON}
    marked = with_caveat_columns(baseline, CaveatContext(baseline_year=2024, flagged_product_reasons=reasons))

    counts = reporting.flagged_products_by_reason(replace(fixture_artefacts, baseline=marked))

    assert counts.rows() == [
        (VOLATILE_REASON, 1, 2),
        (SECTOR_71_REASON, 1, 1),
        (ONE_OFF_REASON, 1, 1),
    ]
    assert counts.columns == ["Reason", "Products", "Opportunities"]


def test_what_market_reach_changes_states_every_figure_or_a_not_run_line(fixture_artefacts):
    # The notebook has no table beside it, so the sentence states every figure, as the model card does.
    without = {k: v for k, v in fixture_artefacts.evaluation.items() if k != "market_reach_effect"}

    text = reporting.market_reach_effect(fixture_artefacts)
    assert text.startswith("Market Reach is the change this version keeps. On 2023, the UK ranking is ")
    assert "old measure" in text and "Lift is" in text and "Market Reach leaves Scenario Effects unchanged" in text
    assert reporting.market_reach_effect(replace(fixture_artefacts, evaluation=without)) == (
        "Market Reach: not run for this version."
    )


def test_what_market_reach_changes_states_the_served_uk_pair_coverage(fixture_artefacts):
    # The notebook prints what the model card and the API state.
    report = fixture_artefacts.evaluation
    effect = {**report["market_reach_effect"], "uk_pair_coverage": {"with": 0.7351, "without": 0.8072}}
    served = {**report["potential_range"], "uk_pair_coverage": {"with_market_reach": 0.7346, "without_market_reach": 0.8068}}
    stale = {**report, "market_reach_effect": effect, "potential_range": served}

    assert "holds 73% with it against 81% without" in reporting.market_reach_effect(
        replace(fixture_artefacts, evaluation=stale))


# --- Blocked Opportunities: the split of the value and how Product-specific its Agreement part is -----


def test_the_blocked_opportunity_split_counts_each_kind_of_tariff_and_parts_sum_to_the_value(blocked_artefacts):
    # The Blocked world: whisky ($18.5m, tariff 150%, Tariff part 40% of its value) and jewellery ($7.1m, tariff
    # above 0, Tariff part 0) in India; jewellery in Hong Kong ($5.1m, tariff 0%).
    split = {row["tariff_kind"]: row for row in reporting.blocked_opportunity_split(blocked_artefacts).iter_rows(named=True)}

    assert list(split) == ["Tariff above 0", "Tariff 0%", "Tariff unknown", "All"]
    assert [split[b]["blocked_opportunities"] for b in split] == [2, 1, 0, 3]
    assert [split[b]["markets"] for b in split] == [1, 1, 0, 2]
    everything = split["All"]
    assert everything["value_usd"] == pytest.approx(30.7e6, rel=0.01)
    assert everything["tariff_part_usd"] == pytest.approx(0.4 * 18.5e6, rel=0.01)
    assert everything["tariff_part_usd"] + everything["agreement_part_usd"] == pytest.approx(everything["value_usd"])
    assert everything["tariff_part_share"] == pytest.approx(7.4 / 30.7, rel=0.01)
    assert split["Tariff 0%"]["tariff_part_usd"] == 0
    assert everything["negative_agreement_parts"] == 0


def test_the_top_blocked_opportunities_table_lists_the_largest_with_their_barrier_and_both_parts(blocked_artefacts):
    table = reporting.top_blocked_opportunities(blocked_artefacts, limit=2)

    assert table.columns == [
        "hs6", "description", "market", "barrier", "baseline_export_potential_usd", "blocked_opportunity_value_usd",
        "tariff_part_usd", "agreement_part_usd", "above_export_potential",
    ]
    assert table["hs6"].to_list() == ["220830", "711319"]  # whisky ($18.5m) then jewellery in India ($7.1m)
    assert table["market"].to_list() == ["India", "India"]
    assert table["barrier"][0] == "No agreement, tariff 150%"
    assert table["tariff_part_usd"][0] == pytest.approx(0.4 * 18.5e6, rel=0.01)


def test_a_tariff_part_above_its_value_is_counted_as_a_negative_agreement_part(blocked_artefacts):
    # The Agreement part is the remainder, so it is negative wherever the Tariff part exceeds the value.
    jewellery_india = (pl.col("hs6") == "711319") & (pl.col("iso3") == "IND")
    baseline = blocked_artefacts.baseline.with_columns(
        tariff_part_usd=pl.when(jewellery_india).then(pl.col("blocked_opportunity_value_usd") * 1.5)
        .otherwise(pl.col("tariff_part_usd"))
    )
    split = reporting.blocked_opportunity_split(replace(blocked_artefacts, baseline=baseline))
    everything = split.filter(pl.col("tariff_kind") == "All").row(0, named=True)

    assert everything["negative_agreement_parts"] == 1
    assert everything["negative_agreement_part_usd"] == pytest.approx(-0.5 * 7.1e6, rel=0.02)


def _description(bundle, hs6):
    return bundle.products.filter(pl.col("hs6") == hs6)["description"].item()


def test_the_agreement_part_variation_spreads_its_share_between_products_within_each_top_market(blocked_artefacts):
    # India: whisky's Agreement part is 18% of its Baseline Export Potential ($11.1m of $61.6m), jewellery's 25%
    # ($7.1m of $28.3m). Whisky leads by value, jewellery by share: sorting by the share reverses India's order.
    variation = reporting.agreement_part_variation(blocked_artefacts)
    india, hong_kong = variation.iter_rows(named=True)

    assert [india["iso3"], hong_kong["iso3"]] == ["IND", "HKG"]  # largest Blocked Opportunity total first
    assert india["blocked_opportunities"] == 2
    assert india["share_min"] == pytest.approx(0.180, abs=0.002)
    assert india["share_max"] == pytest.approx(0.251, abs=0.002)
    assert india["order_correlation"] == pytest.approx(-1.0)
    assert india["first_by_value"] == _description(blocked_artefacts, "220830")
    assert india["first_by_share"] == _description(blocked_artefacts, "711319")
    # One Product: no spread and no order to compare.
    assert hong_kong["share_min"] == hong_kong["share_max"]
    assert hong_kong["order_correlation"] is None


def test_the_agreement_part_variation_counts_the_products_in_both_top_lists(blocked_artefacts):
    # How many of a Market's top Products by value are also among its top Products by share: India has two
    # Products, so with the top one of each it shares none (whisky leads by value, jewellery by share).
    variation = reporting.agreement_part_variation(blocked_artefacts, top=1)

    assert variation["top_shared"].to_list() == [0, 1]


def test_the_agreement_part_variation_keeps_only_the_top_markets(blocked_artefacts):
    assert reporting.agreement_part_variation(blocked_artefacts, markets=1)["iso3"].to_list() == ["IND"]


def test_the_agreement_part_variation_chart_draws_each_markets_median_share_and_its_range(blocked_artefacts):
    chart = reporting.agreement_part_variation_chart(blocked_artefacts)

    assert chart.data["iso3"].to_list() == ["IND", "HKG"]
    assert "India" in chart.svg


def test_an_artefact_without_blocked_opportunities_has_an_empty_split_and_variation(fixture_artefacts):
    # The unscaled fixture's values are all under $1M, so it has no Blocked Opportunity.
    everything = reporting.blocked_opportunity_split(fixture_artefacts).filter(pl.col("tariff_kind") == "All").row(0, named=True)

    assert everything["blocked_opportunities"] == 0 and everything["value_usd"] == 0
    assert reporting.agreement_part_variation(fixture_artefacts).is_empty()
    assert reporting.agreement_part_variation_chart(fixture_artefacts).data.is_empty()


# --- The Trading Bloc presets: the Markets each adds a Trade Agreement with and their Blocked Opportunities --


@pytest.fixture(scope="module")
def india_in_cptpp(blocked_artefacts):
    """The Blocked world with India a CPTPP member, so the CPTPP preset adds a Trade Agreement with India (Japan and
    Canada, the other members, already have one). No Market is an EU member, so the EU presets change nothing."""
    features = blocked_artefacts.baseline_features
    cptpp = features.with_columns(
        pl.when(pl.col("iso3") == "IND").then(1).otherwise(pl.col("cptpp_d")).cast(features.schema["cptpp_d"])
        .alias("cptpp_d")
    )
    return replace(blocked_artefacts, baseline_features=cptpp)


def _preset_results(bundle, names):
    return {
        name: scenarios.run(
            bundle.baseline_features, bundle.baseline, bundle.fold_models, scenarios.Scenario(preset=name),
            form=bundle.model_form,
        )
        for name in names
    }


def test_a_preset_adding_an_agreement_shows_the_precomputed_blocked_opportunities_of_that_market(
    india_in_cptpp, blocked_artefacts_dir
):
    table = reporting.preset_scenarios(india_in_cptpp, _preset_results(india_in_cptpp, ["cptpp"]))
    [cptpp] = table.iter_rows(named=True)
    with TestClient(create_app(blocked_artefacts_dir)) as client:
        served = client.get("/opportunities/blocked", params={"market": "IND"}).json()["summary"]

    assert cptpp["preset"] == "CPTPP"
    assert cptpp["pairs_changed"] > 0
    assert cptpp["added_agreements"] == ["India"]
    # Whisky ($18.5m) and jewellery ($7.1m) in India: one source with the Blocked Opportunities page (ADR 0012).
    assert cptpp["blocked_opportunities"] == 2 == served["blocked_opportunities"]
    assert cptpp["blocked_opportunity_value_usd"] == pytest.approx(25.6e6, rel=0.01)
    assert cptpp["blocked_opportunity_value_usd"] == pytest.approx(served["blocked_opportunity_value_usd"])


def test_a_preset_adding_no_agreement_shows_no_blocked_opportunities_not_zero(india_in_cptpp):
    table = reporting.preset_scenarios(india_in_cptpp, _preset_results(india_in_cptpp, ["eu_customs_union"]))
    [union] = table.iter_rows(named=True)

    assert union["preset"] == "EU Customs Union only"
    assert union["added_agreements"] == []
    assert union["blocked_opportunities"] is None
    assert union["blocked_opportunity_value_usd"] is None


def test_preset_rows_come_in_the_order_given(india_in_cptpp):
    names = ["leave", "cptpp", "eu_rejoin"]
    table = reporting.preset_scenarios(india_in_cptpp, _preset_results(india_in_cptpp, names))

    assert table["preset"].to_list() == ["Leave", "CPTPP", "EU re-join"]
    assert table["blocked_opportunities"].to_list() == [None, 2, None]
