"""Offset Pairs (spec #129): an Opportunity whose Export Gap is at least half offset by the other Products of its
Heading in the same Market, shown with a caveat and scored, ranked and measured as normal.

Most tests use the Offset world (`conftest.offset_world`): Heading 8703 holds three Products in every Market, and
cars (870323) are offset by more than half in Japan, by exactly half in Hong Kong and by just under half in Canada.
"""

import io
import re
from pathlib import Path

import pandas as pd
import polars as pl
import pytest
import streamlit
from conftest import (
    BIG_CARS,
    CARS,
    SMALL_CARS,
    offset_recoded,
    run_page,
    with_gap_period_figures,
)
from fastapi.testclient import TestClient

from api.main import create_app
from exportspredictor import offsets, pipeline
from exportspredictor.caveats import OFFSET_PAIR
from exportspredictor.rankings import rank_opportunities

BIG_CARS_DESCRIPTION = "Motor cars, spark-ignition engine of more than 3,000 cc"
JAPAN_REASON = (
    "the UK already sells Japan $12.0m above Export Potential in the other Products of Heading 8703, mostly "
    + BIG_CARS_DESCRIPTION
)
HONG_KONG_REASON = (
    "the UK already sells Hong Kong $5.0m above Export Potential in the other Products of Heading 8703, mostly "
    + BIG_CARS_DESCRIPTION
)
FIGURES = ["hs6", "iso3", "export_potential_usd", "baseline_export_potential_usd"]


def _marked(bundle) -> set[tuple[str, str]]:
    return set(bundle.baseline.filter("offset_pair").select("hs6", "iso3").iter_rows())


def _reasons(bundle) -> dict[tuple[str, str], str]:
    rows = bundle.baseline.filter("offset_pair").select("hs6", "iso3", "offset_pair_reason").iter_rows()
    return {(hs6, iso3): reason for hs6, iso3, reason in rows}


# --- The fixture pipeline --------------------------------------------------------------------------------------------


def test_a_pair_offset_by_at_least_half_is_marked_and_one_just_under_half_is_not(offset_artefacts):
    marked = _marked(offset_artefacts)

    assert (CARS, "JPN") in marked and (CARS, "HKG") in marked  # offset by 12 of 10, and by exactly 5 of 10
    assert (CARS, "CAN") not in marked  # offset by 4.99 of 10


def test_only_opportunities_in_a_heading_with_a_neighbour_that_offsets_them_are_marked(offset_artefacts):
    # No pair without a positive Export Gap (small cars to Japan: zero, though the Heading over-sells), no Product
    # alone in its Heading (whisky, jewellery), and nothing in India, where every Product of the Heading under-sells.
    assert _marked(offset_artefacts) == {(CARS, "JPN"), (CARS, "HKG")}
    assert offset_artefacts.baseline.filter(~pl.col("offset_pair"))["offset_pair_reason"].is_null().all()


def test_the_reason_names_the_market_the_heading_the_offset_and_the_product_with_the_largest_surplus(
    offset_artefacts,
):
    assert _reasons(offset_artefacts) == {(CARS, "JPN"): JAPAN_REASON, (CARS, "HKG"): HONG_KONG_REASON}
    assert OFFSET_PAIR.caveat_text(JAPAN_REASON) == (
        f"Offset Pair: {JAPAN_REASON}. Its Export Gap may be trade recorded under a neighbouring Product."
    )


def test_recaveat_adds_the_tier_to_a_bundle_saved_without_it_and_changes_no_figure_or_rank(
    offset_world, offset_artefacts
):
    assert not set(OFFSET_PAIR.columns) & set(offset_world.baseline.columns)

    marked = offset_artefacts

    assert set(OFFSET_PAIR.columns) <= set(marked.baseline.columns)
    assert offsets.METADATA_KEY in marked.metadata and offsets.METADATA_KEY not in offset_world.metadata
    for column in ("export_potential_usd", "baseline_export_potential_usd", "actual_exports_usd", "export_gap_usd"):
        assert marked.baseline[column].equals(offset_world.baseline[column]), column
    order = ["hs6", "iso3", "export_gap_usd"]
    assert rank_opportunities(marked).select(order).equals(rank_opportunities(offset_world).select(order))
    assert marked.fold_models is offset_world.fold_models


def test_changing_a_pairs_actual_exports_can_change_its_headings_marks_but_no_export_potential(
    offset_world, offset_artefacts
):
    # Small cars to Canada sell $10,000 more: the rest of the Heading now offsets cars by exactly half.
    changed = pipeline.recaveat(with_gap_period_figures(offset_world, {("CAN", SMALL_CARS): (10, 9)}))

    assert _marked(changed) == {(CARS, "JPN"), (CARS, "HKG"), (CARS, "CAN")}
    assert changed.baseline.select(FIGURES).equals(offset_artefacts.baseline.select(FIGURES))


def test_the_metadata_records_the_threshold_and_the_reach_of_the_marked_baseline(offset_artefacts):
    saved = offset_artefacts.metadata[offsets.METADATA_KEY]
    recorded = offsets.OffsetPairReach.from_dict(saved)
    positive = offset_artefacts.baseline.filter(pl.col("export_gap_usd") > 0)

    assert saved["offset_pair_count"] == 2 and "offset_pairs" not in saved
    assert recorded.threshold == 0.5
    assert recorded.offset_pair_count == 2 and recorded.opportunities == positive.height
    assert recorded.in_top == {100: 2, 1000: 2, 10000: 2}
    assert recorded.offset_export_gap_usd == pytest.approx(20e6)  # cars to Japan and to Hong Kong, +$10m each
    assert recorded.share_of_export_gap == pytest.approx(20e6 / positive["export_gap_usd"].sum())
    assert offsets.OffsetPairReach.from_dict(recorded.to_dict()) == recorded


def test_the_training_run_marks_the_baseline_and_records_the_reach(fixture_artefacts):
    # The fixture world has one Product per Heading, so it has no Offset Pair.
    baseline = fixture_artefacts.baseline

    assert not baseline["offset_pair"].any() and baseline["offset_pair_reason"].is_null().all()
    recorded = offsets.OffsetPairReach.from_dict(fixture_artefacts.metadata[offsets.METADATA_KEY])
    assert (recorded.threshold, recorded.offset_pair_count, recorded.offset_export_gap_usd) == (0.5, 0, 0.0)
    assert recorded.in_top == {100: 0, 1000: 0, 10000: 0}


def test_rereach_marks_offset_pairs_again_from_its_new_figures(fixture_world, offset_artefacts):
    # Scored again by the fold models, cars to Japan sell above Export Potential (no Opportunity), while cars to Hong
    # Kong stay offset and big cars to Canada, now a small Opportunity, become offset by the rest of their Heading.
    scoring = pipeline.fixture_scoring_inputs(
        fixture_world, baseline_year=offset_artefacts.baseline_year,
        exporters=offset_artefacts.metadata["exporter_scope"],
    )
    inputs = pipeline.ScoringInputs(
        grids={year: offset_recoded(grid) for year, grid in scoring.grids.items()},
        calibration=offset_recoded(scoring.calibration),
        year_flows=lambda year: offset_recoded(scoring.year_flows(year)),
    )

    reached = pipeline.rereach(offset_artefacts, inputs, version="fixture-rereach")

    assert reached.baseline.filter(hs6=CARS, iso3="JPN")["export_gap_usd"].item() < 0
    assert _marked(offset_artefacts) == {(CARS, "JPN"), (CARS, "HKG")}
    assert _marked(reached) == {(CARS, "HKG"), (BIG_CARS, "CAN")}
    assert reached.baseline.filter(~pl.col("offset_pair"))["offset_pair_reason"].is_null().all()
    recorded = offsets.OffsetPairReach.from_dict(reached.metadata[offsets.METADATA_KEY])
    assert (recorded.offset_pair_count, recorded.in_top[100]) == (2, 2)
    # The other tiers are kept as the bundle marked them: Hong Kong is still a Re-export Hub.
    assert reached.baseline.filter(hs6=CARS, iso3="HKG")["reexport_hub"].item()


# --- The API ---------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client(offset_artefacts_dir):
    with TestClient(create_app(offset_artefacts_dir)) as client:
        yield client


def _get(client, path: str, **params) -> dict:
    response = client.get(path, params=params)
    assert response.status_code == 200, response.text
    return response.json()


def test_the_api_lists_the_tier_with_its_wording(client):
    tiers = {tier["column"]: tier for tier in _get(client, "/caveats")}

    offset = tiers["offset_pair"]
    assert (offset["badge"], offset["plural"], offset["key"], offset["hide_param"], offset["reason_column"]) == (
        "Offset Pair", "Offset Pairs", "pair", "hide_offset", "offset_pair_reason"
    )
    assert offset["text"] == OFFSET_PAIR.text and "neighbouring Product" in offset["text"]


def test_an_offset_pairs_ranking_row_and_detail_carry_the_flag_reason_and_caveat(client):
    rows = {(row["hs6"], row["iso3"]): row for row in _get(client, "/opportunities/top")["opportunities"]}

    japan = rows[(CARS, "JPN")]
    assert (japan["offset_pair"], japan["offset_pair_reason"]) == (True, JAPAN_REASON)
    assert OFFSET_PAIR.caveat_text(JAPAN_REASON) in japan["caveat_texts"]
    canada = rows[(CARS, "CAN")]
    assert (canada["offset_pair"], canada["offset_pair_reason"]) == (False, None)
    explained = _get(client, f"/explain/{CARS}/JPN")
    assert (explained["offset_pair"], explained["offset_pair_reason"]) == (True, JAPAN_REASON)
    assert explained["caveat_texts"] == japan["caveat_texts"]


@pytest.mark.parametrize("path", ["/opportunities/top", f"/opportunities/by-product/{CARS}", "/opportunities/blocked"])
def test_hiding_offset_pairs_leaves_them_out_and_keeps_every_other_rows_order(client, path):
    everything = _get(client, path)["opportunities"]

    shown = _get(client, path, hide_offset=True)["opportunities"]

    assert any(row["offset_pair"] for row in everything)
    assert shown == [row for row in everything if not row["offset_pair"]]


# --- The app ---------------------------------------------------------------------------------------------------------


def test_the_leaderboard_hides_offset_pairs_by_default_and_shows_their_badge_on_request(monkeypatch, offset_api_url):
    monkeypatch.setenv("EXPORTSPREDICTOR_API_URL", offset_api_url)
    page = run_page(query={"product": CARS})
    [*switch] = page.segmented_control
    switch[0].set_value("Wide table").run()

    assert set(page.dataframe[0].value["Market"]) == {"Canada", "India"}
    assert any("Offset Pairs are hidden by default" in element.value for element in page.main.caption)
    hide = page.sidebar.checkbox(key="hide_offset")
    assert hide.label == "Hide Offset Pairs" and hide.value is True

    hide.uncheck().run()

    caveats = dict(zip(page.dataframe[0].value["Market"], page.dataframe[0].value["Caveats"], strict=True))
    assert caveats["Japan"] == "Offset Pair" and caveats["Canada"] == ""
    assert caveats["Hong Kong"] == "Re-export Hub, Offset Pair"
    assert any("Offset Pairs shown" == button.label for button in page.main.button)
    assert page.query_params["hide_offset"] == "false"


def test_the_detail_panel_shows_the_caveat(monkeypatch, offset_api_url):
    monkeypatch.setenv("EXPORTSPREDICTOR_API_URL", offset_api_url)
    page = run_page(query={"product": CARS, "market": "JPN", "hide_offset": "false"})

    assert OFFSET_PAIR.caveat_text(JAPAN_REASON) in [element.value for element in page.warning]


def test_the_blocked_page_shows_the_badge_on_an_offset_pair(monkeypatch, offset_api_url):
    offered = []
    original = streamlit.download_button

    def record(label, data, *args, **kwargs):
        offered.append(data)
        return original(label, data, *args, **kwargs)

    monkeypatch.setattr(streamlit, "download_button", record)
    monkeypatch.setenv("EXPORTSPREDICTOR_API_URL", offset_api_url)
    page = run_page("pages/blocked_opportunities.py", query={"hide_offset": "false"})

    # The last CSV offered: `run_page` opens the Leaderboard, with its own CSV, before switching pages.
    csv = pd.read_csv(io.StringIO(offered[-1]), dtype={"HS6": str}, keep_default_na=False)
    caveats = {(row["HS6"], row["Market"]): row["Caveats"] for _, row in csv.iterrows()}
    assert caveats[(CARS, "Hong Kong")] == "Re-export Hub, Offset Pair"
    page.session_state["blocked_market"] = "HKG"
    page.session_state["blocked_pair"] = (CARS, "HKG")
    page.run()
    assert OFFSET_PAIR.caveat_text(HONG_KONG_REASON) in [element.value for element in page.warning]


def test_the_scenarios_page_shows_the_badge_on_an_offset_pair(monkeypatch, offset_api_url):
    monkeypatch.setenv("EXPORTSPREDICTOR_API_URL", offset_api_url)
    page = run_page("pages/scenarios.py")
    page.radio(key="kind").set_value("Custom Trade Agreements").run()
    page.multiselect(key="add").set_value(["HKG"]).run()
    page.number_input(key="min_share").set_value(0.0).run()
    page.number_input(key="min_usd").set_value(0.0).run()
    page.button(key="run").click().run()
    assert not page.exception

    pair_tables = [table.value for table in page.dataframe if "Tariff, Scenario %" in table.value.columns]
    caveats = dict(zip(pair_tables[0]["Product"], pair_tables[0]["Caveats"], strict=True))
    assert caveats["Motor cars, spark-ignition engine of 1,500-3,000 cc"] == "Re-export Hub, Offset Pair"


# --- Written up: the known limitation and the notebook's tables -------------------------------------------------------

# The reach the served artefact 20261120T221726Z records (spec #129, docs/research/offset-pairs.md).
SERVED_REACH = {
    "threshold": 0.5, "offset_pair_count": 246_790, "opportunities": 898_686,
    "in_top": {"100": 21, "1000": 159, "10000": 1_709},
    "offset_export_gap_usd": 18.97e9, "opportunity_export_gap_usd": 97.74e9, "share_of_export_gap": 0.194,
}
SPLIT_LIMITATION = "Right about a Heading, wrong about the split between its Products"


def test_a_version_with_offset_pairs_states_the_split_as_a_known_limitation(offset_artefacts):
    from dataclasses import replace

    from exportspredictor import evaluation, governance

    served = replace(offset_artefacts, metadata={**offset_artefacts.metadata, offsets.METADATA_KEY: SERVED_REACH})

    [limit] = [x for x in evaluation.known_limitations(served) if x["title"] == SPLIT_LIMITATION]
    assert "can be right about a Heading and wrong about the split between its Products" in limit["text"]
    assert "246,790 of 898,686 Opportunities (27%)" in limit["text"]
    assert "21 of the top 100, 159 of the top 1,000 and 1,709 of the top 10,000" in limit["text"]
    assert limit in governance.model_card(served)["limitations"]
    # The limitation states only what the version's metadata backs: no worked example from one served Baseline.
    assert not re.search(r"8411|France|Germany|turbo", limit["text"])


def test_a_version_without_offset_pairs_states_no_split_limitation(fixture_artefacts):
    from exportspredictor import evaluation

    assert fixture_artefacts.metadata[offsets.METADATA_KEY]["offset_pair_count"] == 0
    assert SPLIT_LIMITATION not in [x["title"] for x in evaluation.known_limitations(fixture_artefacts)]
    assert evaluation.offset_pair_limitations({}) == []


def test_the_notebook_states_the_reach_from_the_metadata(offset_artefacts):
    from dataclasses import replace

    from exportspredictor import reporting

    served = replace(offset_artefacts, metadata={**offset_artefacts.metadata, offsets.METADATA_KEY: SERVED_REACH})
    unmarked = replace(offset_artefacts, metadata={
        key: value for key, value in offset_artefacts.metadata.items() if key != offsets.METADATA_KEY
    })

    assert reporting.offset_pair_reach(served) == (
        "Offset Pairs: 246,790 of 898,686 Opportunities (27%); 21 of the top 100, 159 of the top 1,000 and 1,709 of "
        "the top 10,000; $19.0bn of $97.7bn of positive Export Gap (19%)."
    )
    assert reporting.offset_pair_reach(unmarked) == repr(reporting.NotRun("The Offset Pair tier"))


def test_the_heading_split_sets_each_product_beside_the_whole_heading(offset_artefacts):
    from exportspredictor import reporting

    split = reporting.heading_split(offset_artefacts, products=(CARS, BIG_CARS), markets=("JPN", "HKG"))

    assert split.columns == ["Market", "Figure", f"{CARS} ($m)", f"{BIG_CARS} ($m)", "Whole Heading 8703 ($m)"]
    assert split.rows() == [  # $m; the whole Heading also counts small cars
        ("Japan", "Actual Exports", 20.0, 22.0, 52.0),
        ("Japan", "Export Potential", 30.0, 10.0, 50.0),
        ("Japan", "Export Gap", 10.0, -12.0, -2.0),
        ("Hong Kong", "Actual Exports", 20.0, 14.0, 45.0),
        ("Hong Kong", "Export Potential", 30.0, 10.0, 50.0),
        ("Hong Kong", "Export Gap", 10.0, -4.0, 5.0),
    ]


@pytest.fixture(scope="module")
def served_bundle():
    from exportspredictor import artefacts as store
    from exportspredictor.paths import artefacts_dir

    try:
        bundle = store.load_latest(artefacts_dir())
    except store.ArtefactsNotFound:
        pytest.skip("No real artefacts under EXPORTSPREDICTOR_ARTEFACTS_DIR")
    except store.UnreadableArtefacts as error:
        pytest.skip(f"The newest real artefacts are not artefact format {store.ARTEFACT_FORMAT}: {error}")
    if bundle.metadata.get("source", "fixture") == "fixture":
        pytest.skip("The artefacts are fixture artefacts")
    return bundle


def test_on_the_served_version_the_two_largest_errors_are_offset_pairs_naming_heading_8411(served_bundle):
    # The worked example of spec #129: turbine parts to France and turbo-jets to Germany, each offset by the other.
    reasons = _reasons(served_bundle)

    france, germany = reasons[("841191", "FRA")], reasons[("841112", "DEU")]
    assert "Heading 8411" in france and "mostly Turbo-jets" in france
    assert "Heading 8411" in germany and "mostly Turbines: parts of turbo-jets" in germany


def test_no_api_or_app_code_names_the_tier():
    root = Path(__file__).resolve().parents[1]
    named = [
        path.relative_to(root).as_posix()
        for folder in ("api", "app")
        for path in sorted((root / folder).rglob("*.py"))
        if re.search(r"offset_pair|hide_offset|Offset Pair", path.read_text(encoding="utf-8"))
    ]

    assert not named