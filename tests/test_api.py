"""API contract tests: FastAPI TestClient against fixture artefacts (the main seam)."""

import math
from dataclasses import replace

import polars as pl
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from exportspredictor import artefacts as store
from exportspredictor.fixture import PRODUCTS
from exportspredictor.pipeline import build_artefacts

FIXTURE_HS6 = "220830"


@pytest.fixture(scope="module")
def client(fixture_artefacts_dir):
    with TestClient(create_app(fixture_artefacts_dir)) as client:
        yield client


def test_health_reports_version_baseline_year_and_gap_period_once_loaded(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "ready": True,
        "artefact_version": "fixture-test",
        "baseline_year": 2024,
        "gap_years": [2022, 2024],
    }


def test_health_is_not_ready_until_artefacts_load(fixture_artefacts, tmp_path):
    now = [0.0]
    with TestClient(create_app(tmp_path, retry_seconds=60, clock=lambda: now[0])) as client:
        before = client.get("/health")
        store.save(fixture_artefacts, tmp_path)
        within_interval = client.get("/health")
        now[0] = 61.0
        after = client.get("/health")

    assert before.status_code == 503
    assert before.json()["ready"] is False
    # A failed load is not retried on every request (no disk scan per call) ...
    assert within_interval.status_code == 503
    # ... but once the retry interval has passed, the artefacts load.
    assert after.status_code == 200
    assert after.json()["ready"] is True


def test_health_is_not_ready_when_the_newest_folder_is_v1(fixture_artefacts, tmp_path):
    store.save(fixture_artefacts, tmp_path)
    v1 = {k: v for k, v in fixture_artefacts.metadata.items() if k != "artefact_format"}
    store.save(replace(fixture_artefacts, metadata={**v1, "version": "zzz-v1"}), tmp_path)

    with TestClient(create_app(tmp_path)) as client:
        response = client.get("/health")

    assert response.status_code == 503
    assert response.json()["ready"] is False
    assert "zzz-v1" in response.json()["reason"]
    assert "no artefact_format" in response.json()["reason"]


def test_opportunities_need_loaded_artefacts(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        response = client.get(f"/opportunities/by-product/{FIXTURE_HS6}")

    assert response.status_code == 503


def test_by_product_returns_every_market_sorted_by_export_gap(client):
    body = client.get(f"/opportunities/by-product/{FIXTURE_HS6}").json()

    gaps = [row["export_gap_usd"] for row in body["opportunities"]]
    assert gaps == sorted(gaps, reverse=True)
    assert {row["iso3"] for row in body["opportunities"]} == {"JPN", "CAN", "IND", "HKG"}
    assert body["product"] == {"hs6": "220830", "description": "Whiskies", "sector": "22"}
    assert body["baseline_year"] == 2024
    assert body["gap_years"] == [2022, 2024]  # the values are Gap Period means
    assert body["artefact_version"] == "fixture-test"


def test_by_product_rows_have_the_documented_shape(client):
    row = client.get(f"/opportunities/by-product/{FIXTURE_HS6}").json()["opportunities"][0]

    assert set(row) == {
        "iso3",
        "market",
        "region",
        "export_potential_usd",
        "actual_exports_usd",
        "export_gap_usd",
        "relative_export_gap",
        "opportunity_type",
        "trade_agreement",
        "blocked_effect_usd",
        "blocked_opportunity",
        "flagged_product",
        "reexport_hub",
        "potential_range_low_usd",
        "potential_range_high_usd",
        "uncertain",
    }


def test_unknown_product_returns_404_with_a_message(client):
    response = client.get("/opportunities/by-product/999999")

    assert response.status_code == 404
    assert "999999" in response.json()["detail"]


def test_products_search_finds_a_product_by_its_code(client):
    response = client.get("/products", params={"q": "851712"})

    assert response.status_code == 200
    assert [product["hs6"] for product in response.json()] == ["851712"]


def test_products_search_finds_a_product_by_an_everyday_word(client):
    # The HS 2012 description says "Telephones for cellular networks", never "smartphone".
    hs6s = [product["hs6"] for product in client.get("/products", params={"q": "Smartphone"}).json()]

    assert hs6s[0] == "851712"


def test_products_search_matches_words_in_the_description(client):
    body = client.get("/products", params={"q": "jewellery precious"}).json()

    assert body == [
        {"hs6": "711319", "description": PRODUCTS["711319"], "sector": "71", "excluded": False, "exclusion_reason": None}
    ]


def test_products_search_by_code_prefix_lists_the_sector(client):
    hs6s = [product["hs6"] for product in client.get("/products", params={"q": "22"}).json()]

    assert hs6s == ["220830"]


def test_unknown_search_text_returns_an_empty_list(client):
    response = client.get("/products", params={"q": "zzz-not-a-product"})

    assert response.status_code == 200
    assert response.json() == []


def test_markets_lists_every_market_with_name_and_region(client):
    response = client.get("/markets")

    assert response.status_code == 200
    assert {market["iso3"]: market for market in response.json()}["JPN"] == {
        "iso3": "JPN",
        "name": "Japan",
        "region": "East Asia & Pacific",
    }
    assert {market["iso3"] for market in response.json()} == {"JPN", "CAN", "IND", "HKG"}


def test_evaluation_returns_the_report_with_headline_breakdowns_and_limitations(client):
    response = client.get("/evaluation")

    assert response.status_code == 200
    body = response.json()
    assert body["artefact_version"] == "fixture-test"
    assert body["test_year"] == 2023
    assert set(body["accuracy"]) == {"all_exporters", "uk_out_of_fold"}
    for scope in body["accuracy"].values():
        # by_income is empty without income levels
        assert set(scope) == {"overall", "by_sector", "by_region", "by_income", "by_tariff"}
        assert scope["by_sector"] and scope["by_region"]
    assert body["headline"]["text"]
    assert isinstance(body["headline"]["beats_naive"], bool)
    assert body["stability"]["years"] == [2023, 2024]
    titles = " ".join(item["title"] for item in body["limitations"]).lower()
    assert "goods only" in titles and "tariffs" in titles


def test_evaluation_returns_the_imputed_observed_split_and_v2s_limitations(client):
    body = client.get("/evaluation").json()

    for scope in body["accuracy"].values():
        split = scope["by_tariff"]
        assert [row["group"] for row in split] == ["observed", "imputed"]
        assert sum(row["rows"] for row in split) == scope["overall"]["rows"]
    assert "2021" in body["tariff_split_text"]
    titles = [item["title"] for item in body["limitations"]]
    assert "No tariffs" not in titles
    assert {"Tariffs carried forward from 2021", "Excluded Products", "Trained on the top 100 Exporters",
            "Values are nominal US dollars", "What the Potential Range assumes"} <= set(titles)
    assert 0 <= body["potential_range"]["coverage"] <= 1
    assert "MFN" in body["event_backtest"]["events"][0]["states"]


def test_evaluation_is_404_when_the_artefact_has_none(fixture_artefacts, tmp_path):
    from dataclasses import replace

    store.save(replace(fixture_artefacts, evaluation=None), tmp_path)
    with TestClient(create_app(tmp_path)) as client:
        response = client.get("/evaluation")

    assert response.status_code == 404
    assert "evaluation" in response.json()["detail"].lower()


def test_by_market_returns_every_product_sorted_by_export_gap(client):
    body = client.get("/opportunities/by-market/JPN").json()

    gaps = [row["export_gap_usd"] for row in body["opportunities"]]
    assert gaps == sorted(gaps, reverse=True)
    assert {row["hs6"] for row in body["opportunities"]} == {"711319", "220830", "300490", "851712", "870323"}
    assert body["market"] == {"iso3": "JPN", "name": "Japan", "region": "East Asia & Pacific"}
    assert body["baseline_year"] == 2024
    assert body["artefact_version"] == "fixture-test"


def test_by_market_rows_name_the_product(client):
    rows = client.get("/opportunities/by-market/JPN").json()["opportunities"]

    assert {"hs6": "220830", "description": "Whiskies", "sector": "22"}.items() <= next(
        row for row in rows if row["hs6"] == "220830"
    ).items()


def test_unknown_market_returns_404_with_a_message(client):
    response = client.get("/opportunities/by-market/XYZ")

    assert response.status_code == 404
    assert "XYZ" in response.json()["detail"]


def _ranked(client, path, **params) -> list[dict]:
    response = client.get(path, params=params)
    assert response.status_code == 200, response.text
    return response.json()["opportunities"]


def test_sector_filter_keeps_only_that_sectors_products(client):
    rows = _ranked(client, "/opportunities/by-market/JPN", sector="85")

    assert [row["hs6"] for row in rows] == ["851712"]


def test_region_filter_keeps_only_that_regions_markets(client):
    rows = _ranked(client, f"/opportunities/by-product/{FIXTURE_HS6}", region="South Asia")

    assert [row["iso3"] for row in rows] == ["IND"]


def test_filters_apply_on_both_views(client):
    # Japan is not in South Asia, and whisky is not in Sector 85.
    assert _ranked(client, "/opportunities/by-market/JPN", region="South Asia") == []
    assert _ranked(client, f"/opportunities/by-product/{FIXTURE_HS6}", sector="85") == []


def test_min_gap_usd_filter_drops_smaller_export_gaps(client):
    everything = _ranked(client, "/opportunities/by-market/JPN")
    rows = _ranked(client, "/opportunities/by-market/JPN", min_gap_usd=0)

    assert rows and all(row["export_gap_usd"] >= 0 for row in rows)
    assert rows == [row for row in everything if row["export_gap_usd"] >= 0]
    assert len(rows) < len(everything)


def test_limit_keeps_the_top_of_the_ranking(client):
    everything = _ranked(client, "/opportunities/by-market/JPN")

    assert _ranked(client, "/opportunities/by-market/JPN", limit=2) == everything[:2]


def _keys(value) -> set[str]:
    """Every key in a JSON body, at any depth."""
    if isinstance(value, dict):
        return set(value) | {key for item in value.values() for key in _keys(item)}
    if isinstance(value, list):
        return {key for item in value for key in _keys(item)}
    return set()


def _every_response(client) -> list:
    """One body from every endpoint (USD only: none may carry £ values or a £ rate)."""
    gets = [
        "/health",
        "/products",
        "/markets",
        "/opportunities/rank-by",
        f"/opportunities/by-product/{FIXTURE_HS6}",
        "/opportunities/by-market/JPN",
        "/evaluation",
        f"/explain/{FIXTURE_HS6}/JPN",
        "/scenarios/presets",
        "/scenarios/thresholds",
    ]
    bodies = [client.get(path).json() for path in gets]
    bodies.append(client.post("/scenarios", json={"add_agreements": ["IND"]}).json())
    return bodies


def test_no_response_carries_gbp_values_or_a_gbp_rate(client):
    for body in _every_response(client):
        keys = {key.lower() for key in _keys(body)}
        assert not {key for key in keys if "gbp" in key}
        assert "£" not in str(body)


def test_min_gap_usd_filters_rankings(client):
    everything = _ranked(client, "/opportunities/by-market/JPN")
    threshold_usd = sorted(row["export_gap_usd"] for row in everything)[len(everything) // 2]  # the median

    rows = _ranked(client, "/opportunities/by-market/JPN", min_gap_usd=threshold_usd)

    assert rows == [row for row in everything if row["export_gap_usd"] >= threshold_usd]
    assert 0 < len(rows) < len(everything)


def test_the_old_gbp_gap_filter_is_gone(client):
    response = client.get("/opportunities/by-market/JPN", params={"min_gap_gbp": 5000})

    assert response.status_code == 422


def test_rows_show_whether_a_uk_trade_agreement_is_in_force_with_the_market(client):
    # The fixture's UK Trade Agreements: Japan from 2022, Canada from 2021; none with India or Hong Kong.
    rows = _ranked(client, f"/opportunities/by-product/{FIXTURE_HS6}")

    assert {row["iso3"]: row["trade_agreement"] for row in rows} == {"JPN": True, "CAN": True, "IND": False, "HKG": False}


def test_every_row_of_a_market_shows_its_trade_agreement(client):
    assert {row["trade_agreement"] for row in _ranked(client, "/opportunities/by-market/JPN")} == {True}
    assert {row["trade_agreement"] for row in _ranked(client, "/opportunities/by-market/IND")} == {False}


def test_explain_returns_factors_history_and_the_baseline_it_explains(client):
    response = client.get(f"/explain/{FIXTURE_HS6}/JPN")

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "artefact_version",
        "baseline_year",
        "gap_years",
        "product",
        "market",
        "fold",
        "base_value",
        "log_export_potential",
        "export_potential_usd",
        "actual_exports_usd",
        "export_gap_usd",
        "baseline_export_potential_usd",
        "relative_export_gap",
        "opportunity_type",
        "flagged_product",
        "reexport_hub",
        "potential_range_low_usd",
        "potential_range_high_usd",
        "uncertain",
        "tariff_caveat",
        "factors",
        "history",
    }
    assert body["product"]["hs6"] == FIXTURE_HS6
    assert body["market"] == {"iso3": "JPN", "name": "Japan", "region": "East Asia & Pacific"}
    assert set(body["factors"][0]) == {"label", "value_text", "contribution", "multiplier", "direction"}
    assert [point["year"] for point in body["history"]] == sorted(point["year"] for point in body["history"])


def test_explain_describes_relatedness_import_growth_and_reexports_in_plain_english(client):
    factors = {f["label"]: f["value_text"] for f in client.get(f"/explain/{FIXTURE_HS6}/JPN").json()["factors"]}

    assert "How close this Product is to what the UK already exports well" in factors
    assert "How fast Japan's imports of this Product are growing" in factors
    assert "How much of this Product Japan re-exports" in factors
    assert factors["How much of this Product Japan re-exports"] != "Yes"


def test_explain_matches_the_ranking_row_of_the_pair(client):
    explained = client.get(f"/explain/{FIXTURE_HS6}/JPN").json()
    ranked = client.get(f"/opportunities/by-product/{FIXTURE_HS6}").json()["opportunities"]
    row = next(r for r in ranked if r["iso3"] == "JPN")

    assert explained["export_potential_usd"] == pytest.approx(row["export_potential_usd"], rel=1e-3)
    assert explained["export_gap_usd"] == pytest.approx(row["export_gap_usd"], rel=1e-3, abs=1e-3)
    total = explained["base_value"] + sum(f["contribution"] for f in explained["factors"])
    assert total == pytest.approx(explained["log_export_potential"], abs=1e-4)


def test_every_pairs_yearly_export_gaps_average_to_its_reported_export_gap(client, fixture_artefacts):
    for hs6, iso3 in fixture_artefacts.baseline.select("hs6", "iso3").iter_rows():
        body = client.get(f"/explain/{hs6}/{iso3}").json()
        gap_years = [p for p in body["history"] if p["export_gap_usd"] is not None]
        ranked = client.get(f"/opportunities/by-product/{hs6}").json()["opportunities"]
        row = next(r for r in ranked if r["iso3"] == iso3)

        assert [p["year"] for p in gap_years] == [2022, 2023, 2024]
        for point in gap_years:
            assert point["export_gap_usd"] == pytest.approx(point["export_potential_usd"] - point["actual_exports_usd"])
        mean = sum(p["export_gap_usd"] for p in gap_years) / 3
        assert body["export_gap_usd"] == pytest.approx(mean, abs=1e-6)
        assert row["export_gap_usd"] == pytest.approx(mean, abs=1e-6)
        assert row["actual_exports_usd"] == pytest.approx(sum(p["actual_exports_usd"] for p in gap_years) / 3)


def test_explain_separates_the_gap_period_means_from_the_baseline_year_it_explains(client):
    body = client.get(f"/explain/{FIXTURE_HS6}/JPN").json()
    latest = next(p for p in body["history"] if p["year"] == 2024)

    assert body["gap_years"] == [2022, 2024]
    assert body["baseline_export_potential_usd"] == pytest.approx(latest["export_potential_usd"])
    assert body["baseline_export_potential_usd"] == pytest.approx(math.exp(body["log_export_potential"]), rel=1e-3)
    assert body["export_potential_usd"] != pytest.approx(body["baseline_export_potential_usd"])


def test_explain_directions_follow_the_sign_of_each_contribution(client):
    factors = client.get(f"/explain/{FIXTURE_HS6}/JPN").json()["factors"]

    for factor in factors:
        expected = "raised" if factor["contribution"] > 0 else "lowered" if factor["contribution"] < 0 else "no effect"
        assert factor["direction"] == expected


@pytest.mark.parametrize("path", ["/explain/999999/JPN", f"/explain/{FIXTURE_HS6}/ZZZ"])
def test_explain_unknown_pair_returns_404_with_a_message(client, path):
    response = client.get(path)

    assert response.status_code == 404
    assert response.json()["detail"]


def test_explain_needs_loaded_artefacts(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        response = client.get(f"/explain/{FIXTURE_HS6}/JPN")

    assert response.status_code == 503


def test_evaluation_includes_lift_and_the_event_backtest(client):
    body = client.get("/evaluation").json()

    lift = body["lift"]
    assert lift["lift"] is not None and len(lift["ci"]) == 2 and "tendency" in lift["text"]
    assert len(lift["size_matched"]["ci"]) == 2 and "lift" in lift["size_matched"]
    assert lift["gap_years"] == [2021, 2022] and lift["held_out_from"] == "v1"
    event = body["event_backtest"]["events"][0]
    direction = event["directions"][0]
    assert direction["controls"] and direction["pooled"]["observed"] is not None
    assert direction["pooled"]["predicted"] is not None
    assert body["event_backtest"]["text"] and body["event_backtest"]["caveat"]


# --- Excluded Products (ADR 0005) -----------------------------------------------------------

CRUDE_OIL = "270900"  # the fixture world's Excluded Product


def test_products_excluded_lists_every_excluded_product_with_its_reason(client):
    response = client.get("/products/excluded")

    assert response.status_code == 200
    [crude] = response.json()
    assert crude["hs6"] == CRUDE_OIL
    assert crude["sector"] == "27"
    assert "pipelines" in crude["reason"]


@pytest.mark.parametrize("path", [f"/opportunities/by-product/{CRUDE_OIL}", f"/explain/{CRUDE_OIL}/JPN"])
def test_asking_for_an_excluded_product_returns_404_with_its_reason(client, path):
    response = client.get(path)

    assert response.status_code == 404
    detail = response.json()["detail"]
    assert "Excluded Product" in detail and "pipelines" in detail


@pytest.mark.parametrize("iso3", ["JPN", "CAN", "IND", "HKG"])
def test_no_ranking_contains_an_excluded_product(client, iso3):
    for path, params in [(f"/opportunities/by-market/{iso3}", {}), ("/opportunities/blocked", {"market": iso3})]:
        body = client.get(path, params=params).json()

        hs6s = {row["hs6"] for row in body["opportunities"]}
        assert hs6s and CRUDE_OIL not in hs6s


def test_product_search_marks_an_excluded_product_rather_than_hiding_it(client):
    [crude] = client.get("/products", params={"q": "crude"}).json()
    [whisky] = client.get("/products", params={"q": FIXTURE_HS6}).json()

    assert crude["hs6"] == CRUDE_OIL
    assert crude["excluded"] is True and "pipelines" in crude["exclusion_reason"]
    assert whisky["excluded"] is False and whisky["exclusion_reason"] is None


# --- Flagged Products and Re-export Hubs (ADR 0005) -------------------------------------

JEWELLERY, HUB = "711319", "HKG"  # the fixture's Flagged Product and Re-export Hub


def _all_rows(client, **params) -> list[dict]:
    """Every ranking row of both views, over every Market and every Product."""
    rows = []
    for iso3 in ("JPN", "CAN", "IND", HUB):
        rows += [{**row, "iso3": iso3} for row in _ranked(client, f"/opportunities/by-market/{iso3}", **params)]
    for hs6 in ("220830", JEWELLERY):
        rows += [{**row, "hs6": hs6} for row in _ranked(client, f"/opportunities/by-product/{hs6}", **params)]
    return rows


def test_every_ranking_row_flags_flagged_products_and_re_export_hubs(client):
    rows = _all_rows(client)

    for row in rows:
        assert row["flagged_product"] is (row["hs6"] == JEWELLERY)
        assert row["reexport_hub"] is (row["iso3"] == HUB)
    assert any(row["flagged_product"] for row in rows) and any(row["reexport_hub"] for row in rows)


def test_hide_filters_are_off_by_default(client):
    by_market = _ranked(client, f"/opportunities/by-market/{HUB}")
    by_product = _ranked(client, f"/opportunities/by-product/{JEWELLERY}")

    assert JEWELLERY in {row["hs6"] for row in by_market}
    assert len(by_market) == 5 and len(by_product) == 4
    assert by_market == _ranked(client, f"/opportunities/by-market/{HUB}", hide_flagged=False, hide_hubs=False)


@pytest.mark.parametrize(("flag", "field"), [("hide_flagged", "flagged_product"), ("hide_hubs", "reexport_hub")])
def test_each_hide_filter_removes_exactly_the_flagged_rows(client, flag, field):
    everything = _all_rows(client)
    shown = _all_rows(client, **{flag: True})

    assert shown == [row for row in everything if not row[field]]
    assert len(shown) < len(everything)


def test_an_explanation_carries_both_flags(client):
    flagged_hub = client.get(f"/explain/{JEWELLERY}/{HUB}").json()
    plain = client.get(f"/explain/{FIXTURE_HS6}/JPN").json()

    assert (flagged_hub["flagged_product"], flagged_hub["reexport_hub"]) == (True, True)
    assert (plain["flagged_product"], plain["reexport_hub"]) == (False, False)


# --- The Leaderboard: every UK pair, a summary, Relative Export Gap and Opportunity Type --------


def _top(client, **params) -> dict:
    response = client.get("/opportunities/top", params=params)
    assert response.status_code == 200, response.text
    return response.json()


def test_leaderboard_ranks_every_uk_pair_by_export_gap_with_a_summary(client, fixture_artefacts):
    body = _top(client)

    rows = body["opportunities"]
    gaps = [row["export_gap_usd"] for row in rows]
    assert gaps == sorted(gaps, reverse=True)
    assert len(rows) == fixture_artefacts.baseline.height == 20
    assert {(row["hs6"], row["iso3"]) for row in rows} == set(
        fixture_artefacts.baseline.select("hs6", "iso3").iter_rows()
    )
    assert body["rank_by"] == "gap"
    assert body["gap_years"] == [2022, 2024]
    assert body["summary"]["pairs"] == 20
    assert body["summary"]["positive_export_gap_usd"] == pytest.approx(sum(gap for gap in gaps if gap > 0))
    assert {"hs6", "description", "sector", "iso3", "market", "region", "opportunity_type"} <= set(rows[0])


def _gap_summary(rows: list[dict]) -> dict:
    return {"pairs": len(rows), "positive_export_gap_usd": pytest.approx(sum(max(r["export_gap_usd"], 0) for r in rows))}


@pytest.mark.parametrize(
    ("params", "keep"),
    [
        ({"sector": "85"}, lambda row: row["sector"] == "85"),
        ({"region": "South Asia"}, lambda row: row["region"] == "South Asia"),
        ({"market": "JPN"}, lambda row: row["iso3"] == "JPN"),
        ({"opportunity_type": "Expand"}, lambda row: row["opportunity_type"] == "Expand"),
        ({"min_gap_usd": 5000}, lambda row: row["export_gap_usd"] >= 5000),
        ({"trade_agreement": True}, lambda row: row["trade_agreement"]),
        ({"trade_agreement": False}, lambda row: not row["trade_agreement"]),
        ({"hide_flagged": True}, lambda row: not row["flagged_product"]),
        ({"hide_hubs": True}, lambda row: not row["reexport_hub"]),
    ],
    ids=lambda value: "-".join(f"{k}={v}" for k, v in value.items()) if isinstance(value, dict) else "",
)
def test_every_leaderboard_filter_keeps_exactly_the_matching_rows_and_the_summary_follows(client, params, keep):
    everything = _top(client)["opportunities"]
    body = _top(client, **params)

    expected = [row for row in everything if keep(row)]
    assert body["opportunities"] == expected
    assert body["summary"] == _gap_summary(expected)


def test_leaderboard_filters_combine_and_limit_keeps_the_top_but_not_the_summary(client):
    everything = _top(client, trade_agreement=False, hide_hubs=True)["opportunities"]
    top = _top(client, trade_agreement=False, hide_hubs=True, limit=2)

    assert everything and all(row["iso3"] == "IND" for row in everything)  # no UK agreement, not a hub
    assert top["opportunities"] == everything[:2]
    assert top["summary"] == _gap_summary(everything)  # the summary sizes every filtered pair


def test_leaderboard_unknown_market_or_opportunity_type_is_rejected(client):
    assert client.get("/opportunities/top", params={"market": "XYZ"}).status_code == 404
    assert client.get("/opportunities/top", params={"opportunity_type": "Exotic"}).status_code == 422


def test_relative_export_gap_is_the_export_gap_over_export_potential(client):
    for row in _top(client)["opportunities"]:
        assert row["relative_export_gap"] == pytest.approx(row["export_gap_usd"] / row["export_potential_usd"])


def test_relative_ranks_only_pairs_above_the_minimum_export_potential(client):
    # Fixture Export Potentials are tens of thousands of dollars, all below the $1M default.
    assert _top(client, rank_by="relative")["opportunities"] == []
    assert _top(client, rank_by="relative")["summary"] == {"pairs": 0, "positive_export_gap_usd": 0.0}

    everything = _top(client)["opportunities"]
    floor = sorted(row["export_potential_usd"] for row in everything)[len(everything) // 2]
    body = _top(client, rank_by="relative", min_potential_usd=floor)

    rows = body["opportunities"]
    assert {(r["hs6"], r["iso3"]) for r in rows} == {
        (r["hs6"], r["iso3"]) for r in everything if r["export_potential_usd"] >= floor
    }
    relative = [row["relative_export_gap"] for row in rows]
    assert relative == sorted(relative, reverse=True)
    assert body["rank_by"] == "relative"
    assert body["summary"] == _gap_summary(rows)


def test_relative_breaks_ties_by_export_gap(fixture_artefacts, tmp_path):
    # Two constructed pairs share a Relative Export Gap of 50%; the larger Export Gap ranks first.
    tied = pl.DataFrame(
        {
            "hs6": ["220830", "300490", "851712"],
            "iso3": ["JPN", "CAN", "IND"],
            "potential": [2e6, 8e6, 3e6],
            "gap": [1e6, 4e6, 0.3e6],
        }
    )
    baseline = (
        fixture_artefacts.baseline.join(tied, on=["hs6", "iso3"], how="left")
        .with_columns(
            export_potential_usd=pl.col("potential").fill_null(1e3),  # every other pair: below the minimum
            export_gap_usd=pl.col("gap").fill_null(0.0),
        )
        .with_columns(relative_export_gap=pl.col("export_gap_usd") / pl.col("export_potential_usd"))
        .drop("potential", "gap")
    )
    store.save(replace(fixture_artefacts, baseline=baseline), tmp_path)

    with TestClient(create_app(tmp_path)) as constructed:
        rows = _top(constructed, rank_by="relative")["opportunities"]

    assert [(row["hs6"], row["iso3"]) for row in rows] == [("300490", "CAN"), ("220830", "JPN"), ("851712", "IND")]


def test_leaderboard_rank_orders_offer_the_minimum_export_potentials(client):
    orders = {order["name"]: order for order in client.get("/opportunities/rank-by").json()}

    assert orders["relative"]["title"] == "Relative Export Gap"
    assert orders["relative"]["min_potential_usd_options"] == [0, 1e5, 1e6, 1e7, 1e8]
    assert orders["relative"]["default_min_potential_usd"] == 1e6
    assert orders["gap"].get("min_potential_usd_options") is None


@pytest.mark.parametrize("path", [f"/opportunities/by-product/{FIXTURE_HS6}", "/opportunities/by-market/JPN"])
def test_by_product_and_by_market_take_relative_and_the_new_filters(client, path):
    everything = _ranked(client, path)
    floor = min(row["export_potential_usd"] for row in everything)

    relative = _ranked(client, path, rank_by="relative", min_potential_usd=floor)
    filtered = _ranked(client, path, opportunity_type="Expand", trade_agreement=True)

    values = [row["relative_export_gap"] for row in relative]
    assert len(relative) == len(everything) and values == sorted(values, reverse=True)
    assert filtered == [row for row in everything if row["opportunity_type"] == "Expand" and row["trade_agreement"]]


# Opportunity Type on constructed pairs: the fixture world with some UK Flows removed in the Gap Period.
NEW_MARKET_PAIR = ("220830", "IND")  # the UK still sells whisky to other Markets
NEW_PRODUCT = "851712"  # the UK sells phones to no Market in 2022–2024


@pytest.fixture(scope="module")
def opportunity_client(fixture_world, tmp_path_factory):
    uk_gap_period = (pl.col("exporter") == "GBR") & (pl.col("year") >= 2022)
    removed = uk_gap_period & (
        ((pl.col("hs6") == NEW_MARKET_PAIR[0]) & (pl.col("importer") == NEW_MARKET_PAIR[1]))
        | (pl.col("hs6") == NEW_PRODUCT)
    )
    world = replace(
        fixture_world,
        flows=fixture_world.flows.with_columns(value_usd=pl.when(removed).then(0.0).otherwise(pl.col("value_usd"))),
    )
    root = tmp_path_factory.mktemp("opportunity-types")
    store.save(build_artefacts(world, version="opportunity-types"), root)
    with TestClient(create_app(root)) as constructed:
        yield constructed


def test_opportunity_type_matches_its_definition_on_constructed_pairs(opportunity_client):
    types = {(row["hs6"], row["iso3"]): row["opportunity_type"] for row in _top(opportunity_client)["opportunities"]}

    assert types[NEW_MARKET_PAIR] == "New Market"
    assert {types[(NEW_PRODUCT, iso3)] for iso3 in ("JPN", "CAN", "IND", "HKG")} == {"New Product"}
    others = {pair: kind for pair, kind in types.items() if pair != NEW_MARKET_PAIR and pair[0] != NEW_PRODUCT}
    assert len(others) == 15 and set(others.values()) == {"Expand"}


def test_opportunity_type_filter_and_explanation_on_constructed_pairs(opportunity_client):
    rows = _top(opportunity_client, opportunity_type="New Product")["opportunities"]
    explanation = opportunity_client.get(f"/explain/{NEW_MARKET_PAIR[0]}/{NEW_MARKET_PAIR[1]}").json()

    assert {row["hs6"] for row in rows} == {NEW_PRODUCT} and len(rows) == 4
    assert all(row["actual_exports_usd"] == 0 for row in rows)
    assert explanation["opportunity_type"] == "New Market"


# --- The Potential Range and Uncertain Opportunities (ADR 0007) -------------------------


def test_every_ranking_row_has_a_potential_range_from_zero_up_and_flags_uncertain_opportunities(client):
    rows = _all_rows(client) + _top(client)["opportunities"]

    for row in rows:
        low, high = row["potential_range_low_usd"], row["potential_range_high_usd"]
        assert 0 <= low <= high
        inside = low <= row["actual_exports_usd"] <= high
        assert row["uncertain"] is (row["export_gap_usd"] > 0 and inside)
    assert any(row["uncertain"] for row in rows)
    assert not all(row["uncertain"] for row in rows if row["export_gap_usd"] > 0)


@pytest.mark.parametrize(
    "path", ["/opportunities/top", f"/opportunities/by-product/{FIXTURE_HS6}", "/opportunities/by-market/JPN"]
)
def test_hide_uncertain_is_off_by_default_and_removes_exactly_the_uncertain_rows_keeping_the_order(client, path):
    def rows(**params):
        body = client.get(path, params=params)
        assert body.status_code == 200, body.text
        return body.json()["opportunities"]

    everything = rows()
    assert everything == rows(hide_uncertain=False)
    for rank_by in ("gap", "relative"):
        ranked = rows(rank_by=rank_by, min_potential_usd=0)
        assert rows(rank_by=rank_by, min_potential_usd=0, hide_uncertain=True) == [
            row for row in ranked if not row["uncertain"]
        ]


def test_hide_uncertain_removes_uncertain_opportunities_from_the_leaderboard_summary(client):
    everything, shown = _top(client), _top(client, hide_uncertain=True)

    assert shown["summary"]["pairs"] == everything["summary"]["pairs"] - sum(
        row["uncertain"] for row in everything["opportunities"]
    )
    assert shown["summary"]["pairs"] < everything["summary"]["pairs"]


def test_an_explanation_carries_the_pairs_potential_range_as_its_ranking_row_does(client):
    for row in _ranked(client, "/opportunities/by-market/JPN"):
        body = client.get(f"/explain/{row['hs6']}/JPN").json()
        for field in ("potential_range_low_usd", "potential_range_high_usd", "uncertain"):
            assert body[field] == row[field]


def test_explain_names_the_applied_tariff_and_its_rate_in_plain_english(client):
    factors = {f["label"]: f["value_text"] for f in client.get(f"/explain/{FIXTURE_HS6}/IND").json()["factors"]}

    # India's 150% MFN rate on whisky, carried forward from 2021 to the 2024 Baseline.
    assert factors["The tariff India charges on this Product from the UK"] == "150%"


def test_explain_carries_the_tariff_carried_forward_caveat_and_the_yearly_mark(client):
    body = client.get("/explain/300490/IND").json()  # medicaments: India 10%, UK exports in 2021

    assert "carried forward from 2021" in body["tariff_caveat"]
    imputed = {point["year"]: point["tariff_imputed"] for point in body["history"]}
    assert imputed[2021] is False
    assert imputed[2022] is imputed[2023] is imputed[2024] is True
