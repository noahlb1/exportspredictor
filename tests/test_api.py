"""API contract tests: FastAPI TestClient against fixture artefacts (the main seam)."""

import math
from dataclasses import replace

import polars as pl
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from exportspredictor import artefacts as store
from exportspredictor import conformal, evaluation, illustrative, loop, realisation
from exportspredictor.fixture import PRODUCTS
from exportspredictor.pipeline import build_artefacts

FIXTURE_HS6 = "220830"


@pytest.fixture(scope="module")
def client(fixture_artefacts_dir):
    with TestClient(create_app(fixture_artefacts_dir)) as client:
        yield client


BLOCKED_READ = "/opportunities/blocked"


@pytest.fixture(scope="module")
def blocked_client(blocked_artefacts_dir):
    """The API over the Blocked world (`conftest.blocked_artefacts`): the fixture alone has no Blocked Opportunity."""
    with TestClient(create_app(blocked_artefacts_dir)) as client:
        yield client


def test_health_reports_version_format_baseline_year_and_gap_period_once_loaded(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "ready": True,
        "artefact_version": "fixture-test",
        "artefact_format": 4,
        "baseline_year": 2024,
        "gap_years": [2022, 2024],
        "gap_period_label": "2022–2024 average",
    }


def test_labels_serve_the_wording_and_options_the_app_shows(client):
    body = client.get("/labels").json()

    assert body["usd_unit"] == "USD, nominal"
    assert "nominal US dollars" in body["usd_caption"]
    assert body["opportunity_types"] == ["Expand", "New Market", "New Product"]
    caveats = {tier["column"]: tier for tier in client.get("/caveats").json()}
    assert {field: caveat["badge"] for field, caveat in caveats.items()} == {
        "flagged_product": "Flagged Product",
        "reexport_hub": "Re-export Hub",
        "sanctioned_market": "Sanctioned Market",
        "restricted_pair": "Restricted Pair",
        "offset_pair": "Offset Pair",
    }
    assert all(caveat["text"].startswith(caveat["badge"]) for caveat in caveats.values())
    # The key terms a caveat names that have their own explanation, by name: the hide filter's `?` explains them too
    assert {field: caveat["terms"] for field, caveat in caveats.items() if caveat["terms"]} == {
        "offset_pair": {"Heading": "heading"}
    }
    assert all(key in body["terms"] for caveat in caveats.values() for key in caveat["terms"].values())
    assert "caveats" not in body  # one source: /caveats
    # The last year of tariff data: later years carry it forward, and it is the MFN year Scenarios restore.
    assert body["last_tariff_year"] == 2021
    assert "carried forward from 2021" in body["tariff_carried_forward"]
    # What "a Trade Agreement" means in the model, said on the Blocked Opportunities page and the Scenario Builder.
    meaning = body["trade_agreement_meaning"]
    assert "average agreement" in meaning and "not a specific deal" in meaning and "only whether a Trade Agreement exists" in meaning
    # The Blocked Opportunity rule at the fixed thresholds (ADR 0012), stated once for every page.
    rule = body["blocked_rule"]
    assert "at least 20% of Baseline and $1M" in rule and "room to grow" in rule
    assert "with no UK Trade Agreement in force" in rule and "room to grow" in rule


def test_labels_serve_the_caution_line_every_page_shows_under_its_heading(client):
    caution = client.get("/labels").json()["caution"]

    assert caution == "Figures are associations the model has learned, not causes or forecasts."


@pytest.mark.parametrize(("share", "about"), [(0.7, "about 70%"), (0.7999, "about 80%")])
def test_labels_serve_the_totals_note_with_the_share_the_artefact_records(fixture_artefacts, tmp_path, share, about):
    # The note that totals understate: the share comes from the artefact, not a constant.
    metadata = {**fixture_artefacts.metadata, "potential_share_of_actual_exports": share}
    store.save(replace(fixture_artefacts, metadata=metadata), tmp_path)
    with TestClient(create_app(tmp_path)) as client:
        note = client.get("/labels").json()["totals_note"]

    assert note["potential_share_of_actual_exports"] == pytest.approx(share)
    assert "a typical value, not an average" in note["text"] and f"{about} of Actual Exports" in note["text"]
    assert "2022–2024 average" in note["text"]


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
    assert body["product"] == {
        "hs6": "220830", "description": "Whiskies", "sector": "22", "sector_name": "Beverages, spirits and vinegar"
    }
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
        "blocked_opportunity_value_usd",
        "blocked_opportunity",
        "flagged_product",
        "flagged_product_reason",
        "reexport_hub",
        "sanctioned_market",
        "sanctioned_market_reason",
        "restricted_pair",
        "restricted_pair_reason",
        "offset_pair",
        "offset_pair_reason",
        "caveat_texts",
        "potential_range_low_usd",
        "potential_range_high_usd",
        "uncertain_opportunity",
        "applied_tariff_pct",
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
        {
            "hs6": "711319",
            "description": PRODUCTS["711319"],
            "sector": "71",
            "sector_name": "Pearls, precious stones and metals",
            "excluded": False,
            "exclusion_reason": None,
        }
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
        "scenario": True,
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


def test_evaluation_serves_the_limitations_the_artefact_shows(client, fixture_artefacts):
    body = client.get("/evaluation").json()

    limitations = {item["title"]: item["text"] for item in body["limitations"]}
    # The artefact's bands are log offsets (ADR 0007, amended), so no fixed dollar width is reported,
    assert "The Potential Range is a fixed width in dollars" not in limitations
    # but what its results show is, as on the model card: the ranking titled by its margin over the Naive Baseline
    headline = fixture_artefacts.evaluation["headline"]
    if not headline["beats_naive"]:
        expected = "Does not beat a naive ranking for UK pairs"
    elif headline["model"] - headline["naive"] < evaluation.NARROW_MARGIN:
        expected = "Only slightly better than a naive ranking for UK pairs"
    else:
        expected = "Ranks Markets better than it sizes them"
    assert expected in limitations
    assert "Trained on a sample of other Exporters' Flows" in limitations


def test_evaluation_serves_the_measure_on_the_largest_importing_markets(client, fixture_artefacts):
    large = client.get("/evaluation").json()["large_markets"]

    assert large == fixture_artefacts.evaluation["large_markets"]
    assert [row["markets"] for row in large["rows"]] == [30, 60]


def test_evaluation_of_an_artefact_without_the_largest_markets_measure_serves_none(fixture_artefacts, tmp_path):
    from dataclasses import replace

    older = {key: value for key, value in fixture_artefacts.evaluation.items() if key != "large_markets"}
    store.save(replace(fixture_artefacts, evaluation=older), tmp_path)
    with TestClient(create_app(tmp_path)) as client:
        body = client.get("/evaluation").json()

    assert body["large_markets"] is None


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
    everything = _ranked(client, "/opportunities/by-market/CAN")
    rows = _ranked(client, "/opportunities/by-market/CAN", min_gap_usd=0)

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
        "/opportunities/blocked/rank-by",
        f"/opportunities/by-product/{FIXTURE_HS6}",
        "/opportunities/by-market/JPN",
        "/evaluation",
        f"/explain/{FIXTURE_HS6}/JPN",
        "/scenarios/presets",
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
        "flagged_product_reason",
        "reexport_hub",
        "sanctioned_market",
        "sanctioned_market_reason",
        "restricted_pair",
        "restricted_pair_reason",
        "offset_pair",
        "offset_pair_reason",
        "caveat_texts",
        "potential_range_low_usd",
        "potential_range_high_usd",
        "uncertain_opportunity",
        "potential_range_text",
        "gap_period_label",
        "tariff_caveat",
        "market_reach",
        "factors",
        "history",
    }
    assert body["product"]["hs6"] == FIXTURE_HS6
    assert body["market"] == {"iso3": "JPN", "name": "Japan", "region": "East Asia & Pacific"}
    assert set(body["factors"][0]) == {"label", "value_text", "contribution", "multiplier", "direction"}
    assert [point["year"] for point in body["history"]] == sorted(point["year"] for point in body["history"])
    assert body["potential_range_text"].startswith("Pairs with about the same Export Potential have Actual Exports")
    assert body["gap_period_label"] == "2022–2024 average"


def test_explain_describes_relatedness_and_reexports_in_plain_english(client):
    factors = {f["label"]: f["value_text"] for f in client.get(f"/explain/{FIXTURE_HS6}/JPN").json()["factors"]}

    assert "How close this Product is to what the UK already exports well" in factors
    # Market import growth was dropped by the feature ablation, so it is no factor.
    assert "How fast Japan's imports of this Product are growing" not in factors
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
    assert lift["lift"] is not None and len(lift["ci"]) == 2 and "times as often" in lift["text"]
    assert len(lift["size_matched"]["ci"]) == 2 and "lift" in lift["size_matched"]
    assert lift["gap_years"] == [2022]
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
    for path, params in [(f"/opportunities/by-market/{iso3}", {}), ("/opportunities/top", {"market": iso3})]:
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
    assert body["gap_years"] == [2022, 2024] and body["gap_period_label"] == "2022–2024 average"
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


def _group_totals(rows: list[dict], code: str, name: str) -> list[dict]:
    """Each group's pair count and positive Export Gap from the pair rows, largest first, ties by name."""
    groups: dict[str, dict] = {}
    for row in rows:
        group = groups.setdefault(row[code], {code: row[code], name: row[name], "pairs": 0, "positive_export_gap_usd": 0.0})
        group["pairs"] += 1
        group["positive_export_gap_usd"] += max(row["export_gap_usd"], 0)
    return sorted(groups.values(), key=lambda group: (-group["positive_export_gap_usd"], group[name]))


@pytest.mark.parametrize(
    "params",
    [{}, {"region": "South Asia"}, {"product": "220830"}, {"market": "JPN"}, {"hide_hubs": True, "min_gap_usd": 0}],
    ids=["all", "region", "product", "market", "hubs-and-min-gap"],
)
def test_leaderboard_market_and_product_totals_cover_every_filtered_pair_not_only_the_limit(client, params):
    everything = _top(client, **params)["opportunities"]
    body = _top(client, **params, limit=2)

    for field, code, name in [("markets", "iso3", "market"), ("products", "hs6", "description")]:
        expected = _group_totals(everything, code, name)
        served = body[field]
        assert [group[code] for group in served] == [group[code] for group in expected], field
        for group, want in zip(served, expected, strict=True):
            assert group == {**want, "positive_export_gap_usd": pytest.approx(want["positive_export_gap_usd"])}
        assert sum(group["pairs"] for group in served) == body["summary"]["pairs"]
        assert sum(group["positive_export_gap_usd"] for group in served) == pytest.approx(
            body["summary"]["positive_export_gap_usd"]
        )
        totals = [group["positive_export_gap_usd"] for group in served]
        assert totals == sorted(totals, reverse=True)
    assert len(body["opportunities"]) == min(2, len(everything))


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


def test_the_blocked_rank_orders_are_the_rise_with_agreement_then_the_tariff_part(client):
    orders = client.get("/opportunities/blocked/rank-by").json()

    assert [(order["name"], order["title"]) for order in orders] == [
        ("blocked", "Rise with agreement"), ("tariff_part", "Tariff part")
    ]
    assert all("associates" in order["description"] for order in orders)


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
        assert row["uncertain_opportunity"] is (row["export_gap_usd"] > 0 and inside)
    assert any(row["uncertain_opportunity"] for row in rows)


@pytest.mark.parametrize(
    "path", ["/opportunities/top", f"/opportunities/by-product/{FIXTURE_HS6}", "/opportunities/by-market/JPN"]
)
def test_the_ranking_query_no_longer_accepts_a_filter_for_uncertain_opportunities(client, path):
    # The Uncertain Opportunity is reported as a share on the Model Evaluation page, not filtered per row
    assert client.get(path).status_code == 200
    assert client.get(path, params={"hide_uncertain": True}).status_code == 422


def test_an_explanation_carries_the_pairs_potential_range_as_its_ranking_row_does(client):
    for row in _ranked(client, "/opportunities/by-market/JPN"):
        body = client.get(f"/explain/{row['hs6']}/JPN").json()
        for field in ("potential_range_low_usd", "potential_range_high_usd", "uncertain_opportunity"):
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


def test_evaluation_never_serves_the_v1_comparison(client, fixture_artefacts):
    assert "v1_comparison" in fixture_artefacts.evaluation

    body = client.get("/evaluation").json()

    assert "v1_comparison" not in body


def test_evaluation_states_the_share_of_opportunities_inside_their_potential_range(client, fixture_artefacts):
    saved = fixture_artefacts.evaluation["potential_range"]
    served = client.get("/evaluation").json()["potential_range"]

    # Worded by the package at serve time, so a version saved before the badge was removed reads the same
    assert served["text"] == conformal.coverage_text(saved)
    assert served["uncertain_opportunity_share"] == saved["uncertain_opportunity_share"]


def test_evaluation_serves_the_potential_range_coverage_by_size(client, fixture_artefacts):
    saved = fixture_artefacts.evaluation["potential_range"]["by_size"]

    served = client.get("/evaluation").json()["potential_range"]["by_size"]

    assert [row["size"] for row in served] == ["under $2m", "$2m–10m", "$10m–100m", "over $100m"]
    assert served == saved


# --- The Scenario note: Illustrative, beside its benchmarks ------------------

SCENARIO = {"add_agreements": ["IND"]}

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


def test_the_note_says_the_event_backtest_and_the_models_association_cover_different_populations(client):
    # The Blend's association over every UK pair is not to be compared with one agreement's backtest
    backtest = client.post("/scenarios", json=SCENARIO).json()["note"]["event_backtest"]

    assert illustrative.DIFFERENT_POPULATIONS in backtest["text"]


def test_the_note_says_a_scenario_moves_only_the_models_share_of_the_blend(client, fixture_artefacts):
    weight = 1 - fixture_artefacts.metadata["model_form"]["naive_weight"]
    primary = next(e for e in fixture_artefacts.evaluation["event_backtest"]["events"] if e["role"] == "primary")
    predicted = primary["directions"][0]["pooled"]["predicted"]

    blend = client.post("/scenarios", json=SCENARIO).json()["note"]["blend"]

    assert blend["model_weight"] == pytest.approx(weight)
    # The served effect is the model's scaled by its weight on the log scale: (1 + e) ** w - 1
    assert blend["served_effect"] == pytest.approx((1 + predicted) ** weight - 1)
    assert "model alone" in blend["text"]


def test_the_blend_note_and_the_event_backtest_part_read_the_same_primary_event_backtest(fixture_artefacts, tmp_path):
    backtest = fixture_artefacts.evaluation["event_backtest"]
    primary = next(e for e in backtest["events"] if e["role"] == "primary")
    direction = primary["directions"][0]
    secondary = {**primary, "role": "secondary", "title": "A secondary check", "directions": [
        {**direction, "pooled": {**direction["pooled"], "predicted": direction["pooled"]["predicted"] + 0.5}},
    ]}
    report = {**fixture_artefacts.evaluation, "event_backtest": {**backtest, "events": [secondary, primary]}}

    note = _note(replace(fixture_artefacts, evaluation=report), tmp_path)

    assert note["event_backtest"]["title"] == primary["title"]
    assert note["blend"]["model_effect"] == pytest.approx(primary["directions"][0]["pooled"]["predicted"])
    assert primary["title"] in note["blend"]["text"]


def test_with_no_saved_benchmark_the_note_keeps_the_label_and_says_the_benchmark_was_not_run(fixture_artefacts, tmp_path):
    note = _note(replace(fixture_artefacts, ppml_benchmark=None), tmp_path)

    assert note["label"] == "Illustrative"
    for part in ("benchmark", "model_association"):
        assert note[part]["run"] is False
        assert "not run" in note[part]["text"]
    assert note["event_backtest"]["run"] is True


def test_with_no_event_backtests_the_note_keeps_the_label_and_says_they_were_not_run(fixture_artefacts, tmp_path):
    report = {key: value for key, value in fixture_artefacts.evaluation.items() if key != "event_backtest"}

    note = _note(replace(fixture_artefacts, evaluation=report), tmp_path)

    assert note["label"] == "Illustrative"
    assert note["event_backtest"]["run"] is False
    assert "not run" in note["event_backtest"]["text"]
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
    assert note["benchmark"]["run"] is False and "not run" in note["benchmark"]["text"]


# --- How the model was chosen, read from the artefact ----------


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
        assert served["label"] == loop.experiment_label(served["name"], [row["arm"] for row in section["arms"]])
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


def _confirmed(fixture_artefacts):
    ranking = {**fixture_artefacts.metadata["ranking_experiments"], "confirmation": CONFIRMATION}
    return replace(fixture_artefacts, metadata={**fixture_artefacts.metadata, "ranking_experiments": ranking})


def test_evaluation_serves_the_five_fold_confirmation_against_the_model_alone(fixture_artefacts, tmp_path):
    confirmation = _served(_confirmed(fixture_artefacts), tmp_path)["model_choice"]["confirmation"]

    assert confirmation == {
        "n_folds": 2, "winner": 0.5635, "model_alone": 0.5206, "naive": 0.5332, "difference": 0.0429,
        "win_threshold": 0.0089, "beats_model_alone": True,
        "folds": [{"fold": 0, "winner": 0.5771, "model_alone": 0.5334}, {"fold": 1, "winner": 0.5745, "model_alone": 0.5295}],
        "text": (
            "Five-fold confirmation on 2022: the chosen Model Form scores 0.5635, the model alone 0.5206 and the Naive "
            "Baseline 0.5332; the difference, +0.0429, beats the Win Threshold of 0.0089 (confirmed)."
        ),
    }


def test_the_confirmation_s_verdict_is_the_record_s(fixture_artefacts, tmp_path):
    # The loop's rule allows for rounding, so a difference just above the threshold can still fall short.
    short = {**CONFIRMATION, "difference": 0.0090, "beats_v2": False}
    ranking = {**fixture_artefacts.metadata["ranking_experiments"], "confirmation": short}
    bundle = replace(fixture_artefacts, metadata={**fixture_artefacts.metadata, "ranking_experiments": ranking})

    confirmation = _served(bundle, tmp_path)["model_choice"]["confirmation"]

    assert confirmation["beats_model_alone"] is False
    assert confirmation["text"].endswith("does not beat the Win Threshold of 0.0089 (not confirmed).")


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


def _with_history(fixture_artefacts):
    """The fixture artefact with every history block the record keeps: earlier versions (with a lower deviance, so
    something would read as traded away), their re-scores, the v1 comparison and a gain ladder."""
    this = fixture_artefacts.evaluation["accuracy"]["all_exporters"]["overall"]["model_tweedie_deviance"]
    report = {**fixture_artefacts.evaluation}
    report["earlier_versions"] = evaluation.earlier_versions(
        report, v2={**V2, "tweedie_deviance_all_exporters": this / 2}, blend=None
    )
    report["v1_comparison"] = {**report.get("v1_comparison", {}), "ladder": {"steps": []}}
    return replace(_confirmed(fixture_artefacts), evaluation=report)


HISTORY_FIELDS = ("earlier_versions", "rescored_earlier_versions", "large_markets_by_version", "v1_comparison", "ladder")


def test_evaluation_serves_no_history_even_when_the_artefact_keeps_it(fixture_artefacts, tmp_path):
    artefacts = _with_history(fixture_artefacts)
    assert {"earlier_versions", "rescored_earlier_versions", "v1_comparison"} <= set(artefacts.evaluation)

    body = _served(artefacts, tmp_path)

    assert not set(HISTORY_FIELDS) & set(body)
    assert not {"test_year_ranking", "traded_away"} & set(body["model_choice"])
    assert not {"v2", "beats_v2"} & set(body["model_choice"]["confirmation"])
    for lift in ("lift", "lift_without_market_reach"):
        assert "held_out_from" not in (body.get(lift) or {})
    assert "Gain ladder" not in [part["title"] for part in body["not_run"]]


def test_evaluation_serves_the_headline_its_benchmarks_and_the_alternatives_considered(client, fixture_artefacts):
    saved = fixture_artefacts.evaluation

    body = client.get("/evaluation").json()

    assert body["artefact_version"] == fixture_artefacts.version
    assert body["headline"]["model"] == pytest.approx(saved["headline"]["model"])
    assert body["headline"]["naive"] == pytest.approx(saved["headline"]["naive"])  # the Naive Baseline beside it
    overall = body["accuracy"]["uk_out_of_fold"]["overall"]
    assert {f"naive_{metric}" for metric in body["metrics"]} <= set(overall)
    alternatives = body["model_choice"]["experiments"]
    assert [e["name"] for e in alternatives] == list(fixture_artefacts.metadata["ranking_experiments"]["experiments"])
    assert body["model_choice"]["validation_year"] == fixture_artefacts.metadata["ranking_experiments"]["validation_year"]


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


# --- Sector names, the Sectors read and the Leaderboard's Product filter -----------


def test_the_sectors_read_lists_each_sector_with_products_by_code_and_name(client):
    sectors = client.get("/sectors").json()

    # The fixture's Products sit in chapters 22, 27 (the Excluded crude oil), 30, 71, 85 and 87.
    assert [sector["code"] for sector in sectors] == ["22", "27", "30", "71", "85", "87"]
    names = {sector["code"]: sector["name"] for sector in sectors}
    assert names["22"] == "Beverages, spirits and vinegar"
    assert names["85"] == "Electrical machinery and equipment"


@pytest.mark.parametrize(
    "path", ["/opportunities/top", BLOCKED_READ, "/opportunities/by-market/JPN"]
)
def test_every_ranking_row_carries_its_sector_name(client, blocked_client, path):
    rows = (blocked_client if path == BLOCKED_READ else client).get(path).json()["opportunities"]

    assert rows
    whisky = [row for row in rows if row["hs6"] == FIXTURE_HS6]
    assert whisky and all(row["sector_name"] == "Beverages, spirits and vinegar" for row in whisky)


def test_every_product_carries_its_sector_name(client):
    [phone] = client.get("/products", params={"q": "851712"}).json()
    [crude] = client.get("/products/excluded").json()
    product = client.get(f"/opportunities/by-product/{FIXTURE_HS6}").json()["product"]

    assert phone["sector_name"] == "Electrical machinery and equipment"
    assert crude["sector_name"] == "Mineral fuels and oils"
    assert product["sector_name"] == "Beverages, spirits and vinegar"


def test_the_leaderboard_product_filter_keeps_that_products_markets_and_the_summary_follows(client):
    everything = _top(client)["opportunities"]
    body = _top(client, product=FIXTURE_HS6)

    expected = [row for row in everything if row["hs6"] == FIXTURE_HS6]
    assert body["opportunities"] == expected and len(expected) == 4
    assert body["summary"] == _gap_summary(expected)
    assert _top(client, product=FIXTURE_HS6, market="JPN")["opportunities"] == [
        row for row in expected if row["iso3"] == "JPN"
    ]


def test_the_leaderboard_product_filter_answers_404_for_an_unknown_or_excluded_product(client):
    unknown = client.get("/opportunities/top", params={"product": "999999"})
    excluded = client.get("/opportunities/top", params={"product": CRUDE_OIL})

    assert unknown.status_code == 404 and "999999" in unknown.json()["detail"]
    assert excluded.status_code == 404
    assert "Excluded Product" in excluded.json()["detail"] and "pipelines" in excluded.json()["detail"]


# --- Row limit, Applied Tariff on ranking rows, term definitions ------------------------------

RANKING_READS = [
    "/opportunities/top",
    BLOCKED_READ,
    f"/opportunities/by-product/{FIXTURE_HS6}",
    "/opportunities/by-market/JPN",
]


@pytest.mark.parametrize("path", RANKING_READS)
def test_every_ranking_read_accepts_a_limit_of_6000_and_refuses_6001(client, path):
    assert client.get(path, params={"limit": 6000}).status_code == 200
    assert client.get(path, params={"limit": 6001}).status_code == 422


@pytest.mark.parametrize("path", RANKING_READS)
def test_every_ranking_row_carries_the_baseline_applied_tariff_in_percent(client, blocked_client, path):
    rows = (blocked_client if path == BLOCKED_READ else client).get(path).json()["opportunities"]

    assert rows and all("applied_tariff_pct" in row for row in rows)
    # The fixture's tariffs: India charges 150% on whisky; Japan, with a UK Trade Agreement, charges nothing.
    tariffs = {(row.get("hs6", FIXTURE_HS6), row.get("iso3", "JPN")): row["applied_tariff_pct"] for row in rows}
    if (FIXTURE_HS6, "IND") in tariffs:
        assert tariffs[(FIXTURE_HS6, "IND")] == pytest.approx(150.0, rel=1e-5)
    if (FIXTURE_HS6, "JPN") in tariffs:
        assert tariffs[(FIXTURE_HS6, "JPN")] == 0.0


def test_the_applied_tariff_is_null_where_it_is_unknown(blocked_artefacts, tmp_path):
    # Whisky in India, a Blocked Opportunity in the Blocked world, has no known tariff here.
    unknown = (pl.col("hs6") == FIXTURE_HS6) & (pl.col("iso3") == "IND")
    features = blocked_artefacts.baseline_features.with_columns(
        log_applied_tariff=pl.when(unknown).then(None).otherwise(pl.col("log_applied_tariff"))
    )
    store.save(replace(blocked_artefacts, baseline_features=features), tmp_path)

    with TestClient(create_app(tmp_path)) as constructed:
        for path in RANKING_READS:
            rows = constructed.get(path).json()["opportunities"]
            tariffs = {(row.get("hs6", FIXTURE_HS6), row.get("iso3", "JPN")): row["applied_tariff_pct"] for row in rows}
            assert tariffs.get((FIXTURE_HS6, "IND"), None) is None
            assert any(tariff is not None for tariff in tariffs.values())


def test_labels_carry_the_term_definitions_with_the_gap_period_from_the_artefact(client):
    terms = client.get("/labels").json()["terms"]

    assert {
        "export_potential", "actual_exports", "export_gap", "opportunity_type", "market_reach", "opportunity",
        "scenario", "baseline", "gap_period", "illustrative", "agreement_part", "blocked_opportunity",
    } <= set(terms)
    assert not {key for key in terms if key.endswith("_in_panel")}  # one wording per term, table and panel alike
    assert all(text.strip() for text in terms.values())
    assert terms["trade_agreement"].startswith("Whether a UK Trade Agreement with the Market is in force in 2024. ")
    assert terms["export_potential"] == (
        "The yearly value of UK exports of this Product to this Market that the model expects from their "
        "fundamentals: economy size, distance, tariffs and existing trade."
    )
    assert terms["gap_period"] == "The recent years (2022–2024) whose yearly Export Gaps are averaged into the Export Gap."
    assert terms["baseline"].endswith("The Baseline year (2024) is the year scored.")
    # The full rule at the fixed thresholds (`scenarios.BLOCKED_MIN_SHARE`, `BLOCKED_MIN_USD`), in its one wording
    assert terms["blocked_opportunity"].startswith("A pair with no UK Trade Agreement in force is a Blocked Opportunity")
    assert "at least 20% of Baseline and $1M" in terms["blocked_opportunity"]
    assert terms["blocked_opportunity"] == client.get("/labels").json()["blocked_rule"]
    assert terms["actual_exports"] == (
        "What the UK actually exported of this Product to this Market (2022–2024 average)."
    )
    assert terms["export_gap"] == (
        "Export Potential minus Actual Exports (2022–2024 average). Positive means the UK exports less than its "
        "fundamentals suggest: an Opportunity."
    )
    assert terms["opportunity_type"].startswith("Expand: the UK already exports the Product to this Market.")


def test_the_term_definitions_take_the_gap_period_label_from_the_artefact(fixture_artefacts, tmp_path):
    store.save(replace(fixture_artefacts, metadata={**fixture_artefacts.metadata, "gap_years": [2021, 2023]}), tmp_path)

    with TestClient(create_app(tmp_path)) as constructed:
        terms = constructed.get("/labels").json()["terms"]

    assert "(2021–2023 average)" in terms["actual_exports"] and "(2021–2023 average)" in terms["export_gap"]
    assert "(2021–2023)" in terms["gap_period"]


# --- One definition of the row cap, one read on Blocked Opportunities, reasons counted ---------------


def test_labels_serve_the_most_rows_a_ranking_read_returns(client):
    assert client.get("/labels").json()["max_rows"] == 6000


def test_the_blocked_opportunities_read_carries_the_default_thresholds(blocked_client):
    thresholds = blocked_client.get(BLOCKED_READ).json()["thresholds"]

    assert thresholds == {"min_effect_share": 0.2, "min_effect_usd": 1_000_000.0}


def test_blocked_opportunity_rows_have_the_documented_shape(blocked_client):
    # The rise in dollars and as a share, its Tariff part and Agreement part, the Barrier and the "above
    # Export Potential" mark; the old uplift field is gone, since room to grow keeps a dollar level always meaningful.
    rows = blocked_client.get(BLOCKED_READ).json()["opportunities"]

    assert rows
    assert set(rows[0]) == {
        "hs6",
        "description",
        "sector",
        "sector_name",
        "iso3",
        "market",
        "region",
        "export_potential_usd",
        "actual_exports_usd",
        "export_gap_usd",
        "relative_export_gap",
        "opportunity_type",
        "trade_agreement",
        "blocked_opportunity_value_usd",
        "blocked_opportunity",
        "flagged_product",
        "flagged_product_reason",
        "reexport_hub",
        "sanctioned_market",
        "sanctioned_market_reason",
        "restricted_pair",
        "restricted_pair_reason",
        "offset_pair",
        "offset_pair_reason",
        "caveat_texts",
        "potential_range_low_usd",
        "potential_range_high_usd",
        "uncertain_opportunity",
        "applied_tariff_pct",
        "baseline_export_potential_usd",
        "baseline_actual_exports_usd",
        "potential_with_agreement_usd",
        "rise_share",
        "tariff_part_usd",
        "agreement_part_usd",
        "barrier",
        "value_split",
        "above_export_potential",
    }
    for row in rows:
        assert isinstance(row["potential_with_agreement_usd"], float)
        assert isinstance(row["rise_share"], float)
        assert isinstance(row["tariff_part_usd"], float) and isinstance(row["agreement_part_usd"], float)
        assert isinstance(row["barrier"], str) and row["barrier"].startswith("No agreement, tariff ")
        assert isinstance(row["above_export_potential"], bool)
        assert isinstance(row["value_split"], str) and "associat" in row["value_split"]


def test_the_blocked_opportunities_read_offers_the_order_by_tariff_part(blocked_client):
    body = blocked_client.get(BLOCKED_READ, params={"rank_by": "tariff_part"}).json()

    assert body["rank_by"] == "tariff_part" and body["order"]["name"] == "tariff_part"
    parts = [row["tariff_part_usd"] for row in body["opportunities"]]
    assert parts == sorted(parts, reverse=True)
    assert blocked_client.get(BLOCKED_READ, params={"rank_by": "gap"}).status_code == 422


def test_excluded_product_reasons_are_counted_most_products_first(fixture_artefacts, tmp_path):
    fuels, bullion = "Mineral fuels: pipeline geography", "Bullion: market geography"
    excluded = pl.DataFrame(
        {
            "hs6": ["271011", "710812", "270900"],
            "description": ["Light petroleum oils", "Gold, unwrought", "Crude petroleum oils"],
            "sector": ["27", "71", "27"],
            "reason": [fuels, bullion, fuels],
        }
    )
    store.save(replace(fixture_artefacts, excluded_products=excluded), tmp_path)
    with TestClient(create_app(tmp_path)) as constructed:
        reasons = constructed.get("/products/excluded/reasons").json()

    assert reasons == [{"reason": fuels, "products": 2}, {"reason": bullion, "products": 1}]


def _jpn_counts(world, year: int) -> tuple[int, int]:
    """Japan's Products imported from other Exporters in `year` that the UK sent too, and all Japan imported from
    other Exporters, from the Flows."""
    from exportspredictor.exclusions import drop_excluded
    from exportspredictor.fixture import UK

    flows = drop_excluded(world.flows.filter((pl.col("year") == year) & (pl.col("importer") == "JPN")))
    flowing = flows.filter(pl.col("value_usd") > 0)
    from_others = set(flowing.filter(pl.col("exporter") != UK)["hs6"])
    from_uk = set(flowing.filter(pl.col("exporter") == UK)["hs6"])
    return len(from_uk & from_others), len(from_others)


def test_the_market_view_serves_the_markets_reach_in_the_reference_year(client, fixture_world, fixture_artefacts):
    year = fixture_artefacts.model_form.market_reach.reference_year
    sent, imported = _jpn_counts(fixture_world, year)

    for read in ("/opportunities/by-market/JPN", "/opportunities/top?market=JPN"):
        reach = client.get(read).json()["market_reach"]
        assert reach["year"] == year
        assert (reach["products_sent"], reach["products_imported"]) == (sent, imported)
        assert 0 < reach["products_expected"] <= imported
        assert reach["text"].startswith(
            f"In {year} Japan imported {imported:,} Products from other Exporters and the UK sent {sent:,} of them"
        )
        assert f"about {reach['products_expected']:,.0f} were expected" in reach["text"]
    assert client.get("/opportunities/top").json()["market_reach"] is None


def test_the_pair_detail_serves_its_markets_reach_and_the_offset_applied_to_it(client, fixture_artefacts):
    body = client.get(f"/explain/{FIXTURE_HS6}/JPN").json()
    market = client.get("/opportunities/by-market/JPN").json()["market_reach"]
    reach = fixture_artefacts.model_form.market_reach
    applied = reach.table.filter((pl.col("fold") == body["fold"]) & (pl.col("importer") == "JPN"))["offset"].item()

    assert {k: body["market_reach"][k] for k in market} == market
    assert body["market_reach"]["pair_offset"] == pytest.approx(applied)
    assert body["market_reach"]["pair_multiplier"] == pytest.approx(math.exp(reach.beta * applied))


def test_evaluation_serves_what_market_reach_changes(client, fixture_artefacts):
    effect = client.get("/evaluation").json()["market_reach_effect"]
    saved = fixture_artefacts.evaluation["market_reach_effect"]

    assert effect["text"] == saved["text"]
    assert effect["selection_measure"] == saved["selection_measure"]
    assert effect["lift"] == saved["lift"]
    assert effect["uk_pair_coverage"] == saved["uk_pair_coverage"]


# --- `/evaluation`'s `itc_comparison`: ITC's realisation test on our model, worded for Advisors by the API ---


def _served_evaluation(bundle, root) -> dict:
    store.save(bundle, root)
    with TestClient(create_app(root)) as test_client:
        response = test_client.get("/evaluation")
    assert response.status_code == 200
    return response.json()


@pytest.fixture(scope="module")
def itc_report():
    return realisation.realisation_report(realisation.fixture_panel())


def test_evaluation_serves_the_itc_comparison_s_hs2_rows_its_years_and_the_api_s_own_text(
    fixture_artefacts, itc_report, tmp_path
):
    served = _served_evaluation(realisation.with_itc_comparison(fixture_artefacts, itc_report), tmp_path)[
        "itc_comparison"
    ]

    # The HS2 rows, the level compared with ITC, for the three tests and both predictors; HS6 stays in the notebook.
    hs2 = [row for row in itc_report["rows"] if row["level"] == "HS2"]
    assert [(row["test"], row["level"], row["predictor"]) for row in served["rows"]] == [
        (test, "HS2", predictor)
        for test in ("intensive_margin", "extensive_margin", "new_flow_size")
        for predictor in ("model", "naive_baseline")
    ]
    for row, saved in zip(served["rows"], hs2, strict=True):
        assert row["coefficient"] == pytest.approx(saved["coefficient"])
    [saved] = [row for row in hs2 if (row["test"], row["predictor"]) == ("intensive_margin", "model")]
    # The base years, and the last year each base year's fold models trained on.
    assert served["base_years"] == [2015, 2016, 2017, 2018, 2019]
    assert served["training_cut_offs"] == [
        {"base_year": t, "training_cut_off": t} for t in (2015, 2016, 2017, 2018, 2019)
    ]
    assert {row["test"]: row["coefficient"] for row in served["published"]} == {
        "intensive_margin": 0.615, "extensive_margin": 0.023, "new_flow_size": 0.243,
    }
    # The text reads the HS2 level, the one compared with ITC.
    assert f"{saved['coefficient']:.3f}" in served["text"] and "0.615" in served["text"]
    assert "(HS2)" in served["text"] and "of 3" in served["text"] and "Naive Baseline" in served["text"]
    assert "2015–2019" in served["text"]
    # Agreement, never validation (CODING_STANDARDS.md: associational language).
    assert "rough benchmark" in served["text"]
    assert "validat" not in served["text"].replace("not a validation", "")
    # The caveats name each difference from ITC.
    for difference in ("UK only", "no exporter fixed effect", "76 sub-sectors", "HS2", "2006–2018", "2015–2019",
                       "learned model", "no forward projection"):
        assert difference in served["caveat"], difference
    # Each predictor runs on the units it scores above zero, so the caveat says where the samples differ.
    assert "units the Naive Baseline scores above zero" in served["caveat"]
    # The base-year models' departures from "never saw a year after the base year" are named.
    for departure in ("Relatedness", "training scope", "no later trade", "round count"):
        assert departure in served["caveat"], departure
    # Advisors see only the level compared with ITC.
    assert {row["Level"] for row in served["table"]} == {"HS2"}
    assert "not compared" not in {row["ITC (76 sub-sectors)"] for row in served["table"]}
    assert [row["Test"] for row in served["table"]] == [
        "Intensive margin", "Intensive margin", "Extensive margin", "New-flow size",
    ]
    assert served["direct_comparison"]["run"] is False
    assert "ITC comparison" not in [part["title"] for part in _served_evaluation(
        realisation.with_itc_comparison(fixture_artefacts, itc_report), tmp_path / "again"
    )["not_run"]]


def test_evaluation_without_the_itc_comparison_omits_it_and_lists_the_part_as_not_run(fixture_artefacts, tmp_path):
    evaluation = {k: v for k, v in fixture_artefacts.evaluation.items() if k != realisation.REPORT_KEY}
    served = _served_evaluation(replace(fixture_artefacts, evaluation=evaluation), tmp_path)

    assert served.get("itc_comparison") is None
    [part] = [part for part in served["not_run"] if part["title"] == "ITC comparison"]
    assert "not run" in part["text"]
    assert "python -m exportspredictor.realisation" in part["text"]


# --- How the Model Works -------------------------------------------------------------------------------------------


def test_how_the_model_works_serves_its_outline_of_ideas_and_layers(client):
    body = client.get("/how-it-works").json()

    assert set(body) == {"intro", "layers", "sections"}
    assert body["layers"] == {"what": "What it is", "why": "Why this project uses it", "how": "How it is done here"}
    assert [section["title"] for section in body["sections"]] == [
        "How the model is built", "How the model is checked", "What it produces and how it is served"
    ]
    ideas = [idea for section in body["sections"] for idea in section["ideas"]]
    assert all(set(idea) == {"name", "summary", "what", "why", "how"} for idea in ideas)
    text = " ".join(idea[field] for idea in ideas for field in ("summary", "what", "why", "how"))
    # The package's settings: 100 Exporters, 5% of Zero Cells, five folds, 20 bands of an 80% range
    for fact in ("100 largest economies", "5% of their Zero Cells", "into five groups", "20 bands", "80%"):
        assert fact in text, fact


def test_how_the_model_works_states_the_served_model_s_blend_weights_market_reach_and_sampling_rate(
    fixture_artefacts, tmp_path
):
    form = fixture_artefacts.metadata["model_form"]
    served_form = {**form, "naive_weight": 0.4, "market_reach": {**form["market_reach"], "beta": 2.0}}
    metadata = {**fixture_artefacts.metadata, "model_form": served_form, "non_uk_sampling_rate": 0.5894}
    store.save(replace(fixture_artefacts, metadata=metadata), tmp_path)

    with TestClient(create_app(tmp_path)) as constructed:
        body = constructed.get("/how-it-works").json()

    ideas = {idea["name"]: idea for section in body["sections"] for idea in section["ideas"]}
    blend = ideas["The blend with the Naive Baseline"]
    assert r"0.6 \log \text{model} + 0.4 \log \text{naive}" in blend["what"]
    assert "The weight 0.4 was chosen" in blend["how"] and "scaled by 0.6" in blend["how"]
    assert r"e^{2 \cdot \text{offset}}" in ideas["Market Reach"]["what"]
    assert "The strength, 2, was chosen" in ideas["Market Reach"]["how"]
    assert "then 58.9% of all their rows" in ideas["The data"]["how"]