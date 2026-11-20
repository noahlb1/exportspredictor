"""Volatile Flagged Products (ADR 0005, spec #113): Products whose UK exports swing most from year to year over the
training years, plus the one-off Products the open review named."""

import polars as pl
import pytest
from conftest import run_page, serve_api
from fastapi.testclient import TestClient

from api.main import create_app
from exportspredictor import artefacts as store
from exportspredictor.caveats import FLAGGED_PRODUCT, CaveatContext, with_caveat_columns
from exportspredictor.exclusions import SECTOR_71_REASON
from exportspredictor.pipeline import recaveat
from exportspredictor.volatility import (
    METADATA_KEY,
    ONE_OFF_REASON,
    VOLATILE_REASON,
    VolatileProducts,
    volatile_products,
)

BASELINE_YEAR = 2024  # validation year 2022, test year 2023: training years are 2016 to 2021
YEARS = list(range(2016, 2025))

# UK exports of each Product by year. Five calm Products and one that swings: the swinging one is the top fifth.
CALM = {f"84{n:04d}": [100.0] * len(YEARS) for n in range(1, 6)}
SWINGING = {"300490": [100.0, 1000.0, 50.0, 900.0, 80.0, 1200.0, 100.0, 100.0, 100.0]}


def _flows(exports: dict[str, list[float]]) -> pl.DataFrame:
    """Flows of the UK (to one Market) and of another Exporter, whose Flows must not count."""
    rows = [
        {"year": year, "exporter": exporter, "importer": "FRA", "hs6": hs6, "value_usd": value}
        for hs6, values in exports.items()
        for year, value in zip(YEARS, values, strict=True)
        for exporter in ("GBR", "DEU")
    ]
    return pl.DataFrame(rows)


def _rule(exports: dict[str, list[float]]) -> VolatileProducts:
    flows = _flows(exports)
    return volatile_products(
        lambda year: flows.filter(pl.col("year") == year), YEARS, baseline_year=BASELINE_YEAR,
        products=sorted(exports),
    )


def test_the_top_fifth_by_volatility_of_uk_exports_is_flagged_as_volatile():
    rule = _rule({**CALM, **SWINGING})

    assert rule.reasons == {"300490": VOLATILE_REASON}
    assert rule.years == [2016, 2017, 2018, 2019, 2020, 2021]


def test_changing_the_validation_test_and_baseline_years_flows_leaves_the_flags_unchanged():
    # The calm Products now swing wildly in 2022, 2023 and 2024 and the swinging one is flat there.
    changed = {hs6: values[:6] + [1.0, 5000.0, 3.0] for hs6, values in CALM.items()}
    changed["300490"] = SWINGING["300490"][:6] + [100.0, 100.0, 100.0]

    assert _rule({**changed}).reasons == _rule({**CALM, **SWINGING}).reasons == {"300490": VOLATILE_REASON}


def test_a_training_year_change_does_change_the_flags():
    calm_swinging = SWINGING["300490"][:6] + [100.0, 100.0, 100.0]
    flipped = {**CALM, "840001": calm_swinging, "300490": [100.0] * len(YEARS)}

    assert _rule(flipped).reasons == {"840001": VOLATILE_REASON}


def test_named_one_off_products_are_flagged_as_one_offs_when_the_rule_misses_them():
    one_offs = {"490700": [100.0] * len(YEARS), "880230": [100.0] * len(YEARS), "284390": [100.0] * len(YEARS)}

    assert _rule({**CALM, **SWINGING, **one_offs}).reasons == {
        "300490": VOLATILE_REASON,
        "490700": ONE_OFF_REASON,
        "880230": ONE_OFF_REASON,
        "284390": ONE_OFF_REASON,
    }


def _flagged(hs6s: list[str], reasons: dict[str, str]) -> dict[str, str | None]:
    """Each Product's Flagged Product reason, None where it is not a Flagged Product."""
    rows = pl.DataFrame({"hs6": hs6s, "iso3": ["FRA"] * len(hs6s)})
    marked = with_caveat_columns(
        rows, CaveatContext(baseline_year=BASELINE_YEAR, flagged_product_reasons=reasons), tiers=(FLAGGED_PRODUCT,)
    )
    return {
        row["hs6"]: row["flagged_product_reason"] if row["flagged_product"] else None
        for row in marked.iter_rows(named=True)
    }


def test_the_flagged_product_tier_flags_the_rules_products_with_their_reasons():
    reasons = {"300490": VOLATILE_REASON, "880230": ONE_OFF_REASON}

    assert _flagged(["300490", "880230", "840001", "711319"], reasons) == {
        "300490": VOLATILE_REASON,
        "880230": ONE_OFF_REASON,
        "840001": None,
        "711319": SECTOR_71_REASON,
    }


def test_without_the_rule_only_sector_71_is_flagged():
    assert _flagged(["300490", "711319"], {}) == {"300490": None, "711319": SECTOR_71_REASON}


def test_a_flagged_products_caveat_names_its_reason():
    text = FLAGGED_PRODUCT.caveat_text(VOLATILE_REASON)

    assert text.startswith("Flagged Product")
    assert "UK exports swing widely from year to year" in text


# --- The pipeline: the rule is recorded in the artefact and re-marked without retraining ----------------------------


def test_the_artefact_records_the_rule_and_reads_only_the_training_years(fixture_artefacts):
    recorded = fixture_artefacts.metadata[METADATA_KEY]

    # The fixture world's years are 2021 to 2024 and its Baseline year 2024: only 2021 is a training year.
    assert recorded["years"] == [2021]
    assert recorded["volatile_reason"] == VOLATILE_REASON and recorded["one_off_reason"] == ONE_OFF_REASON
    assert VolatileProducts.from_dict(recorded).years == [2021]


def test_recaveat_marks_the_rules_products_and_records_the_rule(fixture_artefacts):
    hs6 = "220830"  # whisky: neither in Sector 71 nor a one-off
    rule = VolatileProducts(years=[2016, 2017], share=0.2, threshold=0.5, reasons={hs6: VOLATILE_REASON})

    marked = recaveat(fixture_artefacts, version="volatile", volatile=rule)

    flagged = marked.baseline.filter("flagged_product")
    assert set(flagged["hs6"]) == {hs6, "711319"}
    assert set(flagged.filter(pl.col("hs6") == hs6)["flagged_product_reason"]) == {VOLATILE_REASON}
    assert marked.metadata[METADATA_KEY]["products"] == {hs6: VOLATILE_REASON}
    assert marked.metadata[METADATA_KEY]["threshold"] == 0.5
    scores = ["export_potential_usd", "export_gap_usd", "baseline_export_potential_usd"]
    assert marked.baseline.select("hs6", "iso3", *scores).equals(fixture_artefacts.baseline.select("hs6", "iso3", *scores))

    # A later recaveat (a sanctions table changed, say) keeps the recorded rule.
    again = recaveat(marked, version="again")
    assert again.baseline.equals(marked.baseline)
    assert again.metadata[METADATA_KEY] == marked.metadata[METADATA_KEY]


# --- Served: the API rows and the app's detail panel name the reason ------------------------------------------------

WHISKY, JAPAN = "220830", "JPN"


@pytest.fixture(scope="module")
def volatile_dir(fixture_artefacts, tmp_path_factory):
    rule = VolatileProducts(years=[2016, 2017], share=0.2, threshold=0.5, reasons={WHISKY: VOLATILE_REASON})
    root = tmp_path_factory.mktemp("volatile")
    store.save(recaveat(fixture_artefacts, version="volatile", volatile=rule), root)
    return root


def test_api_rows_carry_each_flagged_products_reason(volatile_dir):
    with TestClient(create_app(volatile_dir)) as client:
        rows = client.get("/opportunities/top").json()["opportunities"]
        explained = client.get(f"/explain/{WHISKY}/{JAPAN}").json()

    reasons = {row["hs6"]: row["flagged_product_reason"] for row in rows if row["flagged_product"]}
    assert reasons[WHISKY] == VOLATILE_REASON
    assert set(reasons.values()) <= {VOLATILE_REASON, SECTOR_71_REASON}
    assert all(row["flagged_product_reason"] is None for row in rows if not row["flagged_product"])
    assert explained["flagged_product_reason"] == VOLATILE_REASON


def test_the_detail_panel_names_the_reason(monkeypatch, volatile_dir):
    with serve_api(volatile_dir) as url:
        monkeypatch.setenv("EXPORTSPREDICTOR_API_URL", url)
        page = run_page(query={"product": WHISKY, "market": JAPAN})

    assert FLAGGED_PRODUCT.caveat_text(VOLATILE_REASON) in [element.value for element in page.warning]
