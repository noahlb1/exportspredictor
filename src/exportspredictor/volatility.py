"""Volatile Flagged Products (ADR 0005, amended): Products whose UK exports swing most from year to year.

A Product's volatility is the standard deviation of the year-to-year log change of the UK's total exports of it over
the training years: the years before the validation year (`training_years`), so neither the validation year, the test
year nor the Baseline year decides which Products are flagged. Products above the 80th percentile of volatility among
the scored Products (the top fifth) are flagged as volatile. The one-off Products the open review named
(`ONE_OFF_PRODUCTS`) are flagged as one-offs if the rule misses them.

The open review measured the same thing over 2012–2021 and found the 2022 Selection Measure falls from 0.76 for the
calmest fifth of Products to 0.30 for the lumpiest: their Export Gaps are mostly timing, not lasting potential.

The result (`VolatileProducts`) is recorded in the artefact metadata (`METADATA_KEY`) and reaches the Flagged Product
Caveat Tier through `caveats.CaveatContext.flagged_product_reasons`.
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

import polars as pl

UK = "GBR"
VOLATILE_SHARE = 0.2  # the top fifth by volatility is flagged
VOLATILE_REASON = "volatile: UK exports swing widely from year to year"
ONE_OFF_REASON = "one-off: few, large contracts"
# HS4 or HS6 codes of the one-off Products the open review named (section 7): banknotes and stamps (sovereign
# contracts), aircraft, vessels and floating platforms, and precious-metal compounds (one producer's Flows).
ONE_OFF_PRODUCTS = ("4907", "8802", "8901", "8905", "284390")
METADATA_KEY = "volatile_products"


def training_years(years: Iterable[int], baseline_year: int) -> list[int]:
    """The years volatility reads: those before the validation year (`baseline_year - 2`)."""
    return sorted(y for y in years if y < baseline_year - 2)


@dataclass(frozen=True)
class VolatileProducts:
    """The volatility rule's result: the years it read, its threshold and each flagged Product's reason."""

    years: list[int]
    share: float
    threshold: float | None  # the volatility above which a Product is flagged; None when no Product has one
    reasons: Mapping[str, str]  # hs6 -> VOLATILE_REASON or ONE_OFF_REASON

    def to_dict(self) -> dict:
        return {
            "years": list(self.years),
            "share": self.share,
            "threshold": self.threshold,
            "volatile_reason": VOLATILE_REASON,
            "one_off_reason": ONE_OFF_REASON,
            "one_off_codes": list(ONE_OFF_PRODUCTS),
            "products": dict(sorted(self.reasons.items())),
        }

    @classmethod
    def from_dict(cls, saved: Mapping) -> "VolatileProducts":
        return cls(
            years=list(saved["years"]), share=saved["share"], threshold=saved["threshold"],
            reasons=dict(saved["products"]),
        )


def product_volatility(exports: pl.DataFrame) -> pl.DataFrame:
    """Each Product's volatility (hs6, volatility) from the UK's exports by year (hs6, year, value_usd).

    A year without exports counts as $1, so a Product that stops and starts swings widely.
    """
    years = exports.select(pl.col("year").unique().sort())
    totals = exports.group_by("hs6", "year").agg(pl.col("value_usd").sum())
    dense = (
        exports.select("hs6").unique().join(years, how="cross")
        .join(totals, on=["hs6", "year"], how="left")
        .with_columns(pl.col("value_usd").fill_null(0.0))
        .sort("hs6", "year")
    )
    return dense.group_by("hs6").agg(volatility=pl.col("value_usd").clip(1.0).log().diff().std())


def volatile_products(
    year_flows: Callable[[int], pl.DataFrame],
    years: Iterable[int],
    *,
    baseline_year: int,
    products: Sequence[str],
    share: float = VOLATILE_SHARE,
    uk: str = UK,
) -> VolatileProducts:
    """The volatility rule for the scored `products`, from every Exporter's Flows of each training year.

    `year_flows(year)` gives a year's Flows (exporter, hs6, value_usd); only the training years among `years` are read.
    """
    read = training_years(years, baseline_year)
    schema = {"hs6": pl.String, "year": pl.Int32, "value_usd": pl.Float64}
    exports = pl.concat(
        [pl.DataFrame(schema=schema)]
        + [
            year_flows(year).filter(pl.col("exporter") == uk).select(
                pl.col("hs6").cast(pl.String), year=pl.lit(year, pl.Int32), value_usd=pl.col("value_usd").cast(pl.Float64)
            )
            for year in read
        ]
    )
    scored = pl.DataFrame({"hs6": sorted(set(products))}, schema={"hs6": pl.String})
    volatility = scored.join(product_volatility(exports), on="hs6", how="left").drop_nulls("volatility")
    threshold = volatility["volatility"].quantile(1 - share, "linear") if volatility.height else None
    volatile = [] if threshold is None else volatility.filter(pl.col("volatility") > threshold)["hs6"].to_list()
    reasons = dict.fromkeys(volatile, VOLATILE_REASON)
    for hs6 in scored["hs6"]:
        if hs6 not in reasons and hs6.startswith(ONE_OFF_PRODUCTS):
            reasons[hs6] = ONE_OFF_REASON
    return VolatileProducts(years=read, share=share, threshold=threshold, reasons=reasons)
