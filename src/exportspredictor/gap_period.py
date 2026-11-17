"""The Gap Period: the recent years whose yearly Export Gaps are averaged into the Export Gap.

Advisors see the mean over the Gap Period (2022–2024 for the 2024 Baseline) of each year's
Export Potential minus that year's Actual Exports, not one year alone (decision #30), so one
unusual shipment or the provisional Baseline-year figures do not decide a ranking. Each
year is scored with that year's features by the pair's out-of-fold model (ADR 0001), and a
year with no Flow counts as $0. The plain mean is used, so Export Gaps still sum across pairs.

2020 and 2021 are left out: COVID-19 and the end of the Brexit transition period distorted UK
trade in those years, so they say little about the UK's current fundamentals.

Scenarios, the Baseline Export Potential they start from and the Blocked Opportunity value
use the Baseline year only.
"""

import polars as pl

GAP_PERIOD_LENGTH = 3  # years, ending with the Baseline year


def gap_years(baseline_year: int) -> list[int]:
    """Every year of the Gap Period ending with `baseline_year`, oldest first."""
    return list(range(baseline_year - GAP_PERIOD_LENGTH + 1, baseline_year + 1))


def gap_period(baseline_year: int) -> list[int]:
    """The first and last year of the Gap Period, as the artefact metadata records it (`gap_years`)."""
    years = gap_years(baseline_year)
    return [years[0], years[-1]]


def export_gaps(yearly: pl.DataFrame) -> pl.DataFrame:
    """The Gap Period means of each pair's yearly Export Potential, Actual Exports and Export Gap.

    `yearly` holds hs6, iso3, year, export_potential_usd and actual_exports_usd (0 for a year
    with no Flow) for every pair in every year of the Gap Period. Returns hs6, iso3 and the
    three means, sorted by hs6 and iso3.
    """
    years = yearly["year"].n_unique()
    counts = yearly.group_by("hs6", "iso3").agg(pl.col("year").n_unique())["year"]
    if yearly.select("hs6", "iso3", "year").is_duplicated().any() or (counts != years).any():
        raise ValueError("Every pair needs exactly one score in every year of the Gap Period")
    return (
        yearly.group_by("hs6", "iso3")
        .agg(pl.col("export_potential_usd", "actual_exports_usd").mean())
        .with_columns(export_gap_usd=pl.col("export_potential_usd") - pl.col("actual_exports_usd"))
        .sort("hs6", "iso3")
    )


def baseline_year_scores(history: pl.DataFrame, baseline_year: int) -> pl.DataFrame:
    """One year's UK scores from the yearly history: hs6, iso3, export_potential_usd, export_gap_usd.

    The evaluation's stability check compares two single years scored by the same models,
    so it reads the Baseline year's scores, not the Gap Period means.
    """
    return history.filter(pl.col("year") == baseline_year).select(
        "hs6", "iso3", "export_potential_usd", "export_gap_usd"
    )


EXPAND, NEW_MARKET, NEW_PRODUCT = OPPORTUNITY_TYPES = ("Expand", "New Market", "New Product")


def with_opportunity_columns(gaps: pl.DataFrame) -> pl.DataFrame:
    """`gaps` (one row per UK pair: Gap Period means) with its Relative Export Gap and Opportunity Type.

    `relative_export_gap` is the Export Gap divided by Export Potential (missing where Export
    Potential is 0). `opportunity_type` says how far the UK already trades the pair over the
    Gap Period: Expand (Actual Exports to this Market), New Market (none here, but some to
    another Market) or New Product (none to any Market). `gaps` must hold every Market of each
    Product, as the UK grid does, so "any Market" is read from the Product's own rows.
    """
    exports_here = pl.col("actual_exports_usd") > 0
    return gaps.with_columns(
        relative_export_gap=pl.when(pl.col("export_potential_usd") > 0).then(
            pl.col("export_gap_usd") / pl.col("export_potential_usd")
        ),
        opportunity_type=pl.when(exports_here)
        .then(pl.lit(EXPAND))
        .when(exports_here.any().over("hs6"))
        .then(pl.lit(NEW_MARKET))
        .otherwise(pl.lit(NEW_PRODUCT)),
    )
