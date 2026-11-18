"""Excluded Products (ADR 0005): Products whose Flows the model cannot explain.

Their recorded Flows follow bullion-market, pipeline or grid geography the model cannot
see, or their values are unreliable. They are dropped as rows from training, scoring and
every ranking (`drop_excluded`), but every country and world total still counts them:
`features.build_features` computes its aggregates from all Flows, so the Exporter total,
the world totals inside RCA and the leave-one-out supply and demand aggregates are the
same with or without the exclusion. The exclusion limits rows, not features.

`EXCLUSIONS` is the one definition of the list. The artefact carries it expanded to HS6
Products with reasons (`excluded_products`), which the API serves at `/products/excluded`.

Flagged Products (the rest of Sector 71) and Re-export Hubs are the other two tiers of
ADR 0005. They stay in training, scoring and rankings; the Baseline carries them as the
boolean `flagged_product` and `reexport_hub` columns (`flag_columns`), and Advisors see
their caveats and can hide them.
"""

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class Exclusion:
    """One entry of the list: the HS codes it covers (as written in ADR 0005) and why."""

    codes: str  # what the entry covers, for people: "Sector 27 except 2712"
    prefixes: tuple[str, ...]  # HS2 or HS4 codes whose Products are excluded
    reason: str
    exceptions: tuple[str, ...] = ()  # HS4 codes inside `prefixes` that stay in


EXCLUSIONS = (
    Exclusion(
        "Sector 27 except 2712",
        ("27",),
        "Mineral fuels: crude and refined oil, gas, coal and electricity. Their trade follows "
        "pipelines, grids and tankers, which the model cannot see.",
        exceptions=("2712",),
    ),
    Exclusion(
        "7106, 7108, 7110, 7112 and 7118",
        ("7106", "7108", "7110", "7112", "7118"),
        "Silver, gold, platinum-group metals, precious-metal scrap and coins: bullion trade, "
        "which is volatile and poorly recorded.",
    ),
    Exclusion("9706", ("9706",), "Antiques: one-off objects, not production capacity."),
    Exclusion("2844", ("2844",), "Radioactive elements: strategic, licensed trade the model cannot see."),
)


def _code(column: str) -> pl.Expr:
    return pl.col(column).cast(pl.String)


def _matches(exclusion: Exclusion, code: pl.Expr) -> pl.Expr:
    covered = pl.any_horizontal(code.str.starts_with(p) for p in exclusion.prefixes)
    if exclusion.exceptions:
        covered = covered & ~pl.any_horizontal(code.str.starts_with(e) for e in exclusion.exceptions)
    return covered


def reason_expr(column: str = "hs6") -> pl.Expr:
    """The exclusion reason of each row's Product, null where the Product is not excluded."""
    code = _code(column)
    expr = pl.lit(None, pl.String)
    for exclusion in reversed(EXCLUSIONS):
        expr = pl.when(_matches(exclusion, code)).then(pl.lit(exclusion.reason)).otherwise(expr)
    return expr


def is_excluded(column: str = "hs6") -> pl.Expr:
    """True where the row's Product is an Excluded Product."""
    code = _code(column)
    return pl.any_horizontal(_matches(exclusion, code) for exclusion in EXCLUSIONS)


def exclusion_reason(hs6: str) -> str | None:
    """Why `hs6` is an Excluded Product, or None if it is not one."""
    return pl.DataFrame({"hs6": [hs6]}).select(reason_expr()).item()


def drop_excluded[T: (pl.DataFrame, pl.LazyFrame)](rows: T) -> T:
    """`rows` without the rows of Excluded Products (keyed by `hs6`)."""
    return rows.filter(~is_excluded())


def excluded_products(products: pl.DataFrame) -> pl.DataFrame:
    """The Excluded Products of a Product lookup (hs6, description, sector), with their reasons."""
    return (
        products.filter(is_excluded())
        .select("hs6", "description", "sector", reason=reason_expr())
        .sort("hs6")
    )


# --- Flagged Products and Re-export Hubs: scored and ranked as normal, shown with a caveat ---

# The rest of Sector 71 after the exclusions (HS4 codes), as listed in ADR 0005.
FLAGGED_PRODUCT_PREFIXES = (
    "7101", "7102", "7103", "7104", "7105", "7107", "7109", "7111",
    "7113", "7114", "7115", "7116", "7117",
)
FLAGGED_PRODUCT_CAVEAT = (
    "Flagged Product: precious stones, jewellery and other Sector 71 goods. Their recorded values are "
    "volatile and entrepôt-driven, so treat the Export Potential with caution."
)
REEXPORT_HUBS = frozenset({"NLD", "BEL", "IRL", "HKG", "SGP", "ARE", "CHE", "LUX"})
REEXPORT_HUB_CAVEAT = (
    "Re-export Hub: this Market re-exports much of what it imports, so part of its demand belongs "
    "to other Markets."
)


def flagged_product_expr(column: str = "hs6") -> pl.Expr:
    """True where the row's Product is a Flagged Product."""
    code = _code(column)
    return pl.any_horizontal(code.str.starts_with(p) for p in FLAGGED_PRODUCT_PREFIXES)


def reexport_hub_expr(column: str = "iso3") -> pl.Expr:
    """True where the row's Market is a Re-export Hub."""
    return pl.col(column).cast(pl.String).is_in(sorted(REEXPORT_HUBS))


def flag_columns(hs6: str = "hs6", iso3: str = "iso3") -> list[pl.Expr]:
    """The boolean `flagged_product` and `reexport_hub` columns of rows keyed by Product and Market."""
    return [flagged_product_expr(hs6).alias("flagged_product"), reexport_hub_expr(iso3).alias("reexport_hub")]


def is_flagged_product(hs6: str) -> bool:
    """Whether `hs6` is a Flagged Product."""
    return hs6.startswith(FLAGGED_PRODUCT_PREFIXES)


def not_ranked_message(excluded: pl.DataFrame, hs6: str) -> str | None:
    """Why `hs6` has no ranking or explanation, from the artefact's `excluded_products` table; None if it has.

    The API answers a ranking or Why? request for an Excluded Product with this message.
    """
    found = excluded.filter(pl.col("hs6") == hs6)
    if found.is_empty():
        return None
    row = found.row(0, named=True)
    return f"{hs6} ({row['description']}) is an Excluded Product, so it is not scored or ranked. {row['reason']}"
