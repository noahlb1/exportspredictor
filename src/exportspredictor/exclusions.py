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
ADR 0005. They stay in training, scoring and rankings. Their lists and caveat text are
here, except the volatile and one-off Flagged Products, which `volatility` derives from the
training years' Flows; `caveats.TIERS` declares them as Caveat Tiers, which mark the Baseline with the
boolean `flagged_product` and `reexport_hub` columns, so Advisors see their caveats and
can hide them.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import polars as pl

from exportspredictor.lookups import as_code


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


def _matches(exclusion: Exclusion, code: pl.Expr) -> pl.Expr:
    covered = pl.any_horizontal(code.str.starts_with(p) for p in exclusion.prefixes)
    if exclusion.exceptions:
        covered = covered & ~pl.any_horizontal(code.str.starts_with(e) for e in exclusion.exceptions)
    return covered


def reason_expr(column: str = "hs6") -> pl.Expr:
    """The exclusion reason of each row's Product, null where the Product is not excluded."""
    code = as_code(column)
    expr = pl.lit(None, pl.String)
    for exclusion in reversed(EXCLUSIONS):
        expr = pl.when(_matches(exclusion, code)).then(pl.lit(exclusion.reason)).otherwise(expr)
    return expr


def is_excluded(column: str = "hs6") -> pl.Expr:
    """True where the row's Product is an Excluded Product."""
    code = as_code(column)
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


def reason_counts(excluded: pl.DataFrame) -> pl.DataFrame:
    """How many Excluded Products each reason leaves out (reason, products), most first; ties keep table order."""
    return (
        excluded.group_by("reason", maintain_order=True)
        .len(name="products")
        .sort("products", descending=True, maintain_order=True)
    )


# --- Flagged Products and Re-export Hubs: scored and ranked as normal, shown with a caveat ---

# The rest of Sector 71 after the exclusions (HS4 codes), as listed in ADR 0005.
FLAGGED_PRODUCT_PREFIXES = (
    "7101", "7102", "7103", "7104", "7105", "7107", "7109", "7111",
    "7113", "7114", "7115", "7116", "7117",
)
FLAGGED_PRODUCT_CAVEAT = (
    "Flagged Product: UK exports of this Product are volatile, one-off or re-export-led (precious stones and "
    "jewellery of Sector 71, Products whose UK exports swing widely from year to year, and Products sold in a few "
    "large contracts), so treat the Export Potential with caution: what looks unmet may be timing rather than "
    "lasting potential."
)
# A Flagged Product row's caveat text, naming its reason: SECTOR_71_REASON here, or a reason of `volatility`.
FLAGGED_PRODUCT_ROW_CAVEAT = (
    "Flagged Product ({reason}): treat the Export Potential with caution, as what looks unmet may be timing rather "
    "than lasting potential."
)
SECTOR_71_REASON = "Sector 71: precious stones and jewellery, whose recorded values are volatile and re-export-led"
REEXPORT_HUBS = frozenset({"NLD", "BEL", "IRL", "HKG", "SGP", "ARE", "CHE", "LUX"})
REEXPORT_HUB_CAVEAT = (
    "Re-export Hub: this Market re-exports much of what it imports, so part of its demand belongs "
    "to other Markets."
)


def flagged_product_expr(column: str = "hs6") -> pl.Expr:
    """True where the row's Product is a Sector 71 Flagged Product (the volatility rule's are in `volatility`)."""
    code = as_code(column)
    return pl.any_horizontal(code.str.starts_with(p) for p in FLAGGED_PRODUCT_PREFIXES)


def flagged_product_reason_expr(reasons: Mapping[str, str], column: str = "hs6") -> pl.Expr:
    """Each row's Flagged Product reason: Sector 71's, else the Product's in `reasons` (hs6 -> reason), else null."""
    code = as_code(column)
    other = (
        code.replace_strict(dict(reasons), default=None, return_dtype=pl.String) if reasons
        else pl.lit(None, pl.String)
    )
    return pl.when(flagged_product_expr(column)).then(pl.lit(SECTOR_71_REASON)).otherwise(other)


def reexport_hub_expr(column: str = "iso3") -> pl.Expr:
    """True where the row's Market is a Re-export Hub."""
    return as_code(column).is_in(sorted(REEXPORT_HUBS))


def not_ranked_message(excluded: pl.DataFrame, hs6: str) -> str | None:
    """Why `hs6` has no ranking or explanation, from the artefact's `excluded_products` table; None if it has.

    The API answers a ranking or Why? request for an Excluded Product with this message.
    """
    found = excluded.filter(pl.col("hs6") == hs6)
    if found.is_empty():
        return None
    row = found.row(0, named=True)
    return f"{hs6} ({row['description']}) is an Excluded Product, so it is not scored or ranked. {row['reason']}"
