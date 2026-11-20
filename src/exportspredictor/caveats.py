"""Caveat Tiers (ADR 0005): groups of UK pairs that are scored and ranked as normal but shown with a caveat.

A tier is one `CaveatTier` declaration in `TIERS`. Everything else reads the declarations:

- the pipeline marks the Baseline with each tier's boolean column, and its reason column if it declares reasons
  (`with_caveat_columns`, which also marks a saved Baseline again without retraining);
- the rankings drop a tier's rows when asked (`rankings.rank_opportunities(hide=...)`);
- the API serves the tiers at `/caveats`, adds each tier's column (and reason) to every ranking row and explanation,
  and takes each tier's hide filter as a query parameter (`api.schemas.tiered`);
- the app draws each tier's badge, Caveats column entry, hide filter, hidden-rows notice and caveat text from
  `/caveats`.

A tier is keyed on the Market (`iso3`), the Product (`hs6`) or the pair (both): its `flags` expression, and its
`reason` expression if any, may read only its key's columns. `flags` and `reason` take a `CaveatContext`, so a tier
can depend on the Baseline year (a sanctions regime in force that year) or, once a field is added to the context, on
other inputs the pipeline holds.

To add a tier: declare it (its membership usually lives in a module of its own, as Flagged Products and Re-export
Hubs live in `exclusions`), append it to `TIERS`, add its term to GLOSSARY.md, and write an artefact version whose
Baseline carries its columns (`with_caveat_columns` over the saved Baseline). No API or app code changes.

Excluded Products are not a tier: they are left out of scoring and rankings altogether (`exclusions`).
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import polars as pl

from exportspredictor.exclusions import (
    FLAGGED_PRODUCT_CAVEAT,
    REEXPORT_HUB_CAVEAT,
    flagged_product_expr,
    reexport_hub_expr,
)

Key = Literal["market", "product", "pair"]
KEY_COLUMNS: dict[str, tuple[str, ...]] = {"market": ("iso3",), "product": ("hs6",), "pair": ("hs6", "iso3")}


@dataclass(frozen=True)
class CaveatContext:
    """What a tier's membership may depend on besides a row's key."""

    baseline_year: int


@dataclass(frozen=True)
class CaveatTier:
    """One Caveat Tier: its Baseline and API row column, its wording, its hide filter and its membership."""

    column: str  # the boolean column on the Baseline and the field on API rows, e.g. "reexport_hub"
    badge: str  # the glossary term shown on a row and in the Caveats column, e.g. "Re-export Hub"
    plural: str  # e.g. "Re-export Hubs": the hide filter reads "Hide <plural>", its notice "<plural> hidden"
    text: str  # what the caveat means for the Advisor, starting with the badge
    key: Key  # what membership depends on: the Market, the Product or the pair
    hide_param: str  # the hide filter's API query parameter and app URL parameter, e.g. "hide_hubs"
    hide_description: str  # the hide filter's API documentation
    flags: Callable[[CaveatContext], pl.Expr]  # True where a row (with `hs6` and `iso3`) is in the tier
    # Optional: each flagged row's reason, a string expression; the Baseline and API rows then carry `reason_column`
    reason: Callable[[CaveatContext], pl.Expr] | None = None
    row_text: str | None = None  # a row's caveat text, with "{reason}" for its reason; `text` when None

    @property
    def reason_column(self) -> str | None:
        """The column holding each flagged row's reason (null elsewhere), or None if the tier declares no reasons."""
        return None if self.reason is None else f"{self.column}_reason"

    @property
    def columns(self) -> list[str]:
        """The tier's columns on the Baseline: its flag, then its reason if it declares one."""
        return [self.column] if self.reason_column is None else [self.column, self.reason_column]

    def caveat_text(self, reason: str | None) -> str:
        """One row's caveat text: `row_text` with the row's reason, or `text` when there is no reason."""
        return self.text if reason is None or self.row_text is None else self.row_text.format(reason=reason)

    def expressions(self, context: CaveatContext) -> list[pl.Expr]:
        """The tier's columns as expressions over rows keyed by `hs6` and `iso3`; refused if one reads outside its key."""
        flags = self.flags(context)
        expressions = [flags.alias(self.column)]
        if self.reason is not None:
            expressions.append(
                pl.when(flags).then(self.reason(context)).otherwise(pl.lit(None, pl.String)).alias(f"{self.column}_reason")
            )
        for expression in expressions:
            outside = set(expression.meta.root_names()) - set(KEY_COLUMNS[self.key])
            if outside:
                raise ValueError(f"Caveat Tier {self.column} is keyed on the {self.key} but reads {sorted(outside)}")
        return expressions


FLAGGED_PRODUCT = CaveatTier(
    column="flagged_product",
    badge="Flagged Product",
    plural="Flagged Products",
    text=FLAGGED_PRODUCT_CAVEAT,
    key="product",
    hide_param="hide_flagged",
    hide_description="Leave out Flagged Products (volatile, entrepôt-driven values)",
    flags=lambda context: flagged_product_expr(),
)
REEXPORT_HUB = CaveatTier(
    column="reexport_hub",
    badge="Re-export Hub",
    plural="Re-export Hubs",
    text=REEXPORT_HUB_CAVEAT,
    key="market",
    hide_param="hide_hubs",
    hide_description="Leave out Re-export Hubs (Markets that re-export much of what they import)",
    flags=lambda context: reexport_hub_expr(),
)

TIERS: tuple[CaveatTier, ...] = (FLAGGED_PRODUCT, REEXPORT_HUB)


def with_caveat_columns[T: (pl.DataFrame, pl.LazyFrame)](
    rows: T, context: CaveatContext, *, tiers: Sequence[CaveatTier] = TIERS
) -> T:
    """`rows` (keyed by `hs6` and `iso3`) with each tier's columns, replacing any already there in place."""
    return rows.with_columns(expression for tier in tiers for expression in tier.expressions(context))
