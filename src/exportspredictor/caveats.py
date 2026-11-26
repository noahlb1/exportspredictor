"""Caveat Tiers (ADR 0005): groups of UK pairs that are scored and ranked as normal but shown with a caveat.

A tier is one `CaveatTier` declaration in `TIERS`. Everything else reads the declarations:

- the pipeline marks the Baseline with each tier's boolean column, and its reason column if it declares reasons
  (`with_caveat_columns`, which also marks a saved Baseline again without retraining);
- the rankings drop a tier's rows when asked (`rankings.rank_opportunities(hide=...)`);
- the API serves the tiers at `/caveats`, adds each tier's column (and reason) to every ranking row and explanation,
  with the row's caveat texts worded here (`caveat_texts`), and takes each tier's hide filter as a query parameter
  (`api.schemas.tiered`);
- the app draws each tier's badge, Caveats column entry, hide filter and hidden-rows notice from `/caveats`, and a
  row's caveats from its `caveat_texts`.

A tier is keyed on the Market (`iso3`), the Product (`hs6`) or the pair (both): its `flags` expression, and its
`reason` expression if any, may read only its key's columns. `flags` and `reason` take a `CaveatContext`, so a tier
can depend on the Baseline year (a sanctions regime in force that year) or on other inputs the pipeline holds, such
as the volatility rule's Flagged Products or the Offset Pairs of the Baseline's own Export Gaps
(`pipeline.caveat_context` builds the whole context from a bundle).

To add a tier: declare it (its membership usually lives in a module of its own, as Flagged Products and Re-export
Hubs live in `exclusions` and `volatility`, Sanctioned Markets and Restricted Pairs in `restrictions`, and Offset Pairs
in `offsets`), append it to `TIERS`, add
its term to GLOSSARY.md, and write an artefact version whose Baseline carries its columns
(`python -m exportspredictor.pipeline recaveat`, which applies `with_caveat_columns` to the saved Baseline). No API
or app code changes.

Excluded Products are not a tier: they are left out of scoring and rankings altogether (`exclusions`).
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import polars as pl

from exportspredictor.exclusions import (
    FLAGGED_PRODUCT_CAVEAT,
    FLAGGED_PRODUCT_ROW_CAVEAT,
    REEXPORT_HUB_CAVEAT,
    flagged_product_expr,
    flagged_product_reason_expr,
    reexport_hub_expr,
)
from exportspredictor.restrictions import (
    PROHIBITIONS,
    SANCTIONS_REGIMES,
    Prohibition,
    SanctionsRegime,
    restricted_pair_expr,
    restriction_reason_expr,
    sanction_reason_expr,
    sanctioned_market_expr,
)

Key = Literal["market", "product", "pair"]
KEY_COLUMNS: dict[str, tuple[str, ...]] = {"market": ("iso3",), "product": ("hs6",), "pair": ("hs6", "iso3")}


@dataclass(frozen=True)
class CaveatContext:
    """What a tier's membership may depend on besides a row's key."""

    baseline_year: int
    # Products flagged by the volatility rule (`volatility.VolatileProducts.reasons`, hs6 -> reason), read from the
    # UK's exports in the training years; Sector 71 is flagged whatever this holds.
    flagged_product_reasons: Mapping[str, str] = field(default_factory=dict)
    # Offset Pairs (`offsets.offset_pairs`, (hs6, iso3) -> reason), read from the Baseline's own Export Gaps.
    offset_pair_reasons: Mapping[tuple[str, str], str] = field(default_factory=dict)


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
    # Whether the app's hide filter starts on: for pairs the UK cannot or should not pursue as ranked. The API's
    # query parameter still defaults to off, so a read shows every pair unless it asks otherwise.
    hidden_by_default: bool = False
    # The key terms `text` names that have their own explanation, each (name, `wording.term_definitions` key): the
    # hide filter's `?` explains them after the caveat.
    terms: tuple[tuple[str, str], ...] = ()

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
    hide_description="Leave out Flagged Products (volatile, one-off or re-export-led values)",
    flags=lambda context: flagged_product_expr() | pl.col("hs6").cast(pl.String).is_in(
        list(context.flagged_product_reasons)
    ),
    reason=lambda context: flagged_product_reason_expr(context.flagged_product_reasons),
    row_text=FLAGGED_PRODUCT_ROW_CAVEAT,
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



def sanctioned_market_tier(regimes: tuple[SanctionsRegime, ...] = SANCTIONS_REGIMES) -> CaveatTier:
    """Sanctioned Markets: Markets under broad UK trade sanctions in the Baseline year (`restrictions`); each row's
    reason names the regime and its start year. Tests pass their own `regimes`."""
    return CaveatTier(
        column="sanctioned_market",
        badge="Sanctioned Market",
        plural="Sanctioned Markets",
        text=(
            "Sanctioned Market: this Market is under broad UK trade sanctions in the Baseline year. It is scored and "
            "ranked as normal, but much UK trade with it may be prohibited: check the sanctions guidance first."
        ),
        key="market",
        hide_param="hide_sanctioned",
        hide_description="Leave out Sanctioned Markets (under broad UK trade sanctions in the Baseline year)",
        flags=lambda context: sanctioned_market_expr(context.baseline_year, regimes),
        reason=lambda context: sanction_reason_expr(context.baseline_year, regimes),
        row_text=(
            "Sanctioned Market: this Market is under broad UK trade sanctions ({reason}). It is scored and ranked as "
            "normal, but much UK trade with it may be prohibited: check the sanctions guidance first."
        ),
        hidden_by_default=True,
    )


def restricted_pair_tier(prohibitions: tuple[Prohibition, ...] = PROHIBITIONS) -> CaveatTier:
    """Restricted Pairs: pairs whose trade the Market prohibits outright (`restrictions`); each row's reason says
    what the Market prohibits. Tests pass their own `prohibitions`."""
    return CaveatTier(
        column="restricted_pair",
        badge="Restricted Pair",
        plural="Restricted Pairs",
        text=(
            "Restricted Pair: the Market prohibits imports of this Product outright. It is scored and ranked as "
            "normal, but the UK cannot sell it there while the prohibition stands."
        ),
        key="pair",
        hide_param="hide_restricted",
        hide_description="Leave out Restricted Pairs (Products the Market prohibits outright, e.g. alcohol)",
        flags=lambda context: restricted_pair_expr(prohibitions),
        reason=lambda context: restriction_reason_expr(prohibitions),
        row_text=(
            "Restricted Pair: {reason}. It is scored and ranked as normal, but the UK cannot sell it there while the "
            "prohibition stands."
        ),
        hidden_by_default=True,
    )


def _pair() -> pl.Expr:
    """Each row's pair as a struct (hs6, iso3), for a set test or lookup against (hs6, iso3) keys (`_pairs`)."""
    return pl.struct(pl.col("hs6").cast(pl.String), pl.col("iso3").cast(pl.String))


def _pairs(pairs: Mapping[tuple[str, str], str]) -> pl.Series:
    """The (hs6, iso3) keys of `pairs` as a Series of structs matching `_pair`."""
    return pl.DataFrame(list(pairs), schema={"hs6": pl.String, "iso3": pl.String}, orient="row").to_struct("pair")


_NEIGHBOURING_PRODUCT = "Its Export Gap may be trade recorded under a neighbouring Product."

OFFSET_PAIR = CaveatTier(
    column="offset_pair",
    badge="Offset Pair",
    plural="Offset Pairs",
    text=(
        "Offset Pair: the UK already sells above Export Potential in the other Products of this Product's Heading in "
        f"this Market, by at least half of its Export Gap. {_NEIGHBOURING_PRODUCT}"
    ),
    key="pair",
    hide_param="hide_offset",
    hide_description=(
        "Leave out Offset Pairs (Opportunities at least half offset by the UK's sales above Export Potential in the "
        "other Products of their Heading in that Market)"
    ),
    # A set test and a lookup on the pair: about 250,000 Offset Pairs on the real Baseline.
    flags=lambda context: _pair().is_in(_pairs(context.offset_pair_reasons).implode()),
    reason=lambda context: _pair().replace_strict(
        _pairs(context.offset_pair_reasons),
        pl.Series(list(context.offset_pair_reasons.values()), dtype=pl.String),
        default=None,
        return_dtype=pl.String,
    ),
    row_text="Offset Pair: {reason}. " + _NEIGHBOURING_PRODUCT,
    hidden_by_default=True,
    terms=(("Heading", "heading"),),
)

SANCTIONED_MARKET = sanctioned_market_tier()
RESTRICTED_PAIR = restricted_pair_tier()

TIERS: tuple[CaveatTier, ...] = (FLAGGED_PRODUCT, REEXPORT_HUB, SANCTIONED_MARKET, RESTRICTED_PAIR, OFFSET_PAIR)


def caveat_texts(row: Mapping, tiers: Sequence[CaveatTier] = TIERS) -> list[str]:
    """The caveat text of each tier one row is in, in tier order (`CaveatTier.caveat_text`, with the row's reason).

    `row` holds each tier's column and, for a tier with reasons, its reason column, as API rows do.
    """
    return [
        tier.caveat_text(row.get(tier.reason_column) if tier.reason_column else None)
        for tier in tiers
        if row.get(tier.column)
    ]


def with_caveat_columns[T: (pl.DataFrame, pl.LazyFrame)](
    rows: T, context: CaveatContext, *, tiers: Sequence[CaveatTier] = TIERS
) -> T:
    """`rows` (keyed by `hs6` and `iso3`) with each tier's columns, replacing any already there in place."""
    return rows.with_columns(expression for tier in tiers for expression in tier.expressions(context))
