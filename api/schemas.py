"""Response models: the API contract the app and the contract tests rely on."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from exportspredictor import scenarios
from exportspredictor.gap_period import OPPORTUNITY_TYPES
from exportspredictor.rankings import (
    BLOCKED,
    DEFAULT_MIN_POTENTIAL_USD,
    DEFAULT_RANK_BY,
    RANK_BY,
)
from exportspredictor.scenarios import DEFAULT_MIN_SHARE, DEFAULT_MIN_USD, PRESETS


class Health(BaseModel):
    ready: bool
    artefact_version: str | None = None
    artefact_format: int | None = None  # the artefact layout the API reads (2 for v2)
    baseline_year: int | None = None
    gap_years: list[int] | None = None  # first and last year the Export Gaps average over
    gap_period_label: str | None = None  # how a Gap Period mean is labelled, e.g. "2022–2024 average"
    reason: str | None = None  # why the artefacts are not loaded, while not ready


class ProductOut(BaseModel):
    hs6: str
    description: str
    sector: str


class ProductMatch(ProductOut):
    """A Product search result: Excluded Products are found and marked, not hidden (ADR 0005)."""

    excluded: bool
    exclusion_reason: str | None


class ExcludedProductOut(ProductOut):
    reason: str  # why it is left out of training, scoring and rankings


class Caveat(BaseModel):
    badge: str  # the glossary term shown on a row, e.g. "Flagged Product"
    text: str  # what it means for the Advisor


class Labels(BaseModel):
    """Wording and options the app shows, each defined once in the package."""

    usd_caption: str  # every value's unit, said once per page
    usd_unit: str  # the unit in column headings, e.g. "Export Gap (USD, nominal)"
    opportunity_types: list[str]  # Expand, New Market, New Product (GLOSSARY.md)
    caveats: dict[str, Caveat]  # by the boolean row field that sets each (ADR 0005, 0007)
    last_tariff_year: int  # tariff data end here; later years carry it forward, and Scenarios restore its MFN rate
    tariff_carried_forward: str  # the caveat of a carried-forward Applied Tariff (ADR 0006)


class MarketOut(BaseModel):
    iso3: str
    name: str
    region: str


RankBy = Literal[tuple(RANK_BY)]  # the ranking orders of `exportspredictor.rankings.RANK_BY`
OpportunityType = Literal[OPPORTUNITY_TYPES]  # Expand, New Market or New Product (GLOSSARY.md)


class RankOrderOut(BaseModel):
    name: str
    title: str
    description: str
    # For an order that ranks only pairs above a minimum Export Potential (Relative Export Gap):
    # the minimums offered, in USD, and the default. Absent for other orders.
    min_potential_usd_options: list[float] | None = None
    default_min_potential_usd: float | None = None


class RankingQuery(BaseModel):
    """Query parameters both ranking views accept; an unknown one (such as the old `min_gap_gbp`) is a 422."""

    model_config = ConfigDict(extra="forbid")

    rank_by: RankBy = Field(
        DEFAULT_RANK_BY, description="; ".join(f"{order.name}: {order.title}" for order in RANK_BY.values())
    )
    sector: str | None = Field(None, description="Sector (HS2 chapter), e.g. 22")
    region: str | None = Field(None, description="Market region, e.g. South Asia")
    opportunity_type: OpportunityType | None = Field(
        None, description="Opportunity Type: " + ", ".join(OPPORTUNITY_TYPES)
    )
    min_gap_usd: float | None = Field(None, description="Minimum Export Gap, in USD")
    trade_agreement: bool | None = Field(
        None, description="Only pairs with (true) or without (false) a UK Trade Agreement in force"
    )
    hide_flagged: bool = Field(False, description="Leave out Flagged Products (volatile, entrepôt-driven values)")
    hide_hubs: bool = Field(False, description="Leave out Re-export Hubs (Markets that re-export much of what they import)")
    hide_uncertain: bool = Field(
        False, description="Leave out Uncertain Opportunities (Actual Exports inside the Potential Range)"
    )
    min_potential_usd: float | None = Field(
        None,
        ge=0,
        description="rank_by=relative only: rank only pairs whose Export Potential is at least this, in USD "
        f"(default {DEFAULT_MIN_POTENTIAL_USD:,.0f})",
    )
    limit: int | None = Field(None, ge=1, description="Keep only the top rows")


class LeaderboardQuery(RankingQuery):
    """The ranking filters plus one Market, for the Leaderboard over every UK pair."""

    market: str | None = Field(None, description="One Market, by ISO3 code, e.g. JPN")


class BlockedQuery(LeaderboardQuery):
    """The Leaderboard's filters for the Blocked Opportunities page, which has one fixed order."""

    rank_by: Literal[BLOCKED.name] = Field(BLOCKED.name, description=f"Always {BLOCKED.name}: {BLOCKED.title}")


class _Values(BaseModel):
    """Export Potential, Actual Exports and Export Gap of one UK Product–Market pair: Gap Period means."""

    export_potential_usd: float
    actual_exports_usd: float
    export_gap_usd: float
    relative_export_gap: float | None  # Export Gap / Export Potential; missing where Export Potential is 0
    opportunity_type: OpportunityType  # how far the UK already trades the pair over the Gap Period
    trade_agreement: bool  # a UK Trade Agreement with the Market is in force in the Baseline year
    blocked_opportunity_value_usd: float  # Scenario Effect of adding a Trade Agreement where none exists (0 if one does)
    blocked_opportunity: bool  # that effect clears the default Blocked Opportunity thresholds
    flagged_product: bool  # a Flagged Product: shown with a caveat (ADR 0005)
    reexport_hub: bool  # the Market is a Re-export Hub: shown with a caveat (ADR 0005)
    # The Potential Range (ADR 0007): where Actual Exports fall 80% of the time for comparable pairs
    potential_range_low_usd: float
    potential_range_high_usd: float
    uncertain_opportunity: bool  # an Uncertain Opportunity: an Opportunity whose Actual Exports lie inside its range


class MarketOpportunity(_Values):
    iso3: str
    market: str
    region: str


class ProductOpportunity(_Values):
    hs6: str
    description: str
    sector: str


class LeaderboardOpportunity(_Values):
    hs6: str
    description: str
    sector: str
    iso3: str
    market: str
    region: str


class LeaderboardSummary(BaseModel):
    pairs: int  # every pair passing the filters (before `limit`)
    positive_export_gap_usd: float  # the sum of their positive Export Gaps


class _Ranking(BaseModel):
    artefact_version: str
    baseline_year: int
    gap_years: list[int]  # first and last year of the Gap Period the values average over
    gap_period_label: str  # e.g. "2022–2024 average", for headings
    rank_by: RankBy


class Leaderboard(_Ranking):
    summary: LeaderboardSummary
    opportunities: list[LeaderboardOpportunity]  # UK Product–Market pairs, sorted by `rank_by`, largest first


class BlockedSummary(LeaderboardSummary):
    blocked_opportunities: int  # the filtered pairs that clear the default Blocked Opportunity thresholds
    blocked_opportunity_value_usd: float  # the sum of their Blocked Opportunity values (Baseline-year Scenario Effects)


class BlockedOpportunities(Leaderboard):
    """The Blocked Opportunities page: every UK pair ranked by Blocked Opportunity value."""

    rank_by: Literal[BLOCKED.name]
    order: RankOrderOut  # the order's title and (associational) description, for the page caption
    summary: BlockedSummary


class ProductOpportunities(_Ranking):
    product: ProductOut
    opportunities: list[MarketOpportunity]  # Markets, sorted by `rank_by`, largest first


class MarketOpportunities(_Ranking):
    market: MarketOut
    opportunities: list[ProductOpportunity]  # Products, sorted by `rank_by`, largest first


class MetricInfo(BaseModel):
    label: str
    better: str  # "lower" or "higher"


class AccuracyRow(BaseModel):
    """Every metric for the model (`model_*`) and the naive baseline (`naive_*`) on one group of rows."""

    group: str | None = None  # Sector (HS2 chapter) or region; absent for the overall row
    rows: int
    non_zero_flows: int
    products: int
    ranked_products: int
    model_tweedie_deviance: float | None
    naive_tweedie_deviance: float | None
    model_rmse_log: float | None
    naive_rmse_log: float | None
    model_spearman_within_product: float | None
    naive_spearman_within_product: float | None


class ScopeAccuracy(BaseModel):
    overall: AccuracyRow
    by_sector: list[AccuracyRow]  # weakest first (lowest within-Product rank correlation)
    by_region: list[AccuracyRow]  # weakest first
    by_income: list[AccuracyRow] = []  # weakest first; World Bank income level of the Market (reporting quality)
    # "observed" then "imputed" (Applied Tariff carried forward from 2021, ADR 0006); an empty side has rows 0
    by_tariff: list[AccuracyRow] = []


class Headline(BaseModel):
    scope: str
    metric: str
    model: float | None
    naive: float | None
    beats_naive: bool
    text: str


class Stability(BaseModel):
    years: list[int]
    pairs: int
    spearman_potential: float | None
    spearman_gap: float | None
    top_n: int
    top_overlap: float
    text: str


class SizeMatchedLift(BaseModel):
    expected_rate: float | None
    lift: float | None
    ci: list[float | None]


class LiftDecile(BaseModel):
    decile: int  # 1 = the largest Export Gaps
    rate: float | None  # weighted share of pairs that grew materially
    pairs: float


class Lift(BaseModel):
    """How much more often the top tenth of pairs by Gap Period Export Gap grew materially than random pairs (#21, #57)."""

    lift: float | None
    ci: list[float | None]
    ci_level: float
    bootstrap_unit: str
    top_rate: float | None
    base_rate: float | None
    size_matched: SizeMatchedLift
    by_decile: list[LiftDecile]
    held_out_exporters: list[str]  # v1's frozen held-out list, within v2's Exporter scope
    held_out_from: str
    training_years: list[int]
    gap_years: list[int]  # the Gap Period the Export Gaps are averaged over
    outcome_year: int
    min_growth_share: float
    min_growth_usd: float
    text: str


class EventEffect(BaseModel):
    observed: float | None  # difference-in-differences, as a share
    observed_ci: list[float | None]
    predicted: float | None  # the model's Scenario Effect, as a share
    predicted_ci: list[float | None]


class EventYear(EventEffect):
    year: int


class EventDirection(BaseModel):
    label: str
    exporters: list[str]
    importers: list[str]
    controls: list[str]  # the control Markets
    by_year: list[EventYear]
    pooled: EventEffect


class EventResult(BaseModel):
    role: str  # "primary" or "secondary"
    name: str
    title: str
    in_force: str
    event_year: int
    training_years: list[int]
    pre_years: list[int]
    post_years: list[int]
    control_rule: str
    caveat: str
    states: str = ""  # what "agreement on" and "off" set, the Applied Tariff included
    directions: list[EventDirection]
    text: str


class EventBacktest(BaseModel):
    events: list[EventResult]
    text: str  # the primary event's summary
    caveat: str


class Limitation(BaseModel):
    title: str
    text: str


class Evaluation(BaseModel):
    """The evaluation artefact. Later checks (Lift, the Event Backtest) add their own fields."""

    model_config = ConfigDict(extra="allow")

    artefact_version: str
    baseline_year: int
    test_year: int
    test_training_years: list[int]
    validation_year: int
    tuning_years: list[int]
    non_uk_sampling_rate: float
    zero_sampling_rate: float
    tweedie_power: float
    metrics: dict[str, MetricInfo]
    naive_baseline: str
    accuracy: dict[str, ScopeAccuracy]  # "all_exporters" and "uk_out_of_fold"
    headline: Headline
    tariff_split_text: str = ""  # how the test rows split between observed and imputed Applied Tariffs
    stability: Stability
    lift: Lift | None = None
    event_backtest: EventBacktest | None = None
    limitations: list[Limitation]


class PresetOut(BaseModel):
    name: str
    title: str
    description: str


PresetName = Literal[tuple(PRESETS)]  # the presets of `exportspredictor.scenarios.PRESETS`


class TariffIn(BaseModel):
    """The Applied Tariff chosen Markets charge on UK goods, optionally in chosen Sectors only."""

    model_config = ConfigDict(extra="forbid")

    markets: list[str] = Field(min_length=1, description="Markets (ISO3) that charge the tariff")
    rate_pct: float = Field(ge=0, allow_inf_nan=False, description="The tariff in percent, e.g. 10 = 10%")
    sectors: list[str] | None = Field(None, description="Sectors (HS2 chapters, e.g. 22); all Sectors if omitted")


class ScenarioIn(BaseModel):
    """A Trading Bloc preset, Markets (ISO3) to add or remove a Trade Agreement with, *or* a tariff change."""

    preset: PresetName | None = None
    add_agreements: list[str] = []
    remove_agreements: list[str] = []
    tariff: TariffIn | None = None

    def to_scenario(self) -> scenarios.Scenario:
        tariff = self.tariff
        return scenarios.Scenario(
            preset=self.preset,
            add_agreements=tuple(self.add_agreements),
            remove_agreements=tuple(self.remove_agreements),
            tariff=None
            if tariff is None
            else scenarios.TariffChange(
                markets=tuple(tariff.markets),
                rate_pct=tariff.rate_pct,
                sectors=None if tariff.sectors is None else tuple(tariff.sectors),
            ),
        )


class Thresholds(BaseModel):
    """The Blocked Opportunity thresholds; the defaults are the package's."""

    min_effect_share: float = Field(DEFAULT_MIN_SHARE, ge=0, description="Blocked Opportunity: effect ≥ this share of Baseline")
    min_effect_usd: float = Field(DEFAULT_MIN_USD, ge=0, description="Blocked Opportunity: effect ≥ this many USD")


class ScenarioRequest(ScenarioIn, Thresholds):
    """The Scenario, the Blocked Opportunity thresholds, and how many rows to return."""

    model_config = ConfigDict(extra="forbid")

    limit: int = Field(50, ge=1, le=1000, description="Rows returned in each list")
    market: str | None = Field(None, description="Drill-down: top pairs of this Market (ISO3) only")


class EffectTotals(BaseModel):
    baseline_potential_usd: float
    scenario_potential_usd: float
    effect_usd: float
    effect_share: float


class MarketEffect(EffectTotals):
    iso3: str
    market: str
    pairs: int


class ProductEffect(EffectTotals):
    hs6: str
    description: str
    pairs: int


class PairEffect(BaseModel):
    hs6: str
    description: str
    iso3: str
    market: str
    baseline_potential_usd: float
    scenario_potential_usd: float
    effect_usd: float
    effect_share: float
    baseline_tariff_pct: float | None  # the Applied Tariff (percent) in the Baseline; None where unknown
    scenario_tariff_pct: float | None  # and under the Scenario (clipped to the training range)
    blocked_opportunity: bool


class ScenarioResponse(BaseModel):
    artefact_version: str
    baseline_year: int
    gap_years: list[int]
    gap_period_label: str
    scenario: ScenarioIn
    thresholds: Thresholds
    market: str | None  # the drill-down Market `top_pairs` is limited to, if any
    changes_nothing: bool
    summary: str  # one associational sentence
    total: EffectTotals  # over the pairs the Scenario changes
    markets: list[MarketEffect]  # every changed Market, largest effect first
    products: list[ProductEffect]  # top Products by effect
    top_pairs: list[PairEffect]  # top Product–Market pairs by effect (of `market` only, if given)
    blocked_opportunities: list[PairEffect]  # pairs clearing both thresholds, largest first
    blocked_count: int  # all Blocked Opportunities, before `limit`
    # Pairs whose Scenario tariff lay beyond the training range and was clipped to it, so
    # their Scenario Effect flattens there (ADR 0006).
    clipped_pairs: int


class FactorOut(BaseModel):
    label: str  # plain-English factor name; raw model input names are never returned
    value_text: str  # the factor's value for this pair, readable
    contribution: float  # SHAP value on the log scale; > 0 raised Export Potential
    multiplier: float  # exp(contribution): the factor multiplies Export Potential by this
    direction: str  # "raised", "lowered" or "no effect"


class HistoryOut(BaseModel):
    year: int
    actual_exports_usd: float
    export_potential_usd: float | None  # Gap Period years only
    export_gap_usd: float | None  # Gap Period years only; their mean is the Export Gap
    tariff_imputed: bool  # the year's Applied Tariff is carried forward from 2021 (ADR 0006)


class Explanation(BaseModel):
    artefact_version: str
    baseline_year: int
    gap_years: list[int]
    gap_period_label: str  # e.g. "2022–2024 average"
    product: ProductOut
    market: MarketOut
    fold: int  # the fold model that held the pair out, scored its Baseline and explains it
    base_value: float  # SHAP base value (log scale)
    log_export_potential: float  # base_value + sum of contributions
    export_potential_usd: float  # Gap Period means, as in the rankings
    actual_exports_usd: float
    export_gap_usd: float
    baseline_export_potential_usd: float  # the Baseline year's, which the factors explain
    relative_export_gap: float | None  # Export Gap / Export Potential (Gap Period means)
    opportunity_type: OpportunityType  # Expand, New Market or New Product, over the Gap Period
    flagged_product: bool  # a Flagged Product: shown with a caveat (ADR 0005)
    reexport_hub: bool  # the Market is a Re-export Hub: shown with a caveat (ADR 0005)
    potential_range_low_usd: float  # the Potential Range around the Gap Period mean (ADR 0007)
    potential_range_high_usd: float
    uncertain_opportunity: bool  # an Uncertain Opportunity
    potential_range_text: str  # the Potential Range in plain words, for the pair's detail view
    tariff_caveat: str | None  # set when the Baseline year's Applied Tariff is carried forward (ADR 0006)
    factors: list[FactorOut]  # largest absolute contribution first
    # By year: every Gap Period year (with Export Potential and Export Gap), and earlier
    # years with UK Actual Exports only
    history: list[HistoryOut]
