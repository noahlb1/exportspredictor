"""Response models: the API contract the app and the contract tests rely on.

Models that carry Caveat Tiers (the ranking queries, ranking rows and explanations) are declared here without them;
`tiered(tiers)` builds the ones the API serves, with each declared tier's hide filter, column and reason column
(`exportspredictor.caveats`).
"""

from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from exportspredictor import scenarios
from exportspredictor.caveats import TIERS, CaveatTier, caveat_texts
from exportspredictor.gap_period import OPPORTUNITY_TYPES
from exportspredictor.rankings import (
    BLOCKED,
    BLOCKED_ORDERS,
    DEFAULT_MIN_POTENTIAL_USD,
    DEFAULT_RANK_BY,
    MAX_ROWS,
    RANK_BY,
)
from exportspredictor.scenarios import BLOCKED_MIN_SHARE, BLOCKED_MIN_USD, PRESETS


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
    sector_name: str  # the Sector's HS 2012 chapter title, shortened (`exportspredictor.sectors`)


class SectorOut(BaseModel):
    code: str  # the HS2 chapter, e.g. 22: the value of the ranking reads' `sector` filter
    name: str  # its HS 2012 chapter title, shortened, e.g. "Beverages, spirits and vinegar"


class ProductMatch(ProductOut):
    """A Product search result: Excluded Products are found and marked, not hidden (ADR 0005)."""

    excluded: bool
    exclusion_reason: str | None


class ExcludedProductOut(ProductOut):
    reason: str  # why it is left out of training, scoring and rankings


class ExcludedReasonOut(BaseModel):
    reason: str  # why Products are left out (`exportspredictor.exclusions.EXCLUSIONS`)
    products: int  # how many Excluded Products it leaves out


class Caveat(BaseModel):
    badge: str  # the glossary term shown on a row, e.g. "Flagged Product"
    text: str  # what it means for the Advisor


class CaveatTierOut(BaseModel):
    """One Caveat Tier (`exportspredictor.caveats`): what the app needs to show its badge, caveat and hide filter."""

    column: str  # the boolean field on ranking rows and explanations that sets it, e.g. "reexport_hub"
    badge: str  # e.g. "Re-export Hub"
    plural: str  # e.g. "Re-export Hubs", for "Hide Re-export Hubs" and "Re-export Hubs hidden"
    text: str  # what the caveat means for the Advisor
    key: str  # "market", "product" or "pair": what membership depends on
    hide_param: str  # the ranking reads' query parameter that leaves the tier's rows out, e.g. "hide_hubs"
    reason_column: str | None  # the row field holding each flagged row's reason, if the tier declares reasons
    row_text: str | None  # a flagged row's caveat text, with "{reason}" for its reason; `text` when absent
    hidden_by_default: bool  # whether the app's hide filter starts on (the query parameter itself defaults to off)


class TotalsNote(BaseModel):
    """Export Potential is a median-type estimate: summed, it falls short of Actual Exports."""

    # Summed Export Potential over summed Actual Exports, every UK pair, Gap Period means; from the artefact
    potential_share_of_actual_exports: float | None
    text: str  # the note, naming the share


class Labels(BaseModel):
    """Wording and options the app shows, each defined once in the package."""

    usd_caption: str  # every value's unit, said once per page
    usd_unit: str  # the unit in column headings, e.g. "Export Gap (USD, nominal)"
    opportunity_types: list[str]  # Expand, New Market, New Product (GLOSSARY.md)
    last_tariff_year: int  # tariff data end here; later years carry it forward, and Scenarios restore its MFN rate
    tariff_carried_forward: str  # the caveat of a carried-forward Applied Tariff (ADR 0006)
    # The label of every Scenario Effect and Blocked Opportunity value, and what it means (ADR 0008)
    illustrative: Caveat
    caution: str  # the grey line under every page heading: figures are associations, not causes or forecasts
    # What Export Potential, Actual Exports, Export Gap and Opportunity Type mean, by row field, for tooltips;
    # those that name the Gap Period take its label from the artefact (`wording.term_definitions`)
    terms: dict[str, str]
    max_rows: int  # the most rows a ranking read returns (a larger `limit` is a 422)
    totals_note: TotalsNote  # beside every summed figure (spec #113, "Totals understate")
    trade_agreement_meaning: str  # what "a Trade Agreement" means in the model (spec #133, ADR 0011)
    blocked_rule: str  # the Blocked Opportunity rule at the fixed thresholds, in one sentence (ADR 0012)


class MarketOut(BaseModel):
    iso3: str
    name: str
    region: str


class MarketListed(MarketOut):
    """A Market as `GET /markets` lists it, flagged when a Scenario accepts it."""

    scenario: bool  # true when the Market has UK feature rows in the Baseline year (never the United Kingdom)


class MarketReachOut(BaseModel):
    """A Market's Market Reach (ADR 0010): how widely the UK sends its Products there against how widely expected."""

    year: int  # the reference year whose UK Flows it is learned from (the Baseline year − 1)
    products_sent: int  # of products_imported, those with a UK Flow too (so at most products_imported)
    products_imported: int  # the Market's Products imported from other Exporters that year
    products_expected: float  # products_imported × the UK's share across every Market
    # Log of the share of the Market's imported Products the UK sends, against that share across Markets, shrunk
    # (mean over the fold tables).
    offset: float
    text: str  # the one sentence the app shows


class PairMarketReachOut(MarketReachOut):
    pair_offset: float  # the offset of the pair's own fold table, the one its Export Potential takes
    pair_multiplier: float  # exp(β · pair_offset), before the rescale constant that keeps the UK's total


# Each choice set is built from the package's own constant, so it is defined once. Pydantic validates
# and documents the Literal; pyright cannot evaluate a Literal built at runtime, so it reads a str.
if TYPE_CHECKING:
    RankBy = OpportunityType = PresetName = BlockedRankBy = str
else:
    RankBy = Literal[tuple(RANK_BY)]  # the ranking orders of `exportspredictor.rankings.RANK_BY`
    OpportunityType = Literal[OPPORTUNITY_TYPES]  # Expand, New Market or New Product (GLOSSARY.md)
    PresetName = Literal[tuple(PRESETS)]  # the presets of `exportspredictor.scenarios.PRESETS`
    BlockedRankBy = Literal[tuple(BLOCKED_ORDERS)]  # the Blocked Opportunities orders (`rankings.BLOCKED_ORDERS`)


class RankOrderOut(BaseModel):
    name: str
    title: str
    description: str
    # For an order that ranks only pairs above a minimum Export Potential (Relative Export Gap):
    # the minimums offered, in USD, and the default. Absent for other orders.
    min_potential_usd_options: list[float] | None = None
    default_min_potential_usd: float | None = None


class _FilterQuery(BaseModel):
    """The filters every ranking read accepts; an unknown parameter (such as the old `min_gap_gbp`) is a 422."""

    model_config = ConfigDict(extra="forbid")

    sector: str | None = Field(None, description="Sector (HS2 chapter), e.g. 22")
    region: str | None = Field(None, description="Market region, e.g. South Asia")
    # `tiered` adds each Caveat Tier's hide filter here, e.g. hide_hubs
    limit: int | None = Field(None, ge=1, le=MAX_ROWS, description=f"Keep only the top rows (at most {MAX_ROWS:,})")


class _OneProduct(BaseModel):
    """One Product, for a read over every UK pair (the Leaderboard, Blocked Opportunities)."""

    product: str | None = Field(
        None, description="One Product, by HS6 code (2012 edition), e.g. 220830; 404 if unknown or Excluded"
    )


class RankingQuery(_FilterQuery):
    """Query parameters both ranking views accept: the shared filters, the order and the Export Gap filters."""

    rank_by: RankBy = Field(
        DEFAULT_RANK_BY, description="; ".join(f"{order.name}: {order.title}" for order in RANK_BY.values())
    )
    opportunity_type: OpportunityType | None = Field(
        None, description="Opportunity Type: " + ", ".join(OPPORTUNITY_TYPES)
    )
    min_gap_usd: float | None = Field(None, description="Minimum Export Gap, in USD")
    trade_agreement: bool | None = Field(
        None, description="Only pairs with (true) or without (false) a UK Trade Agreement in force"
    )
    min_potential_usd: float | None = Field(
        None,
        ge=0,
        description="rank_by=relative only: rank only pairs whose Export Potential is at least this, in USD "
        f"(default {DEFAULT_MIN_POTENTIAL_USD:,.0f})",
    )


class LeaderboardQuery(RankingQuery, _OneProduct):
    """The ranking filters plus one Market and one Product, for the Leaderboard over every UK pair."""

    market: str | None = Field(None, description="One Market, by ISO3 code, e.g. JPN")


class BlockedQuery(_FilterQuery, _OneProduct):
    """The Blocked Opportunities page's filters and order: the shared filters, one Product, one Market, a minimum
    Blocked Opportunity value, and its own orders (by value or by Tariff part). An unknown parameter, such as
    `opportunity_type`, or a ranking order such as `gap`, is a 422."""

    market: str | None = Field(None, description="One Market, by ISO3 code, e.g. IND; 404 if unknown")
    min_value_usd: float | None = Field(None, ge=0, description="Minimum Blocked Opportunity value, in USD")
    rank_by: BlockedRankBy = Field(
        BLOCKED.name, description="; ".join(f"{order.name}: {order.title}" for order in BLOCKED_ORDERS.values())
    )


class _Values(BaseModel):
    """Export Potential, Actual Exports and Export Gap of one UK Product–Market pair: Gap Period means."""

    export_potential_usd: float
    actual_exports_usd: float
    export_gap_usd: float
    relative_export_gap: float | None  # Export Gap / Export Potential; missing where Export Potential is 0
    opportunity_type: OpportunityType  # how far the UK already trades the pair over the Gap Period
    trade_agreement: bool  # a UK Trade Agreement with the Market is in force in the Baseline year
    blocked_opportunity_value_usd: float  # Scenario Effect of adding a Trade Agreement where none exists (0 if one does)
    # that effect clears the fixed Blocked Opportunity thresholds and the pair has room to grow (`is_blocked`)
    blocked_opportunity: bool
    # `tiered` adds each Caveat Tier's column here (and its reason column, if any), e.g. reexport_hub (ADR 0005)
    # The Potential Range (ADR 0007): where Actual Exports fall 80% of the time for comparable pairs
    potential_range_low_usd: float
    potential_range_high_usd: float
    uncertain_opportunity: bool  # an Uncertain Opportunity: an Opportunity whose Actual Exports lie inside its range
    applied_tariff_pct: float | None  # the Baseline Applied Tariff in percent (Baseline policy features); None where unknown


class MarketOpportunity(_Values):
    iso3: str
    market: str
    region: str


class ProductOpportunity(_Values):
    hs6: str
    description: str
    sector: str
    sector_name: str


class LeaderboardOpportunity(_Values):
    hs6: str
    description: str
    sector: str
    sector_name: str
    iso3: str
    market: str
    region: str


class LeaderboardSummary(BaseModel):
    pairs: int  # every pair passing the filters (before `limit`)
    positive_export_gap_usd: float  # the sum of their positive Export Gaps


class MarketTotal(BaseModel):
    """One Market's filtered pairs on the Leaderboard: their count and total positive Export Gap."""

    iso3: str
    market: str
    pairs: int
    positive_export_gap_usd: float


class ProductTotal(BaseModel):
    """One Product's filtered pairs on the Leaderboard: their count and total positive Export Gap."""

    hs6: str
    description: str
    pairs: int
    positive_export_gap_usd: float


class _Ranking(BaseModel):
    artefact_version: str
    baseline_year: int
    gap_years: list[int]  # first and last year of the Gap Period the values average over
    gap_period_label: str  # e.g. "2022–2024 average", for headings
    rank_by: RankBy


class Leaderboard(_Ranking):
    summary: LeaderboardSummary
    market_reach: MarketReachOut | None = None  # the filtered Market's reach; None without a Market filter
    # Per-Market and per-Product totals over every filtered pair (before `limit`), largest first, ties by name; they
    # add up to the summary
    markets: list[MarketTotal]
    products: list[ProductTotal]
    opportunities: list[LeaderboardOpportunity]  # UK Product–Market pairs, sorted by `rank_by`, largest first


class BlockedSummary(BaseModel):
    blocked_opportunities: int  # every filtered Blocked Opportunity (before `limit`)
    blocked_opportunity_value_usd: float  # the sum of their Blocked Opportunity values (Baseline-year Scenario Effects)


class BlockedMarket(BaseModel):
    """One Market's filtered Blocked Opportunities: their count and total Blocked Opportunity value."""

    iso3: str
    market: str
    blocked_opportunities: int
    blocked_opportunity_value_usd: float


class BlockedProduct(BaseModel):
    """One Product's filtered Blocked Opportunities: their count and total Blocked Opportunity value."""

    hs6: str
    description: str
    blocked_opportunities: int
    blocked_opportunity_value_usd: float


class BlockedOpportunity(LeaderboardOpportunity):
    """A Blocked Opportunity: the ranking row plus its Baseline-year figures."""

    baseline_export_potential_usd: float  # Baseline Export Potential (the Baseline year's)
    baseline_actual_exports_usd: float  # UK Actual Exports in the Baseline year (0 where there is no Flow)
    # Export Potential with a Trade Agreement: Baseline Export Potential plus the Blocked Opportunity value (Illustrative);
    # always above Baseline-year Actual Exports (room to grow, ADR 0011)
    potential_with_agreement_usd: float
    rise_share: float | None  # the Rise share: the value as a share of Baseline Export Potential (None where that is 0)
    tariff_part_usd: float  # the Tariff part: the effect of setting a known Applied Tariff to 0, no agreement added
    agreement_part_usd: float  # the Agreement part: the value minus its Tariff part
    barrier: str  # e.g. "No agreement, tariff 7.5%", "No agreement, tariff 0%", "No agreement, tariff unknown"
    value_split: str  # one sentence on how the model splits the value between the Tariff part and the Agreement part
    above_export_potential: bool  # Baseline-year Actual Exports above Baseline Export Potential


class BlockedOpportunities(_Ranking):
    """The Blocked Opportunities page: only Blocked Opportunities (`is_blocked` at the fixed thresholds), by Blocked
    Opportunity value or by Tariff part."""

    rank_by: BlockedRankBy
    order: RankOrderOut  # the order's title and (associational) description, for the page caption
    summary: BlockedSummary
    markets: list[BlockedMarket]  # per-Market totals over every filtered Blocked Opportunity, largest total first
    products: list[BlockedProduct]  # per-Product totals, the same way
    opportunities: list[BlockedOpportunity]
    thresholds: "Thresholds"  # the fixed thresholds every Blocked Opportunity here clears, defined below
    # Blocked Opportunity values are Illustrative: the same note as the Scenario response, defined below
    note: "ScenarioNote"


class ProductOpportunities(_Ranking):
    product: ProductOut
    opportunities: list[MarketOpportunity]  # Markets, sorted by `rank_by`, largest first


class MarketOpportunities(_Ranking):
    market: MarketOut
    market_reach: MarketReachOut | None  # None only for a Market with no UK grid pairs in the reference year
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


class LargeMarketsRow(BaseModel):
    markets: int  # the count of largest importing Markets measured (30, 60)
    model: float | None
    naive: float | None


class LargeMarkets(BaseModel):
    """The headline's measure on the largest importing Markets (`large_markets.large_markets_report`)."""

    test_year: int
    rows: list[LargeMarketsRow]
    text: str


class LargeMarketsByVersion(BaseModel):
    """One version's tie-proof UK ranking on the largest importing Markets (`evaluation.large_markets_beside_earlier`):
    this version, the earlier versions re-scored from their saved scores, and this run's Naive Baseline."""

    name: str
    markets: dict[int, float | None]  # count of largest Markets (30, 60) -> figure; None where not measured
    columns: dict[str, float | None]  # the same figures by column label ("30 largest Markets"), as tables show them


class WithAndWithout(BaseModel):
    with_: float | None = Field(alias="with")
    without: float | None

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)


class LargeMarketsWithAndWithout(WithAndWithout):
    markets: int


class LiftWithAndWithout(WithAndWithout):
    size_matched: WithAndWithout


class MarketReachEffect(BaseModel):
    """What Market Reach, the change this version keeps, does to the test-year figures (`evaluation.market_reach_effect`)."""

    selection_measure: WithAndWithout
    old_measure: WithAndWithout
    large_markets: list[LargeMarketsWithAndWithout]
    lift: LiftWithAndWithout | None = None
    uk_pair_coverage: WithAndWithout | None = None
    scenario_effects: str
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


class NotRun(BaseModel):
    """An opt-in part of the pipeline that this artefact version skipped (`evaluation.parts_not_run`)."""

    title: str
    text: str


class Limitation(BaseModel):
    title: str
    text: str


class ModelFormOut(BaseModel):
    form: str  # "blend" or "model alone"
    naive_weight: float  # the Naive Baseline's weight on the log scale
    text: str


class ExperimentOut(BaseModel):
    name: str  # as the loop recorded it
    label: str | None = None  # as shown: the research note's experiment number and the Glossary's name
    kept: str | None = None  # the kept Arm, None if every Arm was dropped
    best_arm: str | None = None
    selection_measure: float | None = None  # the best Arm's
    change: float | None = None  # the best Arm's change on the best so far


class ConfirmationFold(BaseModel):
    fold: int
    winner: float
    v2: float


class Confirmation(BaseModel):
    """The five-fold confirmation of the chosen form against v2's (the model alone)."""

    n_folds: int
    winner: float
    v2: float
    naive: float
    difference: float
    win_threshold: float
    beats_v2: bool
    folds: list[ConfirmationFold]


class VersionResult(BaseModel):
    """One version's test-year figures: this version or an earlier one (the Blend, v2, v1)."""

    model_config = ConfigDict(extra="allow")  # v2's artefact version

    name: str
    uk_ranking: float | None = None  # on the old measure (ties share their average rank), as every version reported
    uk_ranking_tie_proof: float | None = None  # the Selection Measure; an earlier version's from its saved scores
    lift: float | None = None
    size_matched_lift: float | None = None
    tweedie_deviance_all_exporters: float | None = None
    tweedie_deviance_uk: float | None = None


class NaiveBaselineResult(BaseModel):
    """The Naive Baseline's test-year UK ranking, from this run or as reported for v2."""

    name: str
    uk_ranking: float | None = None  # on the old measure
    uk_ranking_tie_proof: float | None = None  # the Selection Measure, where measured


class TestYearRanking(BaseModel):
    """The test year's UK ranking beside the Naive Baseline and earlier versions' headline figures."""

    test_year: int
    rows: list[VersionResult]  # this version first, then earlier versions
    naive: list[NaiveBaselineResult]
    beats_naive: bool
    text: str


class TradedAway(BaseModel):
    """What choosing on the ranking cost in the size of Export Potential (`model_choice.traded_away`)."""

    deviance: float
    earlier_deviance: float
    earlier_name: str
    size_matched_lift: float | None = None
    earlier_size_matched_lift: float | None = None
    text: str


class EventBacktestNote(BaseModel):
    """With a Blend served: the Event Backtests measured the model alone; served Scenario Effects are smaller."""

    model_weight: float
    model_effect: float | None = None
    served_effect: float | None = None
    text: str


class ModelChoice(BaseModel):
    """How the served model was chosen (ADR 0008), read from the artefact (`model_choice.how_chosen`)."""

    recorded: bool  # False when the artefact records no experiment results
    model_form: ModelFormOut
    selection_measure: str | None = None
    validation_year: int | None = None
    noise_floor_sd: float | None = None
    win_threshold: float | None = None
    experiments: list[ExperimentOut]
    n_experiments: int | None = None  # the experiments the recorded decisions cover (seven for nine decisions)
    confirmation: Confirmation | None = None
    text: str
    test_year_ranking: TestYearRanking | None = None
    traded_away: TradedAway | None = None


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
    large_markets: LargeMarkets | None = None  # None for a version evaluated before #123
    # This version beside the earlier versions re-scored from their saved scores (#128); None before.
    large_markets_by_version: list[LargeMarketsByVersion] | None = None
    market_reach_effect: MarketReachEffect | None = None  # None for a version without Market Reach
    tariff_split_text: str = ""  # how the test rows split between observed and imputed Applied Tariffs
    stability: Stability
    lift: Lift | None = None
    event_backtest: EventBacktest | None = None
    model_choice: ModelChoice
    event_backtest_note: EventBacktestNote | None = None
    not_run: list[NotRun] = []  # the opt-in parts skipped by the run that built this version
    limitations: list[Limitation]


class PresetOut(BaseModel):
    name: str
    title: str
    description: str


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
    """The fixed Blocked Opportunity thresholds (ADR 0012), stated in every response that applies them so prose can
    say "20% and $1M". No request carries them."""

    min_effect_share: float = Field(BLOCKED_MIN_SHARE, description="Blocked Opportunity: effect ≥ this share of Baseline")
    min_effect_usd: float = Field(BLOCKED_MIN_USD, description="Blocked Opportunity: effect ≥ this many USD")


class ScenarioRequest(ScenarioIn):
    """The Scenario and how many rows to return. Threshold fields are refused like any unknown field: the thresholds
    are fixed (ADR 0012)."""

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
    baseline_actual_exports_usd: float  # UK Actual Exports in the Baseline year (0 where there is no Flow)
    # `tiered` adds each Caveat Tier's column here (and its reason column, if any), as on ranking rows


class NotePart(BaseModel):
    """One part of the Scenario note; `run` is False (and the figures absent) when the artefact lacks it."""

    run: bool
    text: str


class BenchmarkPart(NotePart):
    effect_pct: float | None = None  # the PPML benchmark's Trade Agreement effect, % change in Flows
    low_pct: float | None = None
    high_pct: float | None = None
    ci_level: float | None = None
    published_low_pct: float | None = None  # the published estimates it is compared with
    published_high_pct: float | None = None


class ModelAssociationPart(NotePart):
    effect_pct: float | None = None  # the model's Trade Agreement association on the served Model Form, %
    pairs: int | None = None  # UK pairs without a Trade Agreement it is taken over
    model_form: str | None = None


class EventBacktestPart(NotePart):
    title: str | None = None  # the primary Event Backtest
    direction: str | None = None
    predicted: float | None = None  # the model alone's pooled effect, a relative change
    observed: float | None = None  # the observed difference-in-differences
    observed_low: float | None = None
    observed_high: float | None = None


class ScenarioNote(BaseModel):
    """Served beside a Scenario Effect: Illustrative, with its benchmarks (`illustrative.scenario_note`)."""

    label: str  # "Illustrative" (GLOSSARY.md)
    text: str
    benchmark: BenchmarkPart
    model_association: ModelAssociationPart
    event_backtest: EventBacktestPart
    blend: EventBacktestNote | None = None  # for a Blend: a Scenario moves only the model's share


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
    # The Scenario adds a Trade Agreement with a Market that has none in force in the Baseline year (ADR 0012). Only then
    # does it have Blocked Opportunities: those of the Markets it adds one with, as `/opportunities/blocked` serves them
    # (read from the precomputed Blocked Opportunity table, not from this Scenario's own Scenario Effects), with the
    # Caveat Tiers the app hides by default hidden, by Blocked Opportunity value, largest first. Both fields are None
    # (not empty and 0) when `adds_agreement` is false.
    adds_agreement: bool
    blocked_opportunities: list[BlockedOpportunity] | None
    blocked_count: int | None  # all of them, before `limit`
    # Pairs whose Scenario tariff lay beyond the training range and was clipped to it, so
    # their Scenario Effect flattens there (ADR 0006).
    clipped_pairs: int
    # A tariff Scenario's changed pairs whose known Baseline Applied Tariff was below (raised) or above (lowered) the
    # rate it sets; an unknown tariff counts in neither. None for every other kind of Scenario.
    pairs_raised: int | None = None
    pairs_lowered: int | None = None
    # A tariff Scenario's line on the pairs it sets, raises and lowers, naming the rate the model reads when the rate
    # was clipped; None for every other kind of Scenario (`ScenarioResult.tariff_directions_text`).
    tariff_directions: str | None = None
    # The Blocked Opportunity prose (`ScenarioResult.blocked_note`): the Blocked Opportunities section's caption when
    # `adds_agreement`, otherwise the line saying this Scenario adds none; each states the fixed rule.
    blocked_note: str
    note: ScenarioNote  # Scenario Effects are Illustrative: the label and the benchmarks beside them


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
    # `tiered` adds each Caveat Tier's column here (and its reason column, if any), as on ranking rows
    potential_range_low_usd: float  # the Potential Range around the Gap Period mean (ADR 0007)
    potential_range_high_usd: float
    uncertain_opportunity: bool  # an Uncertain Opportunity
    potential_range_text: str  # the Potential Range in plain words, for the pair's detail view
    tariff_caveat: str | None  # set when the Baseline year's Applied Tariff is carried forward (ADR 0006)
    market_reach: PairMarketReachOut | None  # the Market's reach, shown apart from the Product's factors
    factors: list[FactorOut]  # largest absolute contribution first
    # By year: every Gap Period year (with Export Potential and Export Gap), and earlier
    # years with UK Actual Exports only
    history: list[HistoryOut]


BlockedOpportunities.model_rebuild()  # resolves `note`, whose ScenarioNote is defined after it


# --- Caveat Tiers: the models that carry them, built from the declared tiers --------------------------------------


@dataclass(frozen=True)
class Tiered:
    """The query and response models that carry Caveat Tiers, each with every declared tier's fields.

    The classes of the same name above lack those fields; the API serves these.
    """

    RankingQuery: type[RankingQuery]
    LeaderboardQuery: type[LeaderboardQuery]
    BlockedQuery: type[BlockedQuery]
    MarketOpportunity: type[MarketOpportunity]
    ProductOpportunity: type[ProductOpportunity]
    LeaderboardOpportunity: type[LeaderboardOpportunity]
    BlockedOpportunity: type[BlockedOpportunity]
    Leaderboard: type[Leaderboard]
    BlockedOpportunities: type[BlockedOpportunities]
    ProductOpportunities: type[ProductOpportunities]
    MarketOpportunities: type[MarketOpportunities]
    Explanation: type[Explanation]
    PairEffect: type[PairEffect]
    ScenarioResponse: type[ScenarioResponse]


def _rebuilt[M: BaseModel](
    model: type[M], *, after: str | None = None, added: dict | None = None, validators: dict | None = None, **replaced
) -> type[M]:
    """`model` under its own name with `added` fields inserted after the field `after`, and `replaced` fields swapped
    in place, so field order (and the OpenAPI document) reads as if they had been declared there. `validators` are
    pydantic validators the built model runs."""
    fields: dict = {}
    for name, info in model.model_fields.items():
        fields[name] = (replaced[name], info) if name in replaced else (info.annotation, info)
        if name == after:
            fields.update(added or {})
    built = create_model(
        model.__name__, __config__=model.model_config, __doc__=model.__doc__, __module__=model.__module__,
        __validators__=validators, **fields,
    )
    return cast(type[M], built)


def _caveat_texts_validator(tiers: tuple[CaveatTier, ...]) -> dict:
    """A validator that fills a row's `caveat_texts` from its tier columns (`caveats.caveat_texts`), so every row the
    API serves carries its caveats worded once, in the package."""

    @model_validator(mode="before")
    @classmethod
    def with_caveat_texts(cls, data):
        if isinstance(data, dict) and "caveat_texts" not in data:
            return {**data, "caveat_texts": caveat_texts(data, tiers)}
        return data

    return {"with_caveat_texts": with_caveat_texts}


@cache
def tiered(tiers: tuple[CaveatTier, ...] = TIERS) -> Tiered:
    """The models carrying `tiers`: each tier's hide filter on the ranking queries (after `region`), and its column
    (then its reason column, if any) on ranking rows (after `blocked_opportunity`), on Scenario pairs (after
    `baseline_actual_exports_usd`) and on explanations (after `opportunity_type`)."""
    hides = {
        tier.hide_param: (bool, Field(False, description=tier.hide_description)) for tier in tiers
    }
    columns: dict = {}
    for tier in tiers:
        columns[tier.column] = (bool, ...)
        if tier.reason_column is not None:
            columns[tier.reason_column] = (str | None, ...)
    # The caveat text of each tier the row is in, worded by the package (`caveats.caveat_texts`), in tier order.
    columns["caveat_texts"] = (list[str], Field(default_factory=list))
    texts = _caveat_texts_validator(tiers)
    market, product, leaderboard, blocked = (
        _rebuilt(row, after="blocked_opportunity", added=columns, validators=texts)
        for row in (MarketOpportunity, ProductOpportunity, LeaderboardOpportunity, BlockedOpportunity)
    )
    pair = _rebuilt(PairEffect, after="baseline_actual_exports_usd", added=columns, validators=texts)
    return Tiered(
        RankingQuery=_rebuilt(RankingQuery, after="region", added=hides),
        LeaderboardQuery=_rebuilt(LeaderboardQuery, after="region", added=hides),
        BlockedQuery=_rebuilt(BlockedQuery, after="region", added=hides),
        MarketOpportunity=market,
        ProductOpportunity=product,
        LeaderboardOpportunity=leaderboard,
        BlockedOpportunity=blocked,
        Leaderboard=_rebuilt(Leaderboard, opportunities=list[leaderboard]),
        BlockedOpportunities=_rebuilt(BlockedOpportunities, opportunities=list[blocked]),
        ProductOpportunities=_rebuilt(ProductOpportunities, opportunities=list[market]),
        MarketOpportunities=_rebuilt(MarketOpportunities, opportunities=list[product]),
        Explanation=_rebuilt(Explanation, after="opportunity_type", added=columns, validators=texts),
        PairEffect=pair,
        ScenarioResponse=_rebuilt(ScenarioResponse, top_pairs=list[pair], blocked_opportunities=list[blocked] | None),
    )
