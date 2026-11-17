"""Response models: the API contract the app and the contract tests rely on."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from exportspredictor import scenarios
from exportspredictor.rankings import DEFAULT_RANK_BY, RANK_BY
from exportspredictor.scenarios import DEFAULT_MIN_SHARE, DEFAULT_MIN_USD, PRESETS


class Health(BaseModel):
    ready: bool
    artefact_version: str | None = None
    baseline_year: int | None = None
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


class MarketOut(BaseModel):
    iso3: str
    name: str
    region: str


RankBy = Literal[tuple(RANK_BY)]  # the ranking orders of `exportspredictor.rankings.RANK_BY`


class RankOrderOut(BaseModel):
    name: str
    title: str
    description: str


class RankingQuery(BaseModel):
    """Query parameters both ranking views accept; an unknown one (such as the old `min_gap_gbp`) is a 422."""

    model_config = ConfigDict(extra="forbid")

    rank_by: RankBy = Field(
        DEFAULT_RANK_BY, description="; ".join(f"{order.name}: {order.title}" for order in RANK_BY.values())
    )
    sector: str | None = Field(None, description="Sector (HS2 chapter), e.g. 22")
    region: str | None = Field(None, description="Market region, e.g. South Asia")
    min_gap_usd: float | None = Field(None, description="Minimum Export Gap, in USD")
    limit: int | None = Field(None, ge=1, description="Keep only the top rows")


class _Values(BaseModel):
    """Export Potential, Actual Exports and Export Gap of one UK Product–Market pair."""

    export_potential_usd: float
    actual_exports_usd: float
    export_gap_usd: float
    trade_agreement: bool  # a UK Trade Agreement with the Market is in force in the Baseline year
    blocked_effect_usd: float  # Scenario Effect of adding a Trade Agreement where none exists (0 if one does)
    blocked_opportunity: bool  # that effect clears the default Blocked Opportunity thresholds


class MarketOpportunity(_Values):
    iso3: str
    market: str
    region: str


class ProductOpportunity(_Values):
    hs6: str
    description: str
    sector: str


class _Ranking(BaseModel):
    artefact_version: str
    baseline_year: int
    rank_by: RankBy


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
    """How much more often the top tenth of pairs by Export Gap grew materially than random pairs (#21)."""

    lift: float | None
    ci: list[float | None]
    ci_level: float
    bootstrap_unit: str
    top_rate: float | None
    base_rate: float | None
    size_matched: SizeMatchedLift
    by_decile: list[LiftDecile]
    held_out_share: float
    held_out_exporters: list[str]
    training_years: list[int]
    gap_year: int
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
    stability: Stability
    lift: Lift | None = None
    event_backtest: EventBacktest | None = None
    limitations: list[Limitation]


class PresetOut(BaseModel):
    name: str
    title: str
    description: str


PresetName = Literal[tuple(PRESETS)]  # the presets of `exportspredictor.scenarios.PRESETS`


class ScenarioIn(BaseModel):
    """A Trading Bloc preset, *or* Markets (ISO3) to add or remove a Trade Agreement with."""

    preset: PresetName | None = None
    add_agreements: list[str] = []
    remove_agreements: list[str] = []

    def to_scenario(self) -> scenarios.Scenario:
        return scenarios.Scenario(
            preset=self.preset,
            add_agreements=tuple(self.add_agreements),
            remove_agreements=tuple(self.remove_agreements),
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
    blocked_opportunity: bool


class ScenarioResponse(BaseModel):
    artefact_version: str
    baseline_year: int
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


class FactorOut(BaseModel):
    label: str  # plain-English factor name; raw model input names are never returned
    value_text: str  # the factor's value for this pair, readable
    contribution: float  # SHAP value on the log scale; > 0 raised Export Potential
    multiplier: float  # exp(contribution): the factor multiplies Export Potential by this
    direction: str  # "raised", "lowered" or "no effect"


class HistoryOut(BaseModel):
    year: int
    actual_exports_usd: float


class Explanation(BaseModel):
    artefact_version: str
    baseline_year: int
    product: ProductOut
    market: MarketOut
    fold: int  # the fold model that held the pair out, scored its Baseline and explains it
    base_value: float  # SHAP base value (log scale)
    log_export_potential: float  # base_value + sum of contributions
    export_potential_usd: float
    actual_exports_usd: float
    export_gap_usd: float
    factors: list[FactorOut]  # largest absolute contribution first
    history: list[HistoryOut]  # UK Actual Exports by year; years with none are absent
