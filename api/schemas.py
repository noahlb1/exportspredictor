"""Response models: the API contract the app and the contract tests rely on."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from exportspredictor.scenarios import DEFAULT_MIN_SHARE, DEFAULT_MIN_USD


class Health(BaseModel):
    ready: bool
    artefact_version: str | None = None
    baseline_year: int | None = None


class ProductOut(BaseModel):
    hs6: str
    description: str
    sector: str


class MarketOut(BaseModel):
    iso3: str
    name: str
    region: str


RankBy = Literal["gap", "blocked"]


class RankingQuery(BaseModel):
    """Query parameters both ranking views accept."""

    rank_by: RankBy = Field("gap", description="gap: Export Gap; blocked: Blocked Opportunity value")
    sector: str | None = Field(None, description="Sector (HS2 chapter), e.g. 22")
    region: str | None = Field(None, description="Market region, e.g. South Asia")
    min_gap: float | None = Field(None, description="Minimum Export Gap, in USD")
    min_gap_gbp: float | None = Field(None, description="Minimum Export Gap, in GBP")
    limit: int | None = Field(None, ge=1, description="Keep only the top rows")


class _Values(BaseModel):
    """Export Potential, Actual Exports and Export Gap of one UK Product–Market pair."""

    export_potential_usd: float
    actual_exports_usd: float
    export_gap_usd: float
    export_potential_gbp: float
    actual_exports_gbp: float
    export_gap_gbp: float
    trade_agreement: bool  # a UK Trade Agreement with the Market is in force in the Baseline year
    blocked_effect_usd: float  # Scenario Effect of adding a Trade Agreement where none exists (0 if one does)
    blocked_effect_gbp: float
    blocked_opportunity: bool  # that effect clears the default Blocked Opportunity thresholds


class MarketOpportunity(_Values):
    iso3: str
    market: str
    region: str


class ProductOpportunity(_Values):
    hs6: str
    description: str
    sector: str


class GbpRate(BaseModel):
    """The rate GBP values use: the Baseline year's annual average."""

    usd_per_gbp: float
    year: int
    source: str


class _Ranking(BaseModel):
    artefact_version: str
    baseline_year: int
    rank_by: RankBy
    gbp_rate: GbpRate


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
    limitations: list[Limitation]

class PresetOut(BaseModel):
    name: str
    title: str
    description: str


class ScenarioRequest(BaseModel):
    """A Trading Bloc preset, *or* Markets (ISO3) to add or remove a Trade Agreement with."""

    model_config = ConfigDict(extra="forbid")

    preset: Literal["eu_rejoin", "eu_customs_union", "cptpp", "leave"] | None = None
    add_agreements: list[str] = []
    remove_agreements: list[str] = []
    min_effect_share: float = Field(DEFAULT_MIN_SHARE, ge=0, description="Blocked Opportunity: effect ≥ this share of Baseline")
    min_effect_usd: float = Field(DEFAULT_MIN_USD, ge=0, description="Blocked Opportunity: effect ≥ this many USD")
    limit: int = Field(50, ge=1, le=1000, description="Rows returned in each list")


class ScenarioIn(BaseModel):
    preset: str | None
    add_agreements: list[str]
    remove_agreements: list[str]


class Thresholds(BaseModel):
    min_effect_share: float
    min_effect_usd: float


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
    changes_nothing: bool
    summary: str  # one associational sentence
    total: EffectTotals  # over the pairs the Scenario changes
    markets: list[MarketEffect]  # every changed Market, largest effect first
    products: list[ProductEffect]  # top Products by effect
    top_pairs: list[PairEffect]  # top Product–Market pairs by effect
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
