"""Response models: the API contract the app and the contract tests rely on."""

from pydantic import BaseModel, ConfigDict


class Health(BaseModel):
    ready: bool
    artefact_version: str | None = None
    baseline_year: int | None = None


class ProductOut(BaseModel):
    hs6: str
    description: str
    sector: str


class MarketOpportunity(BaseModel):
    iso3: str
    market: str
    region: str
    export_potential_usd: float
    actual_exports_usd: float
    export_gap_usd: float


class ProductOpportunities(BaseModel):
    artefact_version: str
    baseline_year: int
    product: ProductOut
    opportunities: list[MarketOpportunity]  # every Market, sorted by Export Gap, largest first


class MarketOut(BaseModel):
    iso3: str
    name: str
    region: str


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
