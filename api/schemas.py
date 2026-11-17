"""Response models: the API contract the app and the contract tests rely on."""

from pydantic import BaseModel


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
