"""Response models: the API contract the app and the contract tests rely on."""

from pydantic import BaseModel, Field


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


class RankingQuery(BaseModel):
    """Query parameters both ranking views accept."""

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
    gbp_rate: GbpRate


class ProductOpportunities(_Ranking):
    product: ProductOut
    opportunities: list[MarketOpportunity]  # Markets, sorted by Export Gap, largest first


class MarketOpportunities(_Ranking):
    market: MarketOut
    opportunities: list[ProductOpportunity]  # Products, sorted by Export Gap, largest first


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
