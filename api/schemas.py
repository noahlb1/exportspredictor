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


class MarketOut(BaseModel):
    iso3: str
    name: str
    region: str


class _Values(BaseModel):
    """Export Potential, Actual Exports and Export Gap of one UK Product–Market pair."""

    export_potential_usd: float
    actual_exports_usd: float
    export_gap_usd: float


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


class ProductOpportunities(_Ranking):
    product: ProductOut
    opportunities: list[MarketOpportunity]  # Markets, sorted by Export Gap, largest first


class MarketOpportunities(_Ranking):
    market: MarketOut
    opportunities: list[ProductOpportunity]  # Products, sorted by Export Gap, largest first
