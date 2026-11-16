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
