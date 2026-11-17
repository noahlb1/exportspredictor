"""FastAPI service. It loads saved artefacts and serves precomputed tables; no training here.

Run locally:  uv run uvicorn api.main:app
Tests inject an artefact root with `create_app(root)`; the module-level `app` reads
`EXPORTSPREDICTOR_ARTEFACTS_DIR` (default `<repo>/artefacts`).

To add an endpoint: add a route in `create_app`, get the bundle with `Depends(loaded)` (it
answers 503 until artefacts load), and declare a response model in `api/schemas.py`.
Both ranking views go through `_ranking` (and the package's `rank_opportunities`), so a
ranking option or row field is added there once.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import polars as pl
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from api.schemas import (
    Health,
    MarketOpportunities,
    MarketOpportunity,
    MarketOut,
    ProductOpportunities,
    ProductOpportunity,
    ProductOut,
    RankingQuery,
)
from exportspredictor import artefacts as store
from exportspredictor.lookups import search_products
from exportspredictor.paths import artefacts_dir
from exportspredictor.rankings import rank_opportunities


def loaded(request: Request) -> store.Artefacts:
    """The loaded artefact bundle, loading it on first use; 503 while none exists."""
    artefacts = _try_load(request.app)
    if artefacts is None:
        raise HTTPException(status_code=503, detail="Artefacts are not loaded yet.")
    return artefacts


def create_app(artefacts_root: Path | None = None) -> FastAPI:
    root = Path(artefacts_root) if artefacts_root is not None else artefacts_dir()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        _try_load(app)
        yield

    app = FastAPI(title="UK Exports Predictor", lifespan=lifespan)
    app.state.artefacts_root = root
    app.state.artefacts = None

    @app.get("/health", response_model=Health)
    def health(request: Request):
        artefacts = _try_load(request.app)
        if artefacts is None:
            return JSONResponse(status_code=503, content=Health(ready=False).model_dump())
        return Health(ready=True, artefact_version=artefacts.version, baseline_year=artefacts.baseline_year)

    @app.get("/products", response_model=list[ProductOut])
    def products(q: str = Query("", description="HS6 code (or prefix) or words"), limit: int = Query(20, ge=1, le=200), artefacts: store.Artefacts = Depends(loaded)):
        """Products matching a code or words, best first; an empty list when nothing matches."""
        found = search_products(artefacts.products, q, limit=limit)
        return [ProductOut(**{k: row[k] for k in ProductOut.model_fields}) for row in found.iter_rows(named=True)]

    @app.get("/markets", response_model=list[MarketOut])
    def markets(artefacts: store.Artefacts = Depends(loaded)):
        """Every Market, by name."""
        return [MarketOut(**{k: row[k] for k in MarketOut.model_fields}) for row in artefacts.markets.sort("name").iter_rows(named=True)]

    @app.get("/opportunities/by-product/{hs6}", response_model=ProductOpportunities)
    def by_product(hs6: str, query: Annotated[RankingQuery, Query()], artefacts: store.Artefacts = Depends(loaded)):
        """Markets ranked for one Product."""
        product = artefacts.products.filter(pl.col("hs6") == hs6)
        if product.is_empty():
            raise HTTPException(status_code=404, detail=f"Unknown Product: {hs6}")
        return ProductOpportunities(
            **_ranking(artefacts, query, MarketOpportunity, hs6=hs6),
            product=ProductOut(**product.row(0, named=True)),
        )

    @app.get("/opportunities/by-market/{iso3}", response_model=MarketOpportunities)
    def by_market(iso3: str, query: Annotated[RankingQuery, Query()], artefacts: store.Artefacts = Depends(loaded)):
        """Products ranked for one Market."""
        market = artefacts.markets.filter(pl.col("iso3") == iso3)
        if market.is_empty():
            raise HTTPException(status_code=404, detail=f"Unknown Market: {iso3}")
        return MarketOpportunities(
            **_ranking(artefacts, query, ProductOpportunity, iso3=iso3),
            market=MarketOut(**market.row(0, named=True)),
        )

    return app


def _ranking(artefacts: store.Artefacts, query: RankingQuery, row_model: type[BaseModel], **subject) -> dict:
    """The fields both ranking views share: metadata and the ranked, filtered rows."""
    rows = rank_opportunities(
        artefacts,
        **subject,
        sector=query.sector,
        region=query.region,
        min_gap_usd=query.min_gap,
        min_gap_gbp=query.min_gap_gbp,
        limit=query.limit,
    )
    return {
        "artefact_version": artefacts.version,
        "baseline_year": artefacts.baseline_year,
        "gbp_rate": artefacts.gbp_rate,
        "opportunities": [row_model(**{k: row[k] for k in row_model.model_fields}) for row in rows.iter_rows(named=True)],
    }


def _try_load(app: FastAPI) -> store.Artefacts | None:
    if app.state.artefacts is None:
        try:
            app.state.artefacts = store.load_latest(app.state.artefacts_root)
        except store.ArtefactsNotFound:
            return None
    return app.state.artefacts


app = create_app()
