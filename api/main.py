"""FastAPI service. It loads saved artefacts and serves precomputed tables; no training here.

Run locally:  uv run uvicorn api.main:app
Tests inject an artefact root with `create_app(root)`; the module-level `app` reads
`EXPORTSPREDICTOR_ARTEFACTS_DIR` (default `<repo>/artefacts`).

To add an endpoint: add a route in `create_app`, get the bundle with `Depends(loaded)` (it
answers 503 until artefacts load), and declare a response model in `api/schemas.py`.
"""

from contextlib import asynccontextmanager
from pathlib import Path

import polars as pl
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from api.schemas import Health, MarketOpportunity, ProductOpportunities, ProductOut
from exportspredictor import artefacts as store
from exportspredictor.paths import artefacts_dir


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

    @app.get("/opportunities/by-product/{hs6}", response_model=ProductOpportunities)
    def by_product(hs6: str, artefacts: store.Artefacts = Depends(loaded)):
        product = artefacts.products.filter(pl.col("hs6") == hs6)
        if product.is_empty():
            raise HTTPException(status_code=404, detail=f"Unknown Product: {hs6}")
        rows = (
            artefacts.baseline.filter(pl.col("hs6") == hs6)
            .join(artefacts.markets.rename({"name": "market"}), on="iso3", how="left")
            .sort("export_gap_usd", descending=True)
        )
        return ProductOpportunities(
            artefact_version=artefacts.version,
            baseline_year=artefacts.baseline_year,
            product=ProductOut(**product.row(0, named=True)),
            opportunities=[
                MarketOpportunity(**{k: row[k] for k in MarketOpportunity.model_fields})
                for row in rows.iter_rows(named=True)
            ],
        )

    return app


def _try_load(app: FastAPI) -> store.Artefacts | None:
    if app.state.artefacts is None:
        try:
            app.state.artefacts = store.load_latest(app.state.artefacts_root)
        except store.ArtefactsNotFound:
            return None
    return app.state.artefacts


app = create_app()
