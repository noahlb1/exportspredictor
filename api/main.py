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
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from api.schemas import (
    Health,
    MarketOpportunity,
    MarketOut,
    PresetOut,
    ProductOpportunities,
    ProductOut,
    RankBy,
    ScenarioRequest,
    ScenarioResponse,
)
from exportspredictor import artefacts as store
from exportspredictor import scenarios
from exportspredictor.lookups import search_products
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
    def by_product(
        hs6: str,
        rank_by: RankBy = Query("gap", description="gap: Export Gap; blocked: Blocked Opportunity value"),
        artefacts: store.Artefacts = Depends(loaded),
    ):
        product = artefacts.products.filter(pl.col("hs6") == hs6)
        if product.is_empty():
            raise HTTPException(status_code=404, detail=f"Unknown Product: {hs6}")
        rows = rank(
            artefacts.baseline.filter(pl.col("hs6") == hs6).join(
                artefacts.markets.rename({"name": "market"}), on="iso3", how="left"
            ),
            rank_by,
        )
        return ProductOpportunities(
            artefact_version=artefacts.version,
            baseline_year=artefacts.baseline_year,
            rank_by=rank_by,
            product=ProductOut(**product.row(0, named=True)),
            opportunities=[
                MarketOpportunity(**{k: row[k] for k in MarketOpportunity.model_fields})
                for row in rows.iter_rows(named=True)
            ],
        )

    @app.get("/scenarios/presets", response_model=list[PresetOut])
    def presets():
        """The Trading Bloc presets, in display order."""
        return [PresetOut(**vars(preset)) for preset in scenarios.PRESETS.values()]

    @app.post("/scenarios", response_model=ScenarioResponse)
    def run_scenario(request: ScenarioRequest, artefacts: store.Artefacts = Depends(loaded)):
        """Re-predict the Baseline with a preset or custom Trade Agreement changes (422 if invalid)."""
        scenario = scenarios.Scenario(
            preset=request.preset,
            add_agreements=tuple(request.add_agreements),
            remove_agreements=tuple(request.remove_agreements),
        )
        try:
            result = scenarios.run(artefacts.baseline_features, artefacts.baseline, artefacts.fold_models, scenario)
        except scenarios.InvalidScenario as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return scenario_response(result, request, artefacts)

    return app


RANK_COLUMNS = {"gap": "export_gap_usd", "blocked": "blocked_effect_usd"}


def rank(rows: pl.DataFrame, rank_by: str) -> pl.DataFrame:
    """Ranking rows (Baseline columns) with the Blocked Opportunity flag, sorted by `rank_by`, largest first."""
    return rows.with_columns(
        blocked_opportunity=scenarios.is_blocked("blocked_effect_usd", "export_potential_usd")
    ).sort(RANK_COLUMNS[rank_by], descending=True)


def scenario_response(
    result: scenarios.ScenarioResult, request: ScenarioRequest, artefacts: store.Artefacts
) -> ScenarioResponse:
    """Totals, per-Market and per-Product effects, top pairs and Blocked Opportunities."""
    markets = artefacts.markets.select("iso3", market="name")
    products = artefacts.products.select("hs6", "description")
    pairs = (
        result.pairs.with_columns(
            blocked_opportunity=scenarios.is_blocked(
                "effect_usd", "baseline_potential_usd", request.min_effect_share, request.min_effect_usd
            )
        )
        .join(markets, on="iso3", how="left", maintain_order="left")
        .join(products, on="hs6", how="left", maintain_order="left")
    )
    blocked = pairs.filter("blocked_opportunity")
    total = result.total()
    return ScenarioResponse(
        artefact_version=artefacts.version,
        baseline_year=artefacts.baseline_year,
        scenario={
            "preset": request.preset,
            "add_agreements": request.add_agreements,
            "remove_agreements": request.remove_agreements,
        },
        thresholds={"min_effect_share": request.min_effect_share, "min_effect_usd": request.min_effect_usd},
        changes_nothing=result.changes_nothing,
        summary=_summary(result, total, blocked.height, artefacts.baseline_year),
        total=total,
        markets=result.by_market().join(markets, on="iso3", how="left", maintain_order="left").to_dicts(),
        products=result.by_product()
        .head(request.limit)
        .join(products, on="hs6", how="left", maintain_order="left")
        .to_dicts(),
        top_pairs=pairs.head(request.limit).to_dicts(),
        blocked_opportunities=blocked.head(request.limit).to_dicts(),
        blocked_count=blocked.height,
    )


def _summary(result: scenarios.ScenarioResult, total: dict, blocked: int, year: int) -> str:
    """One sentence, worded as an association, never as a cause or a forecast."""
    if result.changes_nothing:
        return (
            "This Scenario changes nothing: the trade-policy conditions it sets are already those of the "
            f"{year} Baseline (for example, the Trade Agreement already exists), so there is no Scenario Effect."
        )
    markets = result.pairs["iso3"].n_unique()
    market_word = "Market" if markets == 1 else "Markets"
    blocked_word = "Blocked Opportunity" if blocked == 1 else "Blocked Opportunities"
    return (
        f"The model associates this Scenario with a change of {format_usd(total['effect_usd'])} "
        f"({total['effect_share']:+.1%}) in UK Export Potential across {markets} {market_word}, "
        f"against the {year} Baseline, with {blocked} {blocked_word}. "
        "This is a learned association, not a forecast."
    )


def format_usd(value: float) -> str:
    """A signed, rounded dollar amount, e.g. +$1.2bn."""
    sign = "-" if value < 0 else "+"
    value = abs(value)
    for unit, size in (("bn", 1e9), ("m", 1e6), ("k", 1e3)):
        if value >= size:
            return f"{sign}${value / size:,.1f}{unit}"
    return f"{sign}${value:,.0f}"


def _try_load(app: FastAPI) -> store.Artefacts | None:
    if app.state.artefacts is None:
        try:
            app.state.artefacts = store.load_latest(app.state.artefacts_root)
        except store.ArtefactsNotFound:
            return None
    return app.state.artefacts


app = create_app()
