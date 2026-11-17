"""FastAPI service. It loads saved artefacts and serves precomputed tables; no training here.

Run locally:  uv run uvicorn api.main:app
Tests inject an artefact root with `create_app(root)`; the module-level `app` reads
`EXPORTSPREDICTOR_ARTEFACTS_DIR` (default `<repo>/artefacts`).

To add an endpoint: add a route in `create_app`, get the bundle with `Depends(loaded)` (it
answers 503 until artefacts load), and declare a response model in `api/schemas.py`.
Both ranking views go through `_ranking`, which calls the package's `rank_opportunities`
(the one filter, sort and limit); ranking orders live in `exportspredictor.rankings.RANK_BY`.

Loading: artefacts load at startup. If none exist yet, `/health` answers 503 and the load
is retried at most once every `retry_seconds` (default 30 s) when a request arrives, so
artefacts saved later are picked up without a restart and without a disk scan per request.
The API serves the lexically newest folder only. If that folder cannot be read (another
artefact format, such as v1, or a missing table), the API stays not ready and `/health`
gives the reason; it does not fall back to an older folder.
"""

import logging
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import polars as pl
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from api.schemas import (
    Evaluation,
    ExcludedProductOut,
    Explanation,
    FactorOut,
    Health,
    HistoryOut,
    MarketOpportunities,
    MarketOpportunity,
    MarketOut,
    PresetOut,
    ProductMatch,
    ProductOpportunities,
    ProductOpportunity,
    ProductOut,
    RankingQuery,
    RankOrderOut,
    ScenarioIn,
    ScenarioRequest,
    ScenarioResponse,
    Thresholds,
)
from exportspredictor import artefacts as store
from exportspredictor import scenarios
from exportspredictor.evaluation import LIMITATIONS
from exportspredictor.explain import UnknownPair, explain_pair
from exportspredictor.lookups import search_products
from exportspredictor.paths import artefacts_dir
from exportspredictor.rankings import RANK_BY, rank_opportunities

RETRY_SECONDS = 30.0
log = logging.getLogger(__name__)


def loaded(request: Request) -> store.Artefacts:
    """The loaded artefact bundle; 503 while none has loaded."""
    artefacts = _try_load(request.app)
    if artefacts is None:
        raise HTTPException(status_code=503, detail="Artefacts are not loaded yet.")
    return artefacts


def create_app(
    artefacts_root: Path | None = None,
    *,
    retry_seconds: float = RETRY_SECONDS,
    clock: Callable[[], float] = time.monotonic,
) -> FastAPI:
    root = Path(artefacts_root) if artefacts_root is not None else artefacts_dir()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        _try_load(app)
        yield

    app = FastAPI(title="UK Exports Predictor", lifespan=lifespan)
    app.state.artefacts_root = root
    app.state.artefacts = None
    app.state.retry_seconds = retry_seconds
    app.state.clock = clock
    app.state.last_attempt = None
    app.state.load_error = None

    @app.get("/health", response_model=Health, response_model_exclude_none=True)
    def health(request: Request):
        artefacts = _try_load(request.app)
        if artefacts is None:
            content = Health(ready=False, reason=request.app.state.load_error).model_dump(exclude_none=True)
            return JSONResponse(status_code=503, content=content)
        return Health(ready=True, artefact_version=artefacts.version, baseline_year=artefacts.baseline_year)

    @app.get("/products", response_model=list[ProductMatch])
    def products(q: str = Query("", description="HS6 code (or prefix) or words"), limit: int = Query(20, ge=1, le=200), artefacts: store.Artefacts = Depends(loaded)):
        """Products matching a code or words, best first; an empty list when nothing matches.

        Excluded Products are listed too, marked `excluded` with their `exclusion_reason`.
        """
        found = search_products(artefacts.products, q, limit=limit, excluded=artefacts.excluded_products)
        return [_pick(ProductMatch, row) for row in found.iter_rows(named=True)]

    @app.get("/products/excluded", response_model=list[ExcludedProductOut])
    def excluded_products(artefacts: store.Artefacts = Depends(loaded)):
        """The Excluded Products (ADR 0005), each with the reason it is left out, by HS6 code."""
        return [_pick(ExcludedProductOut, row) for row in artefacts.excluded_products.iter_rows(named=True)]

    @app.get("/markets", response_model=list[MarketOut])
    def markets(artefacts: store.Artefacts = Depends(loaded)):
        """Every Market, by name."""
        return [_pick(MarketOut, row) for row in artefacts.markets.sort("name").iter_rows(named=True)]

    @app.get("/opportunities/rank-by", response_model=list[RankOrderOut])
    def rank_orders():
        """The ranking orders both Opportunity views accept as `rank_by`, in display order."""
        return [RankOrderOut(name=o.name, title=o.title, description=o.description) for o in RANK_BY.values()]

    @app.get("/opportunities/by-product/{hs6}", response_model=ProductOpportunities)
    def by_product(hs6: str, query: Annotated[RankingQuery, Query()], artefacts: store.Artefacts = Depends(loaded)):
        """Markets ranked for one Product."""
        product = _product(artefacts, hs6)
        return ProductOpportunities(**_ranking(artefacts, query, MarketOpportunity, hs6=hs6), product=product)

    @app.get("/opportunities/by-market/{iso3}", response_model=MarketOpportunities)
    def by_market(iso3: str, query: Annotated[RankingQuery, Query()], artefacts: store.Artefacts = Depends(loaded)):
        """Products ranked for one Market."""
        market = _market(artefacts, iso3)
        return MarketOpportunities(**_ranking(artefacts, query, ProductOpportunity, iso3=iso3), market=market)

    @app.get("/evaluation", response_model=Evaluation)
    def evaluation(artefacts: store.Artefacts = Depends(loaded)):
        """The evaluation artefact: test-year accuracy, breakdowns, stability, and the known limitations."""
        if artefacts.evaluation is None:
            raise HTTPException(status_code=404, detail=f"Artefact version {artefacts.version} has no evaluation.")
        return Evaluation(
            **artefacts.evaluation,
            artefact_version=artefacts.version,
            baseline_year=artefacts.baseline_year,
            limitations=LIMITATIONS,
        )

    @app.get("/explain/{hs6}/{iso3}", response_model=Explanation)
    def explain(hs6: str, iso3: str, artefacts: store.Artefacts = Depends(loaded)):
        """Why? one pair's Export Potential: SHAP factors from its own fold model, plus history."""
        product, market = _product(artefacts, hs6), _market(artefacts, iso3)
        try:
            explanation = explain_pair(artefacts, hs6, iso3)
        except UnknownPair as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        values = (
            artefacts.baseline.filter((pl.col("hs6") == hs6) & (pl.col("iso3") == iso3))
            .select("export_potential_usd", "actual_exports_usd", "export_gap_usd")
            .row(0, named=True)
        )
        return Explanation(
            **_metadata(artefacts),
            **values,
            product=product,
            market=market,
            fold=explanation.fold,
            base_value=explanation.base_value,
            log_export_potential=explanation.log_export_potential,
            factors=[
                FactorOut(
                    label=f.label,
                    value_text=f.value_text,
                    contribution=f.contribution,
                    multiplier=f.multiplier,
                    direction="raised" if f.contribution > 0 else "lowered" if f.contribution < 0 else "no effect",
                )
                for f in explanation.factors
            ],
            history=[
                HistoryOut(year=h.year, actual_exports_usd=h.actual_exports_usd) for h in explanation.history
            ],
        )

    @app.get("/scenarios/presets", response_model=list[PresetOut])
    def presets():
        """The Trading Bloc presets, in display order."""
        return [PresetOut(**vars(preset)) for preset in scenarios.PRESETS.values()]

    @app.get("/scenarios/thresholds", response_model=Thresholds)
    def thresholds():
        """The default Blocked Opportunity thresholds."""
        return Thresholds()

    @app.post("/scenarios", response_model=ScenarioResponse)
    def run_scenario(request: ScenarioRequest, artefacts: store.Artefacts = Depends(loaded)):
        """Re-predict the Baseline with a preset or custom Trade Agreement changes (422 if invalid)."""
        if request.market is not None and artefacts.markets.filter(pl.col("iso3") == request.market).is_empty():
            raise HTTPException(status_code=422, detail=f"Unknown Market: {request.market}")
        try:
            result = scenarios.run(
                artefacts.baseline_features, artefacts.baseline, artefacts.fold_models, request.to_scenario()
            )
        except scenarios.InvalidScenario as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return scenario_response(result, request, artefacts)

    return app


def _pick(model: type[BaseModel], row: dict) -> BaseModel:
    """`model` built from the fields it declares, taken from a table row."""
    return model(**{name: row[name] for name in model.model_fields})


def _product(artefacts: store.Artefacts, hs6: str) -> ProductOut:
    """The Product, or 404: for an unknown Product, or an Excluded Product (with its reason)."""
    found = artefacts.products.filter(pl.col("hs6") == hs6)
    if found.is_empty():
        raise HTTPException(status_code=404, detail=f"Unknown Product: {hs6}")
    excluded = artefacts.excluded_products.filter(pl.col("hs6") == hs6)
    if not excluded.is_empty():
        row = excluded.row(0, named=True)
        raise HTTPException(
            status_code=404,
            detail=f"{hs6} ({row['description']}) is an Excluded Product, so it is not scored or ranked. {row['reason']}",
        )
    return _pick(ProductOut, found.row(0, named=True))


def _market(artefacts: store.Artefacts, iso3: str) -> MarketOut:
    """The Market, or 404."""
    found = artefacts.markets.filter(pl.col("iso3") == iso3)
    if found.is_empty():
        raise HTTPException(status_code=404, detail=f"Unknown Market: {iso3}")
    return _pick(MarketOut, found.row(0, named=True))


def _metadata(artefacts: store.Artefacts) -> dict:
    """The response metadata every value-carrying response shares: version and Baseline year."""
    return {"artefact_version": artefacts.version, "baseline_year": artefacts.baseline_year}


def _ranking(artefacts: store.Artefacts, query: RankingQuery, row_model: type[BaseModel], **subject) -> dict:
    """The fields both ranking views share: metadata and the ranked rows."""
    rows = rank_opportunities(
        artefacts,
        **subject,
        sector=query.sector,
        region=query.region,
        min_gap_usd=query.min_gap_usd,
        rank_by=query.rank_by,
        limit=query.limit,
    )
    return {
        **_metadata(artefacts),
        "rank_by": query.rank_by,
        "opportunities": [_pick(row_model, row) for row in rows.iter_rows(named=True)],
    }


def scenario_response(
    result: scenarios.ScenarioResult, request: ScenarioRequest, artefacts: store.Artefacts
) -> ScenarioResponse:
    """Totals, per-Market and per-Product effects, top pairs and Blocked Opportunities, in nominal USD."""
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
    top = pairs if request.market is None else pairs.filter(pl.col("iso3") == request.market)
    total = result.total()
    return ScenarioResponse(
        **_metadata(artefacts),
        scenario=_pick(ScenarioIn, vars(result.scenario)),
        thresholds=_pick(Thresholds, request.model_dump()),
        market=request.market,
        changes_nothing=result.changes_nothing,
        summary=result.summary(blocked=blocked.height, year=artefacts.baseline_year),
        total=total,
        markets=result.by_market().join(markets, on="iso3", how="left", maintain_order="left").to_dicts(),
        products=result.by_product()
        .head(request.limit)
        .join(products, on="hs6", how="left", maintain_order="left")
        .to_dicts(),
        top_pairs=top.head(request.limit).to_dicts(),
        blocked_opportunities=blocked.head(request.limit).to_dicts(),
        blocked_count=blocked.height,
    )


def _try_load(app: FastAPI) -> store.Artefacts | None:
    """The loaded bundle; loads it if none is loaded and the last attempt is `retry_seconds` old."""
    state = app.state
    if state.artefacts is not None:
        return state.artefacts
    now = state.clock()
    if state.last_attempt is not None and now - state.last_attempt < state.retry_seconds:
        return None
    state.last_attempt = now
    try:
        state.artefacts = store.load_latest(state.artefacts_root)
    except (store.ArtefactsNotFound, store.UnreadableArtefacts) as error:
        state.load_error = str(error)
        log.warning("%s; retrying at most every %.0f s", error, state.retry_seconds)
        return None
    state.load_error = None
    return state.artefacts


app = create_app()
