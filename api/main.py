"""FastAPI service. It loads saved artefacts and serves precomputed tables; no training here.

Run locally:  uv run uvicorn api.main:app
Tests inject an artefact root with `create_app(root)`; the module-level `app` reads
`EXPORTSPREDICTOR_ARTEFACTS_DIR` (default `<repo>/artefacts`).

To add an endpoint: add a route in `create_app`, get the bundle with `Depends(loaded)` (it
answers 503 until artefacts load), and declare a response model in `api/schemas.py`.
Every ranking view (the Leaderboard, Blocked Opportunities, by Product, by Market) goes through the
package's `rank_opportunities` (the one filter, sort and limit; `ranking` adds the summary); ranking orders
live in `exportspredictor.rankings.RANK_BY` (and `BLOCKED`, the Blocked Opportunities order).

Loading: artefacts load at startup. If none exist yet, `/health` answers 503 and the load
is retried at most once every `retry_seconds` (default 30 s) when a request arrives, so
artefacts saved later are picked up without a restart and without a disk scan per request.
The API serves the lexically newest folder only. If that folder cannot be read (another
artefact format, such as v1, or a missing table), the API stays not ready and `/health`
gives the reason; it does not fall back to an older folder.
"""

import logging
import time
from collections.abc import Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import polars as pl
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from api.schemas import (
    BlockedMarket,
    BlockedSummary,
    Caveat,
    CaveatTierOut,
    Evaluation,
    ExcludedProductOut,
    ExcludedReasonOut,
    FactorOut,
    Health,
    HistoryOut,
    Labels,
    LeaderboardQuery,
    LeaderboardSummary,
    MarketOut,
    MarketReachOut,
    PairMarketReachOut,
    PresetOut,
    ProductMatch,
    ProductOut,
    RankingQuery,
    RankOrderOut,
    ScenarioIn,
    ScenarioRequest,
    ScenarioResponse,
    SectorOut,
    Thresholds,
    Tiered,
    TotalsNote,
    tiered,
)
from exportspredictor import artefacts as store
from exportspredictor import market_reach, scenarios
from exportspredictor.caveats import TIERS, CaveatTier
from exportspredictor.conformal import potential_range_text
from exportspredictor.currency import USD_CAPTION, USD_UNIT
from exportspredictor.evaluation import (
    known_limitations,
    large_markets_beside_earlier,
    parts_not_run,
    served_report,
)
from exportspredictor.exclusions import not_ranked_message, reason_counts
from exportspredictor.explain import UnknownPair, explain_pair
from exportspredictor.gap_period import OPPORTUNITY_TYPES, with_baseline_actual_exports
from exportspredictor.gap_period import label as gap_period_label
from exportspredictor.illustrative import ILLUSTRATIVE, ILLUSTRATIVE_TEXT, scenario_note
from exportspredictor.lookups import search_products
from exportspredictor.model_choice import event_backtest_note, how_chosen
from exportspredictor.paths import artefacts_dir
from exportspredictor.rankings import (
    BLOCKED,
    DEFAULT_MIN_POTENTIAL_USD,
    MAX_ROWS,
    MIN_POTENTIAL_USD_OPTIONS,
    RANK_BY,
    blocked_opportunities,
    rank_opportunities,
    ranking,
)
from exportspredictor.sectors import sectors as sectors_of
from exportspredictor.sectors import with_sector_names
from exportspredictor.tariffs import LAST_TRAINS_YEAR, TARIFF_CARRIED_FORWARD_CAVEAT
from exportspredictor.wording import CAUTION, term_definitions, totals_note

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
    tiers: Sequence[CaveatTier] = TIERS,
) -> FastAPI:
    """The API over the newest artefact under `artefacts_root`, serving the caveat `tiers` (the package's by default;
    tests declare more). Every tier's columns must be on the artefact's Baseline."""
    root = Path(artefacts_root) if artefacts_root is not None else artefacts_dir()
    tiers = tuple(tiers)
    models = tiered(tiers)

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
    app.state.tiers = tiers  # a Baseline without one of their columns is refused (`artefacts.load`)

    @app.get("/health", response_model=Health, response_model_exclude_none=True)
    def health(request: Request):
        artefacts = _try_load(request.app)
        if artefacts is None:
            content = Health(ready=False, reason=request.app.state.load_error).model_dump(exclude_none=True)
            return JSONResponse(status_code=503, content=content)
        return Health(
            ready=True,
            artefact_version=artefacts.version,
            artefact_format=artefacts.metadata["artefact_format"],
            baseline_year=artefacts.baseline_year,
            gap_years=list(artefacts.gap_years),
            gap_period_label=gap_period_label(artefacts.gap_years),
        )

    @app.get("/labels", response_model=Labels)
    def labels(artefacts: store.Artefacts = Depends(loaded)):
        """The wording and options the app shows: unit, Opportunity Types, caveats, the last tariff year, the
        caution line under every page heading, the term definitions (with the artefact's Gap Period) and the note beside
        every summed figure, with the share of Actual Exports the artefact records."""
        share = artefacts.potential_share_of_actual_exports
        return Labels(
            usd_caption=USD_CAPTION,
            usd_unit=USD_UNIT,
            opportunity_types=list(OPPORTUNITY_TYPES),
            last_tariff_year=LAST_TRAINS_YEAR,
            tariff_carried_forward=TARIFF_CARRIED_FORWARD_CAVEAT,
            illustrative=Caveat(badge=ILLUSTRATIVE, text=ILLUSTRATIVE_TEXT),
            caution=CAUTION,
            terms=term_definitions(gap_period_label(artefacts.gap_years)),
            max_rows=MAX_ROWS,
            totals_note=TotalsNote(
                potential_share_of_actual_exports=share,
                text=totals_note(share, gap_period_label(artefacts.gap_years)),
            ),
        )

    @app.get("/caveats", response_model=list[CaveatTierOut])
    def caveats():
        """The Caveat Tiers (ADR 0005), in display order: each one's row field, badge, caveat text, key, hide filter
        parameter and, if it gives a reason per row, the row field holding it and the caveat text it goes into."""
        return [
            CaveatTierOut(
                column=tier.column,
                badge=tier.badge,
                plural=tier.plural,
                text=tier.text,
                key=tier.key,
                hide_param=tier.hide_param,
                reason_column=tier.reason_column,
                row_text=tier.row_text,
                hidden_by_default=tier.hidden_by_default,
            )
            for tier in tiers
        ]

    @app.get("/products", response_model=list[ProductMatch])
    def products(q: str = Query("", description="HS6 code (or prefix) or words"), limit: int = Query(20, ge=1, le=200), artefacts: store.Artefacts = Depends(loaded)):
        """Products matching a code or words, best first; an empty list when nothing matches.

        Excluded Products are listed too, marked `excluded` with their `exclusion_reason`.
        """
        found = with_sector_names(
            search_products(artefacts.products, q, limit=limit, excluded=artefacts.excluded_products)
        )
        return [_pick(ProductMatch, row) for row in found.iter_rows(named=True)]

    @app.get("/products/excluded", response_model=list[ExcludedProductOut])
    def excluded_products(artefacts: store.Artefacts = Depends(loaded)):
        """The Excluded Products (ADR 0005), each with the reason it is left out, by HS6 code."""
        excluded = with_sector_names(artefacts.excluded_products)
        return [_pick(ExcludedProductOut, row) for row in excluded.iter_rows(named=True)]

    @app.get("/products/excluded/reasons", response_model=list[ExcludedReasonOut])
    def excluded_reasons(artefacts: store.Artefacts = Depends(loaded)):
        """Each reason Products are excluded for, with how many it leaves out, most first."""
        counted = reason_counts(artefacts.excluded_products)
        return [_pick(ExcludedReasonOut, row) for row in counted.iter_rows(named=True)]

    @app.get("/sectors", response_model=list[SectorOut])
    def sectors(artefacts: store.Artefacts = Depends(loaded)):
        """Each Sector that has Products in the artefact, by code (HS2 chapter) and name, in code order."""
        return [SectorOut(code=row["sector"], name=row["sector_name"]) for row in sectors_of(artefacts.products).iter_rows(named=True)]

    @app.get("/markets", response_model=list[MarketOut])
    def markets(artefacts: store.Artefacts = Depends(loaded)):
        """Every Market, by name."""
        return [_pick(MarketOut, row) for row in artefacts.markets.sort("name").iter_rows(named=True)]

    @app.get("/opportunities/rank-by", response_model=list[RankOrderOut])
    def rank_orders():
        """The ranking orders both Opportunity views accept as `rank_by`, in display order."""
        minimums = {
            "min_potential_usd_options": list(MIN_POTENTIAL_USD_OPTIONS),
            "default_min_potential_usd": DEFAULT_MIN_POTENTIAL_USD,
        }
        return [
            RankOrderOut(name=o.name, title=o.title, description=o.description, **(minimums if o.min_potential else {}))
            for o in RANK_BY.values()
        ]

    @app.get("/opportunities/top", response_model=models.Leaderboard)
    def top(
        query: Annotated[models.LeaderboardQuery, Query()],  # pyright: ignore[reportInvalidTypeForm]
        artefacts: store.Artefacts = Depends(loaded),
    ):
        """The Leaderboard: every UK Product–Market pair ranked (or one Market's, or one Product's), with a summary of
        the filtered pairs. An unknown Market or Product answers 404, as does an Excluded Product, with its reason."""
        return models.Leaderboard(**_leaderboard(artefacts, query, models, tiers), rank_by=query.rank_by)

    @app.get("/opportunities/blocked", response_model=models.BlockedOpportunities)
    def blocked(
        query: Annotated[models.BlockedQuery, Query()],  # pyright: ignore[reportInvalidTypeForm]
        artefacts: store.Artefacts = Depends(loaded),
    ):
        """The Blocked Opportunities page: only Blocked Opportunities (default thresholds), ranked by Blocked Opportunity
        value, in Baseline-year figures, with a summary and per-Market totals over every filtered one. An unknown
        Product answers 404, as does an Excluded Product, with its reason."""
        _check_product(artefacts, query.product)
        found = blocked_opportunities(
            artefacts,
            hs6=query.product,
            sector=query.sector,
            region=query.region,
            min_value_usd=query.min_value_usd,
            hide=_hidden(query, tiers),
            limit=query.limit,
        )
        return models.BlockedOpportunities(
            **_metadata(artefacts),
            summary=BlockedSummary(**found.summary),
            markets=[_pick(BlockedMarket, row) for row in found.markets.iter_rows(named=True)],
            opportunities=[_pick(models.BlockedOpportunity, row) for row in found.rows.iter_rows(named=True)],
            thresholds=Thresholds(),
            rank_by=BLOCKED.name,
            order=RankOrderOut(name=BLOCKED.name, title=BLOCKED.title, description=BLOCKED.description),
            note=scenario_note(artefacts),
        )

    @app.get("/opportunities/by-product/{hs6}", response_model=models.ProductOpportunities)
    def by_product(
        hs6: str,
        query: Annotated[models.RankingQuery, Query()],  # pyright: ignore[reportInvalidTypeForm]
        artefacts: store.Artefacts = Depends(loaded),
    ):
        """Markets ranked for one Product."""
        product = _product(artefacts, hs6)
        return models.ProductOpportunities(
            **_ranking(artefacts, query, models.MarketOpportunity, tiers, hs6=hs6), product=product
        )

    @app.get("/opportunities/by-market/{iso3}", response_model=models.MarketOpportunities)
    def by_market(
        iso3: str,
        query: Annotated[models.RankingQuery, Query()],  # pyright: ignore[reportInvalidTypeForm]
        artefacts: store.Artefacts = Depends(loaded),
    ):
        """Products ranked for one Market."""
        market = _market(artefacts, iso3)
        return models.MarketOpportunities(
            **_ranking(artefacts, query, models.ProductOpportunity, tiers, iso3=iso3),
            market=market,
            market_reach=_market_reach(artefacts, market),
        )

    @app.get("/evaluation", response_model=Evaluation)
    def evaluation(artefacts: store.Artefacts = Depends(loaded)):
        """The evaluation artefact: test-year accuracy, breakdowns, stability, and the known limitations.

        The v1/v2 comparison stays in the artefact and the model card, never in the API (spec #39).
        """
        if artefacts.evaluation is None:
            raise HTTPException(status_code=404, detail=f"Artefact version {artefacts.version} has no evaluation.")
        return Evaluation(
            **served_report(artefacts.evaluation),
            artefact_version=artefacts.version,
            baseline_year=artefacts.baseline_year,
            limitations=known_limitations(artefacts),
            model_choice=how_chosen(artefacts.metadata, artefacts.evaluation),
            event_backtest_note=event_backtest_note(artefacts.metadata, artefacts.evaluation),
            large_markets_by_version=large_markets_beside_earlier(artefacts.evaluation),
            not_run=parts_not_run(artefacts.evaluation, artefacts.ppml_benchmark),
        )

    @app.get("/explain/{hs6}/{iso3}", response_model=models.Explanation)
    def explain(hs6: str, iso3: str, artefacts: store.Artefacts = Depends(loaded)):
        """Why? one pair's Export Potential: SHAP factors from its own fold model, plus history."""
        product, market = _product(artefacts, hs6), _market(artefacts, iso3)
        try:
            explanation = explain_pair(artefacts, hs6, iso3)
        except UnknownPair as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        values = (
            artefacts.baseline.filter((pl.col("hs6") == hs6) & (pl.col("iso3") == iso3))
            .select(
                "export_potential_usd",
                "actual_exports_usd",
                "export_gap_usd",
                "baseline_export_potential_usd",
                "relative_export_gap",
                "opportunity_type",
                *(column for tier in tiers for column in tier.columns),
                "potential_range_low_usd",
                "potential_range_high_usd",
                "uncertain_opportunity",
            )
            .row(0, named=True)
        )
        return models.Explanation(
            **_metadata(artefacts),
            **values,
            potential_range_text=potential_range_text(
                values["potential_range_low_usd"],
                values["potential_range_high_usd"],
                values["actual_exports_usd"],
                uncertain=values["uncertain_opportunity"],
            ),
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
            history=[HistoryOut(**vars(h)) for h in explanation.history],
            tariff_caveat=explanation.tariff_caveat,
            market_reach=_pair_market_reach(artefacts, market, explanation.fold),
        )

    @app.get("/scenarios/presets", response_model=list[PresetOut])
    def presets():
        """The Trading Bloc presets, in display order."""
        return [PresetOut(**vars(preset)) for preset in scenarios.PRESETS.values()]

    @app.get("/scenarios/thresholds", response_model=Thresholds)
    def thresholds():
        """The default Blocked Opportunity thresholds."""
        return Thresholds()

    @app.post("/scenarios", response_model=models.ScenarioResponse)
    def run_scenario(request: ScenarioRequest, artefacts: store.Artefacts = Depends(loaded)):
        """Re-predict the Baseline with a preset, custom Trade Agreements or a tariff change (422 if invalid)."""
        if request.market is not None and artefacts.markets.filter(pl.col("iso3") == request.market).is_empty():
            raise HTTPException(status_code=422, detail=f"Unknown Market: {request.market}")
        try:
            result = scenarios.run(
                artefacts.baseline_features,
                artefacts.baseline,
                artefacts.fold_models,
                request.to_scenario(),
                tariff_range=scenarios.tariff_range(artefacts.metadata),
                form=artefacts.model_form,
            )
        except scenarios.InvalidScenario as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return scenario_response(result, request, artefacts, models, tiers)

    return app


def _pick(model: type[BaseModel], row: dict) -> BaseModel:
    """`model` built from the fields it declares, taken from a table row (fields the row lacks take their default)."""
    return model(**{name: row[name] for name in model.model_fields if name in row})


def _product(artefacts: store.Artefacts, hs6: str) -> ProductOut:
    """The Product, or 404: for an unknown Product, or an Excluded Product (with its reason)."""
    found = artefacts.products.filter(pl.col("hs6") == hs6)
    if found.is_empty():
        raise HTTPException(status_code=404, detail=f"Unknown Product: {hs6}")
    reason = not_ranked_message(artefacts.excluded_products, hs6)
    if reason is not None:
        raise HTTPException(status_code=404, detail=reason)
    return _pick(ProductOut, with_sector_names(found).row(0, named=True))


def _check_product(artefacts: store.Artefacts, hs6: str | None) -> None:
    """A read's one-Product filter: 404 for an unknown or Excluded Product (with its reason); nothing when unset."""
    if hs6 is not None:
        _product(artefacts, hs6)


def _market(artefacts: store.Artefacts, iso3: str) -> MarketOut:
    """The Market, or 404."""
    found = artefacts.markets.filter(pl.col("iso3") == iso3)
    if found.is_empty():
        raise HTTPException(status_code=404, detail=f"Unknown Market: {iso3}")
    return _pick(MarketOut, found.row(0, named=True))


def _metadata(artefacts: store.Artefacts) -> dict:
    """The response metadata every value-carrying response shares: version, Baseline year and Gap Period."""
    return {
        "artefact_version": artefacts.version,
        "baseline_year": artefacts.baseline_year,
        "gap_years": list(artefacts.gap_years),
        "gap_period_label": gap_period_label(artefacts.gap_years),
    }


def _hidden(query: BaseModel, tiers: tuple[CaveatTier, ...]) -> list[str]:
    """The columns of the Caveat Tiers whose hide filter is on in `query`."""
    return [tier.column for tier in tiers if getattr(query, tier.hide_param)]


def _filters(query: RankingQuery, tiers: tuple[CaveatTier, ...]) -> dict:
    """The package's `rank_opportunities` arguments for a ranking query's filters and order."""
    return {
        "sector": query.sector,
        "region": query.region,
        "opportunity_type": query.opportunity_type,
        "min_gap_usd": query.min_gap_usd,
        "trade_agreement": query.trade_agreement,
        "hide": _hidden(query, tiers),
        "rank_by": query.rank_by,
        "min_potential_usd": query.min_potential_usd,
    }


def _market_reach(artefacts: store.Artefacts, market: MarketOut) -> MarketReachOut | None:
    """A Market's Market Reach for the Market view (ADR 0010); None where the artefact has none for it."""
    reach = artefacts.model_form.market_reach
    if reach is None or reach.table is None:
        return None
    figures = market_reach.market_figures(
        reach.table, market.iso3, market=market.name, year=reach.reference_year, n_folds=len(artefacts.fold_models)
    )
    return None if figures is None else MarketReachOut(**figures)


def _pair_market_reach(artefacts: store.Artefacts, market: MarketOut, fold: int) -> PairMarketReachOut | None:
    """`_market_reach` with the offset the pair's own fold applies, for the pair detail."""
    reach = artefacts.model_form.market_reach
    if reach is None or reach.table is None:
        return None
    figures = market_reach.pair_figures(
        reach.table, market.iso3, fold, market=market.name, year=reach.reference_year, beta=reach.beta,
        n_folds=len(artefacts.fold_models),
    )
    return None if figures is None else PairMarketReachOut(**figures)


def _leaderboard(artefacts: store.Artefacts, query: LeaderboardQuery, models: Tiered, tiers: tuple[CaveatTier, ...]) -> dict:
    """The Leaderboard's fields: metadata, summary and top rows, and the filtered Market's reach."""
    reach = None
    if query.market is not None:
        reach = _market_reach(artefacts, _market(artefacts, query.market))
    _check_product(artefacts, query.product)
    ranked = ranking(artefacts, hs6=query.product, iso3=query.market, limit=query.limit, **_filters(query, tiers))
    return {
        **_metadata(artefacts),
        "summary": LeaderboardSummary(**ranked.summary),
        "opportunities": [_pick(models.LeaderboardOpportunity, row) for row in ranked.rows.iter_rows(named=True)],
        "market_reach": reach,
    }


def _ranking(
    artefacts: store.Artefacts,
    query: RankingQuery,
    row_model: type[BaseModel],
    tiers: tuple[CaveatTier, ...],
    **subject,
) -> dict:
    """The fields the by-Product and by-Market views share: metadata and the ranked rows."""
    rows = rank_opportunities(artefacts, **subject, **_filters(query, tiers), limit=query.limit)
    return {
        **_metadata(artefacts),
        "rank_by": query.rank_by,
        "opportunities": [_pick(row_model, row) for row in rows.iter_rows(named=True)],
    }


def scenario_response(
    result: scenarios.ScenarioResult,
    request: ScenarioRequest,
    artefacts: store.Artefacts,
    models: Tiered,
    tiers: tuple[CaveatTier, ...],
) -> ScenarioResponse:
    """Totals, per-Market and per-Product effects, top pairs and Blocked Opportunities, in nominal USD; each pair
    carries its Caveat Tier columns from the Baseline."""
    markets = artefacts.markets.select("iso3", market="name")
    products = artefacts.products.select("hs6", "description")
    caveats = artefacts.baseline.select("hs6", "iso3", *(column for tier in tiers for column in tier.columns))
    pairs = (
        result.pairs.with_columns(
            blocked_opportunity=scenarios.is_blocked(
                "effect_usd", "baseline_potential_usd", request.min_effect_share, request.min_effect_usd
            )
        )
        .join(markets, on="iso3", how="left", maintain_order="left")
        .join(products, on="hs6", how="left", maintain_order="left")
        .join(caveats, on=["hs6", "iso3"], how="left", maintain_order="left")
    )
    blocked = pairs.filter("blocked_opportunity")
    top = pairs if request.market is None else pairs.filter(pl.col("iso3") == request.market)
    total = result.total()

    def shown(rows: pl.DataFrame) -> list[dict]:
        """The top `limit` rows, each with its Baseline-year Actual Exports."""
        return with_baseline_actual_exports(
            rows.head(request.limit), artefacts.history, artefacts.baseline_year
        ).to_dicts()

    return models.ScenarioResponse(
        **_metadata(artefacts),
        scenario=ScenarioIn.model_validate(asdict(result.scenario)),
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
        top_pairs=shown(top),
        blocked_opportunities=shown(blocked),
        blocked_count=blocked.height,
        clipped_pairs=result.clipped_pairs,
        note=scenario_note(artefacts),
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
        state.artefacts = store.load_latest(state.artefacts_root, tiers=state.tiers)
    except (store.ArtefactsNotFound, store.UnreadableArtefacts) as error:
        state.load_error = str(error)
        log.warning("%s; retrying at most every %.0f s", error, state.retry_seconds)
        return None
    state.load_error = None
    return state.artefacts


app = create_app()
