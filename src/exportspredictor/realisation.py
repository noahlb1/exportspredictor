"""ITC's Realisation Test run on our model (#156): does the Export Gap come true?

ITC supports its Export Potential Map (EPM) with a 3-year realisation test (ITC, *Export Potential Assessment
methodology*, June 2023, pp. 17–22, after Périllat 2018). It regresses later trade on its indicator and reports
the coefficients. This module runs the same linear regressions on UK Export Potential, as Lift's regression
counterpart: Lift asks with deciles whether large Export Gaps point at growth, this asks it with ITC's regression.

Three tests, each a linear regression with Market and year × Sector fixed effects, swept out, and errors clustered
by unit (`ppml.fit_fixed_effects_ols`); ITC's exporter fixed effect drops out, because there is one exporter.

**The intensive margin.** For each base year t and each unit with Actual Exports at t:

    log(mean Actual Exports over t+1..t+3 / Actual Exports at t)
        = b · Unrealised Share at t + c · log(Actual Exports at t / mean over t−3..t−1) + Market + year × Sector

The **Unrealised Share** is ITC's floored share, (Export Potential − min(Actual Exports, Export Potential)) / Export
Potential, from year-t values: between 0 and 1, and not the signed Export Gap. Units with no exports in t+1..t+3
drop out, as in ITC, and so do units with none in t−3..t−1, whose past growth is undefined; both are counted.

**The extensive margin**, a linear probability model. For units with no Actual Exports at t, among Products the UK
exports to some Market at t (read from the panel itself): whether any of t+1..t+3 is above zero, on log Export
Potential at t. **New-flow size**: for those that start, the log of their summed Actual Exports over t+1..t+3 on
log Export Potential at t.

**Levels.** A unit is a Product–Market pair (HS6) or a Sector–Market pair (HS2). At HS2, ITC's test level being 76
sub-sectors, Export Potential, Actual Exports and the unrealised potential are summed over the Sector's scored pairs,
the unrealised potential floored at zero pair by pair before summing, so one Product's surplus cannot cancel another's
unrealised potential. The HS2 rows are the ones set beside ITC's figures; the HS6 rows are reported, not compared.

**Predictors.** Each test runs on the model's Export Potential and, through the same code, on the Naive Baseline in
its place (`PREDICTORS`). Each predictor runs on the units it scores above zero (a log needs it), so the model's
row keeps every unit it scores, as ITC's does, and the Naive Baseline's leaves out the units it scores 0; each row
counts the unit-years it left out (`zero_score`), so the two samples differ visibly.

**The panel** (`realisation_report`'s input, `PANEL_SCHEMA`) holds one row per UK pair and year: `hs6`, `iso3`,
`year`, `export_potential_usd` and `naive_baseline_usd` (non-null in base years only, each scored by fold models
trained on years up to that base year, #176), and `actual_exports_usd` (0 where no Flow). Base years are the years
with Export Potential. A panel reaching the test year is refused: the test never reads it, so nothing goes through
the Test-year Ledger (ADR 0009), and the result feeds no modelling choice.

**The report** is JSON-ready and saved in the artefact's evaluation as `itc_comparison`. Its `rows` are keyed by
`test`, `level` and `predictor`. The result is a rough benchmark of agreement, never a validation. Its tables and
Advisor wording are made from it by `realisation_wording`.

`python -m exportspredictor.realisation` adds the report, from the base-year panel (`base_years.load_panel`), to the
newest artefact's evaluation as a new version.
"""

import math
import time
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from exportspredictor import ppml
from exportspredictor.ledger import TEST_YEAR
from exportspredictor.sectors import sector_of

if TYPE_CHECKING:
    from exportspredictor.artefacts import Artefacts

REPORT_KEY = "itc_comparison"  # the evaluation report's block
PAIR = ("hs6", "iso3")
PANEL_SCHEMA = {
    "hs6": pl.String,
    "iso3": pl.String,
    "year": pl.Int16,
    "export_potential_usd": pl.Float64,
    "naive_baseline_usd": pl.Float64,
    "actual_exports_usd": pl.Float64,
}
WINDOW = 3  # years after the base year for growth, and before it for past growth
INTENSIVE_MARGIN = "intensive_margin"
EXTENSIVE_MARGIN = "extensive_margin"
NEW_FLOW_SIZE = "new_flow_size"
TESTS = (INTENSIVE_MARGIN, EXTENSIVE_MARGIN, NEW_FLOW_SIZE)
HS6 = "HS6"
HS2 = "HS2"
# ITC's static realisation test with fixed effects pools every exporter over these years, at this level.
ITC_YEARS = "2006–2018"
ITC_LEVEL = "76 sub-sectors"
# Each level's unit; the HS2 level is the one compared with ITC's level (`ITC_LEVEL`), so it comes first.
LEVELS = {HS2: ("sector", "iso3"), HS6: PAIR}
COMPARED_LEVEL = HS2
# The predictors, and the panel column each one's scores come from.
MODEL = "model"
NAIVE_BASELINE = "naive_baseline"
PREDICTORS = {MODEL: "export_potential_usd", NAIVE_BASELINE: "naive_baseline_usd"}
# The tests' regressors and control, as sample columns and report terms.
UNREALISED_SHARE = "unrealised_share"
PAST_GROWTH = "past_growth"
LOG_EXPORT_POTENTIAL = "log_export_potential"
FIXED_EFFECTS = (("iso3",), ("year", "sector"))
FIXED_EFFECTS_TEXT = ("Market", "year × Sector")
CLUSTER_TEXT = "Product–Market pair (Sector–Market pair at HS2)"

METHODS_PAPER = (
    "ITC, Export Potential Assessment methodology (June 2023), {table}, pp. 17–22; "
    "https://umbraco.exportpotential.intracen.org/media/cklh2pi5/epa-methodology_230627.pdf"
)
# ITC's published coefficients for that test (`ITC_YEARS`, `ITC_LEVEL`).
PUBLISHED = (
    {
        "test": INTENSIVE_MARGIN,
        "regressor": UNREALISED_SHARE,
        "coefficient": 0.615,
        "controls": {PAST_GROWTH: -0.386},
        "level": ITC_LEVEL,
        "source": METHODS_PAPER.format(table="Table 1"),
    },
    {
        "test": EXTENSIVE_MARGIN,
        "regressor": LOG_EXPORT_POTENTIAL,
        "coefficient": 0.023,
        "controls": {},
        "level": ITC_LEVEL,
        "source": METHODS_PAPER.format(table="Table 2"),
    },
    {
        "test": NEW_FLOW_SIZE,
        "regressor": LOG_EXPORT_POTENTIAL,
        "coefficient": 0.243,
        "controls": {},
        "level": ITC_LEVEL,
        "source": METHODS_PAPER.format(table="Table 4"),
    },
)

# Part (c) of #156, the direct comparison with the EPM's UK figures, was dropped at step one (#174); #178 words the
# reason in full for the notebook.
DIRECT_COMPARISON_DROPPED = (
    "dropped: the EPM's public site offers no tabular download, ITC's terms forbid extracting its data by "
    "automated means and allow only limited quotation without written permission, and its figures are a 2031 "
    "estimate from 2021–2025 data, while we score each base year itself"
)


def unrealised_potential(potential: pl.Expr, actual: pl.Expr) -> pl.Expr:
    """ITC's unrealised potential, floored at zero: potential − min(actual, potential)."""
    return potential - pl.min_horizontal(actual, potential)


def base_years(panel: pl.DataFrame, predictor: str = PREDICTORS[MODEL]) -> list[int]:
    """The years in which `panel` holds scores of `predictor`."""
    return sorted(int(y) for y in panel.filter(pl.col(predictor).is_not_null())["year"].unique())


def _unrealised_column(score: str) -> str:
    return f"unrealised_{score}"  # e.g. unrealised_export_potential_usd


_SCORES = tuple(PREDICTORS.values())
_SUMMED = (*_SCORES, "actual_usd", "later_usd", "earlier_usd", *map(_unrealised_column, _SCORES))


def _key(level: str) -> list[str]:
    """A unit-year's columns at `level`, with its Sector for the year × Sector fixed effect."""
    return list(dict.fromkeys([*LEVELS[level], "year", "sector"]))


def _windows(panel: pl.DataFrame) -> pl.DataFrame:
    """Per base year and scored pair: each predictor's score, Actual Exports at t, their means before and after, and
    each predictor's floored unrealised potential at t."""
    years = {int(y) for y in panel["year"].unique()}
    actual, year = pl.col("actual_exports_usd"), pl.col("year")
    frames = []
    for t in base_years(panel):
        missing = sorted(set(range(t - WINDOW, t + WINDOW + 1)) - years)
        if missing:
            raise ValueError(f"Base year {t} needs Actual Exports for {missing}, which the panel does not hold")
        frames.append(
            panel.filter(year.is_between(t - WINDOW, t + WINDOW))
            .group_by(*PAIR)
            .agg(
                *(pl.col(score).filter(year == t).first() for score in _SCORES),
                actual_usd=actual.filter(year == t).first(),
                later_usd=actual.filter(year.is_between(t + 1, t + WINDOW)).mean(),
                earlier_usd=actual.filter(year.is_between(t - WINDOW, t - 1)).mean(),
            )
            .filter(pl.any_horizontal(pl.col(_SCORES).is_not_null()))
            .with_columns(year=pl.lit(t, dtype=pl.Int16))
        )
    return pl.concat(frames).with_columns(
        sector=sector_of("hs6"),
        **{
            _unrealised_column(score): unrealised_potential(pl.col(score), pl.col("actual_usd"))
            for score in _SCORES
        },
    )


def _units(panel: pl.DataFrame, level: str) -> pl.DataFrame:
    """Per base year and unit of `level` (`LEVELS`): each predictor's score and floored unrealised potential, Actual
    Exports at t and their means before and after, and whether the UK exports the unit's Product (or Sector) to some
    Market at t.

    Each figure is summed over the unit's scored pairs (one pair at HS6), the unrealised potential floored pair by
    pair before summing."""
    product = LEVELS[level][0]
    exported = (
        panel.filter(pl.col("actual_exports_usd") > 0)
        .with_columns(sector=sector_of("hs6"))
        .select(product, "year")
        .unique()
        .with_columns(exported=pl.lit(True))
    )
    return (
        _windows(panel)
        .group_by(_key(level))
        .agg(pl.col(_SUMMED).sum())
        .join(exported, on=[product, "year"], how="left")
        .with_columns(pl.col("exported").fill_null(False))
        .sort(_key(level))  # a fixed order, so a report is reproducible to the last digit
    )


def _of(units: pl.DataFrame, predictor: str) -> pl.DataFrame:
    """`units` with `predictor`'s score as `potential_usd`, its unrealised potential as `unrealised_usd`, and whether
    it scores the unit above zero (`scored`): each predictor's sample is its own."""
    score = PREDICTORS[predictor]
    return units.with_columns(
        potential_usd=pl.col(score),
        unrealised_usd=pl.col(_unrealised_column(score)),
        scored=(pl.col(score) > 0).fill_null(False),
    )


def _trading(units: pl.DataFrame) -> pl.DataFrame:
    return units.filter(pl.col("actual_usd") > 0)


def _not_trading(units: pl.DataFrame) -> pl.DataFrame:
    return units.filter(pl.col("actual_usd") == 0)


def _starting_candidates(units: pl.DataFrame) -> pl.DataFrame:
    return _not_trading(units).filter(pl.col("exported") & pl.col("scored"))


def _intensive(units: pl.DataFrame, level: str) -> pl.DataFrame:
    return (
        _trading(units)
        .filter(pl.col("scored") & (pl.col("later_usd") > 0) & (pl.col("earlier_usd") > 0))
        .with_columns(
            growth=(pl.col("later_usd") / pl.col("actual_usd")).log(),
            **{
                UNREALISED_SHARE: pl.col("unrealised_usd") / pl.col("potential_usd"),
                PAST_GROWTH: (pl.col("actual_usd") / pl.col("earlier_usd")).log(),
            },
        )
        .select(*_key(level), "growth", UNREALISED_SHARE, PAST_GROWTH)
    )


def _extensive(units: pl.DataFrame, level: str) -> pl.DataFrame:
    return _starting_candidates(units).select(
        *_key(level),
        started=(pl.col("later_usd") > 0).cast(pl.Float64),
        **{LOG_EXPORT_POTENTIAL: pl.col("potential_usd").log()},
    )


def _new_flow(units: pl.DataFrame, level: str) -> pl.DataFrame:
    return _starting_candidates(units).filter(pl.col("later_usd") > 0).select(
        *_key(level),
        size=(pl.col("later_usd") * WINDOW).log(),
        **{LOG_EXPORT_POTENTIAL: pl.col("potential_usd").log()},
    )


def _sample(sample_of, panel: pl.DataFrame, predictor: str, level: str) -> pl.DataFrame:
    return sample_of(_of(_units(panel, level), predictor), level)


def intensive_margin_sample(panel: pl.DataFrame, *, predictor: str, level: str = HS6) -> pl.DataFrame:
    """The intensive margin's rows: units trading at a base year with exports both before and after it.

    One row per unit and base year with `growth`, `unrealised_share` and `past_growth`, plus `sector` (HS2) for the
    fixed effects. `predictor` (a key of `PREDICTORS`) names the score the Unrealised Share is computed from; it must
    score the unit above zero.
    """
    return _sample(_intensive, panel, predictor, level)


def extensive_margin_sample(panel: pl.DataFrame, *, predictor: str, level: str = HS6) -> pl.DataFrame:
    """The extensive margin's rows: units with no Actual Exports at a base year whose Product (Sector at HS2) the UK
    exports to some Market that year, with `started` (1 if any of the next three years is above zero) and
    `log_export_potential`, the log of `predictor`'s score."""
    return _sample(_extensive, panel, predictor, level)


def new_flow_size_sample(panel: pl.DataFrame, *, predictor: str, level: str = HS6) -> pl.DataFrame:
    """New-flow size's rows: the extensive margin's units that start, with `size`, the log of their Actual Exports
    summed over the next three years, and `log_export_potential`."""
    return _sample(_new_flow, panel, predictor, level)


def _number(value: float) -> float | None:
    return None if value is None or not math.isfinite(value) else float(value)


def _fit(value: float, error: float) -> dict:
    return {"coefficient": _number(value), "std_error": _number(error)}


def _intensive_counts(units: pl.DataFrame) -> dict:
    """How the units reached the intensive margin's sample: who was trading at t, and who dropped out and why."""
    trading = _trading(units)
    scored = trading.filter(pl.col("scored"))
    return {
        "trading": trading.height,  # unit-years trading at the base year
        "zero_score": int((~trading["scored"]).sum()),  # left out: scored 0, so no Unrealised Share
        "no_later_exports": int((scored["later_usd"] == 0).sum()),
        "no_earlier_exports": int(((scored["later_usd"] > 0) & (scored["earlier_usd"] == 0)).sum()),
    }


def _starting_counts(units: pl.DataFrame) -> dict:
    """How the units reached the extensive margin's sample: who was not trading at t, and who was left out and why."""
    not_trading = _not_trading(units)
    return {
        "not_trading": not_trading.height,  # unit-years with no Actual Exports at the base year
        "product_not_exported": int((~not_trading["exported"]).sum()),  # left out: Product sold nowhere at t
        "zero_score": int((not_trading["exported"] & ~not_trading["scored"]).sum()),  # left out: scored 0, so no log
        "started": int((_starting_candidates(units)["later_usd"] > 0).sum()),
    }


# Each test: its sample, its outcome, its regressor and controls, and how units reached its sample.
_TESTS = {
    INTENSIVE_MARGIN: (_intensive, "growth", UNREALISED_SHARE, (PAST_GROWTH,), _intensive_counts),
    EXTENSIVE_MARGIN: (_extensive, "started", LOG_EXPORT_POTENTIAL, (), _starting_counts),
    NEW_FLOW_SIZE: (_new_flow, "size", LOG_EXPORT_POTENTIAL, (), _starting_counts),
}


def _row(units: pl.DataFrame, test: str, level: str, predictor: str) -> dict:
    """One test (`TESTS`) on the units of one level (`LEVELS`) and predictor (`PREDICTORS`), as one report row."""
    sample_of, outcome, regressor, controls, counts = _TESTS[test]
    scored = _of(units, predictor)
    fit = ppml.fit_fixed_effects_ols(
        sample_of(scored, level),
        outcome=outcome,
        regressors=[regressor, *controls],
        fixed_effects=FIXED_EFFECTS,
        cluster=LEVELS[level],
    )
    return {
        "test": test,
        "level": level,
        "predictor": predictor,
        "regressor": regressor,
        **_fit(fit.coefficients[regressor], fit.std_errors[regressor]),
        "controls": {name: _fit(fit.coefficients[name], fit.std_errors[name]) for name in controls},
        **counts(scored),
        "observations": fit.rows,
        "singletons_dropped": fit.dropped_rows,
        "pairs": fit.clusters,
        "absorbed": list(fit.absorbed),
    }


def realisation_report(panel: pl.DataFrame, *, training_cut_offs: Mapping[int, int] | None = None) -> dict:
    """ITC's realisation test on a scored UK panel (see the module docstring), JSON-ready.

    `training_cut_offs` maps each base year to the last year its fold models trained on (`base_years`' records); by
    default it is the base year itself, as the panel's contract states. A cut-off after its base year is refused.

    `rows` holds one entry per test, level and predictor; `published` ITC's own coefficients, cited;
    `direct_comparison` whether part (c) of #156 ran, and why not.
    """
    if panel.height and int(panel["year"].max()) >= TEST_YEAR:  # pyright: ignore[reportArgumentType]
        raise ValueError(f"The panel reaches the test year ({TEST_YEAR}); the realisation test never reads it")
    years = base_years(panel)
    cut_offs = {t: t for t in years} if training_cut_offs is None else {int(t): int(c) for t, c in training_cut_offs.items()}
    if sorted(cut_offs) != years:
        raise ValueError(f"Training cut-offs are given for {sorted(cut_offs)}, but the base years are {years}")
    late = {t: c for t, c in cut_offs.items() if c > t}
    if late:
        raise ValueError(f"Base-year models trained on years after their base year: {late}")
    units = {level: _units(panel, level) for level in LEVELS}
    return {
        "rows": [
            _row(units[level], test, level, predictor)
            for test in TESTS
            for level in LEVELS
            for predictor in PREDICTORS
        ],
        "base_years": years,
        "training_cut_offs": [{"base_year": t, "training_cut_off": cut_offs[t]} for t in years],
        "outcome_years": [min(years) - WINDOW, max(years) + WINDOW],
        "fixed_effects": list(FIXED_EFFECTS_TEXT),
        "cluster": CLUSTER_TEXT,
        "compared_level": COMPARED_LEVEL,
        "published": [dict(row) for row in PUBLISHED],
        "direct_comparison": {"run": False, "reason": DIRECT_COMPARISON_DROPPED},
    }


def with_itc_comparison(bundle: "Artefacts", report: dict) -> "Artefacts":
    """`bundle` with `report` in its evaluation (`REPORT_KEY`); every other block is kept."""
    return replace(bundle, evaluation={**(bundle.evaluation or {}), REPORT_KEY: report})


# --- Fixture mode ----------------------------------------------------------------------------------------------

FIXTURE_YEARS = range(2012, 2023)
FIXTURE_BASE_YEARS = range(2015, 2020)


def fixture_panel(*, effect: float = 0.6, seed: int = 0, products: int = 60, markets: int = 12) -> pl.DataFrame:
    """A synthetic scored UK panel in which a pair's growth over a year rises by `effect` × its Unrealised Share.

    Actual Exports follow a random walk in logs, and about 3% of pair-years have no Flow. Each year's Export
    Potential is Actual Exports times a log-normal factor, so the share is above 0 for about half the pairs; it is
    kept only in `FIXTURE_BASE_YEARS`, as the real panel's is. The Naive Baseline is Export Potential with noise.
    With `effect=0` the share says nothing about growth.

    For the extensive margin, about 40% of Sector–Market cells start with no Flow and open in a later year, sooner
    where their Export Potential is larger, so both HS6 pairs and HS2 cells start trading.
    """
    rng = np.random.default_rng(seed)
    sectors = 6
    pairs = [(f"{1 + p % sectors:02d}{p:04d}", f"M{m:02d}") for p in range(products) for m in range(markets)]
    n, years = len(pairs), list(FIXTURE_YEARS)
    log_actual = np.empty((n, len(years)))
    potential = np.empty((n, len(years)))
    log_actual[:, 0] = rng.normal(13, 1.5, n)
    for k in range(len(years)):
        factor = rng.normal(0, 0.7, n)
        potential[:, k] = np.exp(log_actual[:, k] + factor)
        share = np.maximum(0.0, 1 - np.exp(-factor))
        if k + 1 < len(years):
            log_actual[:, k + 1] = log_actual[:, k] + effect * share + rng.normal(0, 0.3, n)
    actual = np.where(rng.random((n, len(years))) < 0.03, 0.0, np.exp(log_actual))
    # Dormant Sector–Market cells: closed until an opening year drawn with a yearly chance rising with their level.
    cell = np.array([(p % sectors) * markets + m for p in range(products) for m in range(markets)])
    level = np.bincount(cell, weights=log_actual[:, 0]) / np.bincount(cell)
    chance = 1 / (1 + np.exp(-(2 * (level - level.mean()) / level.std() - 1)))
    opens = np.where(rng.random(level.size) < 0.4, 1 + rng.geometric(chance), 0)
    actual[np.arange(len(years))[None, :] < opens[cell][:, None]] = 0.0
    naive = potential * np.exp(rng.normal(0, 0.3, potential.shape))
    scored = np.isin(years, list(FIXTURE_BASE_YEARS))
    return pl.DataFrame(
        {
            "hs6": [hs6 for hs6, _ in pairs for _ in years],
            "iso3": [iso3 for _, iso3 in pairs for _ in years],
            "year": years * n,
            "export_potential_usd": np.where(scored, potential, np.nan).ravel(),
            "naive_baseline_usd": np.where(scored, naive, np.nan).ravel(),
            "actual_exports_usd": actual.ravel(),
        },
        schema=PANEL_SCHEMA,
    ).with_columns(pl.col(_SCORES).fill_nan(None))


# --- Adding the report to a saved artefact ---------------------------------------------------------------------


def add_to_latest(
    artefacts_root: Path | None = None,
    *,
    panel: pl.DataFrame | None = None,
    root: Path | None = None,
    version: str | None = None,
) -> Path:
    """Run the realisation test and save it in the newest artefact's evaluation, as a new version.

    `panel` defaults to the real base-year panel (`base_years.load_panel`, under the data `root`), whose record must
    name the newest artefact or one it was copied from (`base_years.check_panel_version`), and whose base-year models'
    training cut-offs go into the report. Models and tables are copied byte for byte (`artefacts.copy_version`), and
    the rest of the evaluation is kept; nothing is trained, and the test year is never read, so the Test-year Ledger
    is not written. Returns the new folder.
    """
    # Imported here: `base_years` trains through `evaluation`, which serves this module's report.
    from exportspredictor import artefacts as store
    from exportspredictor import base_years as panels
    from exportspredictor.paths import artefacts_dir

    artefacts_root = Path(artefacts_root or artefacts_dir())
    source = store.latest_folder(artefacts_root)
    bundle = store.load(source)
    if bundle.evaluation is None:
        raise ValueError(f"Artefact {bundle.version} has no evaluation; run `python -m exportspredictor.evaluation` first")
    cut_offs = None
    if panel is None:
        info = panels.load_panel_info(root)
        panels.check_panel_version(info, artefacts_root, bundle.version)
        panel = panels.load_panel(root)
        cut_offs = panels.training_cut_offs(info)
    started = time.perf_counter()
    report = realisation_report(panel, training_cut_offs=cut_offs)
    report["seconds"] = round(time.perf_counter() - started, 2)
    metadata = {**bundle.metadata, "version": version or store.new_version(), "copied_from": bundle.version}
    return store.copy_version(
        source, artefacts_root, metadata=metadata, evaluation=with_itc_comparison(bundle, report).evaluation
    )


if __name__ == "__main__":
    print(add_to_latest(), flush=True)
