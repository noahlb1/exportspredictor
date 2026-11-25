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
its place (`PREDICTORS`).

**The panel** (`realisation_report`'s input) holds one row per UK pair and year: `hs6`, `iso3`, `year`,
`export_potential` and `naive_baseline` (non-null in base years only, each scored by fold models trained on years up
to that base year, #176), and `actual_exports` (0 where no Flow). Base years are the years with Export Potential.
A panel reaching the test year is refused: the test never reads it, so nothing goes through the Test-year Ledger
(ADR 0009), and the result feeds no modelling choice.

**The report** is JSON-ready and saved in the artefact's evaluation as `itc_comparison`. Its `rows` are keyed by
`test`, `level` and `predictor`. The result is a rough benchmark of agreement, never a validation.

`python -m exportspredictor.realisation` adds the report, from the base-year panel (`base_years.load_panel`), to the
newest artefact's evaluation as a new version.
"""

import importlib
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from exportspredictor import ppml
from exportspredictor.experiments import TEST_YEAR
from exportspredictor.wording import number_text, years_text

if TYPE_CHECKING:
    from exportspredictor.artefacts import Artefacts

REPORT_KEY = "itc_comparison"  # the evaluation report's block
PAIR = ("hs6", "iso3")
WINDOW = 3  # years after the base year for growth, and before it for past growth
INTENSIVE_MARGIN = "intensive_margin"
EXTENSIVE_MARGIN = "extensive_margin"
NEW_FLOW_SIZE = "new_flow_size"
TESTS = (INTENSIVE_MARGIN, EXTENSIVE_MARGIN, NEW_FLOW_SIZE)
HS6 = "HS6"
HS2 = "HS2"
# Each level's unit; the HS2 level is the one compared with ITC's 76 sub-sectors, so it comes first.
LEVELS = {HS2: ("sector", "iso3"), HS6: PAIR}
COMPARED_LEVEL = HS2
# The panel column each predictor's scores come from.
PREDICTORS = {"model": "export_potential", "naive_baseline": "naive_baseline"}
FIXED_EFFECTS = (("iso3",), ("year", "sector"))
FIXED_EFFECTS_TEXT = ("Market", "year × Sector")
CLUSTER_TEXT = "Product–Market pair (Sector–Market pair at HS2)"

METHODS_PAPER = (
    "ITC, Export Potential Assessment methodology (June 2023), {table}, pp. 17–22; "
    "https://umbraco.exportpotential.intracen.org/media/cklh2pi5/epa-methodology_230627.pdf"
)
# ITC's static realisation test with fixed effects: every exporter pooled, 2006–2018, 76 sub-sectors.
PUBLISHED = (
    {
        "test": INTENSIVE_MARGIN,
        "regressor": "unrealised_share",
        "coefficient": 0.615,
        "controls": {"past_growth": -0.386},
        "level": "76 sub-sectors",
        "source": METHODS_PAPER.format(table="Table 1"),
    },
    {
        "test": EXTENSIVE_MARGIN,
        "regressor": "log_export_potential",
        "coefficient": 0.023,
        "controls": {},
        "level": "76 sub-sectors",
        "source": METHODS_PAPER.format(table="Table 2"),
    },
    {
        "test": NEW_FLOW_SIZE,
        "regressor": "log_export_potential",
        "coefficient": 0.243,
        "controls": {},
        "level": "76 sub-sectors",
        "source": METHODS_PAPER.format(table="Table 4"),
    },
)
TEST_NAMES = {
    INTENSIVE_MARGIN: "Intensive margin",
    EXTENSIVE_MARGIN: "Extensive margin",
    NEW_FLOW_SIZE: "New-flow size",
}
TERM_NAMES = {
    "unrealised_share": "Unrealised Share",
    "past_growth": "past growth",
    "log_export_potential": "log Export Potential",
}
PREDICTOR_NAMES = {"model": "Model", "naive_baseline": "Naive Baseline"}
ITC_COLUMN = "ITC (76 sub-sectors)"  # the coefficient table's column of ITC's published figures

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


def unrealised_share(potential: pl.Expr, actual: pl.Expr) -> pl.Expr:
    """ITC's floored share: (potential − min(actual, potential)) / potential, between 0 and 1."""
    return unrealised_potential(potential, actual) / potential


def base_years(panel: pl.DataFrame, predictor: str = "export_potential") -> list[int]:
    """The years in which `panel` holds scores of `predictor`."""
    return sorted(int(y) for y in panel.filter(pl.col(predictor).is_not_null())["year"].unique())


def _windows(panel: pl.DataFrame, predictor: str) -> pl.DataFrame:
    """Per base year and pair with a score: the score, Actual Exports at t, their means before and after, and the
    floored unrealised potential at t."""
    years = {int(y) for y in panel["year"].unique()}
    frames = []
    for t in base_years(panel, predictor):
        missing = sorted(set(range(t - WINDOW, t + WINDOW + 1)) - years)
        if missing:
            raise ValueError(f"Base year {t} needs Actual Exports for {missing}, which the panel does not hold")
        year = pl.col("year")
        frames.append(
            panel.filter(year.is_between(t - WINDOW, t + WINDOW))
            .group_by(*PAIR)
            .agg(
                potential=pl.col(predictor).filter(year == t).first(),
                actual=pl.col("actual_exports").filter(year == t).first(),
                later=pl.col("actual_exports").filter(year.is_between(t + 1, t + WINDOW)).mean(),
                earlier=pl.col("actual_exports").filter(year.is_between(t - WINDOW, t - 1)).mean(),
            )
            .filter(pl.col("potential").is_not_null())
            .with_columns(year=pl.lit(t, dtype=pl.Int16))
        )
    return pl.concat(frames).sort(*PAIR, "year").with_columns(
        sector=pl.col("hs6").str.slice(0, 2), unrealised=unrealised_potential(pl.col("potential"), pl.col("actual"))
    )


def _units(panel: pl.DataFrame, predictor: str, level: str) -> pl.DataFrame:
    """Per base year and unit of `level` (`LEVELS`): the score, Actual Exports at t and their means before and after,
    the summed floored unrealised potential, and whether the UK exports the unit's Product (or Sector) to some Market at t.

    At HS2 each figure is summed over the Sector–Market's scored pairs, the unrealised potential floored pair by pair before summing."""
    pairs = _windows(panel, predictor)
    units = pairs if level == HS6 else (
        pairs.group_by(*LEVELS[level], "year", maintain_order=True)
        .agg(pl.col("potential", "actual", "later", "earlier", "unrealised").sum())
    )
    product = LEVELS[level][0]
    exported = (
        panel.filter(pl.col("actual_exports") > 0)
        .select(pl.col("hs6").str.slice(0, 2).alias("sector") if product == "sector" else pl.col("hs6"), "year")
        .unique()
        .with_columns(exported=pl.lit(True))
    )
    return (
        units.join(exported, on=[product, "year"], how="left")
        .with_columns(pl.col("exported").fill_null(False))
        .sort(*LEVELS[level], "year")  # a fixed order, so a report is reproducible to the last digit
    )


def _trading(units: pl.DataFrame) -> pl.DataFrame:
    return units.filter((pl.col("actual") > 0) & (pl.col("potential") > 0))


def _not_trading(units: pl.DataFrame) -> pl.DataFrame:
    return units.filter(pl.col("actual") == 0)


def _starting_candidates(units: pl.DataFrame) -> pl.DataFrame:
    return _not_trading(units).filter(pl.col("exported") & (pl.col("potential") > 0))


def _key(level: str) -> list[str]:
    return [*LEVELS[level], "year", "sector"] if level == HS6 else [*LEVELS[level], "year"]


def _intensive(units: pl.DataFrame, level: str) -> pl.DataFrame:
    return (
        _trading(units)
        .filter((pl.col("later") > 0) & (pl.col("earlier") > 0))
        .with_columns(
            growth=(pl.col("later") / pl.col("actual")).log(),
            unrealised_share=pl.col("unrealised") / pl.col("potential"),
            past_growth=(pl.col("actual") / pl.col("earlier")).log(),
        )
        .select(*_key(level), "growth", "unrealised_share", "past_growth")
    )


def _extensive(units: pl.DataFrame, level: str) -> pl.DataFrame:
    return _starting_candidates(units).select(
        *_key(level),
        started=(pl.col("later") > 0).cast(pl.Float64),
        log_export_potential=pl.col("potential").log(),
    )


def _new_flow(units: pl.DataFrame, level: str) -> pl.DataFrame:
    return _starting_candidates(units).filter(pl.col("later") > 0).select(
        *_key(level),
        size=(pl.col("later") * WINDOW).log(),
        log_export_potential=pl.col("potential").log(),
    )


def intensive_margin_sample(panel: pl.DataFrame, *, predictor: str, level: str = HS6) -> pl.DataFrame:
    """The intensive margin's rows: units trading at a base year with exports both before and after it.

    One row per unit and base year with `growth`, `unrealised_share` and `past_growth`, plus `sector` (HS2) for the
    fixed effects. `predictor` is the panel column the Unrealised Share is computed from.
    """
    return _intensive(_units(panel, predictor, level), level)


def extensive_margin_sample(panel: pl.DataFrame, *, predictor: str, level: str = HS6) -> pl.DataFrame:
    """The extensive margin's rows: units with no Actual Exports at a base year whose Product (Sector at HS2) the UK
    exports to some Market that year, with `started` (1 if any of the next three years is above zero) and
    `log_export_potential`, the log of `predictor`'s score."""
    return _extensive(_units(panel, predictor, level), level)


def new_flow_size_sample(panel: pl.DataFrame, *, predictor: str, level: str = HS6) -> pl.DataFrame:
    """New-flow size's rows: the extensive margin's units that start, with `size`, the log of their Actual Exports
    summed over the next three years, and `log_export_potential`."""
    return _new_flow(_units(panel, predictor, level), level)


def _number(value: float) -> float | None:
    return None if value is None or not math.isfinite(value) else float(value)


def _fit(value: float, error: float) -> dict:
    return {"coefficient": _number(value), "std_error": _number(error)}


# Each test: its sample, its outcome, its regressor and controls.
_TESTS = {
    INTENSIVE_MARGIN: (_intensive, "growth", "unrealised_share", ("past_growth",)),
    EXTENSIVE_MARGIN: (_extensive, "started", "log_export_potential", ()),
    NEW_FLOW_SIZE: (_new_flow, "size", "log_export_potential", ()),
}


def _counts(test: str, units: pl.DataFrame) -> dict:
    """How the units reached the test's sample: who was trading or not at t, and who dropped out and why."""
    if test == INTENSIVE_MARGIN:
        trading = _trading(units)
        return {
            "trading": trading.height,  # unit-years trading at the base year
            "no_later_exports": int((trading["later"] == 0).sum()),
            "no_earlier_exports": int(((trading["later"] > 0) & (trading["earlier"] == 0)).sum()),
        }
    not_trading = _not_trading(units)
    candidates = _starting_candidates(units)
    return {
        "not_trading": not_trading.height,  # unit-years with no Actual Exports at the base year
        "product_not_exported": int((~not_trading["exported"]).sum()),  # left out: Product sold nowhere at t
        "zero_score": int((not_trading["exported"] & (not_trading["potential"] <= 0)).sum()),  # no log
        "started": int((candidates["later"] > 0).sum()),
    }


def _row(units: pl.DataFrame, test: str, level: str, predictor: str) -> dict:
    """One test (`TESTS`) on the units of one level (`LEVELS`) and predictor (`PREDICTORS`), as one report row."""
    sample_of, outcome, regressor, controls = _TESTS[test]
    fit = ppml.fit_fixed_effects_ols(
        sample_of(units, level),
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
        **_counts(test, units),
        "observations": fit.rows,
        "singletons_dropped": fit.dropped_rows,
        "pairs": fit.clusters,
        "absorbed": list(fit.absorbed),
    }


def realisation_report(panel: pl.DataFrame) -> dict:
    """ITC's realisation test on a scored UK panel (see the module docstring), JSON-ready.

    `rows` holds one entry per test, level and predictor; `published` ITC's own coefficients, cited;
    `direct_comparison` whether part (c) of #156 ran, and why not.
    """
    if panel.height and int(panel["year"].max()) >= TEST_YEAR:  # pyright: ignore[reportArgumentType]
        raise ValueError(f"The panel reaches the test year ({TEST_YEAR}); the realisation test never reads it")
    years = base_years(panel)
    units = {
        (level, predictor): _units(panel, column, level) for level in LEVELS for predictor, column in PREDICTORS.items()
    }
    return {
        "rows": [
            _row(units[level, predictor], test, level, predictor)
            for test in TESTS
            for level in LEVELS
            for predictor in PREDICTORS
        ],
        "base_years": years,
        "outcome_years": [min(years) - WINDOW, max(years) + WINDOW],
        "fixed_effects": list(FIXED_EFFECTS_TEXT),
        "cluster": CLUSTER_TEXT,
        "compared_level": COMPARED_LEVEL,
        "published": [dict(row) for row in PUBLISHED],
        "direct_comparison": {"run": False, "reason": DIRECT_COMPARISON_DROPPED},
    }


NOT_COMPARED = "not compared"  # ITC's column on HS6 rows: its test runs on 76 sub-sectors, nearer HS2


def coefficient_rows(report: dict, *, levels: tuple[str, ...] = tuple(LEVELS)) -> list[dict]:
    """The coefficient table: one row per test, level (of `levels`) and term, with ITC's published figure and each
    predictor's coefficient, its standard error in brackets ("0.512 (0.031)"). Columns are `Test`, `Level`, `Term`,
    ITC's, then one per predictor ("Model", "Naive Baseline"). ITC's figure sits on the compared level's rows
    (`COMPARED_LEVEL`) only; HS6 rows read "not compared"."""
    published = {row["test"]: row for row in report["published"]}
    columns = list(dict.fromkeys(PREDICTOR_NAMES[row["predictor"]] for row in report["rows"]))
    table: dict[tuple[str, str, str], dict] = {}
    for row in report["rows"]:
        if row["level"] not in levels:
            continue
        itc = published.get(row["test"], {})
        terms = {row["regressor"]: row, **row["controls"]}
        itc_terms = {itc.get("regressor"): itc.get("coefficient"), **itc.get("controls", {})}
        for term, fit in terms.items():
            entry = table.setdefault((row["test"], row["level"], term), {
                "Test": TEST_NAMES[row["test"]],
                "Level": row["level"],
                "Term": TERM_NAMES[term],
                ITC_COLUMN: (
                    number_text(itc_terms.get(term), ".3f") if row["level"] == COMPARED_LEVEL else NOT_COMPARED
                ),
                **dict.fromkeys(columns, ""),
            })
            entry[PREDICTOR_NAMES[row["predictor"]]] = (
                f"{number_text(fit['coefficient'], '.3f')} ({number_text(fit['std_error'], '.3f')})"
            )
    return list(table.values())

def method_rows(report: dict) -> list[dict]:
    """Part (a) of #156: ITC's method and ours side by side, one row per aspect."""
    years = years_text(report["base_years"])
    levels = " and ".join(dict.fromkeys(row["level"] for row in report["rows"]))
    rows = [
        ("Estimator",
         "A decomposition: the exporter's supply, the market's projected demand and the ease of trade between them",
         ("Gradient-boosted trees (XGBoost, Tweedie loss) on global HS6 trade, blended with the Naive Baseline and "
         "scaled by Market Reach")),
        ("Data and years",
         "ITC Trade Map, every exporter; the realisation test rebuilds the indicator for 2006–2018",
         (f"BACI HS6 Flows of the top 100 Exporters; the realisation test reads "
         f"{years_text(report['outcome_years'])} and never the test year")),
        ("Target year",
         "About five years ahead (the current map is an estimate for 2031)",
         "The year scored itself"),
        ("Product level",
         "HS6, some codes grouped; the realisation test runs on 76 sub-sectors",
         f"HS6; the realisation test runs at {levels}"),
        ("Gap definition",
         "Unrealised potential, floored at zero: potential − min(actual, potential)",
         ("The Export Gap is signed (Export Potential − Actual Exports, over the Gap Period); the test uses ITC's "
         "floored Unrealised Share")),
        ("How it is tested",
         ("Three regressions of later trade on the indicator (growth of existing flows, whether new flows start, "
         "their size), every exporter pooled, with exporter, market and year × sub-sector fixed effects"),
         (f"The same three on UK pairs in base years {years}, each scored by fold models trained on years up to "
         f"the base year, with the Naive Baseline run the same way beside the model; at HS2 the HS6 figures are "
         f"summed, unrealised potential floored pair by pair; {' and '.join(report['fixed_effects'])} fixed effects, errors "
         f"clustered by {report['cluster']}")),
    ]
    return [{"Aspect": a, "ITC's Export Potential Map": itc, "This model": ours} for a, itc, ours in rows]


def direct_comparison_text(report: dict) -> str:
    """Part (c) of #156 in one line: whether the direct comparison with the EPM's UK figures ran, and why not."""
    part = report["direct_comparison"]
    return "Direct comparison with the EPM's UK figures: " + ("run." if part["run"] else f"{part['reason']}.")


def caveat(report: dict) -> str:
    """How this test differs from ITC's: the exporters, the fixed effects, the levels, the years, the method and the
    projection."""
    return (
        f"A rough benchmark, not a like-for-like test. ITC pools every exporter over 2006–2018; this covers the UK "
        f"only, in base years {years_text(report['base_years'])}, with no exporter fixed effect. ITC's test runs on 76 "
        f"sub-sectors; the rows set beside it are {COMPARED_LEVEL} Sectors, which group Products differently, and the "
        f"HS6 rows are not compared with it. ITC's indicator is a decomposition that projects about five years ahead; "
        f"ours is a learned model scoring the base year itself, with no forward projection. The coefficients describe "
        f"a tendency across many pairs, not a forecast for any one."
    )


def _agrees(fit: dict | None, published: dict) -> bool | None:
    if fit is None or fit["coefficient"] is None:
        return None
    return (fit["coefficient"] > 0) == (published["coefficient"] > 0)


def text(report: dict) -> str:
    """The plain-language summary the API serves beside the figures, at the level compared with ITC
    (`COMPARED_LEVEL`): agreement in sign and broad size, never a validation."""
    fits = {
        (row["test"], row["predictor"]): row for row in report["rows"] if row["level"] == COMPARED_LEVEL
    }
    published = {row["test"]: row for row in report["published"]}
    years = years_text(report["base_years"])
    row = fits.get((INTENSIVE_MARGIN, "model"))
    if row is None or row["coefficient"] is None:
        return f"ITC's realisation test could not be fitted on the UK pairs of {years}."
    coefficient = row["coefficient"]
    direction = (
        "grew faster" if coefficient > 0 else "grew more slowly" if coefficient < 0 else "grew no differently"
    )
    agreement = {test: _agrees(fits.get((test, "model")), published[test]) for test in TESTS}
    fitted = [test for test, agrees in agreement.items() if agrees is not None]
    naive = fits.get((INTENSIVE_MARGIN, "naive_baseline"))
    naive_text = (
        f" The Naive Baseline, run through the same tests in place of Export Potential, gives "
        f"{number_text(naive['coefficient'], '.3f')} on that first test."
        if naive and naive["coefficient"] is not None else ""
    )
    return (
        f"ITC tests its Export Potential Map by asking whether pairs with more unrealised potential grow faster "
        f"over the next three years. Run on UK Sector–Market pairs ({COMPARED_LEVEL}) in {years}, the model's "
        f"Unrealised Share is associated with later growth: pairs with a larger share {direction} over the next three "
        f"years, other things equal (coefficient {number_text(coefficient, '.3f')}, against "
        f"{number_text(published[INTENSIVE_MARGIN]['coefficient'], '.3f')} published by ITC for its own indicator). "
        f"Across ITC's three tests (whether pairs grow, whether new pairs start, and how large new flows are), the "
        f"model agrees with ITC's sign on {sum(1 for test in fitted if agreement[test])} of {len(fitted)}.{naive_text} "
        f"This is a rough benchmark of agreement, not a validation of either tool."
    )


def served(report: dict) -> dict:
    """The report as the API serves it: with its `text`, `caveat` and coefficient `table` made now, not when saved.
    The table (`coefficient_rows`, by column label as tables show them) holds the level compared with ITC only;
    `rows` keeps every level."""
    return {
        **report,
        "text": text(report),
        "caveat": caveat(report),
        "table": coefficient_rows(report, levels=(COMPARED_LEVEL,)),
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
            "export_potential": np.where(scored, potential, np.nan).ravel(),
            "naive_baseline": np.where(scored, naive, np.nan).ravel(),
            "actual_exports": actual.ravel(),
        },
        schema={
            "hs6": pl.String, "iso3": pl.String, "year": pl.Int16,
            "export_potential": pl.Float64, "naive_baseline": pl.Float64, "actual_exports": pl.Float64,
        },
    ).with_columns(pl.col("export_potential", "naive_baseline").fill_nan(None))


# --- Adding the report to a saved artefact ---------------------------------------------------------------------


def real_panel(root: Path | None = None) -> pl.DataFrame:
    """The scored UK panel of the base-year fold models (`base_years.load_panel`, #176), imported only when read."""
    return importlib.import_module("exportspredictor.base_years").load_panel(root)


def add_to_latest(
    artefacts_root: Path | None = None,
    *,
    panel: pl.DataFrame | None = None,
    root: Path | None = None,
    version: str | None = None,
) -> Path:
    """Run the realisation test and save it in the newest artefact's evaluation, as a new version.

    `panel` defaults to the real base-year panel (`real_panel`, under the data `root`). Models and tables are
    copied byte for byte (`artefacts.copy_version`), and the rest of the evaluation is kept; nothing is trained,
    and the test year is never read, so the Test-year Ledger is not written. Returns the new folder.
    """
    from exportspredictor import artefacts as store
    from exportspredictor.paths import artefacts_dir

    artefacts_root = Path(artefacts_root or artefacts_dir())
    source = store.latest_folder(artefacts_root)
    bundle = store.load(source)
    if bundle.evaluation is None:
        raise ValueError(f"Artefact {bundle.version} has no evaluation; run `python -m exportspredictor.evaluation` first")
    started = time.perf_counter()
    report = realisation_report(real_panel(root) if panel is None else panel)
    report["seconds"] = round(time.perf_counter() - started, 2)
    metadata = {**bundle.metadata, "version": version or store.new_version(), "copied_from": bundle.version}
    return store.copy_version(
        source, artefacts_root, metadata=metadata, evaluation=with_itc_comparison(bundle, report).evaluation
    )


if __name__ == "__main__":
    print(add_to_latest(), flush=True)
