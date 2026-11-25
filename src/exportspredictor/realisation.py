"""ITC's Realisation Test run on our model (#156): does the Export Gap come true?

ITC supports its Export Potential Map (EPM) with a 3-year realisation test (ITC, *Export Potential Assessment
methodology*, June 2023, pp. 17–22, after Périllat 2018). It regresses later trade on its indicator and reports
the coefficients. This module runs the same linear regressions on UK Export Potential, as Lift's regression
counterpart: Lift asks with deciles whether large Export Gaps point at growth, this asks it with ITC's regression.

**The intensive margin** (the one test so far, at HS6 for our model). For each base year t and each UK
Product–Market pair with Actual Exports at t:

    log(mean Actual Exports over t+1..t+3 / Actual Exports at t)
        = b · Unrealised Share at t + c · log(Actual Exports at t / mean over t−3..t−1) + Market + year × Sector

The **Unrealised Share** is ITC's floored share, (Export Potential − min(Actual Exports, Export Potential)) / Export
Potential, from year-t values: between 0 and 1, and not the signed Export Gap. Pairs with no exports in t+1..t+3
drop out, as in ITC, and so do pairs with none in t−3..t−1, whose past growth is undefined; both are counted. The
fixed effects are swept out and errors clustered by pair (`ppml.fit_fixed_effects_ols`); ITC's exporter fixed
effect drops out, because there is one exporter.

**The panel** (`realisation_report`'s input) holds one row per UK pair and year: `hs6`, `iso3`, `year`,
`export_potential` and `naive_baseline` (non-null in base years only, each scored by fold models trained on years up
to that base year, #176), and `actual_exports` (0 where no Flow). Base years are the years with Export Potential.
A panel reaching the test year is refused: the test never reads it, so nothing goes through the Test-year Ledger
(ADR 0009), and the result feeds no modelling choice.

**The report** is JSON-ready and saved in the artefact's evaluation as `itc_comparison`. Its `rows` are keyed by
`test`, `level` and `predictor`, so the extensive margin, new-flow size, HS2 and the Naive Baseline add rows, not
fields. The result is a rough benchmark of agreement, never a validation.

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
HS6 = "HS6"
# The column each predictor's Unrealised Share is computed from.
PREDICTORS = {"model": "export_potential"}
FIXED_EFFECTS = (("iso3",), ("year", "sector"))
FIXED_EFFECTS_TEXT = ("Market", "year × Sector")
CLUSTER_TEXT = "Product–Market pair"

METHODS_PAPER = (
    "ITC, Export Potential Assessment methodology (June 2023), Table 1, pp. 17–22; "
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
        "source": METHODS_PAPER,
    },
    {
        "test": "extensive_margin",
        "regressor": "log_export_potential",
        "coefficient": 0.023,
        "controls": {},
        "level": "76 sub-sectors",
        "source": METHODS_PAPER,
    },
    {
        "test": "new_flow_size",
        "regressor": "log_export_potential",
        "coefficient": 0.243,
        "controls": {},
        "level": "76 sub-sectors",
        "source": METHODS_PAPER,
    },
)
TEST_NAMES = {
    INTENSIVE_MARGIN: "Intensive margin",
    "extensive_margin": "Extensive margin",
    "new_flow_size": "New-flow size",
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


def unrealised_share(potential: pl.Expr, actual: pl.Expr) -> pl.Expr:
    """ITC's floored share: (potential − min(actual, potential)) / potential, between 0 and 1."""
    return (potential - pl.min_horizontal(actual, potential)) / potential


def base_years(panel: pl.DataFrame, predictor: str = "export_potential") -> list[int]:
    """The years in which `panel` holds scores of `predictor`."""
    return sorted(int(y) for y in panel.filter(pl.col(predictor).is_not_null())["year"].unique())


def _windows(panel: pl.DataFrame, predictor: str) -> pl.DataFrame:
    """Per base year and pair with a score: the score, Actual Exports at t, and their means before and after."""
    years = set(int(y) for y in panel["year"].unique())
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
    return pl.concat(frames)


def intensive_margin_sample(panel: pl.DataFrame, *, predictor: str) -> pl.DataFrame:
    """The intensive margin's rows: pairs trading at a base year with exports both before and after it.

    One row per pair and base year with `growth`, `unrealised_share` and `past_growth`, plus `sector` (HS2) for the
    fixed effects. `predictor` is the panel column the Unrealised Share is computed from.
    """
    return (
        _trading(panel, predictor)
        .filter((pl.col("later") > 0) & (pl.col("earlier") > 0))
        .with_columns(
            growth=(pl.col("later") / pl.col("actual")).log(),
            unrealised_share=unrealised_share(pl.col("potential"), pl.col("actual")),
            past_growth=(pl.col("actual") / pl.col("earlier")).log(),
            sector=pl.col("hs6").str.slice(0, 2),
        )
        .select(*PAIR, "year", "sector", "growth", "unrealised_share", "past_growth")
    )


def _trading(panel: pl.DataFrame, predictor: str) -> pl.DataFrame:
    return _windows(panel, predictor).filter((pl.col("actual") > 0) & (pl.col("potential") > 0))


def _number(value: float) -> float | None:
    return None if value is None or not math.isfinite(value) else float(value)


def intensive_margin(panel: pl.DataFrame, *, predictor: str = "model") -> dict:
    """The intensive margin at HS6 for one predictor (`PREDICTORS`), as one report row."""
    column = PREDICTORS[predictor]
    trading = _trading(panel, column)
    sample = intensive_margin_sample(panel, predictor=column)
    fit = ppml.fit_fixed_effects_ols(
        sample,
        outcome="growth",
        regressors=["unrealised_share", "past_growth"],
        fixed_effects=FIXED_EFFECTS,
        cluster=PAIR,
    )
    return {
        "test": INTENSIVE_MARGIN,
        "level": HS6,
        "predictor": predictor,
        "regressor": "unrealised_share",
        "coefficient": _number(fit.coefficients["unrealised_share"]),
        "std_error": _number(fit.std_errors["unrealised_share"]),
        "controls": {
            "past_growth": {
                "coefficient": _number(fit.coefficients["past_growth"]),
                "std_error": _number(fit.std_errors["past_growth"]),
            }
        },
        "trading": trading.height,  # pair-years trading at the base year
        "no_later_exports": int((trading["later"] == 0).sum()),
        "no_earlier_exports": int(((trading["later"] > 0) & (trading["earlier"] == 0)).sum()),
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
    return {
        "rows": [intensive_margin(panel, predictor=predictor) for predictor in PREDICTORS],
        "base_years": years,
        "outcome_years": [min(years) - WINDOW, max(years) + WINDOW],
        "fixed_effects": list(FIXED_EFFECTS_TEXT),
        "cluster": CLUSTER_TEXT,
        "published": [dict(row) for row in PUBLISHED],
        "direct_comparison": {"run": False, "reason": DIRECT_COMPARISON_DROPPED},
    }


def coefficient_rows(report: dict) -> list[dict]:
    """The coefficient table: one row per test and term, ITC's published figure beside each predictor and level's
    coefficient with its standard error in brackets ("0.512 (0.031)"). Columns are `Test`, `Term`, ITC's, then one
    per predictor and level in the report ("Model, HS6"), so more levels and predictors add columns."""
    published = {row["test"]: row for row in report["published"]}
    columns = list(dict.fromkeys(f"{PREDICTOR_NAMES[row['predictor']]}, {row['level']}" for row in report["rows"]))
    table: dict[tuple[str, str], dict] = {}
    for row in report["rows"]:
        itc = published.get(row["test"], {})
        terms = {row["regressor"]: row, **row["controls"]}
        itc_terms = {itc.get("regressor"): itc.get("coefficient"), **itc.get("controls", {})}
        for term, fit in terms.items():
            entry = table.setdefault((row["test"], term), {
                "Test": TEST_NAMES[row["test"]],
                "Term": TERM_NAMES[term],
                ITC_COLUMN: number_text(itc_terms.get(term), ".3f"),
                **dict.fromkeys(columns, ""),
            })
            entry[f"{PREDICTOR_NAMES[row['predictor']]}, {row['level']}"] = (
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
         "Gradient-boosted trees (XGBoost, Tweedie loss) on global HS6 trade, blended with the Naive Baseline and "
         "scaled by Market Reach"),
        ("Data and years",
         "ITC Trade Map, every exporter; the realisation test rebuilds the indicator for 2006–2018",
         f"BACI HS6 Flows of the top 100 Exporters; the realisation test reads "
         f"{years_text(report['outcome_years'])} and never the test year"),
        ("Target year",
         "About five years ahead (the current map is an estimate for 2031)",
         "The year scored itself"),
        ("Product level",
         "HS6, some codes grouped; the realisation test runs on 76 sub-sectors",
         f"HS6; the realisation test runs at {levels}"),
        ("Gap definition",
         "Unrealised potential, floored at zero: potential − min(actual, potential)",
         "The Export Gap is signed (Export Potential − Actual Exports, over the Gap Period); the test uses ITC's "
         "floored Unrealised Share"),
        ("How it is tested",
         "Later trade regressed on the indicator, every exporter pooled, with exporter, market and year × sub-sector "
         "fixed effects",
         f"The same regression on UK pairs in base years {years}, each scored by fold models trained on years up to "
         f"the base year; {' and '.join(report['fixed_effects'])} fixed effects, errors clustered by "
         f"{report['cluster']}"),
    ]
    return [{"Aspect": a, "ITC's Export Potential Map": itc, "This model": ours} for a, itc, ours in rows]


def direct_comparison_text(report: dict) -> str:
    """Part (c) of #156 in one line: whether the direct comparison with the EPM's UK figures ran, and why not."""
    part = report["direct_comparison"]
    return "Direct comparison with the EPM's UK figures: " + ("run." if part["run"] else f"{part['reason']}.")


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
    """
    rng = np.random.default_rng(seed)
    pairs = [(f"{1 + p % 6:02d}{p:04d}", f"M{m:02d}") for p in range(products) for m in range(markets)]
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
