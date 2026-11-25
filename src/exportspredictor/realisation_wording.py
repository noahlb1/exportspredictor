"""The tables and Advisor wording of ITC's Realisation Test report, made from the saved report.

`realisation` estimates and saves the report; this module turns it into the coefficient table, the method table,
the caveat and the plain-language text, for the notebook (through `reporting`) and the API (`served`). Nothing here
fits or reads data.
"""

from exportspredictor import realisation, scope, training
from exportspredictor.realisation import (
    COMPARED_LEVEL,
    HS6,
    INTENSIVE_MARGIN,
    ITC_LEVEL,
    ITC_YEARS,
    MODEL,
    NAIVE_BASELINE,
    TESTS,
)
from exportspredictor.wording import number_text, years_text

TEST_NAMES = {
    realisation.INTENSIVE_MARGIN: "Intensive margin",
    realisation.EXTENSIVE_MARGIN: "Extensive margin",
    realisation.NEW_FLOW_SIZE: "New-flow size",
}
TERM_NAMES = {
    realisation.UNREALISED_SHARE: "Unrealised Share",
    realisation.PAST_GROWTH: "past growth",
    realisation.LOG_EXPORT_POTENTIAL: "log Export Potential",
}
PREDICTOR_NAMES = {MODEL: "Model", NAIVE_BASELINE: "Naive Baseline"}
ITC_COLUMN = f"ITC ({ITC_LEVEL})"  # the coefficient table's column of ITC's published figures
NOT_COMPARED = "not compared"  # ITC's column on HS6 rows: its test runs at `ITC_LEVEL`, nearer HS2
# The model's loss, as its objective names it ("reg:tweedie" is the Tweedie loss).
_LOSS = training.DEFAULT_PARAMS["objective"].removeprefix("reg:").capitalize()


def coefficient_rows(report: dict, *, levels: tuple[str, ...] = tuple(realisation.LEVELS)) -> list[dict]:
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
    """ITC's method and ours side by side, one row per aspect."""
    years = years_text(report["base_years"])
    levels = " and ".join(dict.fromkeys(row["level"] for row in report["rows"]))
    rows = [
        ("Estimator",
         "A decomposition: the exporter's supply, the market's projected demand and the ease of trade between them",
         (f"Gradient-boosted trees (XGBoost, {_LOSS} loss) on global HS6 trade, blended with the Naive Baseline and "
         "scaled by Market Reach")),
        ("Data and years",
         f"ITC Trade Map, every exporter; the realisation test rebuilds the indicator for {ITC_YEARS}",
         (f"BACI HS6 Flows of the top {scope.TOP_EXPORTERS} Exporters; the realisation test reads "
         f"{years_text(report['outcome_years'])} and never the test year")),
        ("Target year",
         "About five years ahead (the current map is an estimate for 2031)",
         "The year scored itself"),
        ("Product level",
         f"HS6, some codes grouped; the realisation test runs on {ITC_LEVEL}",
         f"HS6; the realisation test runs at {levels}"),
        ("Gap definition",
         "Unrealised potential, floored at zero: potential − min(actual, potential)",
         ("The Export Gap is signed (Export Potential − Actual Exports, over the Gap Period); the test uses ITC's "
         "floored Unrealised Share")),
        ("How it is tested",
         ("Three regressions of later trade on the indicator (growth of existing flows, whether new flows start, "
         "their size), every exporter pooled, with exporter, market and year × sub-sector fixed effects"),
         (f"The same three on UK pairs in base years {years}, each scored by fold models trained on years up to "
         f"the base year, with the Naive Baseline run the same way beside the model, each on the units it scores above zero; at {COMPARED_LEVEL} the "
         f"{HS6} figures are summed, unrealised potential floored pair by pair; {' and '.join(report['fixed_effects'])} "
         f"fixed effects, errors clustered by {report['cluster']}")),
    ]
    return [{"Aspect": a, "ITC's Export Potential Map": itc, "This model": ours} for a, itc, ours in rows]


def direct_comparison_text(report: dict) -> str:
    """In one line: whether the direct comparison with the EPM's UK figures ran, and why not."""
    part = report["direct_comparison"]
    return "Direct comparison with the EPM's UK figures: " + ("run." if part["run"] else f"{part['reason']}.")


def caveat(report: dict) -> str:
    """How this test differs from ITC's: the exporters, the fixed effects, the levels, the years, the method and the
    projection."""
    return (
        f"A rough benchmark, not a like-for-like test. ITC pools every exporter over {ITC_YEARS}; this covers the UK "
        f"only, in base years {years_text(report['base_years'])}, with no exporter fixed effect. ITC's test runs on "
        f"{ITC_LEVEL}; the rows set beside it are {COMPARED_LEVEL} Sectors, which group Products differently, and the "
        f"{HS6} rows are not compared with it. ITC's indicator is a decomposition that projects about five years ahead; "
        f"ours is a learned model scoring the base year itself, with no forward projection. The Naive Baseline column "
        f"runs on the units the Naive Baseline scores above zero, so where it scores a unit 0 its sample and the "
        f"model's differ. The base-year models reuse the served feature files, so two slowly moving inputs, "
        f"Relatedness and the training scope, draw on a few later years, though no later trade, and they keep the "
        f"served round count, not scaled down for fewer training years. The coefficients describe "
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
    row = fits.get((INTENSIVE_MARGIN, MODEL))
    if row is None or row["coefficient"] is None:
        return f"ITC's realisation test could not be fitted on the UK pairs of {years}."
    coefficient = row["coefficient"]
    direction = (
        "grew faster" if coefficient > 0 else "grew more slowly" if coefficient < 0 else "grew no differently"
    )
    agreement = {test: _agrees(fits.get((test, MODEL)), published[test]) for test in TESTS}
    fitted = [test for test, agrees in agreement.items() if agrees is not None]
    naive = fits.get((INTENSIVE_MARGIN, NAIVE_BASELINE))
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
    """The report as the API serves it: its rows at the level compared with ITC (`COMPARED_LEVEL`) only, the HS6 rows
    staying in the notebook, with its `text`, `caveat` and coefficient `table` (`coefficient_rows`, by column label as
    tables show them) made now, not when saved."""
    return {
        **report,
        "rows": [row for row in report["rows"] if row["level"] == COMPARED_LEVEL],
        "text": text(report),
        "caveat": caveat(report),
        "table": coefficient_rows(report, levels=(COMPARED_LEVEL,)),
    }
