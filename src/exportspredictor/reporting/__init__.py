"""The notebook's charts and tables, one per main result, read from a loaded artefact.

Each function takes an `Artefacts` and returns a `Chart` (a static image with the rows it draws),
a small table, or `NotRun` where the artefact does not record that part. Nothing here trains,
fits or recomputes a result: every figure is read from what the artefact records (its
evaluation report, the experiment loop's results in `metadata["ranking_experiments"]` and the
saved PPML benchmark). The notebook calls these and defines no functions of its own.

How the served model was reached (every version, every Arm, every read of the test year) is the record, not the
product: its charts and tables are in `reporting.history` (ADR 0013).

Charts are drawn with matplotlib and saved as SVG, which GitHub renders as an image. Text stays
text, so each chart is a few kilobytes in the committed notebook.
"""

import io
import json
import re
import textwrap
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import polars as pl
from matplotlib.figure import Figure

from exportspredictor import (
    offsets,
    ppml,
    rankings,
    realisation,
    realisation_wording,
    scenarios,
)
from exportspredictor.artefacts import (
    METADATA_FILE,
    Artefacts,
    copy_lineage,
    read_evaluation,
)
from exportspredictor.caveats import TIERS, CaveatTier
from exportspredictor.currency import format_usd
from exportspredictor.evaluation import (
    NOT_RUN,
    market_reach_effect_text,
    relabelled,
)
from exportspredictor.loop import experiment_label
from exportspredictor.paths import artefacts_dir
from exportspredictor.rankings import REACH_DEPTHS, in_top, ranked_opportunities

# Colours: one neutral, one accent for "this version" or a gain, one for a loss.
NEUTRAL = "#8a8984"
ACCENT = "#2a78d6"
LOSS = "#e34948"
INK = "#3d3c39"
GRID = "#d9d8d4"


@dataclass(frozen=True)
class Chart:
    """A static chart: its title, the rows it draws and the matplotlib SVG image of them."""

    title: str
    data: pl.DataFrame
    svg: str

    def _repr_svg_(self) -> str:
        return self.svg

    def __repr__(self) -> str:
        return self.title


@dataclass(frozen=True)
class NotRun:
    """A part this artefact version does not record; shown as one line instead of a chart or table."""

    part: str

    def __repr__(self) -> str:
        return f"{self.part}: {NOT_RUN}."

    def _repr_markdown_(self) -> str:
        return f"*{self!r}*"


PROVENANCE_PREFIX = "Outputs below come from artefact version"
_PROVENANCE = re.compile(re.escape(PROVENANCE_PREFIX) + r" (\S+) \(source: ([^;)]+)")


def provenance(bundle: Artefacts, *, artefacts_root: Path | None = None, lineage: bool = True) -> str:
    """Which artefact version the outputs come from, whether it is real data or the fixture, and how it was derived.

    A copy names the version it was copied from (`copied_from`, the latest copy step) and, with `lineage` and where
    `artefacts_root` (by default the served artefacts directory) still holds that version, which evaluation blocks the
    copy added or replaced and the versions it was copied from in turn (`artefacts.copy_lineage`). Without `lineage`
    no other version's folder is read. A version whose values an earlier step rebuilt from another
    (`blocked_values_from`, `ppml_benchmark_from`) names that one too."""
    metadata = bundle.metadata
    steps = []
    copied = metadata.get("copied_from")
    if copied:
        steps.append(f"a copy of version {copied}"
                     + (_copy_step(bundle, copied, artefacts_root or artefacts_dir()) if lineage else ""))
    derived = metadata.get("blocked_values_from") or metadata.get("ppml_benchmark_from")
    if derived and derived != copied:
        steps.append(f"derived from version {derived}")
    return (
        f"{PROVENANCE_PREFIX} {bundle.version} (source: {metadata.get('source')}; "
        f"Model Form: {bundle.model_form.text.rstrip('.')}"
        + "".join(f"; {step}" for step in steps)
        + ")."
    )


def _copy_step(bundle: Artefacts, source: str, artefacts_root: Path) -> str:
    """What a copy changed in its source's evaluation, and what that source was itself copied from; "" if the source's
    folder is gone."""
    folder = Path(artefacts_root) / source
    if not (folder / METADATA_FILE).exists():
        return ""
    before = read_evaluation(folder) or {}
    after = json.loads(json.dumps(bundle.evaluation or {}))  # as saved, so tuples compare equal to lists
    added = [name for name in after if name not in before]
    replaced = [name for name in after if name in before and after[name] != before[name]]
    changes = [f"{', '.join(names)} {verb}" for names, verb in ((added, "added"), (replaced, "replaced")) if names]
    earlier = copy_lineage(artefacts_root, source)[1:]
    return (
        (f" with {' and '.join(changes)} to its evaluation" if changes else "")
        + (f", itself copied from {', then '.join(earlier)}" if earlier else "")
    )


def provenance_versions(printed: str) -> list[tuple[str, str]]:
    """(version, source) of every `provenance` line in some printed output, in order."""
    return _PROVENANCE.findall(printed)


def model_card_excerpt(path: Path) -> str:
    """A saved model card in brief, as Markdown: its title, its Headline section and the names of its other
    sections, with where the whole card is. The whole card runs to tens of kilobytes; the notebook shows this."""
    card = Path(path).read_text(encoding="utf-8")
    title, _, rest = card.partition("\n")
    sections = re.split(r"^## ", rest, flags=re.MULTILINE)[1:]
    headline = next((section for section in sections if section.startswith("Headline\n")), None)
    names = [section.split("\n", 1)[0] for section in sections]
    lines = [
        f"**{title.removeprefix('# ')}**", "",
        (f"An excerpt. The whole card is `{Path(path).parent.name}/{Path(path).name}` in the artefacts directory, "
         "written with the artefact."), "",
    ]
    if headline is not None:
        lines += [f"## {headline.strip()}", ""]
    lines += [f"**Its sections:** {'; '.join(names)}."]
    return "\n".join(lines)


def market_reach_effect(bundle: Artefacts) -> str:
    """What Market Reach changes on the test year (`evaluation.market_reach_effect`), as saved and read by
    `evaluation.relabelled` (the served UK-pair coverage), or the line saying it was not run for a version without
    Market Reach."""
    report = relabelled(bundle.evaluation or {})
    effect = report.get("market_reach_effect")
    return repr(NotRun("Market Reach")) if effect is None else market_reach_effect_text(
        effect, report["test_year"], figures=True)


# A best round under this means the stopping rule, not the configuration, decided the Arm
# (`exportspredictor.loop`, "Pitfalls").
EARLY_STOP_ROUNDS = 50


def experiments(bundle: Artefacts) -> pl.DataFrame | NotRun:
    """Every Arm of every experiment the loop ran: its experiment (`loop.experiment_label`), its Selection
    Measure, its change on the best so far, its best round, whether it was kept, and whether it stopped
    within its first `EARLY_STOP_ROUNDS` rounds."""
    record = bundle.metadata.get("ranking_experiments")
    if not record or not record.get("experiments"):
        return NotRun("The experiment loop")
    rows = [
        {
            "experiment": experiment_label(name, [arm["arm"] for arm in section.get("arms", [])]),
            "arm": row["arm"],
            "selection_measure": row.get("selection_measure"),
            "change": row.get("change"),
            "rounds": row.get("rounds"),
            "kept": row["arm"] == section.get("kept"),
            "stopped_early": row.get("rounds") is not None and row["rounds"] < EARLY_STOP_ROUNDS,
        }
        for name, section in record["experiments"].items()
        for row in section.get("arms", [])
    ]
    return pl.DataFrame(rows, schema={
        "experiment": pl.String, "arm": pl.String, "selection_measure": pl.Float64, "change": pl.Float64,
        "rounds": pl.Int64, "kept": pl.Boolean, "stopped_early": pl.Boolean,
    })


def caveat_tier_reach(
    bundle: Artefacts, *, tops: tuple[int, ...] = REACH_DEPTHS, tiers: tuple[CaveatTier, ...] = TIERS
) -> pl.DataFrame:
    """How many Opportunities each Caveat Tier covers in the Baseline (ADR 0005), and how many of them are in the
    top `tops` of the Leaderboard's default ranking (`rankings.ranked_opportunities`)."""
    ranked = ranked_opportunities(bundle.baseline)
    return pl.DataFrame(
        [
            {
                "Caveat Tier": tier.badge,
                "Opportunities": int(ranked[tier.column].sum()),
                **{f"in top {top:,}": count for top, count in in_top(ranked, tier.column, tops).items()},
            }
            for tier in tiers
        ]
    )


def offset_pair_reach(bundle: Artefacts) -> str:
    """The Offset Pair tier's reach as the artefact metadata records it (`offsets.METADATA_KEY`): how many
    Opportunities are Offset Pairs, how many of the top of the Leaderboard, and their share of positive Export Gap."""
    saved = bundle.metadata.get(offsets.METADATA_KEY)
    if not saved:
        return repr(NotRun("The Offset Pair tier"))
    reach = offsets.OffsetPairReach.from_dict(saved)
    share = "" if reach.share_of_export_gap is None else f" ({reach.share_of_export_gap:.0%})"
    return (
        f"Offset Pairs: {reach.offset_pair_count:,} of {reach.opportunities:,} Opportunities "
        f"({reach.share_of_opportunities:.0%}); {reach.in_top_text()}; "
        f"{format_usd(reach.offset_export_gap_usd)} of {format_usd(reach.opportunity_export_gap_usd)} of positive "
        f"Export Gap{share}."
    )


# The figures of `heading_split`, in the Baseline's columns, each shown in $m (the unit is in each column's name).
_SPLIT_FIGURES = {
    "Actual Exports": "actual_exports_usd", "Export Potential": "export_potential_usd", "Export Gap": "export_gap_usd",
}


def heading_split(bundle: Artefacts, *, products: tuple[str, ...], markets: tuple[str, ...]) -> pl.DataFrame:
    """Some Products of one Heading beside the whole Heading (every scored Product of it) in some Markets, in $m: each
    Market's Actual Exports, Export Potential and Export Gap from the Baseline, as a worked example of an Offset Pair."""
    [code] = pl.DataFrame({"hs6": list(products)}).select(offsets.heading(pl.col("hs6")))["hs6"].unique()
    rows = bundle.baseline.with_columns(pl.col("hs6").cast(pl.String), pl.col("iso3").cast(pl.String)).filter(
        offsets.heading(pl.col("hs6")) == code, pl.col("iso3").is_in(markets)
    )
    names = dict(bundle.markets.select(pl.col("iso3").cast(pl.String), "name").iter_rows())
    table = []
    for iso3 in markets:
        market = rows.filter(pl.col("iso3") == iso3)
        for figure, column in _SPLIT_FIGURES.items():
            values = dict(market.select("hs6", column).iter_rows())
            table.append({
                "Market": names.get(iso3, iso3), "Figure": figure,
                **{f"{hs6} ($m)": round(values.get(hs6, 0.0) / 1e6, 1) for hs6 in products},
                f"Whole Heading {code} ($m)": round(float(market[column].sum()) / 1e6, 1),
            })
    return pl.DataFrame(table)


def flagged_products_by_reason(bundle: Artefacts) -> pl.DataFrame:
    """How many Flagged Products each reason covers in the Baseline (Sector 71, volatile or one-off; ADR 0005), and
    their Opportunities (positive Export Gap); most Products first."""
    return (
        bundle.baseline.filter("flagged_product")
        .group_by(Reason="flagged_product_reason")
        .agg(Products=pl.col("hs6").n_unique(), Opportunities=rankings.IS_OPPORTUNITY.sum().cast(pl.Int64))
        .sort(["Products", "Opportunities", "Reason"], descending=[True, True, False])
    )


def lift_by_tenth(bundle: Artefacts) -> Chart | NotRun:
    """The share of pairs that grew in each tenth of Export Gap (1 = largest), beside the share of all pairs."""
    lift = (bundle.evaluation or {}).get("lift")
    if not lift:
        return NotRun("Lift")
    data = pl.DataFrame(
        [{"tenth": row["decile"], "rate": row["rate"], "pairs": row["pairs"]} for row in lift["by_decile"]],
        schema={"tenth": pl.Int64, "rate": pl.Float64, "pairs": pl.Float64},
    ).with_columns(base_rate=pl.lit(lift["base_rate"], dtype=pl.Float64))
    title = f"Share of pairs that grew by {lift.get('outcome_year')}, by tenth of Export Gap (1 = largest)"
    bars = [
        Bar(f"tenth {row['tenth']}", 0.0, row["rate"], ACCENT if row["tenth"] == 1 else NEUTRAL, f"{row['rate']:.1%}")
        for row in data.iter_rows(named=True)
    ]
    return Chart(title, data, bar_svg(title, bars, lines={"all pairs": lift["base_rate"]}, axis="share that grew"))


ITC_COMPARISON = "ITC's realisation test on our model"


def _from_itc_report[T](bundle: Artefacts, make: Callable[[dict], T]) -> T | NotRun:
    """`make` applied to the saved ITC comparison report, or `NotRun` where this version has none."""
    report = (bundle.evaluation or {}).get(realisation.REPORT_KEY)
    return make(report) if report else NotRun(ITC_COMPARISON)


def itc_coefficients(bundle: Artefacts) -> pl.DataFrame | NotRun:
    """ITC's realisation test on our model: each coefficient, with its standard error, beside ITC's
    published static figure (`realisation_wording.coefficient_rows`), read from the saved evaluation report."""
    return _from_itc_report(bundle, lambda report: pl.DataFrame(realisation_wording.coefficient_rows(report)))


def itc_method(bundle: Artefacts) -> pl.DataFrame | NotRun:
    """ITC's method and ours side by side (`realisation_wording.method_rows`)."""
    return _from_itc_report(bundle, lambda report: pl.DataFrame(realisation_wording.method_rows(report)))


def itc_direct_comparison(bundle: Artefacts) -> str | NotRun:
    """Whether the direct comparison with the EPM's UK figures ran, and if not, why."""
    return _from_itc_report(bundle, realisation_wording.direct_comparison_text)


def event_backtests(bundle: Artefacts) -> Chart | NotRun:
    """Each Event Backtest's predicted change beside the observed change and its interval, by year and pooled."""
    backtest = (bundle.evaluation or {}).get("event_backtest")
    if not backtest or not backtest.get("events"):
        return NotRun("Event Backtests")
    rows = [
        {
            "event": event["name"],
            "direction": direction["label"],
            "year": str(row["year"]),
            "predicted": row["predicted"],
            "observed": row["observed"],
            "observed_low": row["observed_ci"][0],
            "observed_high": row["observed_ci"][1],
        }
        for event in backtest["events"]
        for direction in event["directions"]
        for row in [*direction["by_year"], {**direction["pooled"], "year": "pooled"}]
    ]
    data = pl.DataFrame(rows, schema={
        "event": pl.String, "direction": pl.String, "year": pl.String, "predicted": pl.Float64,
        "observed": pl.Float64, "observed_low": pl.Float64, "observed_high": pl.Float64,
    })
    bars = []
    for row in data.iter_rows(named=True):
        name = f"{row['direction']}, {row['year']}"
        bars.append(Bar(f"{name}: predicted", 0.0, row["predicted"], ACCENT, f"{row['predicted']:+.0%}"))
        bars.append(Bar(
            "observed", 0.0, row["observed"], NEUTRAL, f"{row['observed']:+.0%}",
            low=row["observed_low"], high=row["observed_high"],
        ))
    title = "Event Backtests: the model's predicted change against the observed change (line: 95% interval)"
    return Chart(title, data, bar_svg(title, bars, axis="change in Flows after the Trade Agreement (0.2 = +20%)"))


def ppml_benchmark(bundle: Artefacts) -> pl.DataFrame | NotRun:
    """The saved PPML benchmark beside the model's own Trade Agreement association and the published
    estimates (`ppml.comparison_table`), read from the artefact; nothing is refitted."""
    saved = bundle.ppml_benchmark
    if not saved:
        return NotRun("The PPML benchmark")
    fit = ppml.PpmlFit(**{name: saved["ppml"][name] for name in (
        "coefficient", "std_error", "rows", "dropped_rows", "clusters", "iterations", "converged", "seconds"
    )})
    return ppml.comparison_table(fit, saved["model"])


TOP_BLOCKED = 10  # how many of the largest Blocked Opportunities the write-up lists


def top_blocked_opportunities(bundle: Artefacts, *, limit: int = TOP_BLOCKED) -> pl.DataFrame:
    """The largest Blocked Opportunities by value, as the Blocked Opportunities page orders them
    (`rankings.blocked_opportunities`): each with its Barrier, Baseline-year Export Potential, value, Tariff part and
    Agreement part, and whether its Actual Exports are already above Export Potential."""
    return rankings.blocked_opportunities(bundle, limit=limit).rows.select(
        "hs6", "description", "market", "barrier", "baseline_export_potential_usd", "blocked_opportunity_value_usd",
        "tariff_part_usd", "agreement_part_usd", "above_export_potential",
    )


def blocked_opportunity_split(bundle: Artefacts) -> pl.DataFrame:
    """The Blocked Opportunity value split into its Tariff part and Agreement part, by kind of Applied Tariff and in all
    (`rankings.blocked_opportunity_split`): counts, Markets, dollar sums, the Tariff part's share and how many
    Agreement parts come out negative (it is the remainder)."""
    return rankings.blocked_opportunity_split(bundle)


def preset_scenarios(bundle: Artefacts, results: Mapping[str, scenarios.ScenarioResult]) -> pl.DataFrame:
    """One row per Trading Bloc preset, in the order of `results` (preset name to its `ScenarioResult`): its title,
    the pairs it changes, its Baseline and Scenario Export Potential, Scenario Effect and effect share
    (`ScenarioResult.total`), the Markets it adds a Trade Agreement with (`added_agreements`, by name) and the count
    and total value of the precomputed Blocked Opportunities in those Markets (`rankings.blocked_opportunities`, the
    one source, ADR 0012). Both are null where the preset adds no agreement, never 0.

    No Caveat Tier is hidden, so the counts add up with every other Blocked Opportunity figure the write-up reads from
    the same unfiltered ranking; the Scenario Builder hides the tiers the app hides by default, so its count for the
    same Market can be lower."""
    names = dict(bundle.markets.select("iso3", "name").iter_rows())
    rows = []
    for name, result in results.items():
        blocked = rankings.blocked_opportunities(bundle, markets=result.added_agreements).summary
        if not result.adds_agreement:
            blocked = dict.fromkeys(blocked)  # the one source's summary keys, null not 0 (ADR 0012)
        rows.append({
            "preset": scenarios.PRESETS[name].title,
            "pairs_changed": result.pairs.height,
            **result.total(),
            "added_agreements": [names.get(iso3, iso3) for iso3 in result.added_agreements],
            **blocked,
        })
    return pl.DataFrame(rows, schema={
        "preset": pl.String,
        "pairs_changed": pl.Int64,
        "baseline_potential_usd": pl.Float64,
        "scenario_potential_usd": pl.Float64,
        "effect_usd": pl.Float64,
        "effect_share": pl.Float64,
        "added_agreements": pl.List(pl.String),
        "blocked_opportunities": pl.Int64,
        "blocked_opportunity_value_usd": pl.Float64,
    })


VARIATION_MARKETS = 10  # the top Markets (by Blocked Opportunity total) the variation check reads
VARIATION_TOP = 10  # the top Products by value and by share whose overlap the variation check counts


def agreement_part_variation(
    bundle: Artefacts, *, markets: int = VARIATION_MARKETS, top: int = VARIATION_TOP
) -> pl.DataFrame:
    """How Product-specific the Agreement part is: within each of the top `markets` Markets (largest Blocked
    Opportunity total first), the spread between its Blocked Opportunities of the Agreement part as a share of
    Baseline-year Export Potential, and how their order changes when sorted by that share instead of by value.

    The Trade Agreement is one flag per Market, so an Agreement part that were the same share for every Product would
    say the model's agreement association is a Market-level figure. Columns: iso3, market, blocked_opportunities,
    share_min, share_p10, share_median, share_p90, share_max, order_correlation (Spearman between the order by value
    and the order by share; null with one Product), top_shared (how many of the `top` Products by value are also
    among the `top` by share), first_by_value and first_by_share (Product descriptions).
    """
    rows = rankings.blocked_opportunities(bundle).rows.with_columns(
        share=pl.col("agreement_part_usd") / pl.col("baseline_export_potential_usd")
    )
    share, value = pl.col("share"), pl.col("blocked_opportunity_value_usd")
    by_market = rows.group_by("iso3", "market").agg(
        pl.len().alias("blocked_opportunities"),
        value.sum().alias("value_usd"),
        share.min().alias("share_min"),
        share.quantile(0.1, "linear").alias("share_p10"),
        share.median().alias("share_median"),
        share.quantile(0.9, "linear").alias("share_p90"),
        share.max().alias("share_max"),
        pl.when(pl.len() > 1).then(pl.corr(value, share, method="spearman")).alias("order_correlation"),
        pl.col("hs6").sort_by(value, descending=True).head(top).is_in(
            pl.col("hs6").sort_by(share, descending=True).head(top).implode()
        ).sum().alias("top_shared"),
        pl.col("description").sort_by(value, descending=True).first().alias("first_by_value"),
        pl.col("description").sort_by(share, descending=True).first().alias("first_by_share"),
    )
    return (
        by_market.sort(["value_usd", "market"], descending=[True, False])
        .head(markets)
        .drop("value_usd")
    )


def agreement_part_variation_chart(bundle: Artefacts, *, markets: int = VARIATION_MARKETS) -> Chart:
    """`agreement_part_variation` as bars: each top Market's median Agreement part share of Baseline-year Export
    Potential, with a line from its 10th to its 90th percentile Product."""
    data = agreement_part_variation(bundle, markets=markets)
    bars = [
        Bar(f"{row['market']} ({row['blocked_opportunities']:,} Products)", 0.0, row["share_median"], ACCENT,
            f"{row['share_median']:.0%}", low=row["share_p10"], high=row["share_p90"])
        for row in data.iter_rows(named=True)
    ]
    title = ("Agreement part as a share of Baseline-year Export Potential, between the Blocked Opportunities "
             "of each top Market (bar: median; line: 10th to 90th percentile)")
    return Chart(title, data, bar_svg(title, bars, axis="Agreement part / Baseline-year Export Potential (0.2 = 20%)"))


# --- The one chart form: horizontal bars, optionally floating, with intervals and reference lines ---------


@dataclass(frozen=True)
class Bar:
    """One row of a bar chart: a bar from `start` to `end`, an optional interval and its value label."""

    label: str
    start: float
    end: float
    colour: str
    text: str
    low: float | None = None
    high: float | None = None


LABEL_CHARS = 60  # longer labels are cut, so they stay inside the image
TITLE_CHARS = 100  # longer titles wrap onto a second line
CHART_LIMIT_BYTES = 30_000  # a chart's SVG stays under this, so the committed notebook stays small
WIDTH_INCHES, ROW_INCHES, FRAME_INCHES = 9.5, 0.28, 1.3
# Text as SVG text in a short font list (every label repeats it), ids salted so re-runs match.
SVG_SETTINGS = {
    "svg.fonttype": "none",
    "svg.hashsalt": "exportspredictor",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
}


def bar_svg(
    title: str, bars: list[Bar], *, lines: dict[str, float] | None = None, axis: str = "", origin: float = 0.0
) -> str:
    """A matplotlib SVG image of horizontal bars, one per row, with labelled vertical reference `lines`.

    The axis includes `origin` (0 unless a chart cuts its axis, as the gain ladder does). Text is
    kept as SVG text, not glyph outlines, so the image stays small and its labels searchable; the
    date is left out and ids are salted, so a re-run writes the same bytes."""
    lines = lines or {}
    values = [v for b in bars for v in (b.start, b.end, b.low, b.high) if v is not None] + list(lines.values())
    lo, hi = min([origin, *values]), max([origin, *values])
    span = (hi - lo) or 1.0
    lo, hi = lo - 0.04 * span * (lo < origin), hi + 0.04 * span

    figure = Figure(figsize=(WIDTH_INCHES, FRAME_INCHES + ROW_INCHES * len(bars)), layout="constrained")
    ax = figure.add_subplot()
    rows = list(range(len(bars)))
    ax.barh(rows, [b.end - b.start for b in bars], left=[b.start for b in bars],
            color=[b.colour for b in bars], height=0.66)
    for row, bar in zip(rows, bars, strict=True):
        if bar.low is not None and bar.high is not None:
            ax.hlines(row, bar.low, bar.high, color=INK, linewidth=1.5)
        ax.annotate(bar.text, (1, row), xycoords=("axes fraction", "data"), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.set_yticks(rows, [_cut(bar.label) for bar in bars], fontsize=9, color=INK)
    ax.set_ylim(len(bars) - 0.5, -0.5)  # first row at the top
    ax.set_xlim(lo, hi)
    ax.axvline(origin, color=INK, linewidth=0.8)
    for j, (name, value) in enumerate(lines.items()):
        ax.axvline(value, color=INK, linewidth=1, linestyle=(0, (4, 3)))
        ax.annotate(name, (value, 1), xycoords=("data", "axes fraction"), xytext=(0, 3 + 11 * (j % 2)),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8, color=INK)
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=9, colors=INK)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel(axis, fontsize=9, color=INK)
    figure.suptitle(textwrap.fill(title, TITLE_CHARS), x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)

    buffer = io.StringIO()
    with matplotlib.rc_context(SVG_SETTINGS):
        figure.savefig(buffer, format="svg", metadata={"Date": None})
    return re.sub(r">\s+<", "><", re.sub(r"<!--.*?-->", "", buffer.getvalue(), flags=re.DOTALL))


def _cut(label: str) -> str:
    return label if len(label) <= LABEL_CHARS else label[: LABEL_CHARS - 1] + "…"
