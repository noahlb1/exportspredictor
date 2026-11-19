"""The notebook's charts and tables, one per main result, read from a loaded artefact (spec #66).

Each function takes an `Artefacts` and returns a `Chart` (a static SVG image with the rows it
draws), a small table, or `NotRun` where the artefact does not record that part. Nothing here
trains, fits or recomputes a result: every figure is read from what the artefact records (its
evaluation report, the experiment loop's results in `metadata["ranking_experiments"]` and the
saved PPML benchmark). The notebook calls these and defines no functions of its own.

Charts are SVG so that they render as images on GitHub without a plotting dependency.
"""

import math
from dataclasses import dataclass
from html import escape

import polars as pl

from exportspredictor import model_choice, ppml
from exportspredictor.artefacts import Artefacts
from exportspredictor.evaluation import NOT_RUN, relabelled
from exportspredictor.loop import experiment_label

# Colours: one neutral, one accent for "this version" or a gain, one for a loss.
NEUTRAL = "#8a8984"
ACCENT = "#2a78d6"
LOSS = "#e34948"
INK = "#3d3c39"
GRID = "#d9d8d4"


@dataclass(frozen=True)
class Chart:
    """A static chart: its title, the rows it draws and the SVG image of them."""

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


def provenance(bundle: Artefacts) -> str:
    """Which artefact version the outputs come from, and whether it is real data or the fixture."""
    metadata = bundle.metadata
    derived = metadata.get("ppml_benchmark_from") or metadata.get("copied_from")
    return (
        f"Outputs below come from artefact version {bundle.version} (source: {metadata.get('source')}; "
        f"Model Form: {bundle.model_form.text.rstrip('.')}"
        + (f"; derived from version {derived}" if derived else "")
        + ")."
    )


def ranking_by_version(bundle: Artefacts) -> Chart | NotRun:
    """The test year's UK ranking for this version beside the Naive Baseline and earlier versions."""
    block = relabelled(bundle.evaluation or {}).get("earlier_versions")
    if not block:
        return NotRun("UK ranking by version")
    rows = [{**_pick(row, "name", "uk_ranking"), "kind": "this version" if i == 0 else "earlier version"}
            for i, row in enumerate(block["rows"])]
    rows += [{**_pick(row, "name", "uk_ranking"), "kind": "naive"} for row in block["naive"]]
    data = pl.DataFrame(rows, schema={"name": pl.String, "uk_ranking": pl.Float64, "kind": pl.String})
    colours = {"this version": ACCENT, "earlier version": NEUTRAL, "naive": NEUTRAL}
    title = f"UK within-Product rank correlation on {block['test_year']}, read once (higher is better)"
    return Chart(title, data, bar_svg(
        title,
        [Bar(r["name"], 0.0, r["uk_ranking"], colours[r["kind"]], f"{r['uk_ranking']:.3f}") for r in rows],
    ))



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


def experiments_chart(bundle: Artefacts) -> Chart | NotRun:
    """Each Arm's change in the Selection Measure on the best so far, against the Win Threshold."""
    table = experiments(bundle)
    if isinstance(table, NotRun):
        return table
    record = bundle.metadata["ranking_experiments"]
    threshold = (record.get("noise_floor") or {}).get("win_threshold")
    data = table.with_columns(win_threshold=pl.lit(threshold, dtype=pl.Float64))
    bars = [
        Bar(
            f"{row['experiment']}: {row['arm']}", 0.0, row["change"] or 0.0,
            ACCENT if row["kept"] else LOSS if (row["change"] or 0.0) < 0 else NEUTRAL,
            f"{row['change'] or 0.0:+.4f}" + (" kept" if row["kept"] else "") + (" *" if row["stopped_early"] else ""),
        )
        for row in data.iter_rows(named=True)
    ]
    title = f"Each Arm against the best so far, Selection Measure on {record.get('validation_year')}"
    return Chart(title, data, bar_svg(
        title, bars, lines={} if threshold is None else {"Win Threshold": threshold},
        axis=f"change in the Selection Measure (* stopped within {EARLY_STOP_ROUNDS} rounds)",
    ))

def confirmation(bundle: Artefacts) -> pl.DataFrame | NotRun:
    """The five-fold confirmation fold by fold: the winner's and v2's Selection Measure on each held-out
    fold, then the mean over folds (`fold` "all"), with the winner's margin over v2."""
    summary = model_choice.ranking_summary(bundle.metadata.get("ranking_experiments"))
    confirmed = (summary or {}).get("confirmation")
    if not confirmed:
        return NotRun("The five-fold confirmation")
    rows = [{"fold": str(f["fold"]), "winner": f["winner"], "v2": f["v2"]} for f in confirmed["folds"]]
    rows.append({"fold": "all", "winner": confirmed["winner"], "v2": confirmed["v2"]})
    return pl.DataFrame(rows, schema={"fold": pl.String, "winner": pl.Float64, "v2": pl.Float64}).with_columns(
        difference=pl.col("winner") - pl.col("v2"), win_threshold=pl.lit(confirmed["win_threshold"], dtype=pl.Float64)
    )


def round_count(bundle: Artefacts) -> pl.DataFrame | NotRun:
    """How many boosting rounds the served fold models trained, and what chose them, beside the count the
    next rebuild takes from the Selection Measure: `model_choice.round_count`, which the model card reads too."""
    count = model_choice.round_count(bundle.metadata)
    if count["chosen_by"] is None or count["next_rounds"] is None:
        return NotRun("The round count")
    rows = [
        {"model": "served", "chosen_on": count["chosen_by"], "rounds_chosen": count["rounds_chosen"],
         "rounds_trained": count["rounds"]},
        {"model": "next rebuild", "chosen_on": model_choice.CHOSEN_ON_SELECTION_MEASURE,
         "rounds_chosen": count["next_rounds"], "rounds_trained": None},
    ]
    return pl.DataFrame(rows, schema={
        "model": pl.String, "chosen_on": pl.String, "rounds_chosen": pl.Int64, "rounds_trained": pl.Int64,
    })


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


def gain_ladder(bundle: Artefacts) -> Chart | NotRun:
    """The gain ladder as a waterfall: v1's primary metric, each step's change, v2's (frozen; ADR 0008)."""
    ladder = ((bundle.evaluation or {}).get("v1_comparison") or {}).get("ladder")
    if not ladder:
        return NotRun("The gain ladder")
    values = {rung["name"]: rung["value"] for rung in ladder["rungs"]}
    naive = bundle.evaluation["v1_comparison"]["primary"].get("naive")
    floor = round(min([*values.values(), *([naive] if naive is not None else [])]) - 0.02, 2)
    rows = (
        [{"bar": "v1", "start": floor, "end": values["v1"], "change": None}]
        + [{"bar": f"{s['from']} → {s['to']}: {s['label']}", "start": values[s["from"]], "end": values[s["to"]],
            "change": s["change"]} for s in ladder["steps"]]
        + [{"bar": "v2", "start": floor, "end": values["C"], "change": None}]
    )
    data = pl.DataFrame(rows, schema={"bar": pl.String, "start": pl.Float64, "end": pl.Float64, "change": pl.Float64})
    bars = [
        Bar(row["bar"], row["start"], row["end"],
            NEUTRAL if row["change"] is None else ACCENT if row["change"] >= 0 else LOSS,
            f"{row['end']:.3f}" if row["change"] is None else f"{row['change']:+.3f}")
        for row in data.iter_rows(named=True)
    ]
    title = f"{ladder['metric']}: from v1 to v2 (axis cut, not from 0)"
    return Chart(title, data, bar_svg(
        title, bars, lines={} if naive is None else {"naive": naive}, origin=floor,
    ))


def _pick(row: dict, *keys: str) -> dict:
    return {key: row.get(key) for key in keys}


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


LABEL_WIDTH, PLOT_WIDTH, VALUE_WIDTH = 380, 380, 90
LABEL_CHARS = 60  # longer labels are cut, so they stay inside the image
ROW, TOP, BOTTOM = 24, 44, 40


def bar_svg(
    title: str, bars: list[Bar], *, lines: dict[str, float] | None = None, axis: str = "", origin: float = 0.0
) -> str:
    """An SVG image of horizontal bars, one per row, with labelled vertical reference `lines`.

    The axis includes `origin` (0 unless a chart cuts its axis, as the gain ladder does)."""
    lines = lines or {}
    values = [v for b in bars for v in (b.start, b.end, b.low, b.high) if v is not None] + list(lines.values())
    lo, hi = min([origin, *values]), max([origin, *values])
    span = (hi - lo) or 1.0
    lo, hi = lo - 0.04 * span * (lo < origin), hi + 0.04 * span

    def x(value: float) -> float:
        return LABEL_WIDTH + (value - lo) / (hi - lo) * PLOT_WIDTH

    width, height = LABEL_WIDTH + PLOT_WIDTH + VALUE_WIDTH, TOP + ROW * len(bars) + BOTTOM
    bottom = TOP + ROW * len(bars)
    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" font-family="Helvetica, Arial, sans-serif" font-size="12">'
        ),
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="8" y="20" font-size="13" font-weight="bold" fill="{INK}">{escape(title)}</text>',
    ]
    for tick in _ticks(lo, hi):
        parts.append(f'<line x1="{x(tick):.1f}" x2="{x(tick):.1f}" y1="{TOP - 6}" y2="{bottom}" stroke="{GRID}"/>')
        parts.append(
            f'<text x="{x(tick):.1f}" y="{bottom + 14}" text-anchor="middle" fill="{INK}">{_tick_text(tick)}</text>'
        )
    parts.append(f'<line x1="{x(origin):.1f}" x2="{x(origin):.1f}" y1="{TOP - 6}" y2="{bottom}" stroke="{INK}"/>')
    if axis:
        parts.append(
            f'<text x="{LABEL_WIDTH + PLOT_WIDTH / 2:.1f}" y="{bottom + 32}" text-anchor="middle" fill="{INK}">'
            f'{escape(axis)}</text>'
        )
    for i, bar in enumerate(bars):
        y = TOP + i * ROW
        left, right = sorted((x(bar.start), x(bar.end)))
        parts.append(
            f'<text x="{LABEL_WIDTH - 8}" y="{y + 16}" text-anchor="end" fill="{INK}">{escape(_cut(bar.label))}</text>'
        )
        parts.append(
            f'<rect x="{left:.1f}" y="{y + 4}" width="{max(right - left, 1):.1f}" height="{ROW - 8}" '
            f'fill="{bar.colour}" rx="2"/>'
        )
        if bar.low is not None and bar.high is not None:
            mid = y + ROW / 2
            parts.append(
                f'<line x1="{x(bar.low):.1f}" x2="{x(bar.high):.1f}" y1="{mid}" y2="{mid}" stroke="{INK}" '
                'stroke-width="1.5"/>'
            )
        parts.append(f'<text x="{LABEL_WIDTH + PLOT_WIDTH + 6}" y="{y + 16}" fill="{INK}">{escape(bar.text)}</text>')
    for j, (name, value) in enumerate(lines.items()):
        parts.append(
            f'<line x1="{x(value):.1f}" x2="{x(value):.1f}" y1="{TOP - 6}" y2="{bottom}" stroke="{INK}" '
            'stroke-dasharray="4 3"/>'
        )
        parts.append(
            f'<text x="{x(value):.1f}" y="{TOP - 10 - 12 * (j % 2)}" text-anchor="middle" font-size="11" '
            f'fill="{INK}">{escape(name)}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def _ticks(lo: float, hi: float, count: int = 5) -> list[float]:
    """About `count` round tick values between `lo` and `hi`."""
    raw = (hi - lo) / count
    magnitude = 10 ** math.floor(math.log10(raw))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw)
    first = math.ceil(lo / step) * step
    return [round(first + k * step, 10) for k in range(int((hi - first) / step) + 1)]


def _cut(label: str) -> str:
    return label if len(label) <= LABEL_CHARS else label[: LABEL_CHARS - 1] + "…"


def _tick_text(value: float) -> str:
    return f"{value:g}"
