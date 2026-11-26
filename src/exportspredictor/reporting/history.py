"""The Model History's charts and tables: how the served model was reached (ADR 0013).

Every version on 2023, v2's weakness, the gain ladder, every Arm, the confirmation against v2, the round count, the final
round and every read of 2023. Product surfaces show none of this; `notebooks/model_history.ipynb` does.

Each function reads what is already saved, as product reporting does: the artefact's evaluation report (its
`earlier_versions`, `rescored_earlier_versions` and `v1_comparison` blocks) and metadata (`ranking_experiments`,
`round_scaling`), the experiment loop's saved results and the round's saved probe, and the committed Test-year Ledger.
Nothing trains, fits or reads the test year.
"""

import json
from collections.abc import Mapping
from pathlib import Path

import polars as pl

from exportspredictor import loop, model_choice
from exportspredictor.artefacts import Artefacts
from exportspredictor.evaluation import large_markets_beside_earlier, relabelled
from exportspredictor.ledger import LEDGER_PATH, read_ledger
from exportspredictor.loop import experiment_label
from exportspredictor.paths import data_dir
from exportspredictor.reporting import (
    ACCENT,
    EARLY_STOP_ROUNDS,
    LOSS,
    NEUTRAL,
    Bar,
    Chart,
    NotRun,
    bar_svg,
    experiments,
)

NO_TRADE_OFF = "No earlier version of this artefact records a lower Tweedie deviance."
# The loop's measures that show v2's weakness on Zero Cells beside the Naive Baseline (`zero_cells_against_naive`).
ZERO_CELL_MEASURES = ("selection_measure", "non_zero_spearman", "zero_vs_non_zero_spearman", "tweedie_deviance")


# --- Every version on 2023 ---------------------------------------------------------------------------------------


def ranking_by_version(bundle: Artefacts) -> Chart | NotRun:
    """The test year's UK ranking for this version beside the Naive Baseline and earlier versions.

    Drawn on the tie-proof Selection Measure once every earlier version has been re-scored from its saved scores
    (`evaluation.with_rescored_earlier_versions`), else on the old measure, the one every version reported. A row
    without a figure on the drawn measure (the Naive Baseline as reported for v2) is listed in the data, not drawn."""
    block = relabelled(bundle.evaluation or {}).get("earlier_versions")
    if not block:
        return NotRun("UK ranking by version")
    keys = ("name", "uk_ranking", "uk_ranking_tie_proof")
    rows = [{**{key: row.get(key) for key in keys}, "kind": "this version" if i == 0 else "earlier version"}
            for i, row in enumerate(block["rows"])]
    rows += [{**{key: row.get(key) for key in keys}, "kind": "naive"} for row in block["naive"]]
    data = pl.DataFrame(
        rows, schema={"name": pl.String, "uk_ranking": pl.Float64, "uk_ranking_tie_proof": pl.Float64, "kind": pl.String}
    )
    colours = {"this version": ACCENT, "earlier version": NEUTRAL, "naive": NEUTRAL}
    tie_proof = all(row["uk_ranking_tie_proof"] is not None for row in rows if row["kind"] != "naive")
    column = "uk_ranking_tie_proof" if tie_proof else "uk_ranking"
    measure = ", tie-proof measure" if tie_proof else (", old measure" if "uk_ranking_tie_proof" in block["rows"][0] else "")
    title = f"UK within-Product rank correlation on {block['test_year']}, read once{measure} (higher is better)"
    return Chart(title, data, bar_svg(
        title,
        [Bar(r["name"], 0.0, r[column], colours[r["kind"]], f"{r[column]:.3f}") for r in rows if r[column] is not None],
    ))


def large_markets_by_version(bundle: Artefacts) -> pl.DataFrame | NotRun:
    """The test year's tie-proof UK ranking on the largest importing Markets: this version, the earlier versions
    re-scored from their saved scores, and the Naive Baseline (`evaluation.large_markets_beside_earlier`)."""
    rows = large_markets_beside_earlier(bundle.evaluation or {})
    if not rows:
        return NotRun("The measure on the largest Markets")
    return pl.DataFrame([{"version": row["name"], **row["columns"]} for row in rows])


def traded_away(bundle: Artefacts) -> str:
    """What choosing on the ranking cost in size: this version's Tweedie deviance over all Exporters against the first
    earlier version that records one (`model_choice.traded_away`), or the line saying none records a lower one."""
    traded = model_choice.traded_away(relabelled(bundle.evaluation or {}).get("earlier_versions"))
    return NO_TRADE_OFF if traded is None else traded["text"]


# --- v2 and the gain ladder --------------------------------------------------------------------------------------


def zero_cells_against_naive(bundle: Artefacts) -> pl.DataFrame | NotRun:
    """v2's configuration (the Noise Floor's first seed) beside the Naive Baseline on the validation year: the Selection
    Measure, its parts on non-zero Flows and on any Flow against none, and Tweedie deviance."""
    record = bundle.metadata.get("ranking_experiments")
    if not record or not record.get("noise_floor") or not record.get("naive"):
        return NotRun("v2 beside the Naive Baseline")
    v2 = record["noise_floor"]["arms"][0]
    return pl.DataFrame(
        [
            {"model": f"v2's configuration ({v2['arm']})", **{m: v2.get(m) for m in ZERO_CELL_MEASURES}},
            {"model": "Naive Baseline", **{m: record["naive"].get(m) for m in ZERO_CELL_MEASURES}},
        ],
        schema={"model": pl.String, **dict.fromkeys(ZERO_CELL_MEASURES, pl.Float64)},
    )


def gain_ladder(bundle: Artefacts) -> Chart | NotRun:
    """The gain ladder as a waterfall: v1's primary metric, each step's change, v2's (ADR 0008)."""
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


# --- The experiment loop -----------------------------------------------------------------------------------------


def experiments_chart(bundle: Artefacts) -> Chart | NotRun:
    """Each Arm's change in the Selection Measure on the best so far (`reporting.experiments`), against the Win
    Threshold. The final round's Arms are left to `final_round`, its own table."""
    table = experiments(bundle, by_role=False)
    if isinstance(table, NotRun):
        return table
    record = bundle.metadata["ranking_experiments"]
    final = {name for names in FINAL_ROUND_IDEAS.values() for name in names}
    final_labels = [experiment_label(name, [arm["arm"] for arm in section.get("arms", [])])
                    for name, section in record["experiments"].items() if name in final]
    threshold = (record.get("noise_floor") or {}).get("win_threshold")
    data = table.filter(~pl.col("experiment").is_in(final_labels)).with_columns(
        win_threshold=pl.lit(threshold, dtype=pl.Float64))
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
    """The five-fold confirmation against v2, fold by fold: the winner's and v2's Selection Measure on each held-out
    fold, then over every pair (`fold` "all"), with the winner's margin and the Win Threshold. Read from the loop's
    saved record, not from what the API serves."""
    confirmed = (bundle.metadata.get("ranking_experiments") or {}).get("confirmation")
    if not confirmed or "v2" not in confirmed:
        return NotRun("The five-fold confirmation")
    rows = [{"fold": str(f["fold"]), "winner": f["winner"]["selection_measure"], "v2": f["v2"]["selection_measure"]}
            for f in confirmed["folds"]]
    rows.append({"fold": "all", "winner": confirmed["winner"]["selection_measure"],
                 "v2": confirmed["v2"]["selection_measure"]})
    return pl.DataFrame(rows, schema={"fold": pl.String, "winner": pl.Float64, "v2": pl.Float64}).with_columns(
        difference=pl.col("winner") - pl.col("v2"), win_threshold=pl.lit(confirmed["win_threshold"], dtype=pl.Float64)
    )


def round_count(bundle: Artefacts) -> pl.DataFrame | NotRun:
    """How many boosting rounds the served fold models trained, and what chose them, beside the count the next rebuild
    takes from the Selection Measure (`model_choice.round_count`)."""
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


# --- The final round ---------------------------------------------------------------------------------------------

# One idea per row in the order run (docs/research/final-round-2022.md, "Summary"), each with the loop experiments that
# decided it.
FINAL_ROUND_IDEAS: dict[str, tuple[str, ...]] = {
    "Cap on Market Reach's upside (#163)": (loop.MARKET_REACH_CAP_EXPERIMENT,),
    "Sibling Demand as a feature (#164)": (loop.SIBLING_DEMAND_EXPERIMENT,),
    "A fresh hyperparameter search (#166)": tuple(loop.hyperparameter_experiment(s) for s in loop.HYPERPARAMETER_STAGES),
    "Other model families (#167)": tuple(loop.model_family_experiment(f) for f in loop.MODEL_FAMILY_ARMS),
    "A wider Exporter scope (#169)": (loop.WIDER_SCOPE_EXPERIMENT,),
    "More rows through external memory (#170)": (loop.EXTERNAL_MEMORY_EXPERIMENT,),
    "Training on the Excluded Products (#171)": (loop.EXCLUDED_PRODUCTS_EXPERIMENT,),
}
FINAL_ROUND_PROBE_IDEA = "Tariff 0 where an agreement's preference is unrecorded (#171)"
FINAL_ROUND_PROBE_FILE = Path("derived") / "final_round_2022" / "unrecorded_preferences.json"
# What each row's `threshold` is: the loop's ideas had to beat the Win Threshold; the probe, its own gate
# (half the Win Threshold, pre-registered in docs/research/final-round-2022.md) before a retrained Arm would run.
WIN_THRESHOLD_BAR, PROBE_GATE_BAR = "Win Threshold", "its pre-registered gate"


def final_round_probe(root: Path | None = None) -> dict | None:
    """The unrecorded-preferences probe as `scripts/research/final_round_2022/unrecorded_preferences.py` saved it,
    or None where it was not run (the fixture world)."""
    path = Path(root or data_dir()) / FINAL_ROUND_PROBE_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def final_round(results: Mapping, probe: Mapping | None = None) -> pl.DataFrame | NotRun:
    """The final round of experiments, one row per idea: its experiments and Arms, the best Arm, the best so far as served
    it was paired with (`base`), the best Arm's Selection Measure and gain, the `threshold` it had to beat, what that
    threshold is (`judged_against`) and the decision. Read from the loop's saved results (`loop.load`) and the
    scoring-only probe (`final_round_probe`), which is judged against its pre-registered gate, not the Win Threshold.
    The round ran after the served artefact was built, so the artefact records none of it."""
    recorded = results.get("experiments", {})
    rows = []
    for idea, names in FINAL_ROUND_IDEAS.items():
        sections = [recorded[name] for name in names if name in recorded]
        if not sections:
            continue
        section, arm = max(((s, a) for s in sections for a in s["arms"]), key=lambda pair: pair[1]["change"])
        rows.append({
            "idea": idea, "experiments": len(sections), "arms": sum(len(s["arms"]) for s in sections),
            "best_arm": arm["arm"], "base": section["base"]["selection_measure"], "best": arm["selection_measure"],
            "gain": arm["change"], "threshold": section["win_threshold"], "judged_against": WIN_THRESHOLD_BAR,
            "decision": "kept" if any(s.get("kept") for s in sections) else "dropped",
        })
    if not rows:
        return NotRun("The final round")
    if probe is not None:
        rows.append({
            "idea": FINAL_ROUND_PROBE_IDEA, "experiments": 0, "arms": 0, "best_arm": "scoring-only probe",
            "base": probe["base"]["selection_measure"], "best": probe["probe"]["selection_measure"],
            "gain": probe["gain"], "threshold": probe["gate"], "judged_against": PROBE_GATE_BAR,
            "decision": "retrained Arm run (passed its gate)" if probe["retrained_arm_runs"]
            else "not retrained (missed its gate)",
        })
    return pl.DataFrame(rows, schema={
        "idea": pl.String, "experiments": pl.Int64, "arms": pl.Int64, "best_arm": pl.String, "base": pl.Float64,
        "best": pl.Float64, "gain": pl.Float64, "threshold": pl.Float64, "judged_against": pl.String,
        "decision": pl.String,
    })


# --- Every read of 2023 ------------------------------------------------------------------------------------------


def test_year_ledger(path: Path = LEDGER_PATH) -> pl.DataFrame:
    """Every read of the test year, oldest first, from the Test-year Ledger file alone (ADR 0009); nothing
    of the test year is read. Each row's source record stays in the file's `record` column."""
    return read_ledger(path).drop("record")
