"""Ticket #116, step 2: re-decide the loop's experiments under the tie-proof measure and save the re-check.

Reads rescore.json (from rescore.py), replays the win rule (`loop.decide`) on every experiment with
the re-scored values and the recomputed Win Threshold, and saves the re-check into the loop's
results (`loop.save_recheck`), where the loop reads the threshold and the best so far. The results
file is copied to <data>/derived/loop_recheck_2022/results.before_recheck.json first. Trains nothing,
reads no 2023 data and touches no cached model or feature file.

Writes decisions.json to <data>/derived/loop_recheck_2022/.

    scripts/wt.ps1 python scripts/research/loop_recheck_2022/apply.py
"""
import json
import shutil

from exportspredictor.loop import (
    NOISE_FLOOR,
    RECHECK,
    RESULTS_FILE,
    current_win_threshold,
    decide,
    load,
    loop_dir,
    recheck_results,
    save_recheck,
)
from exportspredictor.paths import data_dir

OUT = data_dir() / "derived" / "loop_recheck_2022"


def main() -> None:
    rescore = json.loads((OUT / "rescore.json").read_text(encoding="utf-8"))
    results = load()
    if RECHECK in results:
        raise SystemExit("The loop's results already carry a re-check")
    arms = rescore["arms"]
    models = rescore["models"]

    def fixed(section: str, arm: str) -> dict:
        return arms[f"{section} / {arm}"]

    seeds = [fixed(NOISE_FLOOR, row["arm"])["fixed"] for row in results[NOISE_FLOOR]["arms"]]
    preview = recheck_results(results, seeds, best=results["best"])
    threshold = current_win_threshold(preview)

    decisions = []
    for name, section in results["experiments"].items():
        base = section["base"]
        base_value = fixed(base["experiment"], base["arm"])["fixed"]
        rows = []
        for row in section["arms"]:
            a = fixed(name, row["arm"])
            model = models[f"0:{row['training_key']}"]
            rows.append({
                "arm": row["arm"],
                "selection_measure": a["fixed"],
                "recorded": row["selection_measure"],
                "fixed_at_recorded_rounds": a["fixed_at_recorded"],
                "recorded_weight": a["recorded_weight"],
                "fixed_weight": a["fixed_weight"],
                "recorded_rounds": model["recorded_rounds"],
                "fixed_rounds": (
                    {part: model["fixed"][part]["rounds"] for part in model["fixed"]} if model["two_part"]
                    else model["fixed"]["rounds"]
                ),
                "censored": (
                    any(model["fixed"][part]["censored"] for part in model["fixed"]) if model["two_part"]
                    else model["fixed"]["censored"]
                ),
            })
        marked, kept = decide(rows, base_value, threshold)
        decisions.append({
            "experiment": name,
            "base": base["arm"],
            "base_recorded": base["selection_measure"],
            "base_fixed": base_value,
            "recorded_threshold": section["win_threshold"],
            "win_threshold": threshold,
            "recorded_kept": section["kept"],
            "kept": kept,
            "stands": kept == section["kept"],
            "arms": marked,
        })

    best = results["best"]
    best_fixed = fixed(best["experiment"], best["arm"])
    flips = [d for d in decisions if not d["stands"]]
    weight_flip = best_fixed["fixed_weight"] != best["settings"]["naive_weight"]
    settings = {**best["settings"], "naive_weight": best_fixed["fixed_weight"]}
    confirmation = rescore["confirmation"]
    phase2_base = {
        "experiment": best["experiment"],
        "arm": best["arm"],
        "settings": settings,
        "selection_measure": best_fixed["fixed"],
    }
    record = {
        "ticket": 116,
        "note": "docs/research/loop-recheck-tie-proof-2022.md",
        "legacy_noise_floor": {k: results[NOISE_FLOOR][k] for k in ("selection_measures", "sd", "win_threshold")},
        "naive": rescore["naive"],
        "decisions": decisions,
        "flipped": [d["experiment"] for d in flips] + (["naive weight"] if weight_flip else []),
        "confirmation": confirmation,
        "phase2_rounds": confirmation["served_rounds_fixed"],
        "rounds_flipped": confirmation["served_rounds_fixed"] != confirmation["served_rounds_recorded"],
    }
    rechecked = recheck_results(results, seeds, best=phase2_base, record=record)

    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(loop_dir() / RESULTS_FILE, OUT / "results.before_recheck.json")
    (OUT / "decisions.json").write_text(json.dumps(rechecked[RECHECK], indent=2), encoding="utf-8")
    save_recheck(loop_dir(), rechecked)
    print(json.dumps({
        "noise_floor": rechecked[RECHECK][NOISE_FLOOR],
        "flipped": record["flipped"],
        "rounds": [confirmation["served_rounds_recorded"], confirmation["served_rounds_fixed"]],
        "phase2_base": {k: phase2_base[k] for k in ("arm", "selection_measure")} | {"naive_weight": settings["naive_weight"]},
    }, indent=2))


if __name__ == "__main__":
    main()
