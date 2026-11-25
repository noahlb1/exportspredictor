"""Step 1 of the re-check: the experiment loop's 2022 decisions re-scored under the tie-proof Selection Measure.

Nothing is trained and nothing reads 2023. Every model is read from the loop's cache
(`data/derived/loop_V202601/models`) and predicts on the CPU; the script stops if one is missing.

For each cached model (the loop's first fold-0 Arms and the four other folds' models of the
five-fold confirmation) the held-out UK grid of 2022 is scored tree by tree, so the Selection
Measure of the model alone is known at every saved round. Early stopping does not change the
trees, only where they are cut, so replaying the loop's rule (stop `patience` rounds after the
last improvement) on the fixed measure gives exactly the round count a re-run would choose, as
long as it stops within the saved trees (each model kept its best round plus `patience`). If it
does not, the round is "censored": the fixed measure was still improving where training ended,
and only a GPU retrain could say where it would stop.

At the re-chosen round each Arm with a blend grid re-chooses its naive weight on the fixed
measure, as the loop does. The old (legacy) measure at the recorded round is recomputed as a
check against `results.json`.

Writes rescore.json and curves.parquet to <data>/derived/loop_recheck_2022/.

    scripts/wt.ps1 python scripts/research/loop_recheck_2022/rescore.py
"""
import gc
import json
import statistics
import time
from dataclasses import replace

import numpy as np
import polars as pl
import xgboost as xgb

from exportspredictor.loop import (
    ANY_FLOW,
    BLEND_WEIGHTS,
    CONFIRMATION,
    MODELS_DIR,
    NOISE_FLOOR,
    SIZE,
    Loop,
    Settings,
    load,
    loop_dir,
    naive_margin,
    real_data,
    v2_plan,
)
from exportspredictor.paths import data_dir
from exportspredictor.prediction import blend
from exportspredictor.selection_measure import (
    legacy_selection_measure,
    selection_measure,
    with_tie_key,
)

OUT = data_dir() / "derived" / "loop_recheck_2022"
MODELS = loop_dir() / MODELS_DIR
THREADS = 12


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def booster(path) -> xgb.Booster:
    if not path.exists():
        raise SystemExit(f"Missing cached model {path}: this script trains nothing")
    model = xgb.Booster(model_file=str(path))
    model.set_param({"device": "cpu", "nthread": THREADS})
    return model


def matrix(lp: Loop, settings: Settings) -> np.ndarray:
    table = lp._with_product_detail(lp.grid, settings)
    return np.ascontiguousarray(table.select(list(settings.features)).to_numpy(), dtype=np.float32)


def tree_margins(model: xgb.Booster, x: np.ndarray, trees: int):
    """Cumulative margin (without base score or base margin) after each tree, one array per round."""
    zeros = np.zeros(x.shape[0], dtype=np.float32)
    total = np.zeros(x.shape[0], dtype=np.float64)
    for i in range(trees):
        total += model.inplace_predict(x, iteration_range=(i, i + 1), predict_type="margin", base_margin=zeros)
        yield i + 1, total


def replay(curve: list[float], patience: int) -> dict:
    """The loop's early stopping replayed on a round-by-round measure (index 0 is round 1)."""
    best, best_round = -np.inf, 0
    for r, value in enumerate(curve, start=1):
        if value > best:
            best, best_round = value, r
        elif r - best_round >= patience:
            return {"rounds": best_round, "measure": best, "censored": False, "stopped_at": r}
    return {"rounds": best_round, "measure": best, "censored": True, "stopped_at": None}


def curve_of(
    lp: Loop, keys: pl.DataFrame, model: xgb.Booster, x: np.ndarray, trees: int, *, times=None, offset=None
) -> list[float]:
    """The fixed measure of the model alone after each round (`times`: a probability it is multiplied by;
    `offset`: the base margin of a model trained from one, which does move the ranking)."""
    values = []
    for _, margin in tree_margins(model, x, trees):
        if offset is not None:
            margin = margin + offset
        prediction = np.exp(margin) if times is not None else margin  # a rank: the base score does not move it
        if times is not None:
            prediction = times * prediction
        values.append(selection_measure(keys.with_columns(prediction=pl.Series(prediction)), "prediction"))
    return [-1.0 if v is None else v for v in values]


def dollars(model: xgb.Booster, x: np.ndarray, rounds: int, margin=None) -> np.ndarray:
    extra = {} if margin is None else {"base_margin": np.asarray(margin, dtype=np.float32)}
    return model.inplace_predict(x, iteration_range=(0, rounds), **extra).astype(np.float64)


def scored(grid: pl.DataFrame, model_usd: np.ndarray, weight: float) -> pl.DataFrame:
    return grid.with_columns(ep=pl.Series(blend(model_usd, grid["naive_usd"].to_numpy(), weight)))


def weight_grid(grid: pl.DataFrame, model_usd: np.ndarray) -> dict[float, float]:
    return {w: selection_measure(scored(grid, model_usd, w), "ep") for w in BLEND_WEIGHTS}


def fold_loop(fold: int) -> Loop:
    log(f"setting up fold {fold}")
    # Device "cuda" only so the cache keys match the saved models; this script predicts on the CPU.
    return Loop(real_data(), replace(v2_plan(), fold=fold), models_dir=MODELS)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results = load()
    curves: list[pl.DataFrame] = []
    out: dict = {"models": {}, "arms": {}, "folds": {}}

    # --- Fold 0: every Arm of the loop's first experiments --------------------------------------
    lp = fold_loop(0)
    grid = lp.grid.select("hs6", "importer", "value_usd", "naive_usd")
    keys = with_tie_key(grid)
    out["naive"] = {
        "legacy": legacy_selection_measure(grid, "naive_usd"),
        "fixed": selection_measure(keys, "naive_usd"),
        "recorded": results["naive"]["selection_measure"],
    }
    log(f"naive {out['naive']}")

    rows = [(NOISE_FLOOR, row) for row in results[NOISE_FLOOR]["arms"]]
    rows += [(name, row) for name, section in results["experiments"].items() for row in section["arms"]]

    def model_curve(key: str, settings: Settings, meta: dict, fold: int, fold_lp: Loop, fold_keys) -> dict:
        tag = f"{fold}:{key}"
        if tag in out["models"]:
            return out["models"][tag]
        x = matrix(fold_lp, settings)
        if settings.two_part:
            binary = booster(MODELS / f"{key}.{ANY_FLOW}.ubj")
            size = booster(MODELS / f"{key}.ubj")
            parts = meta["parts"]
            b_curve = curve_of(fold_lp, fold_keys, binary, x, parts[ANY_FLOW]["trained_rounds"])
            b_choice = replay(b_curve, settings.patience)
            probability = dollars(binary, x, b_choice["rounds"])
            old_probability = dollars(binary, x, parts[ANY_FLOW]["rounds"])
            s_curve = curve_of(fold_lp, fold_keys, size, x, parts[SIZE]["trained_rounds"], times=probability)
            s_choice = replay(s_curve, settings.patience)
            entry = {
                "two_part": True,
                "recorded_rounds": {ANY_FLOW: parts[ANY_FLOW]["rounds"], SIZE: parts[SIZE]["rounds"]},
                "trained_rounds": {ANY_FLOW: parts[ANY_FLOW]["trained_rounds"], SIZE: parts[SIZE]["trained_rounds"]},
                "fixed": {ANY_FLOW: b_choice, SIZE: s_choice},
                "_usd_recorded": old_probability * dollars(size, x, parts[SIZE]["rounds"]),
                "_usd_fixed": probability * dollars(size, x, s_choice["rounds"]),
            }
            curves.append(pl.DataFrame({"fold": fold, "key": key, "part": ANY_FLOW, "round": range(1, len(b_curve) + 1), "fixed": b_curve}))
            curves.append(pl.DataFrame({"fold": fold, "key": key, "part": SIZE, "round": range(1, len(s_curve) + 1), "fixed": s_curve}))
        else:
            model = booster(MODELS / f"{key}.ubj")
            margin = naive_margin(fold_lp.grid["naive_usd"]).to_numpy() if settings.base_margin else None
            curve = curve_of(fold_lp, fold_keys, model, x, meta["trained_rounds"], offset=margin)
            choice = replay(curve, settings.patience)
            entry = {
                "two_part": False,
                "recorded_rounds": meta["rounds"],
                "trained_rounds": meta["trained_rounds"],
                "fixed": choice,
                "curve_at_recorded": curve[meta["rounds"] - 1],  # a check: the model alone at the recorded round
                "_usd_recorded": dollars(model, x, meta["rounds"], margin),
                "_usd_fixed": dollars(model, x, choice["rounds"], margin),
            }
            curves.append(pl.DataFrame({"fold": fold, "key": key, "part": "model", "round": range(1, len(curve) + 1), "fixed": curve}))
        out["models"][tag] = entry
        log(f"model {tag}: recorded {entry['recorded_rounds']} -> fixed {entry['fixed']}")
        return entry

    for section_name, row in rows:
        settings = Settings.from_dict(row["settings"])
        key = row["training_key"]
        meta = json.loads((MODELS / f"{key}.json").read_text(encoding="utf-8"))
        entry = model_curve(key, settings, meta, 0, lp, keys)
        recorded_w = settings.naive_weight
        has_grid = "grid" in row
        old_usd, new_usd = entry["_usd_recorded"], entry["_usd_fixed"]
        arm = {
            "experiment": section_name,
            "arm": row["arm"],
            "training_key": key,
            "recorded": row["selection_measure"],
            "recorded_weight": recorded_w,
            "legacy_at_recorded": legacy_selection_measure(scored(grid, old_usd, recorded_w), "ep"),
            "fixed_at_recorded": selection_measure(scored(keys, old_usd, recorded_w), "ep"),
            "model_alone_fixed_at_recorded": selection_measure(scored(keys, old_usd, 0.0), "ep"),
            "model_alone_fixed_at_fixed_rounds": selection_measure(scored(keys, new_usd, 0.0), "ep"),
        }
        if has_grid:
            by_weight = weight_grid(keys, new_usd)
            chosen = max(by_weight, key=lambda w: by_weight[w])
            arm |= {"fixed_weight_grid": by_weight, "fixed_weight": chosen, "fixed": by_weight[chosen]}
            old_grid = weight_grid(keys, old_usd)
            arm |= {"fixed_weight_grid_at_recorded_rounds": old_grid}
        else:
            arm |= {"fixed_weight": recorded_w, "fixed": selection_measure(scored(keys, new_usd, recorded_w), "ep")}
        out["arms"][f"{section_name} / {row['arm']}"] = arm
        log(f"arm {section_name} / {row['arm']}: {json.dumps({k: arm[k] for k in ('recorded', 'legacy_at_recorded', 'fixed_at_recorded', 'fixed', 'fixed_weight')})}")
        (OUT / "rescore.json").write_text(json.dumps(_public(out), indent=2), encoding="utf-8")

    del keys
    # --- Folds 1-4: the five-fold confirmation's models ----------------------------------------
    confirmation = results[CONFIRMATION]
    winner = Settings.from_dict(confirmation["settings"]["winner"])
    pooled_recorded, pooled_fixed = [], []
    for fold_row in confirmation["folds"]:
        fold = fold_row["fold"]
        key = fold_row["v2"]["training_key"]
        if fold == 0:
            fold_lp, fold_grid = lp, grid
        else:
            lp = None
            gc.collect()
            fold_lp = fold_loop(fold)
            fold_grid = fold_lp.grid.select("hs6", "importer", "value_usd", "naive_usd")
        fold_keys = with_tie_key(fold_grid)
        meta = json.loads((MODELS / f"{key}.json").read_text(encoding="utf-8"))
        entry = model_curve(key, winner, meta, fold, fold_lp, fold_keys)
        rec = fold_grid.with_columns(
            v2=pl.Series(entry["_usd_recorded"]),
            winner=pl.Series(blend(entry["_usd_recorded"], fold_grid["naive_usd"].to_numpy(), winner.naive_weight)),
        )
        fix = fold_grid.with_columns(
            v2=pl.Series(entry["_usd_fixed"]),
            winner=pl.Series(blend(entry["_usd_fixed"], fold_grid["naive_usd"].to_numpy(), winner.naive_weight)),
        )
        saved = pl.read_parquet(loop_dir() / CONFIRMATION / f"fold_{fold}.parquet")
        out["folds"][fold] = {
            "training_key": key,
            "recorded_rounds": meta["rounds"],
            "fixed_rounds": entry["fixed"]["rounds"],
            "censored": entry["fixed"]["censored"],
            "saved_grid_matches": bool(
                saved.height == rec.height
                and np.allclose(saved.sort("hs6", "importer")["winner"].to_numpy(),
                                rec.sort("hs6", "importer")["winner"].to_numpy(), rtol=1e-5)
            ),
            "recorded": fold_row["winner"]["selection_measure"],
            "legacy_at_recorded": legacy_selection_measure(rec, "winner"),
            "fixed_at_recorded": selection_measure(rec, "winner"),
            "fixed_at_fixed_rounds": selection_measure(fix, "winner"),
            "v2_fixed_at_recorded": selection_measure(rec, "v2"),
            "v2_fixed_at_fixed_rounds": selection_measure(fix, "v2"),
        }
        log(f"fold {fold}: {out['folds'][fold]}")
        pooled_recorded.append(rec.with_columns(fold=pl.lit(fold)))
        pooled_fixed.append(fix.with_columns(fold=pl.lit(fold)))
        del fold_lp
        gc.collect()

    rec = with_tie_key(pl.concat(pooled_recorded))
    fix = with_tie_key(pl.concat(pooled_fixed))
    fold_rounds_recorded = [out["folds"][f]["recorded_rounds"] for f in sorted(out["folds"])]
    fold_rounds_fixed = [out["folds"][f]["fixed_rounds"] for f in sorted(out["folds"])]
    out["confirmation"] = {
        "recorded": {k: confirmation[k]["selection_measure"] for k in ("winner", "v2", "naive")},
        "legacy": {k: legacy_selection_measure(rec, c) for k, c in (("winner", "winner"), ("v2", "v2"), ("naive", "naive_usd"))},
        "fixed_at_recorded_rounds": {k: selection_measure(rec, c) for k, c in (("winner", "winner"), ("v2", "v2"), ("naive", "naive_usd"))},
        "fixed_at_fixed_rounds": {k: selection_measure(fix, c) for k, c in (("winner", "winner"), ("v2", "v2"), ("naive", "naive_usd"))},
        "weight_grid_at_recorded_rounds": weight_grid(rec, rec["v2"].to_numpy()),
        "weight_grid_at_fixed_rounds": weight_grid(fix, fix["v2"].to_numpy()),
        "fold_rounds_recorded": fold_rounds_recorded,
        "fold_rounds_fixed": fold_rounds_fixed,
        "served_rounds_recorded": statistics.median_low(fold_rounds_recorded),
        "served_rounds_fixed": statistics.median_low(fold_rounds_fixed),
    }
    log(f"confirmation {out['confirmation']}")
    (OUT / "rescore.json").write_text(json.dumps(_public(out), indent=2), encoding="utf-8")
    pl.concat(curves).write_parquet(OUT / "curves.parquet")
    log("done")


def _public(out: dict) -> dict:
    """The output without the private dollar arrays."""
    def clean(value):
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items() if not str(k).startswith("_")}
        if isinstance(value, float | np.floating):
            return float(value)
        return value
    return clean(out)


if __name__ == "__main__":
    main()
