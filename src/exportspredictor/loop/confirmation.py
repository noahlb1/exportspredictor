"""The five-fold confirmation: the best arm beside v2 on every fold, measured on every held-out UK pair pooled."""

import json
import statistics
import time
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import polars as pl

from exportspredictor import experiments
from exportspredictor.experiments import log_progress, read_results
from exportspredictor.loop.arms import LoopPlan, Settings
from exportspredictor.loop.data import MODELS_DIR, LoopData, save_results
from exportspredictor.loop.runner import Loop
from exportspredictor.loop.selection import ROUNDING, SELECTION_MEASURE, measures

# The best arm after the experiments ("winner") beside the current v2 configuration ("v2") on
# every fold: each fold's model scores only the UK pairs it held out (ADR 0001), and the
# selection measure is taken once over every UK pair pooled, as the headline measure is.
CONFIRMATION = "confirmation"
WINNER, V2 = "winner", "v2"


def confirm(
    data: LoopData, plan: LoopPlan, winner: Settings, v2: Settings, *, threshold: float, folder: Path | None = None
) -> dict:
    """The winner and v2 trained on each of the plan's folds and measured on every held-out UK pair of the validation year.

    Each fold has its own `Loop` (fold `0..n_folds − 1`) sharing the loop's model cache, so
    a fold model the experiments already trained is read, not retrained, and a winner that
    differs from v2 only in scoring (`SCORING_FIELDS`) shares v2's model. Each fold's row
    holds every measure on that fold alone (deviance on its validation rows). The pooled
    rows (`winner`, `v2`, `naive`) hold the measures over every fold's held-out UK pairs
    together; their deviance is the mean of the folds'. `beats_v2` is True when the
    winner beats v2 by more than `threshold`.

    With a `folder`, models are cached in its `models`, each fold's scored grid is saved in
    `confirmation/fold_<k>.parquet` and the section in `results.json` under `confirmation`,
    as each fold finishes; a re-run with the same settings skips finished folds. The
    validation year must not be the test year: `Loop` refuses it.
    """
    experiments._check_years([data.validation_year])
    arms = {V2: v2, WINNER: winner}  # v2 first: a winner that only re-scores reuses its model
    settings = {name: s.to_dict() for name, s in arms.items()}
    results = read_results(folder) or {}
    section = results.get(CONFIRMATION, {})
    if section.get("settings") != settings or section.get("n_folds") != plan.n_folds:
        section = {"settings": settings, "n_folds": plan.n_folds, "validation_year": data.validation_year, "folds": []}
    folds_dir = folder / CONFIRMATION if folder is not None else None
    models_dir = folder / MODELS_DIR if folder is not None else None
    grids: dict[int, pl.DataFrame] = {}
    for fold in range(plan.n_folds):
        saved = folds_dir / f"fold_{fold}.parquet" if folds_dir is not None else None
        if saved is not None and saved.exists() and any(row["fold"] == fold for row in section["folds"]):
            grids[fold] = pl.read_parquet(saved)
            continue
        section["folds"] = [row for row in section["folds"] if row["fold"] != fold]
        row, grids[fold] = _confirm_fold(data, replace(plan, fold=fold), arms, models_dir)
        section["folds"] = sorted([*section["folds"], row], key=lambda r: r["fold"])
        if saved is not None:
            saved.parent.mkdir(parents=True, exist_ok=True)
            grids[fold].write_parquet(saved)
        if folder is not None:
            save_results(folder, results | {CONFIRMATION: section})
        log_progress(f"fold {fold}: {json.dumps({n: row[n][SELECTION_MEASURE] for n in arms})}")
    pooled = pl.concat([grids[fold] for fold in range(plan.n_folds)])
    for name in arms:
        deviances = [row[name]["tweedie_deviance"] for row in section["folds"]]
        section[name] = {
            **measures(pooled, None, name),
            "tweedie_deviance": None if None in deviances else statistics.fmean(deviances),
            "settings": settings[name],
            "rounds": [row[name]["rounds"] for row in section["folds"]],
        }
    section["naive"] = measures(pooled, None, "naive_usd")
    difference = section[WINNER][SELECTION_MEASURE] - section[V2][SELECTION_MEASURE]
    section |= {
        "difference": difference,
        "win_threshold": threshold,
        "beats_v2": difference > threshold + ROUNDING,
    }
    if folder is not None:
        save_results(folder, results | {CONFIRMATION: section})
    return section


def _confirm_fold(
    data: LoopData, plan: LoopPlan, arms: Mapping[str, Settings], models_dir: Path | None
) -> tuple[dict, pl.DataFrame]:
    """One fold: each Arm's model (cached or trained) scores the fold's held-out UK grid and validation rows."""
    started = time.perf_counter()
    lp = Loop(data, plan, models_dir=models_dir)
    grid = lp.grid.select("hs6", "importer", "value_usd", "naive_usd")
    row: dict = {"fold": plan.fold, "uk_pairs": lp.grid.height, "setup_seconds": lp.results["setup_seconds"]}
    for name, settings in arms.items():
        key, trained, _, meta = lp.cached_model(settings)
        grid = grid.with_columns(lp.export_potential(settings, lp.grid).alias(name))
        validation = lp.validation.select("value_usd", "weight").with_columns(
            lp.export_potential(settings, lp.validation).alias(name)
        )
        row[name] = {
            **measures(grid, validation, name),
            "training_key": key,
            "trained": trained,
            **{k: meta[k] for k in ("rounds", "trained_rounds", "training_rows", "train_seconds") if k in meta},
        }
        del validation
    row["seconds"] = round(time.perf_counter() - started, 1)
    return row, grid
