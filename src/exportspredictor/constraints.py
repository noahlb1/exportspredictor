"""What the monotone constraints do to Scenario Effects: a constrained and an unconstrained model compared.

Every policy feature carries a +1 monotone constraint (ADR 0002), so the model can never
associate a Trade Agreement with *lower* Export Potential. `compare` shows what that buys.
It trains two models that differ only in the constraint, with the same data, parameters,
rounds and seed, and asks both the Scenario question for the same UK pairs:

- **add**: for each pair with no Trade Agreement in the Baseline year, add one (the
  Blocked Opportunity question);
- **remove**: for each pair with one, remove it (and any Customs Union).

For each model and change it reports the share of pairs that move the "wrong" way (an
agreement that lowers Export Potential, or a removal that raises it), and the spread and
total of the Scenario Effects.

Both models follow ADR 0001: for each fold `k` in the plan, they train on every non-UK row
and the UK pairs outside fold `k`, and score only fold `k`'s UK pairs. To keep the real-data
cost small, `REAL_PLAN` trains on a few Sectors and one fold (a few minutes on the GPU); the
fixture plan uses every Sector and fold. The comparison is stored in the evaluation report
as `monotone_constraints` (`pipeline` computes it; `add_to_latest` adds it to a saved
artefact without retraining the fold models).

The effects are associations the model has learned, not causal estimates.
"""

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

from exportspredictor.features import FEATURES, FeatureTransform
from exportspredictor.folds import N_FOLDS, outside_fold
from exportspredictor.scenarios import check_only_policy_changed
from exportspredictor.training import Chunk, model_params, preparer, training_matrix

WRONG_SIGN_TOLERANCE = 1e-9  # effect shares smaller than this are treated as no change


@dataclass(frozen=True)
class Plan:
    sectors: tuple[str, ...] | None  # HS2 chapters trained and scored on; None: every Sector
    folds: tuple[int, ...]  # held-out folds: one pair of models per fold, scoring that fold's pairs
    max_rounds: int | None = None  # cap on the tuned round count, to bound the cost


FIXTURE_PLAN = Plan(sectors=None, folds=tuple(range(N_FOLDS)))
# Beverages (whisky), pharmaceuticals and vehicles: three large UK export Sectors with many
# pairs on both sides of a Trade Agreement. One fold keeps the run to two GPU models.
REAL_PLAN = Plan(sectors=("22", "30", "87"), folds=(0,))


def compare(
    chunks: Mapping[int, Chunk],
    baseline_features: pl.DataFrame,
    *,
    plan: Plan,
    uk: str,
    params: dict,
    rounds: int,
    transform: FeatureTransform,
    n_folds: int = N_FOLDS,
    non_uk_rate: float = 1.0,
    device: str = "cpu",
    seed: int = 0,
) -> dict:
    """Scenario Effects of adding and removing a Trade Agreement, with and without the constraint.

    `chunks` are the training years' feature tables (untransformed, as for the fold models);
    `baseline_features` are the Baseline's transformed UK rows with their `fold`; `transform`
    is the fold models' feature transform. Returns the `monotone_constraints` report.
    """
    started = time.perf_counter()
    rounds = min(rounds, plan.max_rounds) if plan.max_rounds else rounds
    years = sorted(chunks)
    in_sectors = _in_sectors(plan.sectors)
    scored = []
    for k in plan.folds:
        held_out = outside_fold(uk, n_folds, k)
        prepare = preparer(
            uk=uk, transform=transform, non_uk_rate=non_uk_rate, seed=seed,
            keep=lambda table, held_out=held_out: held_out(table.filter(in_sectors)),
        )
        matrix = training_matrix([chunks[y] for y in years], prepare)
        constrained = model_params(params, device=device, seed=seed)
        unconstrained = {key: value for key, value in constrained.items() if key != "monotone_constraints"}
        models = {
            name: xgb.train(config, matrix, num_boost_round=rounds)
            for name, config in (("constrained", constrained), ("unconstrained", unconstrained))
        }
        del matrix
        pairs = baseline_features.filter((pl.col("fold") == k) & in_sectors)
        for name, model in models.items():
            model.set_param({"device": "cpu"})
            scored.append(_effects(pairs, model).with_columns(model=pl.lit(name), fold=pl.lit(k)))
    effects = pl.concat(scored)
    report = {
        "sectors": list(plan.sectors) if plan.sectors else None,
        "folds": list(plan.folds),
        "training_years": years,
        "rounds": rounds,
        "non_uk_sampling_rate": non_uk_rate,
        "models": [_summary(effects.filter(pl.col("model") == name), name) for name in ("constrained", "unconstrained")],
        "baseline_spearman": _baseline_agreement(effects),
        "seconds": round(time.perf_counter() - started, 2),
    }
    report["text"] = _text(report)
    return report


def _in_sectors(sectors: Sequence[str] | None) -> pl.Expr:
    if not sectors:
        return pl.lit(True)
    return pl.col("hs6").cast(pl.String).str.slice(0, 2).is_in(list(sectors))


def _effects(pairs: pl.DataFrame, model: xgb.Booster) -> pl.DataFrame:
    """Each pair's Baseline prediction and its Scenario Effect of adding or removing an agreement."""
    without = pairs.filter(pl.col("trade_agreement") == 0)
    with_ = pairs.filter(pl.col("trade_agreement") == 1)
    added = without.with_columns(trade_agreement=pl.lit(1, pairs.schema["trade_agreement"]))
    removed = with_.with_columns(
        trade_agreement=pl.lit(0, pairs.schema["trade_agreement"]),
        customs_union=pl.lit(0, pairs.schema["customs_union"]),
    )
    for before, after in ((without, added), (with_, removed)):
        check_only_policy_changed(before, after)
    frames = []
    for change, before, after in (("add", without, added), ("remove", with_, removed)):
        baseline, scenario = _predict(model, before), _predict(model, after)
        frames.append(
            before.select("hs6", "iso3").with_columns(
                change=pl.lit(change),
                baseline_usd=pl.Series(baseline),
                effect_usd=pl.Series(scenario - baseline),
            )
        )
    return pl.concat(frames).with_columns(
        effect_share=pl.when(pl.col("baseline_usd") > 0).then(pl.col("effect_usd") / pl.col("baseline_usd")).otherwise(0.0)
    )


def _predict(model: xgb.Booster, rows: pl.DataFrame) -> np.ndarray:
    if rows.is_empty():
        return np.zeros(0)
    return model.inplace_predict(rows.select(FEATURES).to_numpy().astype(np.float32)).astype(np.float64)


def _summary(effects: pl.DataFrame, name: str) -> dict:
    summary = {"model": name}
    for change, wrong in (("add", pl.col("effect_share") < -WRONG_SIGN_TOLERANCE),
                          ("remove", pl.col("effect_share") > WRONG_SIGN_TOLERANCE)):
        rows = effects.filter(pl.col("change") == change)
        baseline = float(rows["baseline_usd"].sum())
        summary[change] = {
            "pairs": rows.height,
            "wrong_sign_pairs": rows.filter(wrong).height,
            "wrong_sign_share": rows.filter(wrong).height / rows.height if rows.height else None,
            "min_effect_share": _stat(rows["effect_share"].min()),
            "median_effect_share": _stat(rows["effect_share"].median()),
            "max_effect_share": _stat(rows["effect_share"].max()),
            "effect_usd": float(rows["effect_usd"].sum()),
            "effect_share": float(rows["effect_usd"].sum()) / baseline if baseline else None,
        }
    return summary


def _stat(value) -> float | None:
    return None if value is None else float(value)


def _baseline_agreement(effects: pl.DataFrame) -> float | None:
    """Spearman correlation of the two models' Baseline predictions over the scored pairs."""
    wide = (
        effects.select("hs6", "iso3", "change", "model", "baseline_usd")
        .pivot("model", index=["hs6", "iso3", "change"], values="baseline_usd")
    )
    if wide.height < 2:
        return None
    return _stat(wide.select(pl.corr("constrained", "unconstrained", method="spearman")).item())


def _share(value: float | None) -> str:
    return "–" if value is None else f"{value:.1%}"


def _signed(value: float | None) -> str:
    return "–" if value is None else f"{value:+.1%}"


def _text(report: dict) -> str:
    constrained, unconstrained = report["models"]
    add, remove = unconstrained["add"], unconstrained["remove"]
    wrong = constrained["add"]["wrong_sign_pairs"] + constrained["remove"]["wrong_sign_pairs"]
    spearman = report["baseline_spearman"]
    return (
        f"Without the monotone constraint, the model associates adding a Trade Agreement with lower "
        f"Export Potential for {_share(add['wrong_sign_share'])} of {add['pairs']:,} UK pairs (smallest "
        f"effect {_signed(add['min_effect_share'])}), and removing one with higher Export Potential for "
        f"{_share(remove['wrong_sign_share'])} of {remove['pairs']:,} pairs (largest effect "
        f"{_signed(remove['max_effect_share'])}). With the constraint, {wrong} pairs move the wrong way. "
        f"In total, adding an agreement is associated with {_signed(constrained['add']['effect_share'])} "
        f"Export Potential for these pairs with the constraint and {_signed(add['effect_share'])} without"
        + (f"; the two models' Baselines rank the pairs alike (Spearman {spearman:.3f})." if spearman is not None else ".")
    )


def add_to_latest(
    root: Path | None = None,
    artefacts_root: Path | None = None,
    *,
    plan: Plan = REAL_PLAN,
    device: str = "cuda",
    version: str | None = None,
) -> Path:
    """Run the comparison for the newest evaluated artefact and save it again, with it, as a new version.

    The two models use the artefact's tuned parameters, round count, feature transform,
    non-UK sample and seed. The fold models and every file are copied as they are.
    """
    from exportspredictor import artefacts as store
    from exportspredictor import baci
    from exportspredictor.features import features_dir
    from exportspredictor.paths import artefacts_dir, data_dir

    root = Path(root or data_dir())
    artefacts_root = Path(artefacts_root or artefacts_dir())
    source = store.latest_folder(artefacts_root)
    bundle = store.load(source)
    if bundle.evaluation is None:
        raise ValueError(f"Artefact {bundle.version} has no evaluation; run `python -m exportspredictor.evaluation` first")
    metadata = bundle.metadata
    folder = features_dir(root)
    years = [y for y in json.loads((folder / "features.json").read_text(encoding="utf-8"))["years"]
             if y in metadata["training_years"]]
    result = compare(
        {y: folder / f"training_{y}.parquet" for y in years},
        bundle.baseline_features,
        plan=plan,
        uk=baci.UK,
        params=metadata["params"],
        rounds=metadata["rounds"],
        transform=FeatureTransform.from_dict(metadata["feature_transform"]),
        n_folds=metadata["n_folds"],
        non_uk_rate=metadata["non_uk_sampling_rate"],
        device=device,
        seed=metadata["model_params"].get("seed", 0),
    )
    print(result["text"], flush=True)
    report = {**bundle.evaluation, "monotone_constraints": result}
    new = {**metadata, "version": version or store.new_version(), "copied_from": bundle.version}
    return store.copy_version(source, artefacts_root, metadata=new, evaluation=report)


if __name__ == "__main__":
    print(add_to_latest(), flush=True)
