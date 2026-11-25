"""The figures of the Exporter-count experiment beside the loop's own (GPU for prediction; trains nothing).

The loop (`python -m exportspredictor.loop exporter-count`) decides the experiment and records each
Arm's measures in `results.json`. This script re-scores the same cached fold-0 models, so the note
can put beside them what the loop does not record:

- the cached base (the original files) with and without Market Reach, which reproduces the Market
  Reach experiment's 0.5771 and 0.6169, so the gravity fill's effect is measured on the same footing;
- the 10th-90th spread of the log residual above $10m of Actual Exports (the Market Reach
  experiment's dollar spread);
- the measure on the 30 and 60 largest importing Markets (reported, never selected on);
- the Exporter-count Arm at the base's blend weight, so its gain is not only a re-chosen weight.

Reads the 2022 validation year and the 2021 UK grid of each feature set, and the 2022 BACI Flows (for
the large-Market ranking). No 2023 file is opened and `results.json` is not written. Writes
`<data>/derived/exporter_count_2022/result.json`.

    scripts/wt.ps1 python scripts/research/exporter_count_2022/report.py
"""

import gc
import json
import os
from dataclasses import replace

import numpy as np
import polars as pl

from exportspredictor import loop, market_reach
from exportspredictor.features import FEATURE_SET_ENV, load_year_flows
from exportspredictor.large_markets import restricted_measures
from exportspredictor.loop import (
    EXPORTER_COUNT_EXPERIMENT,
    Loop,
    Settings,
    loop_dir,
    real_data,
    v2_plan,
)
from exportspredictor.loop.selection import measures
from exportspredictor.paths import data_dir

YEAR = 2022
DOLLAR_FLOOR = 1e7
OUT = data_dir() / "derived" / "exporter_count_2022"


def dollar_spread(rows: pl.DataFrame, column: str) -> dict:
    big = rows.filter(pl.col("value_usd") > DOLLAR_FLOOR)
    residual = np.log1p(big["value_usd"].to_numpy()) - np.log1p(big[column].to_numpy())
    p10, p50, p90 = np.percentile(residual, [10, 50, 90])
    return {"pairs": big.height, "p10": p10, "median": p50, "p90": p90, "spread": p90 - p10}


def score(feature_set: str, variants: dict[str, Settings], every_flow: pl.DataFrame) -> dict:
    os.environ[FEATURE_SET_ENV] = feature_set
    lp = Loop(real_data(), v2_plan(), models_dir=loop_dir() / loop.MODELS_DIR)
    out = {}
    for name, settings in variants.items():
        key, trained, _, meta = lp.cached_model(settings)
        assert not trained, f"{name}: its model was not in the cache"
        grid = lp.grid.with_columns(lp.export_potential(settings, lp.grid).alias("ep"))
        validation = lp.validation.with_columns(lp.export_potential(settings, lp.validation).alias("ep"))
        out[name] = {
            "feature_set": feature_set,
            "training_key": key,
            "rounds": meta["rounds"],
            "naive_weight": settings.naive_weight,
            "market_reach_beta": settings.market_reach_beta,
            **measures(grid, validation, "ep"),
            "dollar_spread": dollar_spread(grid, "ep"),
            "large_markets": {str(n): v for n, v in restricted_measures(grid, every_flow, "ep").items()},
            "summed_usd": float(grid["ep"].sum()),
        }
        print(name, json.dumps({k: out[name][k] for k in ("selection_measure", "rounds")}), flush=True)
        del grid, validation
    del lp
    gc.collect()
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results = loop.load()
    section = results["experiments"][EXPORTER_COUNT_EXPERIMENT]
    cached_base = Settings.from_dict(section["previous_best"]["settings"])
    base = Settings.from_dict(section["base_arm"]["settings"])
    (arm_row,) = section["arms"]
    arm = Settings.from_dict(arm_row["settings"])
    beta = market_reach.CHOSEN_BETA
    country_map = pl.read_parquet(data_dir() / "derived" / "country_map.parquet").select("baci_iso3", "iso3").unique()
    every_flow = load_year_flows(YEAR, dict(country_map.iter_rows()), data_dir()).select("importer", "value_usd")

    rows = score("", {
        "cached base": cached_base,
        "cached base + Market Reach": replace(cached_base, market_reach_beta=beta),
    }, every_flow)
    rows |= score("exporter_count", {
        "base retrained": replace(base, market_reach_beta=0.0),
        "base retrained + Market Reach": base,
        "+ Exporter count": replace(arm, market_reach_beta=0.0),
        "+ Exporter count + Market Reach": arm,
        "+ Exporter count + Market Reach, base weight": replace(arm, naive_weight=base.naive_weight),
    }, every_flow)
    sm = {name: row["selection_measure"] for name, row in rows.items()}
    result = {
        "year": YEAR,
        "win_threshold": loop.current_win_threshold(results),
        "gravity_fill_effect": sm["base retrained"] - sm["cached base"],
        "gravity_fill_effect_with_reach": sm["base retrained + Market Reach"] - sm["cached base + Market Reach"],
        "exporter_count_gain": sm["+ Exporter count + Market Reach"] - sm["base retrained + Market Reach"],
        "exporter_count_gain_base_weight": (
            sm["+ Exporter count + Market Reach, base weight"] - sm["base retrained + Market Reach"]
        ),
        "exporter_count_gain_without_reach": sm["+ Exporter count"] - sm["base retrained"],
        "loop_decision": {k: section[k] for k in ("kept", "win_threshold")},
        "rows": rows,
    }
    (OUT / "result.json").write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=1, default=float))


if __name__ == "__main__":
    main()
