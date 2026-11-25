"""Ticket #171, idea b: a scoring-only probe of a tariff of 0 where a Trade Agreement's preference is unrecorded.

TRAINS holds no preference schedule for some Trade Agreement partners, so their pairs carry the MFN rate while
the agreement dummy is 1 (#140's flag H3). This re-predicts fold 0's 2022 UK grid with the best so far as served
(the cached model `e5943f03616c0cf7`: Blend w = 0.3, Market Reach at beta 4, the Market Reach table unchanged)
after `tariffs.zero_unrecorded_preferences`, and compares the Selection Measure with the uncorrected grid. The
Naive Baseline has no policy feature, so it does not move. Nothing is trained (the script refuses a model that is
not cached) and `results.json` is not written. The pre-registered gate (docs/research/final-round-2022.md, section
8): the retrained Arm runs only if the probe gains more than half the Win Threshold.

Reads the 2021 and 2022 files of the served feature set; no 2023 file is opened. Writes
`<data>/derived/final_round_2022/unrecorded_preferences.json`.

    $env:EXPORTSPREDICTOR_FEATURE_SET='exporter_count'; scripts/wt.ps1 python scripts/research/final_round_2022/unrecorded_preferences.py
"""

import json

import polars as pl

from exportspredictor import loop
from exportspredictor.loop import (
    Loop,
    Settings,
    current_win_threshold,
    load,
    loop_dir,
    real_data,
)
from exportspredictor.loop.selection import SELECTION_MEASURE, measures
from exportspredictor.loop.served import kept_feature_set
from exportspredictor.paths import data_dir
from exportspredictor.tariffs import zero_unrecorded_preferences

BASE_KEY = "e5943f03616c0cf7"
OUT = data_dir() / "derived" / "final_round_2022"


def main() -> None:
    results = load()
    data = real_data()
    if data.feature_set != kept_feature_set(results):
        raise SystemExit(f"Run on the served feature files ({kept_feature_set(results)!r}), not {data.feature_set!r}")
    best = Settings.from_dict(results["best"]["settings"])
    settings = loop.served_base_arm(best).apply(best)
    lp = Loop(data, loop.v2_plan(), models_dir=loop_dir() / loop.MODELS_DIR)
    key, trained, _, _ = lp.cached_model(settings)
    assert not trained and key == BASE_KEY, f"expected the cached base {BASE_KEY}, got {key} (trained: {trained})"

    grid = lp.grid
    corrected = zero_unrecorded_preferences(grid)
    corrected = corrected.with_columns(changed=corrected["log_applied_tariff"].ne_missing(grid["log_applied_tariff"]))
    marked = corrected.filter(pl.col("unrecorded_preference"))
    markets = (
        marked.group_by("importer")
        .agg(rows=pl.len(), rows_changed=pl.col("changed").sum(), mean_mfn=pl.col("mfn_tariff").mean())
        .sort("importer")
    )

    scored = {}
    for name, table in (("base", grid), ("probe", corrected.drop("unrecorded_preference", "changed"))):
        potential = lp.export_potential(settings, table)
        scored[name] = measures(grid.with_columns(export_potential_usd=potential), None, "export_potential_usd")
    threshold = current_win_threshold(results)
    gain = scored["probe"][SELECTION_MEASURE] - scored["base"][SELECTION_MEASURE]
    out = {
        "model": key,
        "settings": settings.to_dict(),
        "grid_pairs": grid.height,
        "rows_marked": marked.height,
        "rows_changed": int(corrected["changed"].sum()),
        "markets": markets.to_dicts(),
        "base": scored["base"],
        "probe": scored["probe"],
        "gain": gain,
        "win_threshold": threshold,
        "gate": threshold / 2,
        "retrained_arm_runs": gain > threshold / 2,
        "validation_year": data.validation_year,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "unrecorded_preferences.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("rows_marked", "rows_changed", "gain", "gate", "retrained_arm_runs")}))
    print(markets)


if __name__ == "__main__":
    main()
