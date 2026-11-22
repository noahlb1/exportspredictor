"""Ticket #124: the Market Reach experiment on the 2022 validation year (CPU; trains nothing, reads no 2023 data).

Pre-registered before any result was read (the commit that adds this script and
`exportspredictor.market_reach`):

- **Form**: `market_reach.FORM` ("share"), pseudo-count `market_reach.PSEUDO_COUNT` (5), lag
  `market_reach.LAG_YEARS` (1): fold k's offsets for 2022 come from the UK pairs outside fold k in 2021.
- **Base**: the served Blend (`winner` in the loop's five-fold confirmation, w = 0.3), whose
  tie-proof pooled measure is 0.5635 (#116).
- **Choice of β**: the β of `market_reach.BETA_GRID` with the highest tie-proof Selection Measure on
  the pooled rows of folds 0–2 (each pair with its own fold's offsets); a tie goes to the smaller β.
- **Gate**: adopt only if the chosen β's tie-proof measure on the pooled five-fold 2022 grid beats the
  Blend's by more than `loop.current_win_threshold(loop.load())` (0.008869 after #116).
- **Reported whatever happens**: folds 3–4 pooled and each alone, the loop's diagnostics, the 10th–90th
  spread of the log residual on pairs with Actual Exports above $10m, the measure on the 30 and 60
  largest importing Markets (never selected on), the old measure for continuity, and example Markets.

Reads the loop's confirmation folds (2022 held-out scores), the 2021 UK grid and the 2022 BACI Flows (for
the large-Market measure). Writes `<data>/derived/market_reach_2022/<FORM>/result.json` and `reach_2022.parquet`.

    scripts/wt.ps1 python scripts/research/market_reach_2022/run.py
"""

import json

import numpy as np
import polars as pl

from exportspredictor import market_reach
from exportspredictor.artefacts import latest_folder
from exportspredictor.features import load_year_flows
from exportspredictor.folds import assign_folds
from exportspredictor.large_markets import restricted_measures
from exportspredictor.loop import current_win_threshold, load, loop_dir
from exportspredictor.loop.selection import measures
from exportspredictor.paths import artefacts_dir, data_dir
from exportspredictor.selection_measure import (
    legacy_selection_measure,
    selection_measure,
    with_tie_key,
)

YEAR = 2022
BASE = "winner"  # the served Blend
FIT_FOLDS, CHECK_FOLDS = (0, 1, 2), (3, 4)
DOLLAR_FLOOR = 1e7
EXAMPLES = ["USA", "DEU", "FRA", "CHN", "NLD", "IRL", "JPN", "IND", "KOR", "MEX", "BRA", "VNM", "COL", "BGR",
            "RUS", "IRN", "SAU", "MLT", "CYP", "FLK"]
# #124 wrote the "share" form to the folder itself; later forms write to a subfolder named after the form.
OUT = data_dir() / "derived" / "market_reach_2022" / market_reach.FORM


def held_out_2022() -> pl.DataFrame:
    folds = []
    for k in range(5):
        part = pl.read_parquet(loop_dir() / "confirmation" / f"fold_{k}.parquet").with_columns(
            pl.col("hs6").cast(pl.String), pl.col("importer").cast(pl.String), fold=pl.lit(k, pl.Int8)
        )
        folds.append(part)
    return pl.concat(folds)


def uk_rows_before() -> pl.DataFrame:
    """The 2021 UK grid with each pair's fold (the same hash as the confirmation's)."""
    lag = YEAR - market_reach.LAG_YEARS
    grid = pl.read_parquet(
        data_dir() / "derived" / "features_V202601" / f"uk_grid_{lag}.parquet",
        columns=["hs6", "importer", "value_usd", market_reach.DEMAND],
    ).select(pl.col("hs6").cast(pl.String), iso3=pl.col("importer").cast(pl.String), value_usd="value_usd",
             log_demand_loo=market_reach.DEMAND)
    return assign_folds(grid).rename({"iso3": "importer"}).with_columns(year=pl.lit(lag, pl.Int16))


def dollar_spread(rows: pl.DataFrame, column: str) -> dict:
    big = rows.filter(pl.col("value_usd") > DOLLAR_FLOOR)
    residual = np.log1p(big["value_usd"].to_numpy()) - np.log1p(big[column].to_numpy())
    p10, p50, p90 = np.percentile(residual, [10, 50, 90])
    return {"pairs": big.height, "p10": p10, "median": p50, "p90": p90, "spread": p90 - p10}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = held_out_2022()
    before = uk_rows_before()
    check_folds = rows.select("hs6", "importer", "fold").join(
        before.select("hs6", "importer", pl.col("fold").alias("hash_fold")), on=["hs6", "importer"], how="left"
    )
    mismatched = check_folds.filter(pl.col("fold") != pl.col("hash_fold")).height
    assert mismatched == 0, f"{mismatched} pairs' confirmation fold differs from the pair hash"

    table = market_reach.reach_table(before, YEAR)
    table.write_parquet(OUT / f"reach_{YEAR}.parquet")
    rows = with_tie_key(rows)
    potential = rows[BASE].to_numpy()
    for beta in market_reach.BETA_GRID:
        constant = market_reach.rescale_constant(potential, rows, table, beta=beta)
        rows = rows.with_columns(pl.Series(f"reach_{beta:g}", market_reach.apply(potential, rows, table, beta=beta,
                                                                                  constant=constant)))
    threshold = current_win_threshold(load())

    def subset(folds) -> pl.DataFrame:
        return rows.filter(pl.col("fold").is_in(list(folds)))

    columns = [BASE] + [f"reach_{b:g}" for b in market_reach.BETA_GRID]
    fit = {c: selection_measure(subset(FIT_FOLDS), c) for c in columns}
    chosen = max(market_reach.BETA_GRID, key=lambda b: (round(fit[f"reach_{b:g}"], 12), -b))
    col = f"reach_{chosen:g}"

    every_flow = load_year_flows(
        YEAR, dict(pl.read_parquet(data_dir() / "derived" / "country_map.parquet")
                   .select("baci_iso3", "iso3").unique().iter_rows()), data_dir()
    ).select("importer", "value_usd")

    by_beta = []
    for c in columns:
        entry = {
            "column": c,
            "fit_folds_0_2": fit[c],
            "check_folds_3_4": selection_measure(subset(CHECK_FOLDS), c),
            **{f"fold_{k}": selection_measure(subset([k]), c) for k in range(5)},
            "pooled": measures(rows, None, c),
            "pooled_legacy": legacy_selection_measure(rows, c),
            "dollar_spread": dollar_spread(rows, c),
            "large_markets": {str(n): v for n, v in restricted_measures(rows, every_flow, c).items()},
            "summed_usd": float(rows[c].sum()),
        }
        entry["pooled"].pop("tweedie_deviance")
        by_beta.append(entry)
        print({k: v for k, v in entry.items() if k != "pooled"} | {"pooled": entry["pooled"]["selection_measure"]},
              flush=True)

    base, best = by_beta[0], by_beta[columns.index(col)]
    gain = best["pooled"]["selection_measure"] - base["pooled"]["selection_measure"]
    names = pl.read_parquet(latest_folder(artefacts_dir()) / "markets.parquet").select(importer="iso3", market="name")
    examples = table.filter(pl.col("importer").is_in(EXAMPLES)).group_by("importer").agg(
        products_sent=pl.col("products_sent").mean(), products_imported=pl.col("products_imported").mean(),
        products_expected=pl.col("products_expected").mean(), offset=pl.col("offset").mean(),
        offset_min=pl.col("offset").min(), offset_max=pl.col("offset").max(),
    ).sort("offset")
    examples = examples.join(names, on="importer", how="left")
    result = {
        "year": YEAR, "pairs": rows.height, "form": market_reach.FORM, "pseudo_count": market_reach.PSEUDO_COUNT,
        "lag_years": market_reach.LAG_YEARS, "beta_grid": list(market_reach.BETA_GRID),
        "chosen_beta": chosen, "win_threshold": threshold, "pooled_gain": gain,
        "check_gain": best["check_folds_3_4"] - base["check_folds_3_4"],
        "fold_gains": {k: best[f"fold_{k}"] - base[f"fold_{k}"] for k in range(5)},
        "decision": "adopt" if gain > threshold else "drop",
        "by_beta": by_beta, "examples": examples.to_dicts(),
        "offset_quantiles": {str(q): table["offset"].quantile(q) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
    }
    (OUT / "result.json").write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("by_beta",)}, indent=1, default=float))


if __name__ == "__main__":
    main()
