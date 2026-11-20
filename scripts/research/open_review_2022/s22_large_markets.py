"""The Selection Measure on the 30 and 60 largest importing Markets of 2022 (ticket #123); reads 2022 only.

Reproduces the open review's figures (the Blend 0.507 on the top 30 and 0.544 on the top 60, measured
before ties were broken, #115) with the package's `large_markets.restricted_measures`, and gives the
tie-proof figures beside them. Markets are ranked by their total 2022 imports from every Exporter.
From the worktree root:

    scripts/wt.ps1 python scripts/research/open_review_2022/s22_large_markets.py
"""

import json

import polars as pl
from common import DATA, LOOP, OUT, YEAR

from exportspredictor.features import load_year_flows
from exportspredictor.large_markets import largest_importers, restricted_measures
from exportspredictor.selection_measure import (
    legacy_selection_measure,
    selection_measure,
)

COUNTS = (30, 60, None)  # None: every Market, the headline measure
COLUMNS = {"Blend (served, w = 0.3)": "winner", "v2, the model alone": "v2", "Naive Baseline": "naive_usd"}

pairs = pl.concat(
    pl.read_parquet(LOOP / "confirmation" / f"fold_{k}.parquet", columns=["hs6", "importer", "value_usd", *COLUMNS.values()])
    .with_columns(pl.col("hs6").cast(pl.String), pl.col("importer").cast(pl.String))
    for k in range(5)
)
country_map = pl.read_parquet(DATA / "derived" / "country_map.parquet").select("baci_iso3", "iso3").unique()
# How the evaluation ranks Markets: total imports, every Exporter's Flows of every Product.
every_flow = load_year_flows(YEAR, dict(country_map.iter_rows()), DATA).select("importer", "value_usd")
# How the review ranked them (s05_anyflow): each Market's leave-one-out imports of the UK grid's Products,
# summed over the grid; that leaves out the UK's own Flows and the Products the grid does not hold.
review = pl.read_parquet(DATA / "derived" / "features_V202601" / f"uk_grid_{YEAR}.parquet",
                         columns=["importer", "log_demand_loo"]).select(
    pl.col("importer").cast(pl.String), value_usd=pl.col("log_demand_loo").exp() - 1
)
RANKINGS = {"as the review ranked them (s05)": review, "as the evaluation ranks them": every_flow}
markets = pairs["importer"].unique().to_list()
counts = [n or len(markets) for n in COUNTS]

result = {"year": YEAR, "pairs": pairs.height, "markets": len(markets), "measures": [],
          "top_30": {name: largest_importers(flows, markets, 30) for name, flows in RANKINGS.items()}}
for ranking, flows in RANKINGS.items():
    for name, column in COLUMNS.items():
        for label, measure in (("legacy (before #115)", legacy_selection_measure), ("tie-proof", selection_measure)):
            measured = restricted_measures(pairs, flows, column, counts=counts, measure=measure)
            row = {"markets ranked": ranking, "predictor": name, "measure": label,
                   **{f"top_{n}": measured[n] for n in counts[:-1]}, "every_market": measured[counts[-1]]}
            result["measures"].append(row)
            print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
review_30, evaluation_30 = (set(top) for top in result["top_30"].values())
print("Top 30 only as the review ranked them:", sorted(review_30 - evaluation_30),
      "; only as the evaluation ranks them:", sorted(evaluation_30 - review_30))
(OUT / "s22_large_markets.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
