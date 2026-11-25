"""The evidence for the Offset Pair tier. CPU only; trains nothing, reads no 2023 file.

The scripts used when the idea was first explored were not kept, so this one recomputes every figure
in `docs/research/offset-pairs.md`. It applies the tier's rule on its own, without the package's
Offset Pair code, so the two can be checked against each other.

From the served artefact (the lexically newest version folder, `20261120T221726Z` when written):

- **Heading 8411 in France and Germany**: Actual Exports and Export Potential for turbo-jets (841112),
  turbine parts (841191) and every scored Product of the Heading summed.
- **The rule.** A Heading is the first four digits of a Product's HS6 code. For each Market and Heading,
  the Heading gap is the summed Export Gap of the Heading's scored Products in that Market. A pair is an
  Offset Pair when its Export Gap is positive, the Heading has at least two scored Products in that
  Market, and the Heading gap is at most (1 - `THRESHOLD`) times its Export Gap.
- **Reach**: how many Opportunities the rule marks, in all and in the top 100, 1,000 and 10,000 by Export
  Gap (ties broken by Product, then Market), and their share of the summed positive Export Gap.

From the experiment loop's five-fold confirmation of 2022 (`<loop_dir>/confirmation/fold_k.parquet`,
each pair scored by the fold model that held it out):

- for pairs in a Heading with at least two Products in the Market, the residual
  log(1 + Actual Exports) - log(1 + Export Potential), and the same log ratio of the rest of the
  Heading's summed figures in that Market;
- their Pearson correlation among pairs whose Actual Exports or Export Potential exceeds $10m;
- the same correlation within each Heading holding at least `MIN_PAIRS` such pairs, and how many of
  those Headings are negative.

The 2022 Export Potential is the served Blend (`winner`). The served Model Form adds Market Reach
(ADR 0010), so the correlation is also reported with Reach applied at the served beta, as
`scripts/research/market_reach_2022/run.py` applies it (offsets from the 2021 UK grid).

Writes `<data>/derived/offset_pairs/result.json`.

    scripts/wt.ps1 python scripts/research/offset_pairs/run.py
"""

import json

import numpy as np
import polars as pl

from exportspredictor import market_reach
from exportspredictor.artefacts import latest_folder
from exportspredictor.folds import assign_folds
from exportspredictor.loop import loop_dir
from exportspredictor.paths import artefacts_dir, data_dir

THRESHOLD = 0.5
DEPTHS = (100, 1_000, 10_000)
YEAR = 2022
BASE = "winner"  # the served Blend in the loop's confirmation
DOLLAR_FLOOR = 1e7
MIN_PAIRS = 15
HEADING_EXAMPLE = "8411"
PRODUCT_EXAMPLES = ("841112", "841191")
MARKET_EXAMPLES = ("FRA", "DEU")
OUT = data_dir() / "derived" / "offset_pairs"


def with_heading_sums(rows: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """Adds each pair's Heading, its Market's count of the Heading's Products, and the Heading's sums."""
    rows = rows.with_columns(heading=pl.col("hs6").str.slice(0, 4))
    group = ["iso3", "heading"]
    return rows.with_columns(
        products_in_heading=pl.len().over(group),
        **{f"heading_{c}": pl.col(c).sum().over(group) for c in columns},
    )


def mark_offset_pairs(baseline: pl.DataFrame) -> pl.DataFrame:
    """The tier's rule, applied to the Baseline's own Export Gaps."""
    rows = with_heading_sums(baseline, ["export_gap_usd"])
    return rows.with_columns(
        offset_pair=(pl.col("export_gap_usd") > 0)
        & (pl.col("products_in_heading") >= 2)
        & (pl.col("heading_export_gap_usd") <= (1 - THRESHOLD) * pl.col("export_gap_usd")),
        offset_usd=pl.col("export_gap_usd") - pl.col("heading_export_gap_usd"),
    )


def heading_table(baseline: pl.DataFrame, products: pl.DataFrame) -> list[dict]:
    rows = baseline.filter(pl.col("iso3").is_in(MARKET_EXAMPLES), pl.col("hs6").str.starts_with(HEADING_EXAMPLE))
    table = []
    for market in MARKET_EXAMPLES:
        in_market = rows.filter(pl.col("iso3") == market)
        for hs6 in PRODUCT_EXAMPLES:
            row = in_market.filter(pl.col("hs6") == hs6).row(0, named=True)
            table.append({"market": market, "column": hs6,
                          "description": products.filter(pl.col("hs6") == hs6)["description"].item(),
                          "actual_exports_usd": row["actual_exports_usd"],
                          "export_potential_usd": row["export_potential_usd"],
                          "export_gap_usd": row["export_gap_usd"]})
        table.append({"market": market, "column": f"Heading {HEADING_EXAMPLE}", "products": in_market.height,
                      "actual_exports_usd": in_market["actual_exports_usd"].sum(),
                      "export_potential_usd": in_market["export_potential_usd"].sum(),
                      "export_gap_usd": in_market["export_gap_usd"].sum()})
    return table


def reach(marked: pl.DataFrame) -> dict:
    opportunities = marked.filter(pl.col("export_gap_usd") > 0).sort(
        ["export_gap_usd", "hs6", "iso3"], descending=[True, False, False]
    )
    offset = opportunities.filter("offset_pair")
    return {
        "threshold": THRESHOLD,
        "pairs": marked.height,
        "opportunities": opportunities.height,
        "offset_pairs": offset.height,
        "share_of_opportunities": offset.height / opportunities.height,
        "top": {str(n): int(opportunities.head(n)["offset_pair"].sum()) for n in DEPTHS},
        "offset_gap_usd": offset["export_gap_usd"].sum(),
        "positive_gap_usd": opportunities["export_gap_usd"].sum(),
        "share_of_positive_gap": offset["export_gap_usd"].sum() / opportunities["export_gap_usd"].sum(),
        "fully_offset_top": {
            str(n): int(opportunities.head(n).select((pl.col("offset_pair") & (pl.col("heading_export_gap_usd") <= 0))
                                                      .sum()).item()) for n in DEPTHS
        },
        "largest": offset.head(10).select("hs6", "iso3", "export_gap_usd", "offset_usd").to_dicts(),
    }


def held_out_2022() -> pl.DataFrame:
    return pl.concat([
        pl.read_parquet(loop_dir() / "confirmation" / f"fold_{k}.parquet").select(
            pl.col("hs6").cast(pl.String), iso3=pl.col("importer").cast(pl.String), value_usd="value_usd",
            potential=BASE, fold=pl.lit(k, pl.Int8),
        )
        for k in range(5)
    ])


def with_reach(rows: pl.DataFrame, beta: float) -> pl.DataFrame:
    """The Blend with Market Reach at `beta`, offsets from the 2021 UK grid, one constant over the five folds."""
    lag = YEAR - market_reach.LAG_YEARS
    grid = pl.read_parquet(
        data_dir() / "derived" / "features_V202601" / f"uk_grid_{lag}.parquet",
        columns=["hs6", "importer", "value_usd", market_reach.DEMAND],
    ).select(pl.col("hs6").cast(pl.String), iso3=pl.col("importer").cast(pl.String), value_usd="value_usd",
             log_demand_loo=market_reach.DEMAND)
    before = assign_folds(grid).rename({"iso3": "importer"}).with_columns(year=pl.lit(lag, pl.Int16))
    table = market_reach.reach_table(before, YEAR)
    keyed = rows.rename({"iso3": "importer"})
    potential = keyed["potential"].to_numpy()
    constant = market_reach.rescale_constant(potential, keyed, table, beta=beta)
    return rows.with_columns(potential=pl.Series(market_reach.apply(potential, keyed, table, beta=beta,
                                                                    constant=constant)))


def sibling_correlation(rows: pl.DataFrame) -> dict:
    rows = with_heading_sums(rows, ["value_usd", "potential"]).filter(pl.col("products_in_heading") >= 2)
    rows = rows.with_columns(
        residual=pl.col("value_usd").log1p() - pl.col("potential").log1p(),
        rest_residual=(pl.col("heading_value_usd") - pl.col("value_usd")).clip(lower_bound=0).log1p()
        - (pl.col("heading_potential") - pl.col("potential")).clip(lower_bound=0).log1p(),
    ).filter((pl.col("value_usd") > DOLLAR_FLOOR) | (pl.col("potential") > DOLLAR_FLOOR))
    by_heading = rows.group_by("heading").agg(pairs=pl.len(), r=pl.corr("residual", "rest_residual")).filter(
        pl.col("pairs") >= MIN_PAIRS
    ).sort("heading")
    return {
        "pairs": rows.height,
        "correlation": float(np.corrcoef(rows["residual"], rows["rest_residual"])[0, 1]),
        "headings": by_heading.height,
        "negative_headings": by_heading.filter(pl.col("r") < 0).height,
        "median_heading_r": by_heading["r"].median(),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    folder = latest_folder(artefacts_dir())
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    baseline = pl.read_parquet(
        folder / "baseline.parquet",
        columns=["hs6", "iso3", "export_potential_usd", "actual_exports_usd", "export_gap_usd"],
    ).with_columns(pl.col("hs6").cast(pl.String), pl.col("iso3").cast(pl.String))
    products = pl.read_parquet(folder / "products.parquet").with_columns(pl.col("hs6").cast(pl.String))

    marked = mark_offset_pairs(baseline)
    beta = metadata["market_reach"]["beta"]
    rows = held_out_2022()
    result = {
        "artefact": folder.name,
        "baseline_year": metadata["baseline_year"],
        "heading_8411": heading_table(baseline, products),
        "reach": reach(marked),
        "correlation_2022": {
            "year": YEAR, "dollar_floor": DOLLAR_FLOOR, "min_pairs": MIN_PAIRS,
            "blend": sibling_correlation(rows),
            "blend_with_market_reach": {"beta": beta, **sibling_correlation(with_reach(rows, beta))},
        },
    }
    (OUT / "result.json").write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    print(json.dumps(result, indent=1, default=float))


if __name__ == "__main__":
    main()
