"""UK×Market affinity: simpler, lagged and Sector-excluding variants, and which Markets move."""
import json

import polars as pl
from common import DATA, OUT, fit_check_test


def build_frame():
    pl.Config.set_tbl_rows(40); pl.Config.set_tbl_formatting("ASCII_MARKDOWN")
    df = pl.read_parquet(OUT / "frame_ranked.parquet")
    # 2021 UK Flows of every pair (the UK grid of 2021 carries the year's value): a lagged source, no 2022 Actual Exports at all
    g21 = pl.read_parquet(DATA / "derived" / "features_V202601" / "uk_grid_2021.parquet", columns=["importer", "hs6", "value_usd", "log_demand_loo"]).with_columns(
        pl.col("hs6").cast(pl.String)).rename({"value_usd": "value_2021", "log_demand_loo": "log_demand_2021"})
    df = df.join(g21, on=["hs6", "importer"], how="left").with_columns(pl.col("value_2021", "log_demand_2021").fill_null(0.0))
    df.write_parquet(OUT / "frame_ranked21.parquet")
    return df


def breadth_offset(value_col, demand_col, keys=("importer",), a=5.0):
    """log of the UK's share of the Market's imported Products: (# Products the UK sends) / (# Products the Market
    imports from anyone else), from the source pairs only."""
    keys = list(keys)
    def fn(src):
        return src.group_by(keys).agg(uk=(pl.col(value_col) > 0).sum(), mk=(pl.col(demand_col) > 0).sum()).select(
            *keys, offset=((pl.col("uk") + a) / (pl.col("mk") + 2 * a)).log())
    return fn


def other_sector(value_col, demand_col, a=5.0):
    """As breadth_offset by Market, but per (Market, Sector) from the *other* Sectors only."""
    def fn(src):
        per = src.group_by("importer", "hs2").agg(uk=(pl.col(value_col) > 0).sum(), mk=(pl.col(demand_col) > 0).sum())
        tot = per.group_by("importer").agg(ukt=pl.col("uk").sum(), mkt=pl.col("mk").sum())
        return per.join(tot, on="importer").select(
            "importer", "hs2", offset=((pl.col("ukt") - pl.col("uk") + a) / (pl.col("mkt") - pl.col("mk") + 2 * a)).log())
    return fn



def main():
    df = build_frame()
    B = (0.0, 0.5, 1, 1.5, 2, 3, 4)
    res = [
        fit_check_test(df, breadth_offset("value_usd", "log_demand_loo"), label="breadth share by Market, 2022 other folds", betas=B),
        fit_check_test(df, breadth_offset("value_2021", "log_demand_2021"), label="breadth share by Market, 2021 other folds (lagged)", betas=B),
        fit_check_test(df, other_sector("value_usd", "log_demand_loo"), label="breadth share by Market from other Sectors, 2022", betas=B),
        fit_check_test(df, breadth_offset("value_usd", "log_demand_loo", ("importer", "hs2")), label="breadth share by Market x Sector, 2022", betas=B),
    ]
    (OUT / "s09_affinity_variants.json").write_text(json.dumps(res, indent=1, default=float))
    off = breadth_offset("value_usd", "log_demand_loo")(df).join(df.select("importer", "market").unique(), on="importer")
    big = ["USA", "CHN", "DEU", "FRA", "IRL", "NLD", "JPN", "IND", "MEX", "BRA", "RUS", "VNM", "KOR", "AUS", "CAN", "TUR", "EGY", "ZAF", "NGA", "ARE", "SAU", "IRN", "COL", "BGR", "MLT", "GIB", "FLK"]
    print(off.filter(pl.col("importer").is_in(big)).sort("offset"))


if __name__ == "__main__":
    main()
