"""The Review Pack: the served outputs an Advisor sees, plus each Market's total imports, for a fresh reviewer.

`build` selects and joins what `rankings`, `scenarios` and the artefact already compute, and adds no scoring,
threshold or ranking rule of its own. It hides no Caveat Tier and applies no app filter, so the reviewer sees every
row the app would badge or hide. Every table names its Products and Markets beside their codes, in nominal USD, and
holds no feature, fold or model column: the reviewer judges the outputs, not how they were made.

`python -m exportspredictor.review` builds the pack from the newest artefact, with each Market's total imports summed
from the Baseline year's BACI Flows from all Exporters and keyed by standard ISO3 (`market_total_imports`), and writes one CSV per table and a
README naming each table and its columns (`write`) to `data/derived/review_pack_<version>/`.
"""

import time
from pathlib import Path

import polars as pl

from exportspredictor import baci, rankings, scenarios
from exportspredictor.artefacts import Artefacts, load_latest
from exportspredictor.caveats import TIERS
from exportspredictor.downloads import writing
from exportspredictor.paths import artefacts_dir, data_dir
from exportspredictor.sectors import with_sector_names

TOP = 50  # the top Opportunities and Blocked Opportunities overall, and the largest Scenario Effects per preset
GROUPS = 10  # the top Markets and Products (by total Export Gap among Opportunities) given their own top rows
GROUP_TOP = 10  # the top Opportunities in each of those Markets and Products
SAMPLE_SIZE = 50  # the Product–Market pairs drawn at random from the whole Baseline
SEED = 0  # the pair sample's seed for `python -m exportspredictor.review`

# Each table, in the README's order, with what it holds in plain domain terms.
TABLES = {
    "top_opportunities": f"The {TOP} largest Opportunities, by Export Gap, as the Leaderboard ranks them.",
    "top_opportunities_by_market": (
        f"The {GROUP_TOP} largest Opportunities in each of the {GROUPS} Markets with the largest total Export Gap "
        "among their Opportunities, Markets in that order."
    ),
    "top_opportunities_by_product": (
        f"The {GROUP_TOP} largest Opportunities in each of the {GROUPS} Products with the largest total Export Gap "
        "among their Opportunities, Products in that order."
    ),
    "top_blocked_opportunities": (
        f"The {TOP} largest Blocked Opportunities, by Blocked Opportunity value, as the Blocked Opportunities page "
        f"ranks them, in Baseline-year figures. {scenarios.BLOCKED_RULE}"
    ),
    "blocked_opportunities_by_market": "Every Market's count and total value of Blocked Opportunities.",
    "blocked_opportunities_by_product": "Every Product's count and total value of Blocked Opportunities.",
    "largest_scenario_effects": (
        f"For each Trading Bloc preset Scenario the Scenario Builder offers, the {TOP} pairs with the largest "
        "Scenario Effect, in Baseline-year figures. A Scenario Effect is a learned association, not a forecast."
    ),
    "pair_sample": (
        f"{SAMPLE_SIZE} Product–Market pairs drawn at random from every pair the model scores, whatever their Export "
        "Gap, so the tables above can be judged against ordinary pairs."
    ),
    "market_total_imports": (
        "Each Market's total imports of goods from all Exporters in the Baseline year, to judge the scale of the "
        "figures above."
    ),
}

# Every column a table holds, with its meaning in plain domain terms (the README's column list).
COLUMNS = {
    "rank": "Position within the table (or within its Market or Product), from 1.",
    "scenario": "The Trading Bloc preset Scenario.",
    "hs6": "The Product's six-digit HS code.",
    "description": "The Product's name.",
    "sector_name": "The Product's Sector (HS2 chapter).",
    "iso3": "The Market's three-letter country code.",
    "market": "The Market's name.",
    "region": "The Market's world region.",
    "opportunity_type": "Expand, New Market or New Product: how far the UK already trades the pair.",
    "export_potential_usd": "Export Potential, the mean over the Gap Period (USD).",
    "actual_exports_usd": "UK Actual Exports, the mean over the Gap Period (USD).",
    "export_gap_usd": "Export Gap: Export Potential minus Actual Exports, the mean over the Gap Period (USD).",
    "relative_export_gap": "Relative Export Gap: Export Gap as a share of Export Potential.",
    "potential_range_low_usd": "The low end of the Potential Range (USD).",
    "potential_range_high_usd": "The high end of the Potential Range (USD).",
    "uncertain_opportunity": "Whether the pair is an Uncertain Opportunity.",
    "baseline_export_potential_usd": "Export Potential in the Baseline year (USD).",
    "baseline_actual_exports_usd": "UK Actual Exports in the Baseline year (USD; 0 where there was no Flow).",
    "trade_agreement_in_force": "Whether a UK Trade Agreement with the Market is in force in the Baseline year.",
    "applied_tariff_pct": "The Applied Tariff the Market charges on the UK Product in the Baseline year (percent; "
    "empty where unknown).",
    "tariff_imputed": "Whether the Baseline-year Applied Tariff is carried forward from 2021.",
    "blocked_opportunity": "Whether the pair is a Blocked Opportunity.",
    "blocked_opportunity_value_usd": "Blocked Opportunity value (Rise with agreement): the rise in Baseline-year "
    "Export Potential associated with adding a Trade Agreement (USD; 0 where one is in force).",
    "rise_share": "Rise share: the Blocked Opportunity value as a share of Baseline-year Export Potential.",
    "potential_with_agreement_usd": "Export Potential with a Trade Agreement: Baseline-year Export Potential plus "
    "the Blocked Opportunity value (USD).",
    "tariff_part_usd": "Tariff part of the Blocked Opportunity value (USD).",
    "agreement_part_usd": "Agreement part of the Blocked Opportunity value (USD).",
    "barrier": "The Barrier, in words.",
    "value_split": "How the Blocked Opportunity value splits between its Tariff part and Agreement part, in words.",
    "above_export_potential": "Whether Baseline-year Actual Exports are above Baseline-year Export Potential.",
    "blocked_opportunities": "The number of Blocked Opportunities.",
    "baseline_potential_usd": "Export Potential in the Baseline year, before the Scenario (USD).",
    "scenario_potential_usd": "Export Potential in the Baseline year under the Scenario (USD).",
    "effect_usd": "Scenario Effect: Export Potential under the Scenario minus before it (USD).",
    "effect_share": "Scenario Effect as a share of Export Potential before the Scenario.",
    "baseline_tariff_pct": "The Applied Tariff before the Scenario (percent; empty where unknown).",
    "scenario_tariff_pct": "The Applied Tariff under the Scenario (percent; empty where unknown).",
    "total_imports_usd": "The Market's total imports of goods from all Exporters in the Baseline year (USD).",
    **{
        column: description
        for tier in TIERS
        for column, description in (
            (tier.column, f"Whether the pair carries the {tier.badge} caveat (a Caveat Tier)."),
            (f"{tier.column}_reason", f"The reason for the {tier.badge} caveat, where it has one."),
        )
    },
}

_PRODUCT = ["hs6", "description", "sector_name"]
_MARKET = ["iso3", "market", "region"]
_PAIR = [*_PRODUCT, *_MARKET]
_OPPORTUNITY = [
    *_PAIR,
    "opportunity_type",
    "export_potential_usd",
    "actual_exports_usd",
    "export_gap_usd",
    "relative_export_gap",
    "potential_range_low_usd",
    "potential_range_high_usd",
    "uncertain_opportunity",
    "baseline_export_potential_usd",
    "baseline_actual_exports_usd",
    "trade_agreement_in_force",
    "applied_tariff_pct",
    "tariff_imputed",
    "blocked_opportunity",
    "blocked_opportunity_value_usd",
]
_BLOCKED = [
    *_PAIR,
    "baseline_export_potential_usd",
    "baseline_actual_exports_usd",
    "blocked_opportunity_value_usd",
    "rise_share",
    "potential_with_agreement_usd",
    "tariff_part_usd",
    "agreement_part_usd",
    "applied_tariff_pct",
    "tariff_imputed",
    "barrier",
    "value_split",
    "above_export_potential",
    "export_gap_usd",
]
_SAMPLE = [
    *_PAIR,
    "opportunity_type",
    "export_potential_usd",
    "actual_exports_usd",
    "export_gap_usd",
    "potential_range_low_usd",
    "potential_range_high_usd",
]


def _with_caveats(columns: list[str], table: pl.DataFrame) -> list[str]:
    """`columns` followed by each Caveat Tier's flag and reason columns that `table` holds."""
    caveats = [column for tier in TIERS for column in (tier.column, f"{tier.column}_reason")]
    return columns + [column for column in caveats if column in table.columns]


def _ranked(rows: pl.DataFrame) -> pl.DataFrame:
    """`rank_opportunities` rows as the pack shows them: its Opportunities only, ranked from 1, without model inputs."""
    rows = rows.filter(pl.col("export_gap_usd") > 0).rename({"trade_agreement": "trade_agreement_in_force"})
    return rows.select(_with_caveats(_OPPORTUNITY, rows)).with_row_index("rank", offset=1)


def _top_in_each(bundle: Artefacts, groups: pl.DataFrame, key: str) -> pl.DataFrame:
    """The top `GROUP_TOP` Opportunities in each group of `groups` (`rankings.Ranking.markets` or `.products`, keyed
    on `key`), in the groups' order, each ranked from 1 within its group."""
    tables = [
        _ranked(rankings.ranking(bundle, **{key: code}, limit=GROUP_TOP).rows)
        for code in groups.head(GROUPS)[key]
    ]
    return pl.concat(tables)


def _named(table: pl.DataFrame, bundle: Artefacts) -> pl.DataFrame:
    """`table` with each Product's and Market's name and group beside its code."""
    products = with_sector_names(bundle.products).select(_PRODUCT)
    markets = bundle.markets.rename({"name": "market"}).select(_MARKET)
    if "hs6" in table.columns:
        table = table.join(products, on="hs6", how="left", maintain_order="left")
    if "iso3" in table.columns:
        table = table.join(markets, on="iso3", how="left", maintain_order="left")
    return table


def _scenario_effects(bundle: Artefacts) -> pl.DataFrame:
    """The `TOP` largest Scenario Effects of each preset the artefact offers (`scenarios.offered_presets`), each run
    through `scenarios.run` on the served Model Form, as the Scenario Builder runs it."""
    tariff_range = scenarios.tariff_range(bundle.metadata)
    tables = []
    for preset in scenarios.offered_presets(bundle.baseline_features, tariff_range):
        result = scenarios.run(
            bundle.baseline_features,
            bundle.baseline,
            bundle.fold_models,
            scenarios.Scenario(preset=preset.name),
            tariff_range=tariff_range,
            form=bundle.model_form,
        )
        top = result.pairs.head(TOP).with_row_index("rank", offset=1)
        tables.append(top.with_columns(scenario=pl.lit(preset.title)))
    effects = _named(pl.concat(tables), bundle)
    return effects.select(
        "scenario", "rank", *_PAIR, "baseline_potential_usd", "scenario_potential_usd", "effect_usd", "effect_share",
        "baseline_tariff_pct", "scenario_tariff_pct",
    )


def market_total_imports(flows: pl.LazyFrame, year: int, *, country_map: pl.DataFrame | None = None) -> pl.DataFrame:
    """Each Importer's total imports in `year` from all Exporters (`iso3`, `total_imports_usd`), from Flows with
    `year`, `importer` and `value_usd`, largest first.

    `country_map` (`baci_iso3`, `iso3`: the derived `country_map.parquet`) keys each Importer by the standard ISO3 the
    artefact uses (BACI's `S19` becomes `TWN`); a code it does not list is kept as it is."""
    imports = (
        flows.filter(pl.col("year") == year)
        .group_by(baci_iso3=pl.col("importer").cast(pl.String))
        .agg(total_imports_usd=pl.col("value_usd").sum())
        .collect()
    )
    if country_map is not None:
        codes = country_map.select("baci_iso3", "iso3").unique("baci_iso3")
        imports = imports.join(codes, on="baci_iso3", how="left")
        imports = imports.with_columns(pl.col("iso3").fill_null(pl.col("baci_iso3")))
    else:
        imports = imports.with_columns(iso3="baci_iso3")
    return imports.select("iso3", "total_imports_usd").sort("total_imports_usd", "iso3", descending=[True, False])


def build(
    bundle: Artefacts, market_imports: pl.DataFrame, seed: int, *, sample_size: int = SAMPLE_SIZE
) -> dict[str, pl.DataFrame]:
    """The Review Pack from an artefact bundle: one table per name in `TABLES`, its columns those in `COLUMNS`.

    `market_imports` holds each Market's total imports in the Baseline year (`iso3`, `total_imports_usd`, as
    `market_total_imports` returns them); they pass through unchanged, with the Market's name. The pair sample draws
    `sample_size` pairs (every pair, in random order, where the Baseline holds fewer) from the whole Baseline with
    `seed`.
    """
    leaderboard = rankings.ranking(bundle)
    blocked = rankings.blocked_opportunities(bundle, limit=TOP)
    sample = bundle.baseline.sample(n=min(sample_size, bundle.baseline.height), seed=seed, shuffle=True)
    sample = _named(sample, bundle)
    imports = _named(market_imports.select("iso3", "total_imports_usd"), bundle)
    return {
        "top_opportunities": _ranked(leaderboard.rows).head(TOP),
        "top_opportunities_by_market": _top_in_each(bundle, leaderboard.markets, "iso3"),
        "top_opportunities_by_product": _top_in_each(bundle, leaderboard.products, "hs6"),
        "top_blocked_opportunities": blocked.rows.select(_with_caveats(_BLOCKED, blocked.rows)).with_row_index(
            "rank", offset=1
        ),
        "blocked_opportunities_by_market": blocked.markets,
        "blocked_opportunities_by_product": blocked.products,
        "largest_scenario_effects": _scenario_effects(bundle),
        "pair_sample": sample.select(_with_caveats(_SAMPLE, sample)),
        "market_total_imports": imports.select("iso3", "market", "total_imports_usd"),
    }


def readme(pack: dict[str, pl.DataFrame], *, version: str, baseline_year: int) -> str:
    """The Review Pack's README: what it is, then each table's file, what it holds and its columns, in domain terms."""
    lines = [
        "# Review Pack",
        "",
        (
            "The outputs an Advisor sees in the UK Exports Predictor app, plus each Market's total imports so their "
            "scale can be judged. GLOSSARY.md defines every term."
        ),
        "",
        f"- Model version: {version}. Baseline year: {baseline_year}.",
        "- Every amount is in nominal US dollars (USD). Shares are fractions (0.2 is 20%).",
        "- Every Product and Market is named beside its code.",
        "- No Caveat Tier is hidden and no filter is applied, so rows the app marks with a caveat are included.",
        (
            "- Export Potential, Scenario Effects and Blocked Opportunity values are associations the model has "
            "learned, not forecasts."
        ),
    ]
    for name, table in pack.items():
        lines += ["", f"## {name}.csv", "", TABLES[name], "", f"{table.height:,} rows. Columns:", ""]
        lines += [f"- `{column}`: {COLUMNS[column]}" for column in table.columns]
    return "\n".join(lines) + "\n"


def write(pack: dict[str, pl.DataFrame], folder: Path, *, version: str, baseline_year: int) -> Path:
    """Write the Review Pack to `folder`: one `<table>.csv` per table and `README.md` (`readme`). Returns `folder`."""
    folder = Path(folder)
    for name, table in pack.items():
        with writing(folder / f"{name}.csv") as part:
            table.write_csv(part)
    with writing(folder / "README.md") as part:
        part.write_text(readme(pack, version=version, baseline_year=baseline_year), encoding="utf-8")
    return folder


def folder_for(version: str, root: Path | None = None) -> Path:
    """Where `python -m exportspredictor.review` writes the Review Pack of an artefact version."""
    return Path(root or data_dir()) / "derived" / f"review_pack_{version}"


def main() -> None:
    """Build the Review Pack from the newest artefact and write it under `data/derived/`."""
    started = time.perf_counter()
    bundle = load_latest(artefacts_dir())
    country_map = pl.read_parquet(data_dir() / "derived" / "country_map.parquet")
    imports = market_total_imports(baci.scan_flows(), bundle.baseline_year, country_map=country_map)
    pack = build(bundle, imports, SEED)
    folder = write(pack, folder_for(bundle.version), version=bundle.version, baseline_year=bundle.baseline_year)
    print(f"Wrote the Review Pack of {bundle.version} to {folder} in {time.perf_counter() - started:.0f} s:")
    for name, table in pack.items():
        print(f"  {name}.csv: {table.height:,} rows")


if __name__ == "__main__":
    main()
