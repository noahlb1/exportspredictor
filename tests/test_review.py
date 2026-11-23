"""The Review Pack: the served outputs a fresh reviewer reads, built from an artefact bundle (spec #140, #159).

Tested on the fixture artefact for what each table holds, against the modules the app and API read
(`rankings`, `scenarios`), not for how the pack is assembled.
"""

import polars as pl
import pytest

from exportspredictor import rankings, review, scenarios
from exportspredictor.features import ALL_FEATURES

TABLES = {
    "top_opportunities",
    "top_opportunities_by_market",
    "top_opportunities_by_product",
    "top_blocked_opportunities",
    "blocked_opportunities_by_market",
    "blocked_opportunities_by_product",
    "largest_scenario_effects",
    "pair_sample",
    "market_total_imports",
}


@pytest.fixture(scope="module")
def bundle(blocked_artefacts):
    """The Blocked world: the fixture artefacts scaled so that three pairs are Blocked Opportunities."""
    return blocked_artefacts


@pytest.fixture(scope="module")
def market_imports(fixture_world, bundle):
    return review.market_total_imports(fixture_world.flows.lazy(), bundle.baseline_year)


@pytest.fixture(scope="module")
def pack(bundle, market_imports):
    return review.build(bundle, market_imports, seed=0)


def test_the_pack_holds_every_named_table_with_names_beside_codes(pack):
    assert set(pack) == TABLES
    for name, table in pack.items():
        assert not table.is_empty(), name
        if "hs6" in table.columns:
            assert "description" in table.columns, name
            assert table["description"].null_count() == 0, name
        if "iso3" in table.columns:
            assert "market" in table.columns, name
            assert table["market"].null_count() == 0, name


def test_every_column_is_described_for_the_readme_and_none_is_a_model_input(pack):
    model_columns = {*ALL_FEATURES, "fold", "naive_usd"}
    for name, table in pack.items():
        assert set(table.columns) <= set(review.COLUMNS), (name, set(table.columns) - set(review.COLUMNS))
        assert not set(table.columns) & model_columns, (name, set(table.columns) & model_columns)
        assert not any("model" in column for column in table.columns), name


def _pairs(table: pl.DataFrame) -> list[tuple[str, str]]:
    return list(table.select("hs6", "iso3").iter_rows())


def test_the_top_opportunities_are_the_leaderboards_top_opportunities_in_order(bundle, pack):
    leaderboard = rankings.ranking(bundle).rows.filter(pl.col("export_gap_usd") > 0)

    table = pack["top_opportunities"]
    assert _pairs(table) == _pairs(leaderboard.head(review.TOP))
    assert table["rank"].to_list() == list(range(1, table.height + 1))
    assert table["export_gap_usd"].to_list() == leaderboard.head(review.TOP)["export_gap_usd"].to_list()


@pytest.mark.parametrize(("table", "key", "groups"), [
    ("top_opportunities_by_market", "iso3", "markets"),
    ("top_opportunities_by_product", "hs6", "products"),
])
def test_each_top_group_lists_its_top_opportunities_as_its_own_ranking_does(bundle, pack, table, key, groups):
    # The groups are those with the largest total Export Gap among their Opportunities, in that order.
    top_groups = getattr(rankings.ranking(bundle), groups).head(review.GROUPS)[key].to_list()
    expected = []
    for code in top_groups:
        own = rankings.ranking(bundle, **{key: code}).rows.filter(pl.col("export_gap_usd") > 0)
        expected += _pairs(own.head(review.GROUP_TOP))

    assert _pairs(pack[table]) == expected
    assert pack[table][key].unique(maintain_order=True).to_list() == [
        code for code in top_groups if code in set(pack[table][key])
    ]


def test_the_blocked_opportunity_tables_are_the_one_sources(bundle, pack):
    blocked = rankings.blocked_opportunities(bundle)

    assert pack["top_blocked_opportunities"].height == 3  # the Blocked world's three
    assert _pairs(pack["top_blocked_opportunities"]) == _pairs(blocked.rows.head(review.TOP))
    assert pack["top_blocked_opportunities"]["blocked_opportunity_value_usd"].to_list() == (
        blocked.rows.head(review.TOP)["blocked_opportunity_value_usd"].to_list()
    )
    assert pack["blocked_opportunities_by_market"].equals(rankings.blocked_by(blocked.rows, "iso3", "market"))
    assert pack["blocked_opportunities_by_product"].equals(rankings.blocked_by(blocked.rows, "hs6", "description"))


def test_each_offered_presets_largest_scenario_effects_are_those_scenarios_run_gives(bundle, pack):
    tariff_range = scenarios.tariff_range(bundle.metadata)
    offered = scenarios.offered_presets(bundle.baseline_features, tariff_range)
    table = pack["largest_scenario_effects"]

    assert offered  # the fixture offers presets, so the check below compares something
    assert table["scenario"].unique(maintain_order=True).to_list() == [preset.title for preset in offered]
    for preset in offered:
        result = scenarios.run(
            bundle.baseline_features, bundle.baseline, bundle.fold_models, scenarios.Scenario(preset=preset.name),
            tariff_range=tariff_range, form=bundle.model_form,
        )
        expected = result.pairs.head(review.TOP)
        shown = table.filter(pl.col("scenario") == preset.title)
        assert _pairs(shown) == _pairs(expected)
        assert shown["effect_usd"].to_list() == pytest.approx(expected["effect_usd"].to_list())


def test_the_pair_sample_has_the_requested_size_and_the_baselines_own_figures(bundle, market_imports):
    sample = review.build(bundle, market_imports, seed=0, sample_size=7)["pair_sample"]

    assert sample.height == 7
    assert sample.select("hs6", "iso3").n_unique() == 7
    columns = ["export_potential_usd", "actual_exports_usd", "export_gap_usd", "potential_range_low_usd",
               "potential_range_high_usd"]
    joined = sample.join(bundle.baseline, on=["hs6", "iso3"], how="left", suffix="_baseline")
    for column in columns:
        assert joined[column].to_list() == joined[f"{column}_baseline"].to_list(), column


def test_the_pair_sample_draws_from_the_whole_baseline_not_only_opportunities(bundle, market_imports):
    everything = review.build(bundle, market_imports, seed=0, sample_size=bundle.baseline.height)["pair_sample"]

    assert sorted(_pairs(everything)) == sorted(_pairs(bundle.baseline))
    assert (bundle.baseline["export_gap_usd"] <= 0).any()  # so the whole Baseline holds pairs that are not Opportunities


def test_the_pair_sample_is_the_same_for_the_same_seed_and_different_for_another(bundle, market_imports):
    def sample(seed: int) -> list[tuple[str, str]]:
        return _pairs(review.build(bundle, market_imports, seed=seed, sample_size=7)["pair_sample"])

    assert sample(1) == sample(1)
    assert sample(1) != sample(2)


def test_market_total_imports_sum_the_years_flows_into_each_market_from_every_exporter():
    flows = pl.LazyFrame({
        "year": [2024, 2024, 2024, 2023],
        "exporter": ["GBR", "DEU", "DEU", "DEU"],
        "importer": ["FRA", "FRA", "JPN", "FRA"],
        "hs6": ["220830", "870323", "220830", "220830"],
        "value_usd": [10.0, 5.0, 2.0, 100.0],
    })

    imports = review.market_total_imports(flows, 2024)

    assert imports.to_dicts() == [{"iso3": "FRA", "total_imports_usd": 15.0}, {"iso3": "JPN", "total_imports_usd": 2.0}]


def test_market_total_imports_pass_through_unchanged_and_cover_every_market_in_the_pack(pack, market_imports):
    shown = pack["market_total_imports"]

    assert shown.select("iso3", "total_imports_usd").equals(market_imports)
    for name, table in pack.items():
        if "iso3" in table.columns:
            assert set(table["iso3"]) <= set(shown["iso3"]), name


def test_the_folder_holds_one_csv_per_table_and_a_readme_naming_each_table_and_column(bundle, pack, tmp_path):
    folder = review.write(pack, tmp_path / "review_pack", version="fixture-test", baseline_year=bundle.baseline_year)

    readme = (folder / "README.md").read_text(encoding="utf-8")
    assert "fixture-test" in readme
    for name, table in pack.items():
        written = pl.read_csv(folder / f"{name}.csv", infer_schema_length=None)
        assert written.columns == table.columns
        assert written.height == table.height
        assert f"{name}.csv" in readme
        assert review.TABLES[name] in readme
        for column in table.columns:
            assert f"`{column}`: {review.COLUMNS[column]}" in readme
    assert sorted(path.name for path in folder.iterdir()) == sorted([*(f"{name}.csv" for name in pack), "README.md"])


def test_the_readme_speaks_in_domain_terms_not_code_or_modelling_history(bundle, pack, tmp_path):
    folder = review.write(pack, tmp_path / "review_pack", version="fixture-test", baseline_year=bundle.baseline_year)
    readme = (folder / "README.md").read_text("utf-8")

    for word in ("ADR", "rankings.", "scenarios.", "fold", "XGBoost", "spec #", "artefact"):
        assert word not in readme, word


def test_market_total_imports_key_each_market_by_its_standard_code_through_the_country_map():
    flows = pl.LazyFrame({
        "year": [2024, 2024],
        "exporter": ["GBR", "DEU"],
        "importer": ["S19", "S19"],  # BACI's code for Taiwan
        "hs6": ["220830", "870323"],
        "value_usd": [10.0, 5.0],
    })
    country_map = pl.DataFrame({"baci_iso3": ["S19", "FRA"], "iso3": ["TWN", "FRA"]})

    imports = review.market_total_imports(flows, 2024, country_map=country_map)

    assert imports.to_dicts() == [{"iso3": "TWN", "total_imports_usd": 15.0}]
