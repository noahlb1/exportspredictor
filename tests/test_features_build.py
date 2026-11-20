"""The real-data feature build, run on a tiny data root laid out like the derived BACI, gravity and policy tables."""

import json
import math

import polars as pl
import pytest

from exportspredictor import baci, features
from exportspredictor.fixture import UK


def test_build_writes_training_rows_and_the_uk_grid_per_year(derived_data_root):
    out = features.build(derived_data_root, years=[2023, 2024], rate=0.5, seed=1)

    training = pl.read_parquet(out / "training_2023.parquet")
    grid = pl.read_parquet(out / "uk_grid_2024.parquet")
    assert set(training.columns) >= {*features.KEYS, *features.FEATURES, "value_usd", "weight"}
    assert set(grid["exporter"]) == {UK}
    assert grid.height == 5 * 4
    assert set(training["weight"]) == {1.0, 2.0}


def test_build_maps_baci_codes_to_standard_iso3(derived_data_root):
    out = features.build(derived_data_root, years=[2023], rate=1.0)

    taiwan = pl.read_parquet(out / "training_2023.parquet").filter(pl.col("exporter") == "TWN")
    assert taiwan.height > 0
    assert taiwan["log_gdp_o"].null_count() == 0
    assert taiwan["log_distance"].null_count() == 0


def test_build_records_its_settings(derived_data_root):
    out = features.build(derived_data_root, years=[2023], rate=0.5, seed=4)

    settings = json.loads((out / "features.json").read_text())
    assert settings["zero_sampling_rate"] == 0.5
    assert settings["zero_sampling_seed"] == 4
    assert settings["years"] == [2023]
    assert settings["features"] == features.FEATURES


def test_build_drops_excluded_products_and_records_the_list(derived_data_root):
    out = features.build(derived_data_root, years=[2024], rate=1.0)

    for name in ("training_2024", "uk_grid_2024"):
        assert "270900" not in set(pl.read_parquet(out / f"{name}.parquet")["hs6"].cast(pl.String))
    settings = json.loads((out / "features.json").read_text())
    assert settings["excluded_products"][0] == "Sector 27 except 2712"


def test_build_trains_only_on_the_recorded_exporter_scope(derived_data_root):
    out = features.build(derived_data_root, years=[2023], rate=0.5, scope_size=2)

    settings = json.loads((out / "features.json").read_text())
    scope = settings["exporter_scope"]
    assert UK in scope and len(scope) in (2, 3)
    training = pl.read_parquet(out / "training_2023.parquet")
    assert set(training["exporter"]) == set(scope) & {UK, "DEU", "FRA", "TWN"}
    assert len(set(training["exporter"])) < 4  # an Exporter fell out of scope
    assert settings["uk_zero_sampling_rate"] == 1.0
    uk_zeros = training.filter((pl.col("exporter") == UK) & (pl.col("value_usd") == 0))
    assert uk_zeros.height > 0 and (uk_zeros["weight"] == 1.0).all()


def test_build_scores_the_union_of_the_gap_period_grids_in_each_gap_year(derived_data_root):
    # Nobody trades whisky in 2023, so the 2023 grid alone would have no whisky pair.
    path = baci.flows_dir(derived_data_root) / "flows_2023.parquet"
    flows = pl.read_parquet(path)
    flows.filter(pl.col("hs6").cast(pl.String) != "220830").write_parquet(path)

    out = features.build(derived_data_root, years=[2021, 2022, 2023, 2024], rate=1.0)

    grid = pl.read_parquet(out / "uk_grid_2023.parquet")
    assert grid.height == 5 * 4
    assert grid.filter(pl.col("hs6").cast(pl.String) == "220830")["value_usd"].to_list() == [0.0] * 4
    settings = json.loads((out / "features.json").read_text())
    assert settings["gap_years"] == [2022, 2024]


def test_build_adds_relatedness_import_growth_and_the_reexport_ratio(derived_data_root):
    out = features.build(derived_data_root, years=[2024], rate=1.0)

    grid = pl.read_parquet(out / "uk_grid_2024.parquet")
    training = pl.read_parquet(out / "training_2024.parquet")
    for table in (grid, training):
        for name in ("relatedness", "market_import_growth", "reexport_ratio"):
            assert table[name].null_count() < table.height, name
    assert grid["market_import_growth"].null_count() == 0  # 2021 is on disk: three years earlier
    settings = json.loads((out / "features.json").read_text())
    assert settings["proximity_years"] == [2021]  # the fixture's only year before 2022
    assert pl.read_parquet(out / "proximity.parquet").height > 0


# --- Feature sets: a rebuild writes its own folder (#126) -------------------------------------


def _filled_gravity(root, iso3: str, gdp_usd: float) -> None:
    """A `country_year_filled.parquet` in which one country's GDP came from a secondary source."""
    derived = root / "derived"
    filled = pl.read_parquet(derived / "country_year.parquet").with_columns(
        gdp_usd=pl.when(pl.col("iso3") == iso3).then(gdp_usd).otherwise("gdp_usd"),
        gdp_source=pl.when(pl.col("iso3") == iso3).then(pl.lit("carried (cia_factbook)")).otherwise(pl.lit("wdi")),
    )
    filled.write_parquet(derived / "country_year_filled.parquet")


def test_a_feature_set_is_written_to_its_own_folder_and_leaves_the_original_alone(derived_data_root):
    original = features.build(derived_data_root, years=[2023], rate=1.0)
    before = pl.read_parquet(original / "training_2023.parquet")
    _filled_gravity(derived_data_root, "JPN", 1e6)

    out = features.build(derived_data_root, years=[2023], rate=1.0, feature_set="exporter_count")

    assert out == derived_data_root / "derived" / f"features_{baci.BACI_VERSION}_exporter_count"
    assert out != original
    assert pl.read_parquet(original / "training_2023.parquet").equals(before)
    rows = pl.read_parquet(out / "training_2023.parquet")
    assert "log_exporters_loo" in rows.columns
    # The set reads the filled gravity table: Japan's GDP is the secondary source's.
    japan = rows.filter(pl.col("importer") == "JPN")["log_gdp_d"].unique().to_list()
    assert japan == [pytest.approx(math.log(1e6))]


def test_a_feature_set_records_its_features_constraints_and_gravity_fill(derived_data_root):
    _filled_gravity(derived_data_root, "JPN", 1e6)

    out = features.build(derived_data_root, years=[2023], rate=1.0, feature_set="exporter_count")

    settings = json.loads((out / "features.json").read_text())
    assert settings["feature_set"] == "exporter_count"
    assert settings["features"] == features.FEATURES
    assert settings["candidate_features"] == ["log_exporters_loo"]
    assert settings["monotone_constraints"] == features.MONOTONE_CONSTRAINTS
    assert "log_exporters_loo" not in settings["monotone_constraints"]
    assert settings["gravity"]["country_year"] == "country_year_filled.parquet"
    assert settings["gravity"]["filled"] == {"JPN": {"gdp_usd": ["carried (cia_factbook)"]}}


def test_a_built_feature_set_is_never_overwritten_unless_asked(derived_data_root):
    features.build(derived_data_root, years=[2023], rate=1.0)

    with pytest.raises(FileExistsError):
        features.build(derived_data_root, years=[2023], rate=0.5)
    features.build(derived_data_root, years=[2023], rate=0.5, overwrite=True)


def test_every_reader_opens_the_feature_set_the_environment_names(derived_data_root, monkeypatch):
    derived = derived_data_root / "derived"
    assert features.features_dir(derived_data_root) == derived / f"features_{baci.BACI_VERSION}"

    monkeypatch.setenv(features.FEATURE_SET_ENV, "exporter_count")
    assert features.features_dir(derived_data_root) == derived / f"features_{baci.BACI_VERSION}_exporter_count"

    monkeypatch.setenv(features.FEATURE_SET_ENV, "nonsense")
    with pytest.raises(ValueError, match="nonsense"):
        features.features_dir(derived_data_root)


def test_a_build_of_earlier_years_never_opens_a_later_years_flows(derived_data_root, monkeypatch):
    """The rebuild for the experiment loop stops at the validation year: the test year's file stays unread."""
    opened = []
    original = pl.scan_parquet

    def spy(source, *args, **kwargs):
        opened.append(str(source))
        return original(source, *args, **kwargs)

    monkeypatch.setattr(pl, "scan_parquet", spy)

    features.build(derived_data_root, years=[2021, 2022], rate=1.0)

    flows = [path for path in opened if "flows_" in path]
    assert flows and all("flows_2021" in path or "flows_2022" in path for path in flows), flows


def test_building_the_training_rows_in_parts_gives_the_same_rows(derived_data_root):
    """Parts keep peak memory down on real data; every feature of a row reads the whole year's Flows."""
    whole = features.build(derived_data_root, years=[2023], rate=0.5, seed=2)
    expected = pl.read_parquet(whole / "training_2023.parquet").sort(features.KEYS)

    parted = features.build(derived_data_root, years=[2023], rate=0.5, seed=2, parts=3, overwrite=True)

    assert pl.read_parquet(parted / "training_2023.parquet").sort(features.KEYS).equals(expected)


def test_a_resumed_build_keeps_the_years_already_written(derived_data_root, monkeypatch):
    """A build killed after some years (memory, on a shared machine) resumes without redoing them."""
    out = features.features_dir(derived_data_root)
    out.mkdir(parents=True)
    for name in ("training_2023", "uk_grid_2023"):
        pl.DataFrame({"marker": [1]}).write_parquet(out / f"{name}.parquet")

    features.build(derived_data_root, years=[2023, 2024], rate=1.0, resume=True)

    assert pl.read_parquet(out / "training_2023.parquet").columns == ["marker"]
    assert "log_exporters_loo" in pl.read_parquet(out / "training_2024.parquet").columns
    assert json.loads((out / "features.json").read_text())["years"] == [2023, 2024]


def test_a_resumed_build_reuses_the_proximity_it_saved(derived_data_root):
    out = features.build(derived_data_root, years=[2023], rate=1.0)
    saved = pl.read_parquet(out / "proximity.parquet")
    saved.head(0).write_parquet(out / "proximity.parquet")  # a stand-in that refitting would replace

    features.build(derived_data_root, years=[2023, 2024], rate=1.0, resume=True)

    assert pl.read_parquet(out / "proximity.parquet").is_empty()
    assert pl.read_parquet(out / "training_2024.parquet")["relatedness"].null_count() > 0
