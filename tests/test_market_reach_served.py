"""Market Reach served (spec #113, Phase 1, ticket #125): the fixture pipeline's Model Form applies it."""

import json
from dataclasses import replace

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from exportspredictor import artefacts as store
from exportspredictor import market_reach, pipeline, scenarios
from exportspredictor.fixture import UK
from exportspredictor.folds import assign_folds
from exportspredictor.gap_period import gap_years
from exportspredictor.pipeline import build_artefacts
from exportspredictor.prediction import export_potential


def test_the_served_model_form_carries_market_reach_learned_from_the_year_before_the_baseline(fixture_artefacts):
    reach = fixture_artefacts.model_form.market_reach

    assert reach is not None
    assert reach.beta == market_reach.CHOSEN_BETA
    assert reach.reference_year == fixture_artefacts.baseline_year - market_reach.LAG_YEARS
    assert reach.table is not None
    assert tuple(reach.table.columns) == market_reach.TABLE_COLUMNS
    assert sorted(reach.table["fold"].unique().to_list()) == list(range(fixture_artefacts.metadata["n_folds"]))
    assert fixture_artefacts.metadata["market_reach"] == {
        "form": market_reach.FORM,
        "beta": market_reach.CHOSEN_BETA,
        "pseudo_count": market_reach.PSEUDO_COUNT,
        "lag_years": market_reach.LAG_YEARS,
    }
    fits = fixture_artefacts.metadata["market_reach_fits"]
    assert [int(year) for year in fits] == gap_years(fixture_artefacts.baseline_year)
    assert all(fit["reference_year"] == int(year) - 1 for year, fit in fits.items())


def test_the_uks_summed_export_potential_is_unchanged_by_market_reach(fixture_artefacts):
    rows, models, form = (
        fixture_artefacts.baseline_features, fixture_artefacts.fold_models, fixture_artefacts.model_form
    )

    reached = export_potential(rows, models, form=form).to_numpy()
    before = export_potential(rows, models, form=form.without_market_reach()).to_numpy()

    assert reached.sum() == pytest.approx(before.sum(), rel=1e-9)
    assert fixture_artefacts.baseline["baseline_export_potential_usd"].sum() == pytest.approx(before.sum(), rel=1e-9)
    assert not np.allclose(reached, before)  # it moves the ranking, not the level


def test_a_scenarios_relative_effect_is_unchanged_by_market_reach(fixture_artefacts):
    rows, models, form = (
        fixture_artefacts.baseline_features, fixture_artefacts.fold_models, fixture_artefacts.model_form
    )
    scenario = scenarios.Scenario(add_agreements=tuple(rows["iso3"].unique().sort()))
    plain = form.without_market_reach()
    plain_baseline = rows.select(
        "hs6", "iso3", baseline_export_potential_usd=export_potential(rows, models, form=plain)
    )

    reached = scenarios.run(rows, fixture_artefacts.baseline, models, scenario, form=form).pairs
    without = scenarios.run(rows, plain_baseline, models, scenario, form=plain).pairs

    assert reached.height == without.height > 0
    joined = reached.join(without, on=["hs6", "iso3"], suffix="_plain")
    assert joined["effect_share"].to_numpy() == pytest.approx(joined["effect_share_plain"].to_numpy(), rel=1e-6)


def _changed_world(world, held_fold: int, n_folds: int):
    """The fixture world with every UK Flow of the Baseline year changed, and those of the year before for the
    pairs of `held_fold`: none of them may reach `held_fold`'s Market Reach offsets."""
    baseline_year = int(world.flows["year"].max())
    flows = assign_folds(world.flows.rename({"importer": "iso3"}), n_folds).rename({"iso3": "importer"})
    is_uk = pl.col("exporter") == UK
    same_year = is_uk & (pl.col("year") == baseline_year)
    own_pairs = is_uk & (pl.col("year") == baseline_year - 1) & (pl.col("fold") == held_fold)
    changed = flows.with_columns(
        value_usd=pl.when(same_year | own_pairs)
        .then(pl.when(pl.col("value_usd") > 0).then(0.0).otherwise(5e6))
        .otherwise(pl.col("value_usd"))
    ).select(world.flows.columns)
    return replace(world, flows=changed)


def test_a_pairs_own_flows_and_same_year_uk_flows_never_reach_its_served_market_reach_offset(
    fixture_world, fixture_default_artefacts
):
    n_folds = fixture_default_artefacts.metadata["n_folds"]
    held = 0
    changed = build_artefacts(
        _changed_world(fixture_world, held, n_folds),
        version="fixture-reach-leak",
        search_results=fixture_default_artefacts.metadata["search"],
    )

    def held_rows(bundle):
        table = bundle.model_form.market_reach.table
        return table.filter(pl.col("fold") == held).sort("importer")

    assert held_rows(changed).equals(held_rows(fixture_default_artefacts))
    other = changed.model_form.market_reach.table.filter(pl.col("fold") != held).sort("fold", "importer")
    assert not other.equals(
        fixture_default_artefacts.model_form.market_reach.table.filter(pl.col("fold") != held).sort("fold", "importer")
    )


def _as_format_3(bundle):
    """`bundle` as a format-3 version: no Market Reach, and its Export Potential scrambled, so a test can see that
    `rereach` scores it again."""
    dropped = ("market_reach", "market_reach_fits")
    metadata = {k: v for k, v in bundle.metadata.items() if k not in dropped} | {
        "artefact_format": 3,
        "version": "fixture-format-3",
        "model_form": bundle.model_form.without_market_reach().to_dict(),
    }
    baseline = bundle.baseline.with_columns(pl.col("export_potential_usd", "baseline_export_potential_usd") * 3)
    return replace(bundle, metadata=metadata, baseline=baseline, market_reach=None)


def test_rereach_turns_a_format_3_version_into_the_format_4_version_a_training_run_builds(
    fixture_world, fixture_default_artefacts
):
    built = fixture_default_artefacts
    inputs = pipeline.fixture_scoring_inputs(
        fixture_world, baseline_year=built.baseline_year, exporters=built.metadata["exporter_scope"]
    )

    reached = pipeline.rereach(_as_format_3(built), inputs, version="fixture-rereach")

    assert reached.metadata["artefact_format"] == 4
    assert reached.metadata["refreshed_from"] == "fixture-format-3"
    for key in ("model_form", "market_reach", "market_reach_fits"):
        assert reached.metadata[key] == built.metadata[key], key
    assert reached.metadata["potential_share_of_actual_exports"] == pytest.approx(
        built.metadata["potential_share_of_actual_exports"]
    )
    assert reached.market_reach is not None and reached.market_reach.equals(built.market_reach)
    for name in ("baseline", "history", "baseline_features", "potential_range_bands"):
        assert_frame_equal(getattr(reached, name), getattr(built, name), check_exact=False, rel_tol=1e-9)
    coverage = reached.metadata["market_reach_pair_coverage"]
    assert 0 <= coverage["before"] <= 1 and 0 <= coverage["after"] <= 1


def test_only_a_format_3_reader_opens_a_format_3_folder(fixture_default_artefacts, tmp_path):
    folder = store.save(fixture_default_artefacts, tmp_path)
    metadata = json.loads((folder / store.METADATA_FILE).read_text(encoding="utf-8")) | {"artefact_format": 3}
    (folder / store.METADATA_FILE).write_text(json.dumps(metadata), encoding="utf-8")
    (folder / "market_reach.parquet").unlink()

    with pytest.raises(store.UnreadableArtefacts, match="artefact format 3.*rereach"):
        store.load(folder)
    assert store.load_format_3(folder).market_reach is None
