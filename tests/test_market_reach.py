"""Market Reach (spec #113, Phase 1, ticket #124): a per-Market offset learned out-of-fold from the year before."""

import math

import numpy as np
import polars as pl
import pytest

from exportspredictor import market_reach


def _uk_rows() -> pl.DataFrame:
    """UK grid rows of 2021 and 2022 in two folds and two Markets, four Products each.

    In 2021, fold 1's pairs: the UK sends 3 of the 4 Products KEN imports from others and 1 of the
    4 PER imports. Fold 0's pairs, and every 2022 row, say something quite different, so any of them
    reaching fold 0's offsets would change them.
    """
    rows = []
    for year in (2021, 2022):
        for fold, sent in ((1, {"KEN": 3, "PER": 1}), (0, {"KEN": 0, "PER": 4})):
            for market in ("KEN", "PER"):
                for i in range(4):
                    hs6 = f"{fold}{i:05d}"
                    uk = 10.0 if i < sent[market] else 0.0
                    if year == 2022:
                        uk = 50.0 - uk  # the scored year: the opposite pattern
                    rows.append((year, fold, hs6, market, uk, 1.0))
    return pl.DataFrame(
        rows, schema=["year", "fold", "hs6", "importer", "value_usd", "log_demand_loo"], orient="row"
    ).with_columns(pl.col("fold").cast(pl.Int8))


def _fold_table(table: pl.DataFrame, fold: int) -> dict[str, dict]:
    return {row["importer"]: row for row in table.filter(pl.col("fold") == fold).iter_rows(named=True)}


def test_a_markets_offset_is_its_uk_share_against_the_pooled_share_from_other_folds_the_year_before():
    table = market_reach.reach_table(_uk_rows(), 2022, n_folds=2, pseudo_count=1.0)
    fold_0 = _fold_table(table, 0)
    # From fold 1's 2021 pairs: KEN 3 of 4, PER 1 of 4; pooled 4 of 8, so 2 expected in each.
    # Offset = log((sent + 1) / (imported + 2)) − log((4 + 1) / (8 + 2)).
    assert fold_0["KEN"]["products_sent"] == 3
    assert fold_0["KEN"]["products_imported"] == 4
    assert fold_0["KEN"]["products_expected"] == pytest.approx(2.0)
    assert fold_0["KEN"]["offset"] == pytest.approx(math.log(4 / 3))
    assert fold_0["PER"]["offset"] == pytest.approx(math.log(2 / 3))


def test_a_pairs_own_flows_and_every_uk_flow_of_the_scored_year_leave_its_offset_unchanged():
    rows = _uk_rows()
    before = market_reach.reach_table(rows, 2022, n_folds=2)
    changed = rows.with_columns(
        value_usd=pl.when((pl.col("fold") == 0) | (pl.col("year") == 2022))
        .then(pl.col("value_usd") * 7 + 3)
        .otherwise(pl.col("value_usd"))
    )
    after = market_reach.reach_table(changed, 2022, n_folds=2)
    assert after.filter(pl.col("fold") == 0).sort("importer").equals(before.filter(pl.col("fold") == 0).sort("importer"))


def test_a_market_with_no_products_in_the_other_folds_gets_no_offset():
    rows = _uk_rows().filter(~((pl.col("fold") == 1) & (pl.col("importer") == "PER")))
    table = market_reach.reach_table(rows, 2022, n_folds=2, pseudo_count=1.0)
    potential = np.array([5.0])
    one_row = pl.DataFrame({"fold": [0], "importer": ["PER"]}, schema={"fold": pl.Int8, "importer": pl.String})
    assert market_reach.apply(potential, one_row, table, beta=2.0, constant=1.0) == pytest.approx([5.0])


def _scored() -> tuple[np.ndarray, pl.DataFrame, pl.DataFrame]:
    table = market_reach.reach_table(_uk_rows(), 2022, n_folds=2, pseudo_count=1.0)
    rows = pl.DataFrame(
        {"fold": [0, 0, 1, 1, None], "importer": ["KEN", "PER", "KEN", "PER", "KEN"]},
        schema={"fold": pl.Int8, "importer": pl.String},
    )
    return np.array([100.0, 100.0, 40.0, 10.0, 7.0]), rows, table


def test_export_potential_is_multiplied_by_exp_beta_times_the_offset_and_the_constant():
    potential, rows, table = _scored()
    adjusted = market_reach.apply(potential, rows, table, beta=2.0, constant=0.5)
    # Fold 0: KEN offset log(4/3), PER log(2/3); a row with no fold (another Exporter's) gets none.
    assert adjusted[:2] == pytest.approx([100 * (4 / 3) ** 2 * 0.5, 100 * (2 / 3) ** 2 * 0.5])
    assert adjusted[4] == pytest.approx(7.0 * 0.5)


def test_the_constant_keeps_the_uks_summed_export_potential_unchanged():
    potential, rows, table = _scored()
    constant = market_reach.rescale_constant(potential, rows, table, beta=4.0)
    adjusted = market_reach.apply(potential, rows, table, beta=4.0, constant=constant)
    assert adjusted.sum() == pytest.approx(potential.sum())
    assert not np.allclose(adjusted, potential)


def test_beta_zero_leaves_export_potential_unchanged():
    potential, rows, table = _scored()
    constant = market_reach.rescale_constant(potential, rows, table, beta=0.0)
    assert market_reach.apply(potential, rows, table, beta=0.0, constant=constant) == pytest.approx(potential)


def test_a_scenarios_relative_effect_is_unchanged_by_market_reach():
    potential, rows, table = _scored()
    constant = market_reach.rescale_constant(potential, rows, table, beta=2.0)
    scenario = potential * np.array([1.2, 1.0, 1.5, 0.9, 1.1])  # a Scenario's model output on the same rows
    before = market_reach.apply(potential, rows, table, beta=2.0, constant=constant)
    after = market_reach.apply(scenario, rows, table, beta=2.0, constant=constant)
    assert after / before == pytest.approx(scenario / potential)


def test_the_pre_registered_settings_are_fixed_in_code():
    """Committed before any 2022 result was read (#124); the note cites the commit."""
    assert market_reach.FORM == "share"
    assert market_reach.BETA_GRID == (0.0, 0.5, 1.0, 2.0, 4.0)
    assert market_reach.PSEUDO_COUNT == 5.0
    assert market_reach.LAG_YEARS == 1


def test_the_adopted_beta_comes_from_the_pre_registered_grid():
    assert market_reach.CHOSEN_BETA in market_reach.BETA_GRID
