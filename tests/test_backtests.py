"""Lift and Event Backtests: Opportunities and Scenario Effects checked against what actually happened."""

import polars as pl
import pytest

from exportspredictor import backtests


def _pairs(gaps, grew, exporters=None, potential=None):
    n = len(gaps)
    return pl.DataFrame(
        {
            "exporter": exporters or ["A", "B"] * (n // 2),
            "export_gap_usd": [float(g) for g in gaps],
            "export_potential_usd": [float(p) for p in (potential or [100.0] * n)],
            "grew": grew,
            "weight": [1.0] * n,
        }
    )


# --- Lift ------------------------------------------------------------------------------


def test_lift_compares_the_top_decile_by_export_gap_with_all_pairs():
    # 20 pairs; the two largest gaps both grew, 5 of 20 grew overall.
    gaps = list(range(20, 0, -1))
    grew = [True, True, True, False, False, True, True] + [False] * 13

    result = backtests.lift(_pairs(gaps, grew), resamples=200)

    assert result["top_rate"] == pytest.approx(1.0)
    assert result["base_rate"] == pytest.approx(5 / 20)
    assert result["lift"] == pytest.approx(4.0)
    assert result["by_decile"][0] == {"decile": 1, "rate": pytest.approx(1.0), "pairs": pytest.approx(2.0)}


def test_lift_weights_sampled_zero_cells_by_their_weight():
    pairs = _pairs([10, 9, 1, 0], [True, False, False, True]).with_columns(weight=pl.Series([1.0, 1.0, 1.0, 7.0]))

    result = backtests.lift(pairs, resamples=50)

    # 10 weighted pairs: the top tenth is the largest gap alone; 8 of 10 grew.
    assert result["top_rate"] == pytest.approx(1.0)
    assert result["base_rate"] == pytest.approx(8 / 10)


def test_lift_has_a_bootstrap_confidence_interval_around_it():
    gaps = list(range(200, 0, -1))
    grew = [i % 3 == 0 or i < 10 for i in range(200)]
    exporters = [f"E{i % 10}" for i in range(200)]

    result = backtests.lift(_pairs(gaps, grew, exporters), resamples=300, seed=1)

    low, high = result["ci"]
    assert low <= result["lift"] <= high
    assert low < high
    assert result["ci_level"] == 0.95
    assert result["bootstrap_unit"] == "Exporter"


def test_size_matched_lift_compares_with_pairs_of_similar_export_potential():
    # The largest gaps are also the largest pairs, and every large pair grew: the
    # size-matched Lift is 1 (no better than other pairs of the same size), the plain Lift is not.
    n = 100
    gaps = list(range(n, 0, -1))
    potential = list(range(n, 0, -1))
    grew = [i < 10 or i % 4 == 0 for i in range(n)]

    result = backtests.lift(_pairs(gaps, grew, potential=potential), resamples=50)

    assert result["lift"] > 1.5
    assert result["size_matched"]["lift"] == pytest.approx(1.0)


def test_grows_materially_needs_both_the_share_and_the_dollar_threshold():
    frame = pl.DataFrame({"before": [100.0, 0.0, 1e6, 1e6], "after": [1e6, 2e5, 1.2e6, 1.6e6]})

    grew = frame.select(backtests.grows_materially("before", "after", min_share=0.5, min_usd=1e5)).to_series()

    assert grew.to_list() == [True, True, False, True]


# --- Gap-Period Lift -----------------------------------------------------------------------

GAP_YEARS = (2015, 2016, 2017)
GAP_SETTINGS = backtests.LiftSettings(training_years=tuple(range(2012, 2018)), gap_years=GAP_YEARS, outcome_year=2023)


def _yearly(rows):
    return pl.DataFrame(
        rows, schema=["exporter", "importer", "hs6", "year", "value_usd", "weight", "export_potential_usd"], orient="row"
    )


def _later(rows):
    return pl.DataFrame(rows, schema=["exporter", "importer", "hs6", "value_usd"], orient="row")


def test_the_export_gap_is_the_mean_of_the_three_yearly_gaps():
    yearly = _yearly(
        [
            ("A", "J", "1", 2015, 100.0, 1.0, 400.0),  # yearly gaps 300, 500, 700: mean 500
            ("A", "J", "1", 2016, 200.0, 1.0, 700.0),
            ("A", "J", "1", 2017, 300.0, 1.0, 1000.0),
        ]
    )

    pairs = backtests.gap_period_pairs(yearly, _later([]), GAP_SETTINGS)

    assert pairs["export_gap_usd"].to_list() == [500.0]
    assert pairs["export_potential_usd"].to_list() == [700.0]
    assert pairs["actual_exports_usd"].to_list() == [200.0]


def test_growth_is_measured_from_the_three_year_mean_of_actual_exports():
    # Mean Actual Exports 2015–2017 = 200,000. B rose to 320,000 (+60%, +$120,000): grew.
    # C's 2017 value alone (300,000) would make 320,000 too small a rise; D rose by +50% but only $50,000.
    yearly = _yearly(
        [
            *[("A", "J", hs6, y, v, 1.0, 1.0) for hs6 in ("B", "C") for y, v in zip(GAP_YEARS, (100e3, 200e3, 300e3))],
            *[("A", "J", "D", y, 100e3, 1.0, 1.0) for y in GAP_YEARS],
        ]
    )
    later = _later([("A", "J", "B", 320e3), ("A", "J", "C", 290e3), ("A", "J", "D", 150e3)])

    pairs = backtests.gap_period_pairs(yearly, later, GAP_SETTINGS).sort("hs6")

    assert pairs["grew"].to_list() == [True, False, False]


def test_the_gap_period_pairs_are_the_last_gap_years_rows_with_their_weights():
    # A sampled Zero Cell of 2017 (weight 20) with no 2015 or 2016 row: those years' Flows were 0,
    # and its Export Potential is the mean of the years it has. A pair seen only in 2015 is not a 2017 pair.
    yearly = _yearly(
        [
            ("A", "J", "Z", 2017, 0.0, 20.0, 90.0),
            ("A", "J", "P", 2016, 60.0, 1.0, 80.0),
            ("A", "J", "P", 2017, 0.0, 20.0, 100.0),
            ("A", "J", "Q", 2015, 50.0, 1.0, 10.0),
        ]
    )

    pairs = backtests.gap_period_pairs(yearly, _later([]), GAP_SETTINGS).sort("hs6")

    assert pairs["hs6"].to_list() == ["P", "Z"]
    assert pairs["weight"].to_list() == [20.0, 20.0]
    assert pairs["actual_exports_usd"].to_list() == [20.0, 0.0]
    assert pairs["export_gap_usd"].to_list() == [70.0, 90.0]


def test_lift_settings_refuse_a_training_year_after_the_gap_period():
    with pytest.raises(backtests.LeakageError):
        backtests.LiftSettings(training_years=tuple(range(2012, 2019)), gap_years=GAP_YEARS, outcome_year=2023)
    with pytest.raises(backtests.LeakageError):
        backtests.LiftSettings(training_years=(2015, 2016, 2017), gap_years=GAP_YEARS, outcome_year=2017)


def test_the_held_out_exporters_are_the_frozen_v1_list_within_scope():
    v1 = ["AUS", "GIB", "NIU", "QAT", "SVK"]

    held = backtests.lift_exporters(v1, scope=["AUS", "DEU", "GBR", "QAT", "SVK", "USA"], uk="GBR")

    assert held == ["AUS", "QAT", "SVK"]
    with pytest.raises(ValueError, match="scope"):
        backtests.lift_exporters(["GIB"], scope=["GBR", "DEU"], uk="GBR")
    with pytest.raises(ValueError, match="GBR"):
        backtests.lift_exporters(["GBR", "AUS"], scope=["GBR", "AUS"], uk="GBR")


def test_held_out_exporters_are_a_deterministic_fifth():
    exporters = [f"C{i:02d}" for i in range(50)]

    held = backtests.held_out_exporters(exporters, share=0.2)

    assert len(held) == 10
    assert held == backtests.held_out_exporters(list(reversed(exporters)), share=0.2)
    assert backtests.held_out_exporters(["GBR", "DEU", "FRA", "USA"], share=0.2) != []


# --- Event Backtest: the observed difference-in-differences ------------------------------


def _flows(rows):
    return pl.DataFrame(rows, schema=["exporter", "importer", "hs6", "year", "value_usd"], orient="row")


def test_observed_effect_is_the_treated_change_relative_to_the_control_change():
    # EU -> JPN rises 50% from the pre-period, EU -> USA rises 20%: the observed effect is 1.5 / 1.2 - 1 = +25%.
    flows = _flows(
        [
            ("DEU", "JPN", "1", 2018, 60.0), ("FRA", "JPN", "2", 2018, 40.0),
            ("DEU", "JPN", "1", 2019, 90.0), ("FRA", "JPN", "2", 2019, 60.0),
            ("DEU", "USA", "1", 2018, 50.0), ("FRA", "USA", "2", 2018, 50.0),
            ("DEU", "USA", "1", 2019, 60.0), ("FRA", "USA", "2", 2019, 60.0),
            ("DEU", "CHN", "1", 2019, 999.0),  # not a control Market
        ]
    )
    predicted = pl.DataFrame({"hs6": ["1", "2"], "year": [2019, 2019], "on_usd": [75.0, 50.0], "off_usd": [60.0, 40.0], "weight": [1.0, 1.0]})

    result = backtests.compare(
        flows, predicted, exporters=["DEU", "FRA"], treated=["JPN"], controls=["USA"],
        pre_years=[2018], post_years=[2019], resamples=100,
    )

    year = result["by_year"][0]
    assert year["year"] == 2019
    assert year["observed"] == pytest.approx(0.25)
    assert year["predicted"] == pytest.approx(0.25)
    assert result["pooled"]["observed"] == pytest.approx(0.25)
    low, high = year["observed_ci"]
    assert low <= 0.25 <= high


def test_predicted_effect_weights_sampled_zero_cells():
    flows = _flows([("A", "J", "1", 2018, 1.0), ("A", "J", "1", 2019, 1.0), ("A", "U", "1", 2018, 1.0), ("A", "U", "1", 2019, 1.0)])
    predicted = pl.DataFrame({"hs6": ["1", "1"], "year": [2019, 2019], "on_usd": [12.0, 3.0], "off_usd": [10.0, 1.0], "weight": [1.0, 10.0]})

    result = backtests.compare(
        flows, predicted, exporters=["A"], treated=["J"], controls=["U"], pre_years=[2018], post_years=[2019], resamples=10
    )

    # (12 - 10 + 10 * (3 - 1)) / (10 + 10 * 1) = 22 / 20
    assert result["by_year"][0]["predicted"] == pytest.approx(1.1)


def test_control_markets_are_similar_markets_without_an_agreement_largest_first():
    rows = pl.DataFrame(
        {
            "exporter": ["DEU"] * 5,
            "importer": ["USA", "CAN", "AUS", "IND", "JPN"],
            "year": [2018] * 5,
            "value_usd": [500.0, 400.0, 100.0, 900.0, 800.0],
            "trade_agreement": [0, 1, 0, 0, 0],
        }
    )

    controls = backtests.control_markets(
        rows, exporters=["DEU"], treated=["JPN"], pre_years=[2018], high_income={"USA", "CAN", "AUS", "JPN"}, n=5
    )

    # CAN has an agreement, IND is not high income, JPN is treated.
    assert controls == ["USA", "AUS"]


# --- Backtest models on the fixture world --------------------------------------------------

from dataclasses import replace

from exportspredictor import pipeline
from exportspredictor.fixture import UK

PLAN = pipeline.FIXTURE_BACKTESTS
EVENT = PLAN.event  # the fixture's UK–Japan agreement, in force from 2022


def _train_event(chunks):
    return backtests.train_backtest_model(
        chunks, name=EVENT.name, training_years=EVENT.training_years, before=EVENT.event_year,
        uk=UK, params={"max_depth": 2}, rounds=20, device="cpu",
    )


def test_no_event_backtest_model_sees_data_from_after_its_event(fixture_world):
    after = pl.col("year") >= EVENT.event_year
    distorted = replace(
        fixture_world,
        flows=fixture_world.flows.with_columns(
            value_usd=pl.when(after).then(pl.col("value_usd") * 1000).otherwise("value_usd")
        ),
    )
    chunks = pipeline.fixture_training(fixture_world)
    distorted_chunks = pipeline.fixture_training(distorted)

    model, distorted_model = _train_event(chunks), _train_event(distorted_chunks)

    assert bytes(model.model.save_raw()) == bytes(distorted_model.model.save_raw())
    # ...while the backtest itself does read the post-event years.
    report = backtests.event_backtest(chunks, EVENT, model, high_income=None, resamples=20)
    distorted_report = backtests.event_backtest(distorted_chunks, EVENT, distorted_model, high_income=None, resamples=20)
    assert report["directions"][0]["by_year"] != distorted_report["directions"][0]["by_year"]


def test_the_on_and_off_rows_differ_only_in_policy_columns_the_tariff_included(fixture_world):
    from exportspredictor import features, policy
    from exportspredictor.training import read_chunk

    chunks = pipeline.fixture_training(fixture_world)
    post = read_chunk(chunks[2023]).filter(
        pl.col("exporter").cast(pl.String) == UK, pl.col("importer").cast(pl.String) == "JPN"
    )
    # Pretend the pair pays its MFN rate, so the "on" state visibly changes the tariff.
    post = post.with_columns(log_applied_tariff=(pl.col("mfn_tariff") / 100).log1p().cast(pl.Float32))

    on, off = backtests.agreement_states(post)

    assert on.columns == off.columns == post.columns and on.height == off.height == post.height
    fixed = [c for c in post.columns if c not in features.POLICY_FEATURES]
    assert on.select(fixed).equals(post.select(fixed)) and off.select(fixed).equals(post.select(fixed))
    assert on["trade_agreement"].to_list() == [1] * post.height
    assert off["trade_agreement"].to_list() == [0] * post.height
    assert on["log_applied_tariff"].to_list() == [0.0] * post.height  # zeroed under the agreement
    mfn = post.select((pl.col("mfn_tariff") / 100).log1p().cast(pl.Float32))["mfn_tariff"]
    assert off["log_applied_tariff"].to_list() == mfn.to_list()  # MFN restored without it
    for flag in policy.POLICY_FEATURES:
        if flag != "trade_agreement":
            assert on[flag].equals(post[flag]) and off[flag].equals(post[flag])


def test_the_off_state_keeps_the_tariff_where_no_mfn_rate_is_known():
    rows = pl.DataFrame(
        {"trade_agreement": [1, 1], "log_applied_tariff": [0.1, None], "mfn_tariff": [None, 12.0]},
        schema_overrides={"trade_agreement": pl.Int8, "log_applied_tariff": pl.Float32},
    )

    on, off = backtests.agreement_states(rows)

    assert on["log_applied_tariff"].to_list() == [0.0, None]  # an unknown tariff stays unknown (ADR 0006)
    assert off["log_applied_tariff"].to_list() == pytest.approx([0.1, 0.11332869])
    assert on.schema == off.schema == rows.schema


def test_the_event_backtest_says_how_the_tariff_moves_with_the_agreement(report):
    for event in report["event_backtest"]["events"]:
        assert "tariff" in event["states"] and "MFN" in event["states"]


def test_a_backtest_model_refuses_years_from_its_event_onwards(fixture_world):
    chunks = pipeline.fixture_training(fixture_world)

    with pytest.raises(backtests.LeakageError):
        backtests.train_backtest_model(
            chunks, name="late", training_years=[2021, 2022], before=2022, uk=UK, params={}, rounds=5, device="cpu"
        )


def test_an_event_backtest_refuses_a_model_trained_after_the_event(fixture_world):
    chunks = pipeline.fixture_training(fixture_world)
    late = backtests.train_backtest_model(
        chunks, name="late", training_years=[2021, 2022], before=2023, uk=UK, params={}, rounds=5, device="cpu"
    )

    with pytest.raises(backtests.LeakageError):
        backtests.event_backtest(chunks, EVENT, late, high_income=None, resamples=10)


def test_the_lift_model_never_trains_on_the_held_out_exporters(fixture_world):
    chunks = pipeline.fixture_training(fixture_world)
    settings = PLAN.lift
    held = backtests.lift_exporters(pipeline.FIXTURE_V1_LIFT_EXPORTERS, scope=["DEU", "FRA", UK, "USA"], uk=UK)

    def train(chunks):
        return backtests.train_backtest_model(
            chunks, name="lift", training_years=settings.training_years, before=settings.before,
            uk=UK, params={"max_depth": 2}, rounds=20, device="cpu", held_out=held,
        )

    shifted = {
        year: chunk.with_columns(
            value_usd=pl.when(pl.col("exporter").is_in(held)).then(pl.col("value_usd") * 1000).otherwise("value_usd")
        )
        for year, chunk in chunks.items()
    }
    model = train(chunks)

    assert model.held_out_exporters == held
    assert bytes(model.model.save_raw()) == bytes(train(shifted).model.save_raw())


def test_the_lift_model_sees_no_held_out_exporter_and_no_year_after_the_gap_period(fixture_world):
    chunks = pipeline.fixture_training(fixture_world, baseline_year=2025)  # 2021–2024
    settings = backtests.LiftSettings(training_years=(2021, 2022), gap_years=(2021, 2022), outcome_year=2024)
    held = ["FRA"]
    poisoned = (pl.col("exporter").cast(pl.String).is_in(held)) | (pl.col("year") > max(settings.gap_years))
    distorted = {
        year: chunk.with_columns(value_usd=pl.when(poisoned).then(pl.col("value_usd") * 1000).otherwise("value_usd"))
        for year, chunk in chunks.items()
    }

    def train(chunks, years=settings.training_years):
        return backtests.train_backtest_model(
            chunks, name="lift", training_years=years, before=settings.before,
            uk=UK, params={"max_depth": 2}, rounds=20, device="cpu", held_out=held,
        )

    assert bytes(train(chunks).model.save_raw()) == bytes(train(distorted).model.save_raw())
    with pytest.raises(backtests.LeakageError):
        train(chunks, years=(2021, 2022, 2023))
    late = backtests.train_backtest_model(
        chunks, name="lift", training_years=(2021, 2022, 2023), before=2024, uk=UK, params={}, rounds=5, device="cpu",
        held_out=held,
    )
    with pytest.raises(backtests.LeakageError):
        backtests.lift_backtest(chunks, settings, late, resamples=10)


def test_on_the_fixture_each_pairs_export_gap_is_the_mean_of_its_three_yearly_gaps(fixture_world):
    chunks = pipeline.fixture_training(fixture_world, baseline_year=2025)  # 2021–2024
    settings = backtests.LiftSettings(training_years=(2021, 2022, 2023), gap_years=(2021, 2022, 2023), outcome_year=2024)
    model = backtests.train_backtest_model(
        chunks, name="lift", training_years=settings.training_years, before=settings.before,
        uk=UK, params={"max_depth": 2}, rounds=20, device="cpu", held_out=["FRA"],
    )

    yearly = backtests.lift_rows(chunks, settings, model)
    pairs = backtests.gap_period_pairs(yearly, pl.DataFrame(schema=yearly.select("exporter", "importer", "hs6", "value_usd").schema), settings)

    keys = ["exporter", "importer", "hs6"]
    full = yearly.group_by(keys).agg(n=pl.len(), gap=(pl.col("export_potential_usd") - pl.col("value_usd")).mean(),
                                     actual=pl.col("value_usd").mean()).filter(pl.col("n") == 3)
    assert set(yearly["year"]) == {2021, 2022, 2023} and full.height > 0
    checked = full.join(pairs, on=keys)
    assert checked.height == full.height
    assert checked["export_gap_usd"].to_numpy() == pytest.approx(checked["gap"].to_numpy())
    assert checked["actual_exports_usd"].to_numpy() == pytest.approx(checked["actual"].to_numpy())
    result = backtests.lift_backtest(chunks, settings, model, resamples=20)
    assert result["gap_years"] == [2021, 2022, 2023] and result["held_out_exporters"] == ["FRA"]


# --- The backtests in the evaluation report -------------------------------------------------------


@pytest.fixture
def report(fixture_artefacts):
    return fixture_artefacts.evaluation


def test_lift_is_reported_with_a_confidence_interval_as_a_tendency(report):
    lift = report["lift"]

    low, high = lift["ci"]
    assert low <= lift["lift"] <= high
    assert lift["ci_level"] == 0.95
    assert "tendency" in lift["text"] and "not a promise" in lift["text"]
    assert lift["training_years"] == [2021, 2022] and lift["outcome_year"] == 2023
    assert lift["gap_years"] == [2021, 2022]


def test_the_lift_report_holds_out_the_frozen_v1_list_within_scope(report, fixture_artefacts):
    lift = report["lift"]
    scope = fixture_artefacts.metadata["exporter_scope"]

    assert lift["held_out_exporters"] == ["FRA"]  # the fixture's synthetic v1 list is FRA and MEX; MEX is out of scope
    assert "MEX" not in scope and "FRA" in scope
    assert lift["held_out_from"] == "v1"


def test_lift_reports_raw_and_size_matched_lift_each_with_an_interval(report):
    lift = report["lift"]

    assert lift["lift"] is not None and len(lift["ci"]) == 2
    assert set(lift["size_matched"]) >= {"lift", "ci"} and len(lift["size_matched"]["ci"]) == 2


def test_the_event_backtest_reports_predicted_and_observed_effects_and_its_control_group(report):
    event = report["event_backtest"]["events"][0]
    direction = event["directions"][0]

    assert event["role"] == "primary"
    assert direction["controls"] == ["IND", "HKG"]  # Canada already has a UK agreement
    assert event["control_rule"]
    for year in direction["by_year"]:
        assert year["predicted"] is not None and year["observed"] is not None
    assert direction["pooled"]["predicted"] >= 0  # monotone: switching the agreement on never lowers it
    assert "difference-in-differences" in report["event_backtest"]["text"]
    assert report["event_backtest"]["caveat"]


def test_every_event_backtest_model_trained_only_before_its_event(report):
    for event in report["event_backtest"]["events"]:
        assert max(event["training_years"]) < event["event_year"]


def test_the_real_plan_trains_before_each_event_and_reads_the_years_the_ticket_names():
    plan = backtests.REAL_PLAN

    assert plan.lift.training_years == tuple(range(2012, 2018))
    assert (plan.lift.gap_years, plan.lift.outcome_year) == ((2015, 2016, 2017), 2023)
    assert (plan.lift.min_growth_share, plan.lift.min_growth_usd) == (0.5, 100_000)
    assert plan.event.training_years == tuple(range(2012, 2019)) and plan.event.event_year == 2019
    assert plan.event.post_years == (2019, 2020, 2021)
    assert "COVID" in plan.event.caveat
    assert plan.secondary.event_year == 2023 and plan.secondary.post_years == (2024,)
    assert max(plan.secondary.training_years) < plan.secondary.event_year


def test_event_text_says_when_the_model_overstates_the_observed_effect():
    direction = {
        "label": "EU exports to Japan", "controls": ["USA"],
        "pooled": {"predicted": 0.45, "observed": 0.03, "observed_ci": [-0.06, 0.15]},
    }
    report = {"title": "T", "in_force": "2019", "training_years": [2012, 2018], "post_years": [2019, 2021], "directions": [direction]}

    text = backtests.event_text(report)

    assert "larger than" in text and "outside" in text


def test_lift_text_says_when_large_gaps_do_no_better_than_pairs_of_similar_size():
    result = {
        "lift": 6.6, "ci": [4.8, 10.4], "top_rate": 0.045, "base_rate": 0.007, "gap_years": [2015, 2016, 2017], "outcome_year": 2023,
        "min_growth_share": 0.5, "min_growth_usd": 1e5,
        "size_matched": {"lift": 0.83, "ci": [0.80, 0.87], "expected_rate": 0.054},
    }

    text = backtests.lift_text(result)

    assert "less often than pairs of similar Export Potential" in text
    assert "tendency" in text


def test_control_markets_leave_out_excluded_markets():
    rows = pl.DataFrame(
        {"exporter": ["GBR"] * 2, "importer": ["RUS", "USA"], "year": [2022] * 2, "value_usd": [9.0, 1.0], "trade_agreement": [0, 0]}
    )

    controls = backtests.control_markets(rows, exporters=["GBR"], treated=["AUS"], pre_years=[2022], high_income=None, n=5, exclude=["RUS"])

    assert controls == ["USA"]
