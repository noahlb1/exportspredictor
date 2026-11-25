"""The final round's account (spec #155, #172): one row per idea, read from the loop's saved results and the
unrecorded-preferences probe, so the notebook's table cannot drift from what the loop recorded."""

import pytest

from exportspredictor import reporting


def _section(base: float, arms: dict[str, float], kept: str | None = None) -> dict:
    return {
        "base": {"selection_measure": base},
        "base_arm": {"arm": "best so far as served"},
        "arms": [{"arm": arm, "selection_measure": base + change, "change": change} for arm, change in arms.items()],
        "kept": kept,
        "win_threshold": 0.0089,
        "decided": True,
    }


def _results() -> dict:
    return {"experiments": {
        "naive baseline": _section(0.53, {"blend": 0.04}, kept="blend"),  # before the round: not shown
        "market reach cap": _section(0.62, {"cap Market Reach": -0.0006}),
        "hyperparameters: depth": _section(0.62, {"max_depth 8": -0.0002, "max_depth 14": 0.0004}),
        "hyperparameters: subsampling": _section(0.62, {"colsample_bytree 0.6": 0.0005}),
        "model family: lightgbm": _section(0.62, {"LightGBM": -0.0005}),
        "model family: catboost": _section(0.62, {"CatBoost": -0.0019}),
    }}


def test_each_idea_of_the_round_is_one_row_with_its_best_arm_against_the_win_threshold():
    table = reporting.final_round(_results())

    rows = {row["idea"]: row for row in table.iter_rows(named=True)}
    assert list(rows) == [
        "Cap on Market Reach's upside (#163)", "A fresh hyperparameter search (#166)", "Other model families (#167)",
    ]
    search = rows["A fresh hyperparameter search (#166)"]
    assert search["experiments"] == 2
    assert search["arms"] == 3
    assert search["best_arm"] == "colsample_bytree 0.6"
    assert search["gain"] == pytest.approx(0.0005)
    assert search["best"] == pytest.approx(0.6205)
    assert search["threshold"] == pytest.approx(0.0089)
    assert search["decision"] == "dropped"
    assert rows["Other model families (#167)"]["best_arm"] == "LightGBM"


def test_a_kept_arm_is_reported_as_kept():
    results = _results()
    results["experiments"]["market reach cap"]["kept"] = "cap Market Reach"

    table = reporting.final_round(results)

    assert table.row(0, named=True)["decision"] == "kept"


def test_the_probe_is_a_row_decided_against_its_gate_not_the_win_threshold():
    probe = {"base": {"selection_measure": 0.62099}, "probe": {"selection_measure": 0.62089}, "gain": -0.0001,
             "gate": 0.0044, "retrained_arm_runs": False}

    table = reporting.final_round(_results(), probe)

    row = table.row(-1, named=True)
    assert row["idea"] == "Tariff 0 where an agreement's preference is unrecorded (#171)"
    assert row["gain"] == pytest.approx(-0.0001)
    assert row["threshold"] == pytest.approx(0.0044)
    # The row says which bar it is judged against, so its threshold is not read as the Win Threshold.
    assert row["judged_against"] == "its pre-registered gate"
    assert row["decision"] == "not retrained (missed its gate)"
    assert {r["judged_against"] for r in table.head(-1).iter_rows(named=True)} == {"Win Threshold"}


def test_without_any_idea_of_the_round_it_is_not_run():
    results = {"experiments": {"naive baseline": _section(0.53, {"blend": 0.04}, kept="blend")}}

    assert isinstance(reporting.final_round(results), reporting.NotRun)


def test_the_probe_is_read_from_its_saved_file_or_is_none(tmp_path):
    assert reporting.final_round_probe(tmp_path) is None

    saved = tmp_path / "derived" / "final_round_2022"
    saved.mkdir(parents=True)
    (saved / "unrecorded_preferences.json").write_text('{"gain": -0.0001}', encoding="utf-8")

    assert reporting.final_round_probe(tmp_path) == {"gain": -0.0001}
