"""Opt-in pipeline parts: a default run trains only what the headline needs.

The Event Backtests (Lift included), the monotone-constraint comparison and the gain ladder
each run only when asked for; the v1 comparison's primary metric is computed by default.
"""

import math

import pytest
import xgboost

from exportspredictor import realisation
from exportspredictor.pipeline import (
    Parts,
    build_artefacts,
    fixture_rescore,
    parse_parts,
)


@pytest.fixture(scope="module")
def tuned(fixture_artefacts):
    """The fixture's search and experiment loop results, so the runs below train nothing for either."""
    return fixture_artefacts.metadata


def _run(world, tuned, monkeypatch, parts: Parts):
    """Build the fixture artefacts with `parts`; return them and the number of models trained."""
    trained = []
    train = xgboost.train

    def counting(*args, **kwargs):
        trained.append(1)
        return train(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(xgboost, "train", counting)
        bundle = build_artefacts(
            world, version="parts", search_results=tuned["search"], ranking_results=tuned["ranking_experiments"],
            parts=parts, rescore=fixture_rescore(world),  # as the session fixture artefacts are built
        )
    return bundle, len(trained)


@pytest.fixture(scope="module")
def default_run(fixture_world, tuned):
    with pytest.MonkeyPatch.context() as monkeypatch:
        return _run(fixture_world, tuned, monkeypatch, Parts())


def test_a_default_run_trains_only_the_fold_and_test_year_models(default_run):
    bundle, trained = default_run
    assert trained == 2 * bundle.metadata["n_folds"]


OWN_PART = {
    "event_backtests": {"lift", "lift_without_market_reach", "event_backtest"},
    "constraint_comparison": {"monotone_constraints"},
}


def _blocks(report: dict) -> set[str]:
    """The report's blocks, with the gain ladder as its own."""
    return set(report) | ({"gain_ladder"} if "ladder" in report.get("v1_comparison", {}) else set())


@pytest.fixture(scope="module")
def runs(fixture_world, tuned, default_run):
    """One run per opt-in part, and one with all three."""
    out = {"default": default_run}
    with pytest.MonkeyPatch.context() as monkeypatch:
        for name in ("event_backtests", "constraint_comparison", "gain_ladder"):
            out[name] = _run(fixture_world, tuned, monkeypatch, Parts(**{name: True}))
        out["every"] = _run(fixture_world, tuned, monkeypatch, Parts.every())
    return out


def test_a_default_run_skips_every_opt_in_part(default_run):
    report = default_run[0].evaluation
    assert not _blocks(report) & {"lift", "event_backtest", "monotone_constraints", "gain_ladder"}


def test_a_default_run_still_computes_the_v1_comparisons_primary_metric(default_run):
    block = default_run[0].evaluation["v1_comparison"]
    assert block["primary"]["v2"] is not None and block["primary"]["v1"] is not None
    assert block["secondaries_not_run"] == ["Lift", "Event Backtest"]
    assert "not run" in block["verdict"]["text"]


@pytest.mark.parametrize("name", ["event_backtests", "constraint_comparison", "gain_ladder"])
def test_each_opt_in_adds_exactly_its_own_part(runs, name):
    (default, default_trained), (bundle, trained) = runs["default"], runs[name]
    own = OWN_PART.get(name, {"gain_ladder"})
    assert _blocks(bundle.evaluation) - _blocks(default.evaluation) == own
    assert trained > default_trained


def _comparable(value):
    """A report without its timings, floats rounded: two runs differ in the last bits of a sum."""
    if isinstance(value, dict):
        return {k: _comparable(v) for k, v in value.items() if "seconds" not in k}
    if isinstance(value, list):
        return [_comparable(v) for v in value]
    if isinstance(value, float):
        return "nan" if math.isnan(value) else round(value, 9)
    return value


def test_all_three_parts_together_reproduce_the_full_run(runs, fixture_artefacts):
    default_trained = runs["default"][1]
    added = sum(runs[name][1] - default_trained for name in ("event_backtests", "constraint_comparison", "gain_ladder"))
    bundle, trained = runs["every"]
    assert trained == default_trained + added
    # The ITC comparison is added to the fixture's full run after the pipeline (`realisation`), not by a part.
    full = {key: value for key, value in fixture_artefacts.evaluation.items() if key != realisation.REPORT_KEY}
    assert _comparable(bundle.evaluation) == _comparable(full)


@pytest.mark.parametrize(
    ("argv", "parts"),
    [
        ([], Parts()),
        (["--event-backtests"], Parts(event_backtests=True)),
        (["--constraint-comparison", "--gain-ladder"], Parts(constraint_comparison=True, gain_ladder=True)),
        (["--every-part"], Parts.every()),
    ],
)
def test_the_command_line_switches_on_the_parts_asked_for(argv, parts):
    assert parse_parts(argv) == parts


def test_the_ppml_benchmark_is_opt_in_and_trains_no_model(fixture_world, tuned, default_run):
    default, default_trained = default_run
    assert default.ppml_benchmark is None
    with pytest.MonkeyPatch.context() as monkeypatch:
        bundle, trained = _run(fixture_world, tuned, monkeypatch, Parts(ppml_benchmark=True))
    assert trained == default_trained
    assert bundle.ppml_benchmark is not None and bundle.ppml_benchmark["ppml"]["converged"]
    assert bundle.ppml_benchmark["model"]["model_form"] == bundle.model_form.to_dict()
    assert _comparable(bundle.evaluation) == _comparable(default.evaluation)


def test_the_command_line_switches_on_the_ppml_benchmark():
    assert parse_parts(["--ppml-benchmark"]) == Parts(ppml_benchmark=True)
    assert parse_parts(["--every-part"]).ppml_benchmark


@pytest.mark.parametrize("command", ["refresh", "rerange", "compare"])
def test_the_command_line_help_names_the_commands_that_copy_the_newest_version(command, capsys):
    with pytest.raises(SystemExit):
        parse_parts(["--help"])
    assert f"python -m exportspredictor.pipeline {command}" in capsys.readouterr().out
