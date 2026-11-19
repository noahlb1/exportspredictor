"""Illustrative Scenario Effects (ADR 0008): the Scenario note served beside every Scenario Effect.

A Scenario Effect or Blocked Opportunity value is **Illustrative**: an upper-end association the
model has learned, shown beside its benchmarks, never a forecast and never selected on. The
Scenario note puts beside it the PPML benchmark's effect with its interval, the model's own Trade
Agreement association on the served Model Form (both from `Artefacts.ppml_benchmark`), the primary
Event Backtest's predicted and observed change, and, for a Blend, what the Event Backtests measured
(`model_choice.event_backtest_note`). A part the artefact lacks is said to be not run, never hidden.
"""

from typing import TYPE_CHECKING

from exportspredictor.evaluation import NOT_RUN
from exportspredictor.model_choice import event_backtest_note, primary_event_backtest

if TYPE_CHECKING:
    from exportspredictor.artefacts import Artefacts

ILLUSTRATIVE = "Illustrative"
ILLUSTRATIVE_TEXT = (
    "Scenario Effects and Blocked Opportunity values are illustrative: upper-end associations the model has "
    "learned from past trade, shown beside their benchmarks. They are not forecasts, and the model was not chosen "
    "on them."
)


def scenario_note(artefacts: "Artefacts") -> dict:
    """The Scenario note for an artefact, JSON-ready: the Illustrative `label` and its `text`, then the
    `benchmark`, `model_association` and `event_backtest` parts (each {run, text, figures...}) and the
    `blend` note (None for the model alone)."""
    report = artefacts.evaluation or {}
    return {
        "label": ILLUSTRATIVE,
        "text": ILLUSTRATIVE_TEXT,
        "benchmark": _benchmark(artefacts.ppml_benchmark),
        "model_association": _model_association(artefacts.ppml_benchmark),
        "event_backtest": _event_backtest(report),
        "blend": event_backtest_note(artefacts.metadata, report),
    }


def _pct(value: float) -> str:
    return f"{value:+.1f}%"


def _benchmark(saved: dict | None) -> dict:
    if not saved:
        return {"run": False, "text": f"PPML benchmark: {NOT_RUN}."}
    fit, published = saved["ppml"], saved["published_range"]
    years = saved["years"]
    return {
        "run": True,
        "effect_pct": fit["effect_pct"],
        "low_pct": fit["effect_low_pct"],
        "high_pct": fit["effect_high_pct"],
        "ci_level": saved["ci_level"],
        "published_low_pct": published["low_pct"],
        "published_high_pct": published["high_pct"],
        "text": (
            f"A PPML gravity benchmark ({saved['level']} level, {years[0]}–{years[-1]}) associates a Trade Agreement "
            f"with {_pct(fit['effect_pct'])} in Flows ({saved['ci_level']:.0%} interval {_pct(fit['effect_low_pct'])} "
            f"to {_pct(fit['effect_high_pct'])}); published estimates range from {_pct(published['low_pct'])} to "
            f"{_pct(published['high_pct'])}."
        ),
    }


def _model_association(saved: dict | None) -> dict:
    if not saved:
        return {"run": False, "text": f"The model's own Trade Agreement association: {NOT_RUN}."}
    model = saved["model"]
    return {
        "run": True,
        "effect_pct": model["effect_pct"],
        "pairs": model["pairs"],
        "model_form": model["model_form"]["form"],
        "text": (
            f"On the served Model Form ({model['model_form']['form'].capitalize()}), the model associates adding a Trade Agreement with "
            f"{_pct(model['effect_pct'])} in Export Potential over the {model['pairs']:,} UK pairs without one "
            "(weighted by Baseline Export Potential)."
        ),
    }


def _event_backtest(report: dict) -> dict:
    primary = primary_event_backtest(report)
    if primary is None:
        return {"run": False, "text": f"Event Backtest: {NOT_RUN}."}
    direction = primary["directions"][0]
    pooled = direction["pooled"]
    low, high = pooled["observed_ci"]
    predicted, observed = pooled["predicted"], pooled["observed"]
    interval = "" if low is None or high is None else f" (95% interval {low:+.0%} to {high:+.0%})"
    return {
        "run": True,
        "title": primary["title"],
        "direction": direction["label"],
        "predicted": predicted,
        "observed": observed,
        "observed_low": low,
        "observed_high": high,
        "text": (
            f"In the primary Event Backtest ({primary['title']}), the model alone associated the agreement with "
            f"{_share(predicted)} for {direction['label']}; the observed change against Control Markets was "
            f"{_share(observed)}{interval}."
        ),
    }


def _share(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.0%}"
