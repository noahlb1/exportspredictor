"""How the served model was chosen (ADR 0008): what the Model Evaluation page and the model card show.

Everything here is read from what an artefact already records: `metadata["ranking_experiments"]`
(the experiment loop's Selection Measure, Noise Floor, Arms and five-fold confirmation),
`metadata["model_form"]`, `metadata["round_scaling"]` and the evaluation report's
`earlier_versions` and `event_backtest` blocks. Nothing is recomputed and no test-year figure is
read beyond the report. The model card (`governance`) and the notebook (`reporting`) read the
same functions, so they cannot state different figures.
"""

from exportspredictor.evaluation import relabelled
from exportspredictor.loop import (
    count_experiments,
    experiment_label,
    served_configuration,
)
from exportspredictor.prediction import MODEL_ALONE, ModelForm
from exportspredictor.wording import SELECTION_MEASURE_PLAIN, number_text

CHOSEN_ON_DEVIANCE = "Tweedie deviance (the hyperparameter search)"
CHOSEN_ON_SELECTION_MEASURE = "Selection Measure"
NOT_RECORDED = "No experiment results are recorded for this version, so how its Model Form was chosen cannot be shown."


def how_chosen(metadata: dict, report: dict) -> dict:
    """The Selection Measure, the Model Form, the Win Threshold, every experiment, the confirmation
    and the test-year ranking beside earlier versions, JSON-ready. `recorded` is False (and the
    experiment fields empty) when the artefact records no experiment results."""
    form = served_model_form(metadata)
    summary = ranking_summary(metadata.get("ranking_experiments"))
    report = relabelled(report)  # today's names for a report saved earlier
    return {
        "recorded": summary is not None,
        "model_form": model_form_summary(metadata),
        **(summary or {"experiments": []}),
        "text": _text(summary, form),
        "test_year_ranking": report.get("earlier_versions"),
        "traded_away": traded_away(report.get("earlier_versions")),
    }


def served_model_form(metadata: dict) -> ModelForm:
    """The served Model Form (ADR 0008); the model alone where none is recorded."""
    form = metadata.get("model_form")
    return ModelForm.from_dict(form) if form else MODEL_ALONE


def model_form_summary(metadata: dict) -> dict:
    """The served Model Form, JSON-ready, with its plain-words text."""
    form = served_model_form(metadata)
    return {**form.to_dict(), "text": form.text}


def primary_event_backtest(report: dict) -> dict | None:
    """The primary Event Backtest of an evaluation report (its `role` is "primary"), or None where none was run."""
    events = (report.get("event_backtest") or {}).get("events") or []
    return next((event for event in events if event.get("role") == "primary"), None)


def round_count(metadata: dict) -> dict:
    """How many boosting rounds the served fold models trained (`rounds`), what chose them (`chosen_by`,
    `rounds_chosen`, read from `round_scaling`) and the count the next rebuild takes (`next_rounds`, from
    `loop.served_configuration`, None without experiment results), with the model card's sentence.

    The pipeline records `selected_rounds` since it serves the loop's chosen form (chosen by the Selection Measure). An artefact
    recording the search's `tuned_rounds` was trained at v2's count, which early stopping on Tweedie
    deviance chose; both are scaled by the ratio of training rows."""
    scaling, rounds = metadata.get("round_scaling") or {}, metadata.get("rounds")
    record = metadata.get("ranking_experiments") or {}
    next_rounds = served_configuration(record).rounds if record.get("best") else None
    ratio = number_text(scaling.get("ratio"), ".4f")
    if "selected_rounds" in scaling:
        chosen = scaling["selected_rounds"]
        return {"chosen_by": CHOSEN_ON_SELECTION_MEASURE, "rounds_chosen": chosen, "rounds": rounds,
                "next_rounds": next_rounds, "text": (
            f"The {number_text(rounds, ',')} rounds were chosen by the Selection Measure ({scaling['rule']}: "
            f"{number_text(chosen, ',')}), scaled by the ratio of training rows ({ratio}). Deviance chose none of it.")}
    if "tuned_rounds" in scaling:
        chosen = scaling["tuned_rounds"]
        then = (f"the median of the five-fold confirmation's rounds ({number_text(next_rounds, ',')}, each chosen "
                "by the Selection Measure), scaled the same way" if record.get("confirmation") and next_rounds
                else "the Selection Measure's count")
        return {"chosen_by": CHOSEN_ON_DEVIANCE, "rounds_chosen": chosen, "rounds": rounds,
                "next_rounds": next_rounds, "text": (
            f"Disclosed deviation: the served models were trained for {number_text(rounds, ',')} rounds, the "
            f"hyperparameter search's {number_text(chosen, ',')} (early stopping on Tweedie deviance) "
            f"scaled by the ratio of training rows ({ratio}). Under ADR 0008 deviance should choose nothing. The "
            f"artefact was not rebuilt, so that the test year stays read once; the next rebuild uses {then}.")}
    return {"chosen_by": None, "rounds_chosen": None, "rounds": rounds, "next_rounds": next_rounds,
            "text": "How the round count was chosen is not recorded for this version."}


def traded_away(block: dict | None) -> dict | None:
    """What choosing on the ranking cost: this version's Tweedie deviance over all Exporters against the
    first earlier version that records one, read from the report's `earlier_versions`. None when no
    earlier version records a deviance, or when this version's is no worse."""
    if not block:
        return None
    this, *earlier = block["rows"]
    other = next((row for row in earlier if row.get("tweedie_deviance_all_exporters") is not None), None)
    deviance = this.get("tweedie_deviance_all_exporters")
    if other is None or deviance is None or deviance <= other["tweedie_deviance_all_exporters"]:
        return None
    lift, other_lift = this.get("size_matched_lift"), other.get("size_matched_lift")
    lift_text = (
        f" Size-matched Lift is {lift:.2f} ({other['name']} {other_lift:.2f})."
        if lift is not None and other_lift is not None else ""
    )
    text = (
        f"Choosing on the ranking traded away accuracy on size: Tweedie deviance over all Exporters is "
        f"{deviance:,.1f} for this version against {other['tweedie_deviance_all_exporters']:,.1f} for {other['name']} "
        f"(lower is better), so the size of Export Potential is estimated less well.{lift_text} Treat dollar values "
        "with more caution than rankings."
    )
    return {
        "deviance": deviance,
        "earlier_deviance": other["tweedie_deviance_all_exporters"],
        "earlier_name": other["name"],
        "size_matched_lift": lift,
        "earlier_size_matched_lift": other_lift,
        "text": text,
    }


def event_backtest_note(metadata: dict, report: dict) -> dict | None:
    """When a Blend is served: the Event Backtests measured the model alone, and a served Scenario
    Effect is the model's effect scaled by the model's weight on the log scale, so it is smaller.
    Gives the primary Event Backtest's pooled predicted effect both ways. None for the model alone."""
    form = served_model_form(metadata)
    if not form.naive_weight:
        return None
    weight = 1 - form.naive_weight
    primary = primary_event_backtest(report)
    model_effect = primary["directions"][0]["pooled"]["predicted"] if primary else None
    served = form.served_effect(model_effect) if model_effect is not None else None
    example = (
        f": the model's {model_effect:+.0%} in the primary Event Backtest ({primary['title']}) would show as "
        f"{served:+.0%}"
        if primary is not None and model_effect is not None and served is not None else ""
    )
    return {
        "model_weight": weight,
        "model_effect": model_effect,
        "served_effect": served,
        "text": (
            f"The Event Backtests measured the model alone. The app blends it (weight {weight:g}) with the Naive "
            f"Baseline, which ignores trade policy, so the Scenario Effects shown are smaller{example}. Both are "
            "associations, not forecasts."
        ),
    }


def ranking_summary(record: dict | None) -> dict | None:
    """How the experiment loop chose the served form: the Selection Measure, the Noise Floor's Win
    Threshold, each recorded decision's best Arm, whether it was kept and its label (the research
    note's experiment number and the Glossary's name), the number of experiments those decisions
    cover (`n_experiments`; the loop records nine decisions for seven) and the five-fold
    confirmation fold by fold (None where not run). None when nothing is recorded."""
    if not record:
        return None
    floor = record.get("noise_floor") or {}
    experiments, decisions = [], {}
    for name, section in (record.get("experiments") or {}).items():
        best = next((row for row in section.get("arms", []) if row.get("best")), None)
        decisions[name] = [row["arm"] for row in section.get("arms", [])]
        experiments.append({
            "name": name,
            "label": experiment_label(name, decisions[name]),
            "kept": section.get("kept"),
            "best_arm": best["arm"] if best else None,
            "selection_measure": best["selection_measure"] if best else None,
            "change": best.get("change") if best else None,
        })
    return {
        "selection_measure": record.get("selection_measure"),
        "validation_year": record.get("validation_year"),
        "noise_floor_sd": floor.get("sd"),
        "win_threshold": floor.get("win_threshold"),
        "experiments": experiments,
        "n_experiments": count_experiments(decisions),
        "confirmation": _confirmation(record.get("confirmation")),
    }


def _confirmation(confirmation: dict | None) -> dict | None:
    if not confirmation:
        return None
    return (
        {key: confirmation[key] for key in ("n_folds", "difference", "win_threshold", "beats_v2")}
        | {name: confirmation[name]["selection_measure"] for name in ("winner", "v2", "naive")}
        | {"folds": [
            {"fold": fold["fold"], "winner": fold["winner"]["selection_measure"], "v2": fold["v2"]["selection_measure"]}
            for fold in confirmation.get("folds", [])
        ]}
    )


def selection_rule_text(summary: dict) -> str:
    """How the experiment loop chose (`ranking_summary`): one Selection Measure, kept only above the Win Threshold."""
    return (
        f"Every choice was made on one Selection Measure, on {summary['validation_year']} only: "
        f"{SELECTION_MEASURE_PLAIN} (the headline's rank correlation). An Arm, one tried setting, was kept only if it "
        f"beat the best so far by more than the Win Threshold, {number_text(summary['win_threshold'], '.4f')}: twice "
        "the Noise Floor, how far the measure moves when only the random seed changes."
    )


def _text(summary: dict | None, form: ModelForm) -> str:
    """The section's lead sentence; without experiment results it says so and names the served form."""
    if summary is None:
        return f"{NOT_RECORDED} {form.text}"
    kept = [f"{e['label']}: {e['kept']}" for e in summary["experiments"] if e["kept"]]
    count, decisions = summary["n_experiments"], len(summary["experiments"])
    steps = f", decided in {decisions} steps" if decisions != count else ""
    return (
        f"{selection_rule_text(summary)} Of {count} experiment{'s' if count != 1 else ''}{steps}, {len(kept)} kept a change"
        + (f" ({', '.join(kept)})" if kept else "")
        + "."
    )
