"""How the served model was chosen (ADR 0008): what the Model Evaluation page and the model card show.

Everything here is read from what an artefact already records: `metadata["ranking_experiments"]`
(the experiment loop's Selection Measure, Noise Floor, Arms and five-fold confirmation),
`metadata["model_form"]` and the evaluation report's `earlier_versions` block. Nothing is
recomputed and no test-year figure is read beyond the report.
"""

from exportspredictor.prediction import ModelForm

NOT_RECORDED = "No experiment results are recorded for this version, so how its Model Form was chosen cannot be shown."


def how_chosen(metadata: dict, report: dict) -> dict:
    """The Selection Measure, the Model Form, the Win Threshold, every experiment, the confirmation
    and the test-year ranking beside earlier versions, JSON-ready. `recorded` is False (and the
    experiment fields empty) when the artefact records no experiment results."""
    form = ModelForm.from_dict(metadata.get("model_form") or {"form": "model alone", "naive_weight": 0.0})
    summary = ranking_summary(metadata.get("ranking_experiments"))
    return {
        "recorded": summary is not None,
        "model_form": {**form.to_dict(), "text": form.text},
        **(summary or {"experiments": []}),
        "text": _text(summary, form),
        "test_year": report.get("earlier_versions"),
        "traded_away": traded_away(report.get("earlier_versions")),
    }


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
    form = metadata.get("model_form")
    if not form or not ModelForm.from_dict(form).naive_weight:
        return None
    model_form = ModelForm.from_dict(form)
    weight = 1 - model_form.naive_weight
    events = (report.get("event_backtest") or {}).get("events") or []
    model_effect = events[0]["directions"][0]["pooled"]["predicted"] if events else None
    served = model_form.served_effect(model_effect) if model_effect is not None else None
    example = (
        f" In the primary Event Backtest ({events[0]['title']}), the model's {model_effect:+.0%} would be served as "
        f"{served:+.0%}."
        if served is not None else ""
    )
    return {
        "model_weight": weight,
        "model_effect": model_effect,
        "served_effect": served,
        "text": (
            "The Event Backtests measured the model alone, before the Blend. The Naive Baseline has no policy "
            f"features, so a served Scenario Effect is the model's effect scaled by its weight, {weight:g}, on the log "
            f"scale, and is smaller.{example} Both are associations, not forecasts."
        ),
    }


def ranking_summary(record: dict | None) -> dict | None:
    """How the experiment loop chose the served form: the Selection Measure, the Noise Floor's Win
    Threshold, each experiment's best Arm and whether it was kept, and the five-fold confirmation
    fold by fold (None where not run). None when nothing is recorded."""
    if not record:
        return None
    floor = record.get("noise_floor") or {}
    experiments = []
    for name, section in (record.get("experiments") or {}).items():
        best = next((row for row in section.get("arms", []) if row.get("best")), None)
        experiments.append({
            "name": name,
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


def _text(summary: dict | None, form: ModelForm) -> str:
    if summary is None:
        return f"{NOT_RECORDED} {form.text}"
    kept = [e["name"] for e in summary["experiments"] if e["kept"]]
    return (
        f"Every choice was made on one Selection Measure, on {summary['validation_year']} only: "
        f"{summary['selection_measure']}. An Arm was kept only if it beat the best so far by more than the Win "
        f"Threshold, {summary['win_threshold']:.4f} (twice the Noise Floor's standard deviation). "
        f"Of {len(summary['experiments'])} experiments, {len(kept)} kept a change"
        + (f" ({', '.join(kept)})" if kept else "")
        + f". {form.text}"
    )
