"""How the served model was chosen, read from an artefact's recorded experiment loop (ADR 0008)."""

from dataclasses import replace

from exportspredictor import governance, model_choice

# The real loop's nine recorded decisions (artefact 20261119T001736Z), Arms in recorded order.
# The research note numbers them as seven experiments: 1-2 share one decision (the blend and the
# starting point), and experiments 5 and 6 decide each Arm on its own.
RECORDED = {
    "naive baseline": (["blend", "starting point"], "blend"),
    "select on ranking": (["v1", "search trial 15", "search trial 23", "search trial 19", "depth 6"], None),
    "two-part model": (["two-part"], None),
    "scope: UK Zero Cells": (["UK Zero Cells sampled"], None),
    "scope: zero rate": (["zero rate 2%"], None),
    "product detail: trading_pair_share": (["+ trading_pair_share"], None),
    "product detail: importing_markets": (["+ importing_markets"], None),
    "product detail: hs4": (["+ hs4"], None),
    "monotone constraints": (["constraints on", "constraints off"], None),
}


def _record(recorded=RECORDED) -> dict:
    return {
        "selection_measure": "the UK ranking",
        "validation_year": 2022,
        "noise_floor": {"sd": 0.0044, "win_threshold": 0.0089},
        "experiments": {
            name: {
                "arms": [{"arm": arm, "selection_measure": 0.57, "change": 0.0, "best": i == 0} for i, arm in enumerate(arms)],
                "kept": kept,
            }
            for name, (arms, kept) in recorded.items()
        },
    }


def test_the_lead_sentence_counts_the_research_note_s_seven_experiments_not_the_nine_decisions():
    text = model_choice.how_chosen({"ranking_experiments": _record()}, {})["text"]

    assert "Of 7 experiments" in text
    assert "9 experiments" not in text


def test_a_loop_that_ran_only_the_blend_counts_one_experiment():
    text = model_choice.how_chosen({"ranking_experiments": _record({"naive baseline": (["blend"], "blend")})}, {})["text"]

    assert "Of 1 experiments" not in text and "Of 1 experiment," in text


def test_each_decision_is_labelled_with_its_experiment_number_and_glossary_names():
    summary = model_choice.ranking_summary(_record())

    labels = [e["label"] for e in summary["experiments"]]
    assert labels[0] == "1–2. Naive Baseline"
    assert labels[1:4] == ["3. select on ranking", "4. two-part model", "5. scope: UK Zero Cells"]
    assert labels[-1] == "7. monotone constraints"
    assert summary["n_experiments"] == 7


def test_the_model_card_labels_the_nine_decisions_as_seven_experiments(fixture_artefacts):
    bundle = replace(fixture_artefacts, metadata={**fixture_artefacts.metadata, "ranking_experiments": _record()})

    text = governance.render_model_card(governance.model_card(bundle))

    section = text.split("\n## Alternatives considered on 2022\n", 1)[1].split("\n## ", 1)[0]
    assert "9 decisions for 7 experiments" in section
    assert "| 1–2. Naive Baseline | blend |" in section
    assert "| naive baseline |" not in section
