"""What the pipeline serves of the loop (#77): the chosen form, its configuration and round count, and the record an artefact keeps."""

import statistics
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from exportspredictor import search
from exportspredictor.features import FEATURES, ZERO_SAMPLING_RATE
from exportspredictor.loop.arms import (
    FIXTURE_LOOP,
    NAIVE_EXPERIMENT,
    NOISE_FLOOR,
    LoopPlan,
    Settings,
    blend_arm,
)
from exportspredictor.loop.confirmation import CONFIRMATION, WINNER
from exportspredictor.loop.data import fixture_data
from exportspredictor.loop.runner import Loop
from exportspredictor.prediction import ModelForm

RECORDED = ("selection_measure", "validation_year", "fold", "n_folds", "naive", "noise_floor", "experiments", "best",
            CONFIRMATION)  # the parts of the results an artefact records


def chosen_form(results: Mapping) -> ModelForm:
    """The model form the loop chose (its best so far), as the pipeline serves it.

    Only the forms the pipeline can serve: the model alone or a blend, on the package's
    features, with the monotone constraints. Any other winner raises ValueError, so an
    artefact never serves a form other than the one its experiments chose.
    """
    settings = Settings.from_dict(results["best"]["settings"])
    if settings.base_margin is not None or settings.two_part:
        raise ValueError(f"The loop chose a form the pipeline does not serve: {results['best']['arm']!r}")
    if list(settings.features) != list(FEATURES) or not settings.monotone:
        raise ValueError("The loop's best so far changed the features or the constraints; the pipeline serves neither")
    return ModelForm(naive_weight=settings.naive_weight)


@dataclass(frozen=True)
class ServedConfiguration:
    """What the pipeline serves of the loop's choice: the form, the configuration and its round count.

    `rounds` was chosen by the selection measure on the loop's training rows (the years before
    the validation year, at `non_uk_rate`); the pipeline scales it to its own rows. Deviance
    chose none of it.
    """

    form: ModelForm
    params: dict
    rounds: int
    rounds_rule: str  # how `rounds` was taken from the loop's results
    non_uk_rate: float
    validation_year: int


def served_configuration(results: Mapping) -> ServedConfiguration:
    """The loop's best so far as the pipeline serves it, with the round count its selection measure chose.

    With a five-fold confirmation, the rounds are the median (lower) of the winner's per-fold
    rounds, each chosen by the selection measure on that fold's held-out UK pairs; the
    confirmation must be of the best so far's settings. Without one (the fixture), they are the
    best arm's own. Raises ValueError for a form the pipeline cannot serve (`chosen_form`) or a
    best so far that changed how Zero Cells are sampled (the pipeline trains on the rows as built).
    """
    form = chosen_form(results)
    best = Settings.from_dict(results["best"]["settings"])
    start = Settings.from_dict(results[NOISE_FLOOR]["arms"][0]["settings"])
    if best.zero_rate != start.zero_rate or best.uk_zeros_sampled:
        raise ValueError("The loop's best so far changed how Zero Cells are sampled; the pipeline trains on the built rows")
    confirmation = results.get(CONFIRMATION)
    if confirmation is not None:
        confirmed = Settings.from_dict(confirmation["settings"][WINNER])
        if confirmed.to_dict() != best.to_dict():
            raise ValueError("The five-fold confirmation is of other settings than the loop's best so far; re-run it")
        fold_rounds = confirmation[WINNER]["rounds"]
        rounds = statistics.median_low(fold_rounds)
        rule = f"median of the five-fold confirmation's rounds {fold_rounds}"
    else:
        section = results[NOISE_FLOOR] if results["best"]["experiment"] == NOISE_FLOOR else (
            results["experiments"][results["best"]["experiment"]]
        )
        rounds = next(row["rounds"] for row in section["arms"] if row["arm"] == results["best"]["arm"])
        rule = "the best arm's own rounds (no five-fold confirmation)"
    return ServedConfiguration(
        form=form, params=dict(best.params), rounds=int(rounds), rounds_rule=rule, non_uk_rate=best.non_uk_rate,
        validation_year=results["validation_year"],
    )


def ranking_record(results: Mapping) -> dict:
    """What an artefact records of the loop: the selection measure, the noise floor, every experiment's
    rows, the best so far and the five-fold confirmation (where run)."""
    return {key: results[key] for key in RECORDED if key in results}


CONFIRMED_NAIVE_WEIGHT = 0.3  # the real loop's blend weight, confirmed on five folds (#76)


def fixture_ranking(world) -> dict:
    """The fixture world's loop results: the noise floor and the blend experiment (CPU, under a second).

    The blend sweeps only `CONFIRMED_NAIVE_WEIGHT`, the weight the real loop chose and confirmed (#76),
    so the fixture artefact serves the same form as the real one; the fixture's 20 UK pairs cannot
    choose a weight of their own.
    """
    lp = Loop(fixture_data(world), FIXTURE_LOOP)
    lp.noise_floor()
    return lp.run_experiment(NAIVE_EXPERIMENT, [blend_arm((CONFIRMED_NAIVE_WEIGHT,))])


def v2_plan(root: Path | None = None) -> LoopPlan:
    """The real loop's start: v2's configuration (the saved search winner) at the search's rates, on fold 0."""
    saved = search.load(root)
    return LoopPlan(
        start=Settings(
            params=dict(saved["winner"]["params"]),
            non_uk_rate=saved["non_uk_rate"],
            zero_rate=saved["plan"]["zero_rate"] or ZERO_SAMPLING_RATE,
        )
    )
