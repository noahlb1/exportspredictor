"""The loop's decisions re-checked under the tie-proof Selection Measure (spec #113, ticket #116).

The noise floor and every experiment of spec #65 were measured with the old measure. Ticket #116
re-scored them from the cached models (nothing trained, 2023 never read) and recorded the result
in `results.json` under `RECHECK`: the noise floor's seeds re-scored, the Win Threshold they give,
and the best so far re-scored, which is the base later experiments (Phase 2) start from. The
original sections stay as they were measured, except `best`, which becomes the re-scored one (its
old value is kept as `legacy_best`), so an experiment compares like with like.

Every later decision reads the threshold through `current_win_threshold`.
"""

import statistics
from collections.abc import Mapping, Sequence

from exportspredictor.loop.arms import NOISE_FLOOR
from exportspredictor.loop.data import save_results
from exportspredictor.loop.selection import (
    MIN_WIN_THRESHOLD,
    SELECTION_MEASURE_TEXT,
    win_threshold,
)

RECHECK = "recheck"


def current_win_threshold(results: Mapping) -> float:
    """The Win Threshold a new decision uses: the re-checked one if the loop was re-checked, else the noise floor's."""
    if RECHECK in results:
        return results[RECHECK][NOISE_FLOOR]["win_threshold"]
    return results[NOISE_FLOOR]["win_threshold"]


def recheck_results(
    results: Mapping,
    noise_floor_measures: Sequence[float],
    *,
    best: Mapping,
    minimum: float = MIN_WIN_THRESHOLD,
    record: Mapping | None = None,
) -> dict:
    """`results` with the re-check recorded: the noise floor's seeds re-scored (in seed order), the
    Win Threshold they give, and `best` (the best so far re-scored, perhaps with a corrected
    setting) as the new best so far. `record` adds anything else worth keeping (old against new)."""
    values = list(noise_floor_measures)
    section = {
        "selection_measure": SELECTION_MEASURE_TEXT,
        NOISE_FLOOR: {
            "seeds": list(results[NOISE_FLOOR].get("seeds", [])),
            "selection_measures": values,
            "mean": statistics.fmean(values),
            "sd": statistics.stdev(values),
            "win_threshold": win_threshold(values, minimum),
        },
        "legacy_best": dict(results[RECHECK]["legacy_best"]) if RECHECK in results else dict(results["best"]),
        "best": dict(best),
        **dict(record or {}),
    }
    return {**results, RECHECK: section, "best": dict(best)}


def save_recheck(folder, results: Mapping) -> None:
    """Save results carrying a re-check (`recheck_results`) where the loop reads them."""
    if RECHECK not in results:
        raise ValueError("These results carry no re-check")
    save_results(folder, dict(results))
