"""Scenarios: re-predict the Baseline with trade-policy conditions changed (ADR 0002).

A Scenario is either a Trading Bloc preset or a list of Markets with which a Trade
Agreement is added or removed. Applying one copies the Baseline year's UK feature rows
for the affected Markets, changes only the policy columns and re-predicts each pair with
its own fold model (ADR 0001), so a Scenario Effect never comes from a model that trained
on that pair. `check_only_policy_changed` guards that nothing else moves.

**Presets** (wayfinder #8). Bloc membership comes from the Baseline year's `<bloc>_d`
flags, and a bloc Scenario changes only the rows of the bloc's member Markets:

- EU re-join: a Trade Agreement with every EU member, the Customs Union flag and the
  UK's EU membership flag (`eu_o`), as for any pair of EU members in training.
- EU Customs Union only: the Customs Union flag alone.
- CPTPP: a Trade Agreement with every member and the UK's CPTPP flag, with accession
  treated as complete. The 2024 Baseline has no UK CPTPP membership (the 1 July rule).
- Leave: for each bloc the UK is in, its membership flag, Trade Agreement and Customs
  Union with that bloc's members are switched off.

**Blocked Opportunities**: pairs whose Scenario Effect is at least `min_share` of their
Baseline Export Potential and at least `min_usd`. `blocked_effects` precomputes, for every
pair without a Trade Agreement, the Scenario Effect of adding one: the value that the
`rank_by=blocked` rankings sort by.

Scenario Effects are associations the model has learned, not causal estimates.
"""

import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

from exportspredictor.currency import format_change
from exportspredictor.features import FEATURES, POLICY_FEATURES

DEFAULT_MIN_SHARE = 0.2
DEFAULT_MIN_USD = 1_000_000.0


class InvalidScenario(ValueError):
    """The Scenario is malformed, e.g. it names an unknown Market or preset."""


class LeakageError(AssertionError):
    """A Scenario changed something other than the policy columns."""


@dataclass(frozen=True)
class Preset:
    name: str
    title: str
    description: str


PRESETS = {
    preset.name: preset
    for preset in [
        Preset(
            "eu_rejoin",
            "EU re-join",
            "A Trade Agreement with every EU member, plus the Customs Union and EU membership.",
        ),
        Preset(
            "eu_customs_union",
            "EU Customs Union only",
            "The Customs Union with every EU member, with no other change.",
        ),
        Preset(
            "cptpp",
            "CPTPP",
            "A Trade Agreement with every CPTPP member, with accession treated as complete.",
        ),
        Preset(
            "leave",
            "Leave",
            "Leave every Trading Bloc the UK belongs to in the Baseline year, with its Trade Agreements.",
        ),
    ]
}


@dataclass(frozen=True)
class Scenario:
    """A preset name, or Markets (ISO3) to add or remove a Trade Agreement with."""

    preset: str | None = None
    add_agreements: tuple[str, ...] = ()
    remove_agreements: tuple[str, ...] = ()

    def validate(self, markets: Iterable[str]) -> None:
        """Raise InvalidScenario unless this is exactly one well-formed kind of Scenario."""
        custom = bool(self.add_agreements or self.remove_agreements)
        if (self.preset is None) == (not custom):
            raise InvalidScenario("Give either a preset or Markets to add or remove a Trade Agreement with, not both.")
        if self.preset is not None and self.preset not in PRESETS:
            raise InvalidScenario(f"Unknown preset {self.preset!r}; choose one of {sorted(PRESETS)}.")
        both = set(self.add_agreements) & set(self.remove_agreements)
        if both:
            raise InvalidScenario(f"A Trade Agreement cannot be both added and removed: {sorted(both)}.")
        unknown = (set(self.add_agreements) | set(self.remove_agreements)) - set(markets)
        if unknown:
            raise InvalidScenario(f"Unknown Markets: {sorted(unknown)}.")


def apply(baseline_features: pl.DataFrame, scenario: Scenario) -> pl.DataFrame:
    """The Baseline feature rows the Scenario changes, with their policy columns changed.

    Rows whose policy columns end up as in the Baseline are left out, so a Scenario that
    changes nothing returns no rows. Columns and their types are those of the input.
    """
    scenario.validate(baseline_features["iso3"].unique())
    changed = baseline_features
    for column, condition, value in _changes(baseline_features, scenario):
        changed = changed.with_columns(
            pl.when(condition).then(pl.lit(value)).otherwise(pl.col(column)).cast(changed.schema[column]).alias(column)
        )
    differs = pl.any_horizontal(*(changed[c] != baseline_features[c] for c in POLICY_FEATURES))
    after = changed.filter(differs)
    check_only_policy_changed(baseline_features, after)
    return after


@dataclass(frozen=True)
class ScenarioResult:
    """Scenario Effects for every pair the Scenario changes (none if it changes nothing).

    `pairs`: hs6, iso3, baseline_potential_usd, scenario_potential_usd, effect_usd,
    effect_share (of Baseline Export Potential), sorted by effect_usd, largest first.
    """

    scenario: Scenario
    pairs: pl.DataFrame

    @property
    def changes_nothing(self) -> bool:
        return self.pairs.is_empty()

    def by_market(self) -> pl.DataFrame:
        """Baseline, Scenario and effect totals per Market, largest effect first."""
        return _totals(self.pairs, "iso3")

    def by_product(self) -> pl.DataFrame:
        """Baseline, Scenario and effect totals per Product, largest effect first."""
        return _totals(self.pairs, "hs6")

    def total(self) -> dict[str, float]:
        """Baseline and Scenario Export Potential of the changed pairs, and the effect."""
        sums = self.pairs.select(
            pl.col("baseline_potential_usd", "scenario_potential_usd", "effect_usd").sum()
        ).row(0, named=True)
        baseline = sums["baseline_potential_usd"]
        return {**sums, "effect_share": sums["effect_usd"] / baseline if baseline else 0.0}

    def blocked_opportunities(
        self, min_share: float = DEFAULT_MIN_SHARE, min_usd: float = DEFAULT_MIN_USD
    ) -> pl.DataFrame:
        """Pairs whose Scenario Effect clears both Blocked Opportunity thresholds."""
        return self.pairs.filter(is_blocked("effect_usd", "baseline_potential_usd", min_share, min_usd))

    def summary(self, *, blocked: int, year: int, usd_per_gbp: float) -> str:
        """One sentence, worded as an association, never as a cause or a forecast.

        `blocked` is the number of Blocked Opportunities, `year` the Baseline year, and the
        effect is given in GBP (at `usd_per_gbp`) and in USD.
        """
        if self.changes_nothing and self.scenario.preset == "leave":
            return (
                f"This Scenario changes nothing: in the {year} Baseline the UK belongs to no tracked "
                "Trading Bloc, so there is none to leave and no Scenario Effect."
            )
        if self.changes_nothing:
            return (
                "This Scenario changes nothing: the trade-policy conditions it sets are already those of the "
                f"{year} Baseline (for example, the Trade Agreement already exists), so there is no Scenario Effect."
            )
        total = self.total()
        markets = self.pairs["iso3"].n_unique()
        market_word = "Market" if markets == 1 else "Markets"
        blocked_word = "Blocked Opportunity" if blocked == 1 else "Blocked Opportunities"
        effect_gbp = format_change(total["effect_usd"] / usd_per_gbp, "£")
        return (
            f"The model associates this Scenario with a change of {effect_gbp} "
            f"({format_change(total['effect_usd'])}, {total['effect_share']:+.1%}) in UK Export Potential "
            f"across {markets} {market_word}, against the {year} Baseline, with {blocked} {blocked_word}. "
            "This is a learned association, not a forecast."
        )


def is_blocked(effect: str, baseline: str, min_share: float = DEFAULT_MIN_SHARE, min_usd: float = DEFAULT_MIN_USD) -> pl.Expr:
    """The Blocked Opportunity rule: effect ≥ `min_share` × Baseline and ≥ `min_usd`."""
    return (pl.col(effect) >= min_share * pl.col(baseline)) & (pl.col(effect) >= min_usd)


def run(
    baseline_features: pl.DataFrame,
    baseline: pl.DataFrame,
    models: Sequence[xgb.Booster],
    scenario: Scenario,
) -> ScenarioResult:
    """Apply a Scenario and re-predict the changed pairs with their own fold models.

    `baseline` holds each pair's Baseline `export_potential_usd`, which the same fold
    models produced from the same feature rows, so only the changed rows are predicted.
    """
    changed = apply(baseline_features, scenario)
    pairs = (
        changed.select("hs6", "iso3")
        .with_columns(scenario_potential_usd=predict(changed, models))
        .join(baseline.select("hs6", "iso3", baseline_potential_usd="export_potential_usd"), on=["hs6", "iso3"])
        .with_columns(effect_usd=pl.col("scenario_potential_usd") - pl.col("baseline_potential_usd"))
        .with_columns(effect_share=_share("effect_usd", "baseline_potential_usd"))
        .select(
            "hs6", "iso3", "baseline_potential_usd", "scenario_potential_usd", "effect_usd", "effect_share"
        )
        .sort("effect_usd", "hs6", "iso3", descending=[True, False, False])
    )
    return ScenarioResult(scenario=scenario, pairs=pairs)


def blocked_effects(
    baseline_features: pl.DataFrame, baseline: pl.DataFrame, models: Sequence[xgb.Booster]
) -> pl.DataFrame:
    """Every pair's Scenario Effect of adding a Trade Agreement where none exists (0 where one does).

    Returns hs6, iso3, blocked_effect_usd: the value `rank_by=blocked` rankings sort by.
    """
    without = baseline_features.filter(pl.col("trade_agreement") == 0)
    added = without.with_columns(trade_agreement=pl.lit(1, without.schema["trade_agreement"]))
    check_only_policy_changed(baseline_features, added)
    effects = added.select("hs6", "iso3").with_columns(scenario=predict(added, models))
    return (
        baseline.select("hs6", "iso3", "export_potential_usd")
        .join(effects, on=["hs6", "iso3"], how="left")
        .select(
            "hs6",
            "iso3",
            blocked_effect_usd=pl.when(pl.col("scenario").is_null())
            .then(0.0)
            .otherwise(pl.col("scenario") - pl.col("export_potential_usd")),
        )
    )


def with_blocked_effects(baseline: pl.DataFrame, effects: pl.DataFrame) -> pl.DataFrame:
    """The Baseline table with its `blocked_effect_usd` column (replaced if already there)."""
    return baseline.drop("blocked_effect_usd", strict=False).join(
        effects.select("hs6", "iso3", "blocked_effect_usd"), on=["hs6", "iso3"], how="left", maintain_order="left"
    )


def _share(numerator: str, denominator: str) -> pl.Expr:
    return pl.when(pl.col(denominator) > 0).then(pl.col(numerator) / pl.col(denominator)).otherwise(0.0)


def _totals(pairs: pl.DataFrame, by: str) -> pl.DataFrame:
    return (
        pairs.group_by(by)
        .agg(pl.col("baseline_potential_usd", "scenario_potential_usd", "effect_usd").sum(), pairs=pl.len())
        .with_columns(effect_share=_share("effect_usd", "baseline_potential_usd"))
        .sort("effect_usd", by, descending=[True, False])
    )


def predict(rows: pl.DataFrame, models: Sequence[xgb.Booster]) -> pl.Series:
    """Export Potential (USD) for feature rows, each from its pair's fold model (`fold`)."""
    fold = rows["fold"].to_numpy()
    data = rows.select(FEATURES).to_numpy().astype(np.float32)
    potential = np.zeros(rows.height)
    for k, model in enumerate(models):
        mask = fold == k
        if mask.any():
            potential[mask] = model.inplace_predict(data[mask])
    return pl.Series("export_potential_usd", potential)


def check_only_policy_changed(before: pl.DataFrame, after: pl.DataFrame) -> None:
    """Raise LeakageError unless `after`'s rows match `before`'s pairs outside the policy columns."""
    keys = ["hs6", "iso3"]
    fixed = ["fold", *(f for f in FEATURES if f not in POLICY_FEATURES)]
    if after.columns != before.columns:
        raise LeakageError("A Scenario must keep the Baseline feature columns.")
    joined = after.select(*keys, *fixed).join(
        before.select(*keys, *fixed), on=keys, how="left", suffix="_baseline", validate="1:1"
    )
    for column in fixed:
        same = joined[column].eq_missing(joined[f"{column}_baseline"])
        if not same.all():
            raise LeakageError(f"A Scenario changed the non-policy column {column!r}.")


def _changes(rows: pl.DataFrame, scenario: Scenario) -> list[tuple[str, pl.Expr, int]]:
    """(column, rows to change, new value) for each policy change, applied in order."""
    if scenario.preset is None:
        add = pl.col("iso3").is_in(list(scenario.add_agreements))
        remove = pl.col("iso3").is_in(list(scenario.remove_agreements))
        return [
            ("trade_agreement", add, 1),
            ("trade_agreement", remove, 0),
            # A Customs Union implies a Trade Agreement, so removing one removes both.
            ("customs_union", remove, 0),
        ]
    if scenario.preset == "eu_rejoin":
        eu = pl.col("eu_d") == 1
        return [("trade_agreement", eu, 1), ("customs_union", eu, 1), ("eu_o", eu, 1)]
    if scenario.preset == "eu_customs_union":
        return [("customs_union", pl.col("eu_d") == 1, 1)]
    if scenario.preset == "cptpp":
        cptpp = pl.col("cptpp_d") == 1
        return [("trade_agreement", cptpp, 1), ("cptpp_o", cptpp, 1)]
    # Leave: every bloc the UK belongs to in the Baseline year.
    changes = []
    for bloc in _uk_blocs(rows):
        members = pl.col(f"{bloc}_d") == 1
        changes += [(f"{bloc}_o", members, 0), ("trade_agreement", members, 0), ("customs_union", members, 0)]
    return changes


def _uk_blocs(rows: pl.DataFrame) -> list[str]:
    return [c.removesuffix("_o") for c in POLICY_FEATURES if c.endswith("_o") and rows[c].max() == 1]


# --- Rebuilding the Blocked Opportunity values of a saved artefact ---------------------


def rebuild_blocked_values(root: Path, *, version: str | None = None) -> Path:
    """Recompute `blocked_effect_usd` from the newest artefact's saved fold models.

    Writes a new artefact version (models and tables copied) and returns its folder, so
    this step re-runs on its own without retraining. Nothing is trained here.
    """
    from exportspredictor import artefacts as store

    source = store.load_latest(root)
    started = time.perf_counter()
    effects = blocked_effects(source.baseline_features, source.baseline, source.fold_models)
    metadata = {
        **source.metadata,
        "version": version or store.new_version(),
        "blocked_values_from": source.version,
        "blocked_values_seconds": round(time.perf_counter() - started, 2),
    }
    rebuilt = replace(source, metadata=metadata, baseline=with_blocked_effects(source.baseline, effects))
    return store.save(rebuilt, root)


def main() -> None:
    from exportspredictor.paths import artefacts_dir

    print(rebuild_blocked_values(artefacts_dir()), flush=True)


if __name__ == "__main__":
    main()
