"""Scenarios: re-predict the Baseline with trade-policy conditions changed (ADR 0002).

A Scenario is a Trading Bloc preset, a list of Markets with which a Trade Agreement is
added or removed, or a tariff Scenario that sets the Applied Tariff chosen Markets charge
on UK goods. Applying one copies the Baseline year's UK feature rows
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

**Tariffs** (ADR 0006). Adding a Trade Agreement (directly, in a preset or in a Blocked
Opportunity value) sets a known Applied Tariff to 0 (an unknown one stays unknown), and removing one restores the Importer's
2021 MFN rate (`mfn_tariff`), so the model is never asked about an agreement with full
tariffs, a combination it rarely saw in training. A tariff Scenario sets `ln(1 + t)` for
the chosen pairs. Every tariff a Scenario sets is clipped to the training range, so the
Scenario Effect of a very large hike flattens; `ScenarioResult.clipped_pairs` counts them.

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
from exportspredictor.features import (
    FEATURES,
    POLICY_FEATURES,
    TARIFF_METADATA,
    FeatureTransform,
)

DEFAULT_MIN_SHARE = 0.2
DEFAULT_MIN_USD = 1_000_000.0
TARIFF = "log_applied_tariff"  # ln(1 + t): the policy feature a Scenario's tariffs are set in


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
class TariffChange:
    """The Applied Tariff (percent, e.g. 10.0 = 10%) chosen Markets charge on UK goods.

    With `sectors` (HS2 chapters, e.g. "22"), only those Sectors' Products; otherwise all.
    """

    markets: tuple[str, ...]
    rate_pct: float
    sectors: tuple[str, ...] | None = None


@dataclass(frozen=True)
class Scenario:
    """A preset name, Markets (ISO3) to add or remove a Trade Agreement with, or a tariff change."""

    preset: str | None = None
    add_agreements: tuple[str, ...] = ()
    remove_agreements: tuple[str, ...] = ()
    tariff: TariffChange | None = None

    def validate(self, markets: Iterable[str], sectors: Iterable[str] = ()) -> None:
        """Raise InvalidScenario unless this is exactly one well-formed kind of Scenario.

        `markets` and `sectors` are those the Baseline holds UK pairs in.
        """
        kinds = [self.preset is not None, bool(self.add_agreements or self.remove_agreements), self.tariff is not None]
        if sum(kinds) != 1:
            raise InvalidScenario(
                "Give exactly one of a preset, Markets to add or remove a Trade Agreement with, or a tariff change."
            )
        if self.preset is not None and self.preset not in PRESETS:
            raise InvalidScenario(f"Unknown preset {self.preset!r}; choose one of {sorted(PRESETS)}.")
        both = set(self.add_agreements) & set(self.remove_agreements)
        if both:
            raise InvalidScenario(f"A Trade Agreement cannot be both added and removed: {sorted(both)}.")
        named = set(self.add_agreements) | set(self.remove_agreements)
        if self.tariff is not None:
            named |= set(self.tariff.markets)
            if not self.tariff.markets:
                raise InvalidScenario("A tariff change needs at least one Market.")
            if not np.isfinite(self.tariff.rate_pct) or self.tariff.rate_pct < 0:
                raise InvalidScenario(f"A tariff rate must be 0% or more, not {self.tariff.rate_pct}%.")
            unknown_sectors = set(self.tariff.sectors or ()) - set(sectors)
            if unknown_sectors:
                raise InvalidScenario(f"Unknown Sectors: {sorted(unknown_sectors)}.")
        unknown = named - set(markets)
        if unknown:
            raise InvalidScenario(f"Unknown Markets: {sorted(unknown)}.")


def tariff_range(metadata: dict) -> tuple[float, float]:
    """The training range of `ln(1 + t)` in artefact metadata: the bounds Scenario tariffs are clipped to."""
    low, high = FeatureTransform.from_dict(metadata["feature_transform"]).bounds[TARIFF]
    return float(low), float(high)


def apply(
    baseline_features: pl.DataFrame, scenario: Scenario, tariff_range: tuple[float, float] | None = None
) -> pl.DataFrame:
    """The Baseline feature rows the Scenario changes, with their policy columns changed.

    Rows whose policy columns end up as in the Baseline are left out, so a Scenario that
    changes nothing returns no rows. Columns and their types are those of the input. Every
    tariff the Scenario sets is clipped to `tariff_range` (of `ln(1 + t)`), when given.
    """
    return _apply(baseline_features, scenario, tariff_range)[0]


def _apply(
    baseline_features: pl.DataFrame, scenario: Scenario, tariff_range: tuple[float, float] | None
) -> tuple[pl.DataFrame, int]:
    """`apply`, and the number of pairs whose Scenario tariff was clipped to the training range."""
    scenario.validate(baseline_features["iso3"].unique(), _sectors(baseline_features))
    changed = baseline_features
    clipped = pl.repeat(False, baseline_features.height, eager=True)
    for column, condition, value in _changes(baseline_features, scenario):
        if column == TARIFF and tariff_range is not None:
            low, high = tariff_range
            wanted = changed.select(pl.when(condition).then(value)).to_series()
            clipped |= ((wanted < low) | (wanted > high)).fill_null(False)
            value = value.clip(low, high)
        changed = changed.with_columns(
            pl.when(condition).then(value).otherwise(pl.col(column)).cast(changed.schema[column]).alias(column)
        )
    differs = pl.any_horizontal(*(changed[c].ne_missing(baseline_features[c]) for c in POLICY_FEATURES))
    after = changed.filter(differs)
    check_only_policy_changed(baseline_features, after)
    return after, int(clipped.sum())


@dataclass(frozen=True)
class ScenarioResult:
    """Scenario Effects for every pair the Scenario changes (none if it changes nothing).

    `pairs`: hs6, iso3, baseline_potential_usd, scenario_potential_usd, effect_usd,
    effect_share (of Baseline Export Potential), baseline_tariff_pct, scenario_tariff_pct
    (Applied Tariff, percent; null where unknown), sorted by effect_usd, largest first.
    `clipped_pairs`: pairs whose Scenario tariff lay beyond the training range and was
    clipped to it, so their Scenario Effect is that of the range's edge.
    """

    scenario: Scenario
    pairs: pl.DataFrame
    clipped_pairs: int = 0

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

    def summary(self, *, blocked: int, year: int) -> str:
        """One sentence, worded as an association, never as a cause or a forecast.

        `blocked` is the number of Blocked Opportunities and `year` the Baseline year; the
        effect is in nominal USD.
        """
        if self.changes_nothing and self.scenario.preset == "leave":
            return (
                f"This Scenario changes nothing: in the {year} Baseline the UK belongs to no tracked "
                "Trading Bloc, so there is none to leave and no Scenario Effect."
            )
        if self.changes_nothing:
            return (
                "This Scenario changes nothing: the trade-policy conditions it sets are already those of the "
                f"{year} Baseline (for example, the Trade Agreement already exists or the tariff is already that "
                "rate), so there is no Scenario Effect."
            )
        total = self.total()
        markets = self.pairs["iso3"].n_unique()
        market_word = "Market" if markets == 1 else "Markets"
        blocked_word = "Blocked Opportunity" if blocked == 1 else "Blocked Opportunities"
        return (
            f"The model associates this Scenario with a change of {format_change(total['effect_usd'])} "
            f"({total['effect_share']:+.1%}) in UK Export Potential "
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
    tariff_range: tuple[float, float] | None = None,
) -> ScenarioResult:
    """Apply a Scenario and re-predict the changed pairs with their own fold models.

    `baseline` holds each pair's `baseline_export_potential_usd` (the Baseline year's, never
    the Gap Period mean), which the same fold models produced from the same feature rows,
    so only the changed rows are predicted. Scenario tariffs are clipped to `tariff_range`
    (the training range, `tariff_range(metadata)`).
    """
    changed, clipped_pairs = _apply(baseline_features, scenario, tariff_range)
    pairs = (
        changed.select("hs6", "iso3", scenario_tariff_pct=_percent(TARIFF))
        .with_columns(scenario_potential_usd=predict(changed, models))
        .join(
            baseline.select("hs6", "iso3", baseline_potential_usd="baseline_export_potential_usd"), on=["hs6", "iso3"]
        )
        .join(baseline_features.select("hs6", "iso3", baseline_tariff_pct=_percent(TARIFF)), on=["hs6", "iso3"])
        .with_columns(effect_usd=pl.col("scenario_potential_usd") - pl.col("baseline_potential_usd"))
        .with_columns(effect_share=_share("effect_usd", "baseline_potential_usd"))
        .select(
            "hs6",
            "iso3",
            "baseline_potential_usd",
            "scenario_potential_usd",
            "effect_usd",
            "effect_share",
            "baseline_tariff_pct",
            "scenario_tariff_pct",
        )
        .sort("effect_usd", "hs6", "iso3", descending=[True, False, False])
    )
    return ScenarioResult(scenario=scenario, pairs=pairs, clipped_pairs=clipped_pairs)


def blocked_effects(
    baseline_features: pl.DataFrame, baseline: pl.DataFrame, models: Sequence[xgb.Booster]
) -> pl.DataFrame:
    """Every pair's Scenario Effect of adding a Trade Agreement where none exists (0 where one does).

    It is the Scenario that adds a Trade Agreement with every Market, so the agreement removes
    a known tariff exactly as in every Scenario and leaves an unknown one unknown (ADR 0006).
    Measured against the Baseline year's Export Potential (`baseline_export_potential_usd`).

    Returns hs6, iso3, blocked_effect_usd: the Blocked Opportunity value.
    """
    every_market = Scenario(add_agreements=tuple(baseline_features["iso3"].unique().sort()))
    added = apply(baseline_features, every_market)
    effects = added.select("hs6", "iso3").with_columns(scenario=predict(added, models))
    return (
        baseline.select("hs6", "iso3", "baseline_export_potential_usd")
        .join(effects, on=["hs6", "iso3"], how="left")
        .select(
            "hs6",
            "iso3",
            blocked_effect_usd=pl.when(pl.col("scenario").is_null())
            .then(0.0)
            .otherwise(pl.col("scenario") - pl.col("baseline_export_potential_usd")),
        )
    )


def with_blocked_effects(baseline: pl.DataFrame, effects: pl.DataFrame) -> pl.DataFrame:
    """The Baseline table with its `blocked_effect_usd` column (replaced if already there)."""
    return baseline.drop("blocked_effect_usd", strict=False).join(
        effects.select("hs6", "iso3", "blocked_effect_usd"), on=["hs6", "iso3"], how="left", maintain_order="left"
    )


def _percent(log_tariff: str) -> pl.Expr:
    """The Applied Tariff in percent from its `ln(1 + t)` feature (null where unknown)."""
    return (pl.col(log_tariff).cast(pl.Float64).exp() - 1) * 100


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
    non_policy = [f for f in FEATURES if f not in POLICY_FEATURES]
    fixed = ["fold", *non_policy, *(c for c in TARIFF_METADATA if c in before.columns)]
    if after.columns != before.columns:
        raise LeakageError("A Scenario must keep the Baseline feature columns.")
    joined = after.select(*keys, *fixed).join(
        before.select(*keys, *fixed), on=keys, how="left", suffix="_baseline", validate="1:1"
    )
    for column in fixed:
        same = joined[column].eq_missing(joined[f"{column}_baseline"])
        if not same.all():
            raise LeakageError(f"A Scenario changed the non-policy column {column!r}.")


_ON, _OFF = pl.lit(1), pl.lit(0)
# The Applied Tariff under an added Trade Agreement: 0 where the tariff is known (observed or
# carried forward), and still unknown (null) where it is unknown. A missing tariff is not a
# rate, so removing it is no defined policy change; and XGBoost's monotone constraints do not
# cover the missing-value branch, so setting it to 0 could lower a prediction (ADR 0006).
AGREEMENT_TARIFF = pl.when(pl.col(TARIFF).is_not_null()).then(pl.lit(0.0))
# The Importer's 2021 MFN rate, never below the current tariff (MFN is the ceiling of the
# rates an agreement grants). Where the MFN rate is unknown, the current tariff stays.
_MFN_TARIFF = pl.max_horizontal(pl.col(TARIFF), (pl.col("mfn_tariff") / 100).log1p())


def _changes(rows: pl.DataFrame, scenario: Scenario) -> list[tuple[str, pl.Expr, pl.Expr]]:
    """(column, rows to change, new value) for each policy change, applied in order.

    Tariff changes come before the flag changes, so their conditions read the Baseline
    flags: the tariff follows an agreement only where one is really added or removed.
    """
    if scenario.tariff is not None:
        change = scenario.tariff
        chosen = pl.col("iso3").is_in(list(change.markets))
        if change.sectors is not None:
            chosen &= pl.col("hs6").str.slice(0, 2).is_in(list(change.sectors))
        return [(TARIFF, chosen, pl.lit(float(np.log1p(change.rate_pct / 100))))]
    if scenario.preset is None:
        add = pl.col("iso3").is_in(list(scenario.add_agreements))
        remove = pl.col("iso3").is_in(list(scenario.remove_agreements))
        return [
            (TARIFF, add & (pl.col("trade_agreement") == 0), AGREEMENT_TARIFF),
            (TARIFF, remove & _has_agreement(), _MFN_TARIFF),
            ("trade_agreement", add, _ON),
            ("trade_agreement", remove, _OFF),
            # A Customs Union implies a Trade Agreement, so removing one removes both.
            ("customs_union", remove, _OFF),
        ]
    if scenario.preset == "eu_rejoin":
        eu = pl.col("eu_d") == 1
        return [(TARIFF, eu, AGREEMENT_TARIFF), ("trade_agreement", eu, _ON), ("customs_union", eu, _ON), ("eu_o", eu, _ON)]
    if scenario.preset == "eu_customs_union":
        # Customs Union pairs trade tariff-free in training (ADR 0006).
        eu = pl.col("eu_d") == 1
        return [(TARIFF, eu, AGREEMENT_TARIFF), ("customs_union", eu, _ON)]
    if scenario.preset == "cptpp":
        cptpp = pl.col("cptpp_d") == 1
        new = cptpp & (pl.col("trade_agreement") == 0)
        return [(TARIFF, new, AGREEMENT_TARIFF), ("trade_agreement", cptpp, _ON), ("cptpp_o", cptpp, _ON)]
    # Leave: every bloc the UK belongs to in the Baseline year, every bloc's tariffs first.
    blocs = {bloc: pl.col(f"{bloc}_d") == 1 for bloc in _uk_blocs(rows)}
    tariffs = [(TARIFF, members & _has_agreement(), _MFN_TARIFF) for members in blocs.values()]
    flags = [
        (column, members, _OFF)
        for bloc, members in blocs.items()
        for column in (f"{bloc}_o", "trade_agreement", "customs_union")
    ]
    return [*tariffs, *flags]


def _has_agreement() -> pl.Expr:
    return (pl.col("trade_agreement") == 1) | (pl.col("customs_union") == 1)


def _uk_blocs(rows: pl.DataFrame) -> list[str]:
    return [c.removesuffix("_o") for c in POLICY_FEATURES if c.endswith("_o") and rows[c].max() == 1]


def _sectors(rows: pl.DataFrame) -> list[str]:
    """The Sectors (HS2 chapters) the Baseline holds UK pairs in."""
    return rows["hs6"].str.slice(0, 2).unique().to_list()


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
