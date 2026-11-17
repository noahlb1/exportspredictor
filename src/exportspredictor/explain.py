""""Why?" explanations: which factors raised or lowered one UK pair's Export Potential.

The explanation uses SHAP values (TreeSHAP) computed by XGBoost itself
(`predict(..., pred_contribs=True)`), from **the pair's own fold model**: the one that held
the pair out and scored its Baseline (ADR 0001). Explaining with any other model would
explain a number the Advisor never sees.

Scale: the model has a Tweedie objective with a log link, so SHAP values are additive on
the log scale. `base_value` plus every factor's `contribution` equals
`log_export_potential`, and `exp` of that is the Baseline Export Potential (USD). On the
dollar scale a factor multiplies Export Potential by `exp(contribution)` (`multiplier`).
Contributions are associations the model has learned, not causes.

Every model input gets a plain-English label naming the Market (`factor_labels`), and a
readable value (`value_text`), so the app never shows a raw feature name.
"""

import math
from dataclasses import dataclass

import numpy as np
import polars as pl
import xgboost as xgb

from exportspredictor.artefacts import Artefacts
from exportspredictor.features import FEATURES, GROWTH_YEARS, REEXPORT_RATIO_CAP
from exportspredictor.policy import TRADING_BLOCS

_BLOC_NAMES = {"eu": "the EU", "cptpp": "CPTPP", "efta": "EFTA"}
_UK = "the UK"


class UnknownPair(LookupError):
    """The Product–Market pair is not in the Baseline."""


@dataclass(frozen=True)
class Factor:
    feature: str  # model input name, for developers; never shown to Advisors
    label: str  # plain-English name of the factor
    value_text: str  # the factor's value for this pair, readable
    contribution: float  # SHAP value on the log scale: > 0 raised Export Potential
    multiplier: float  # exp(contribution): the factor's multiplicative association


@dataclass(frozen=True)
class HistoryPoint:
    year: int
    actual_exports_usd: float


@dataclass(frozen=True)
class Explanation:
    hs6: str
    iso3: str
    market: str
    fold: int  # the fold model used: the one that held the pair out and scored its Baseline
    base_value: float  # the model's average log Export Potential (SHAP bias term)
    log_export_potential: float  # base_value + sum of contributions
    export_potential_usd: float  # exp(log_export_potential): the Baseline Export Potential
    factors: list[Factor]  # every model input, largest absolute contribution first
    history: list[HistoryPoint]  # UK Actual Exports of the pair by year (non-zero years)


def factor_labels(market: str) -> dict[str, str]:
    """A plain-English label for every model input, naming the Market."""
    labels = {
        "log_gdp_o": f"Size of {_UK} economy (GDP)",
        "log_gdp_d": f"Size of {market}'s economy (GDP)",
        "log_population_o": f"Population of {_UK}",
        "log_population_d": f"Population of {market}",
        "log_distance": f"Distance from {_UK} to {market}",
        "contiguous": f"{market} shares a land border with {_UK}",
        "common_language": f"{market} shares a language with {_UK}",
        "colonial_tie": f"Historical colonial tie between {_UK} and {market}",
        "common_colonizer": f"{_UK} and {market} share a former colonial power",
        "log_supply_loo": f"How much {_UK} exports this Product to other Markets",
        "rca_loo": f"How specialised {_UK} is in this Product, against the world average",
        "log_demand_loo": f"How much {market} imports this Product from other countries",
        "high_income_share_loo": f"Share of {market}'s imports of this Product that come from high-income countries",
        "sector": "The Product's Sector (HS2 chapter)",
        "relatedness": f"How close this Product is to what {_UK} already exports well",
        "market_import_growth": f"How fast {market}'s imports of this Product are growing",
        "reexport_ratio": f"How much of this Product {market} re-exports",
        "trade_agreement": f"Trade Agreement between {_UK} and {market}",
        "customs_union": f"Customs Union between {_UK} and {market}",
    }
    for bloc in TRADING_BLOCS:
        name = _BLOC_NAMES.get(bloc, bloc.upper())
        labels[f"{bloc}_o"] = f"{_UK} is a member of {name}"
        labels[f"{bloc}_d"] = f"{market} is a member of {name}"
    return labels


def explain_pair(artefacts: Artefacts, hs6: str, iso3: str) -> Explanation:
    """SHAP factors and Actual Exports history for one UK Product–Market pair of the Baseline."""
    row = artefacts.baseline_features.filter((pl.col("hs6") == hs6) & (pl.col("iso3") == iso3))
    if row.is_empty():
        raise UnknownPair(f"No Baseline for Product {hs6} in Market {iso3}")
    fold = int(row["fold"].item())
    model = artefacts.fold_models[fold]
    if model.attr("held_out_fold") not in (None, str(fold)):
        raise ValueError(f"Fold model {fold} held out fold {model.attr('held_out_fold')}")

    values = row.select(pl.col(FEATURES).cast(pl.Float32)).to_numpy()
    matrix = xgb.DMatrix(values, feature_names=FEATURES, missing=np.nan)
    contributions = model.predict(matrix, pred_contribs=True)[0]
    log_potential = float(model.predict(matrix, output_margin=True)[0])
    base_value = float(contributions[-1])

    market = _market_name(artefacts, iso3)
    labels = factor_labels(market)
    factors = sorted(
        (
            Factor(
                feature=feature,
                label=labels[feature],
                value_text=value_text(feature, values[0, i]),
                contribution=float(contributions[i]),
                multiplier=math.exp(float(contributions[i])),
            )
            for i, feature in enumerate(FEATURES)
        ),
        key=lambda f: abs(f.contribution),
        reverse=True,
    )
    history = artefacts.history.filter((pl.col("hs6") == hs6) & (pl.col("iso3") == iso3)).sort("year")
    return Explanation(
        hs6=hs6,
        iso3=iso3,
        market=market,
        fold=fold,
        base_value=base_value,
        log_export_potential=log_potential,
        export_potential_usd=math.exp(log_potential),
        factors=factors,
        history=[HistoryPoint(int(y), float(v)) for y, v in history.select("year", "actual_exports_usd").iter_rows()],
    )


def value_text(feature: str, value: float) -> str:
    """The value of one model input, in words an Advisor reads (inputs are as the model saw them)."""
    if value is None or math.isnan(value):
        return "Not known"
    if feature in ("log_gdp_o", "log_gdp_d"):
        return usd(math.exp(value))
    if feature in ("log_supply_loo", "log_demand_loo"):
        return usd(math.expm1(value))
    if feature in ("log_population_o", "log_population_d"):
        return f"{math.exp(value) / 1e6:,.1f} million"
    if feature == "log_distance":
        return f"{math.exp(value):,.0f} km"
    if feature == "rca_loo":
        return f"{value:,.2f}× the world average share (above 1 means specialised)"
    if feature == "high_income_share_loo":
        return f"{value:.0%}"
    if feature == "sector":
        return f"Chapter {int(value):02d}"
    if feature == "relatedness":
        return f"{value:.0%} of its related Products are ones {_UK} exports competitively"
    if feature == "market_import_growth":
        return f"{math.expm1(value):+.0%} over {GROWTH_YEARS} years"
    if feature == "reexport_ratio":
        ratio = math.expm1(value)
        if ratio >= REEXPORT_RATIO_CAP * 0.999:
            return f"Exports {REEXPORT_RATIO_CAP:.0f}× or more what it imports from other countries"
        return f"Exports {ratio:.0%} of what it imports from other countries"
    return "Yes" if value >= 0.5 else "No"


def usd(amount: float) -> str:
    """A dollar amount, rounded to a readable unit."""
    for size, unit in ((1e12, "trillion"), (1e9, "billion"), (1e6, "million")):
        if abs(amount) >= size:
            return f"US${amount / size:,.1f} {unit}"
    return f"US${amount:,.0f}"


def _market_name(artefacts: Artefacts, iso3: str) -> str:
    found = artefacts.markets.filter(pl.col("iso3") == iso3)
    return found["name"].item(0) if not found.is_empty() else iso3


def global_importance(artefacts: Artefacts, *, sample: int | None = 20_000, seed: int = 0) -> pl.DataFrame:
    """Global feature importance: mean absolute SHAP contribution (log scale) over Baseline pairs.

    Each pair's contributions come from its own fold model, as in `explain_pair`. `sample`
    limits the work to a random sample of pairs (None: every pair). Returns feature, label
    (Market-neutral), mean_abs_contribution and pairs, largest first. A feature near zero
    contributes nothing to any prediction (course: mean |SHAP| supports data minimisation).
    """
    rows = artefacts.baseline_features
    if sample is not None and rows.height > sample:
        rows = rows.sample(sample, seed=seed)
    totals = np.zeros(len(FEATURES))
    for fold, part in rows.group_by("fold"):
        values = part.select(pl.col(FEATURES).cast(pl.Float32)).to_numpy()
        matrix = xgb.DMatrix(values, feature_names=FEATURES, missing=np.nan)
        contributions = artefacts.fold_models[int(fold[0])].predict(matrix, pred_contribs=True)
        totals += np.abs(contributions[:, : len(FEATURES)]).sum(axis=0)
    labels = factor_labels("the Market")
    return pl.DataFrame(
        {
            "feature": FEATURES,
            "label": [labels[f] for f in FEATURES],
            "mean_abs_contribution": (totals / max(rows.height, 1)).tolist(),
            "pairs": [rows.height] * len(FEATURES),
        }
    ).sort("mean_abs_contribution", descending=True)
