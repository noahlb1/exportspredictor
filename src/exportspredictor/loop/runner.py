"""The experiment loop on one fold: the noise floor, then experiments, each arm trained (or read from the cache) and measured."""

import json
import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import polars as pl

from exportspredictor import experiments, market_reach, search
from exportspredictor.evaluation import naive_potential
from exportspredictor.exclusions import drop_excluded
from exportspredictor.features import (
    KEYS,
    PRODUCT_DETAIL_FEATURES,
    FeatureTransform,
    subsample_zeros,
)
from exportspredictor.folds import outside_fold, with_folds
from exportspredictor.loop.arms import (
    ANY_FLOW,
    EXPORTER_COUNT_EXPERIMENT,
    MARKET_REACH_CAP_EXPERIMENT,
    MAX_TRAINED_MODELS,
    NOISE_FLOOR,
    SIBLING_DEMAND_EXPERIMENT,
    SIZE,
    Arm,
    LoopPlan,
    Settings,
    exporter_count_arms,
    market_reach_cap_arm,
    models_trained_by,
    product_detail_arm,
    product_detail_experiment,
    served_base_arm,
    sibling_demand_arm,
    training_key,
)
from exportspredictor.loop.data import (
    MODELS_DIR,
    LoopData,
    naive_from_features,
    read_results,
    save_results,
)
from exportspredictor.loop.recheck import current_win_threshold
from exportspredictor.loop.selection import (
    SELECTION_MEASURE,
    SELECTION_MEASURE_TEXT,
    decide,
    measures,
    rank_average_measures,
    win_threshold,
)
from exportspredictor.loop.trainers import (
    Model,
    Trainer,
    TrainingJob,
    TrainingRows,
    trainer_for,
)
from exportspredictor.prediction import (
    MIN_PREDICTION_USD,
    blend,
    export_potential,
    flow_probability,
    two_part_potential,
)
from exportspredictor.selection_measure import selection_measure, with_tie_key
from exportspredictor.training import (
    BASE_MARGIN,
    preparer,
    read_chunk,
    scan_chunks,
)

NO_MEASURE = -1.0  # what a round scores when no Product can be ranked (a tiny fixture fold)


def naive_margin(naive_usd: pl.Series) -> pl.Series:
    """The naive baseline as a starting point on the log scale, floored at MIN_PREDICTION_USD (a Zero Cell's is 0)."""
    return naive_usd.cast(pl.Float64).clip(lower_bound=MIN_PREDICTION_USD).log()


@dataclass(frozen=True)
class TwoPart:
    """A two-part model's Boosters."""

    any_flow: Model
    size: Model


def run_product_detail(lp: "Loop") -> dict:
    """Run the three Product detail experiments in order, each from the best so far."""
    for feature in PRODUCT_DETAIL_FEATURES:
        lp.noise_floor()
        best = Settings.from_dict(lp.results["best"]["settings"])
        lp.run_experiment(product_detail_experiment(feature), [product_detail_arm(feature, best)])
    return lp.results


def run_exporter_count(lp: "Loop") -> dict:
    """#127: the Exporter count against the best so far retrained on this loop's files, both with Market Reach."""
    lp.noise_floor()
    best = Settings.from_dict(lp.results["best"]["settings"])
    base, arms = exporter_count_arms(best)
    return lp.run_experiment(EXPORTER_COUNT_EXPERIMENT, arms, base_arm=base)


def run_market_reach_cap(lp: "Loop") -> dict:
    """#163: the upward Market Reach multiplier capped, against the best so far as served (`served_base_arm`).

    Both are scored from the best so far's cached model on this loop's files, so nothing is trained once that
    model exists. A kept cap becomes the best so far, with Market Reach and the cap in its settings.
    """
    lp.noise_floor()
    best = Settings.from_dict(lp.results["best"]["settings"])
    return lp.run_experiment(MARKET_REACH_CAP_EXPERIMENT, [market_reach_cap_arm()], base_arm=served_base_arm(best))


def run_sibling_demand(lp: "Loop") -> dict:
    """#164: Sibling Demand as a feature, against the best so far as served retrained on this loop's files.

    The files (`sibling_demand`) are the served ones with the feature stored, so the pair differs by one feature.
    A dropped Arm leaves the served files unchanged; a kept one serves these files.
    """
    lp.noise_floor()
    best = Settings.from_dict(lp.results["best"]["settings"])
    return lp.run_experiment(
        SIBLING_DEMAND_EXPERIMENT, [sibling_demand_arm(best)], base_arm=served_base_arm(best),
        files_kept_if_dropped=False,
    )


class Loop:
    """The experiment loop on one fold: the noise floor, then experiments, each against the best so far.

    With a `folder`, results are saved there as each arm finishes, models are cached in
    it, and a new `Loop` on the same folder resumes; without one, both live in memory.
    `models_dir` caches models elsewhere: the confirmation's fold loops share the loop's
    cache but keep no results of their own.
    """

    def __init__(
        self, data: LoopData, plan: LoopPlan, folder: Path | None = None, *, models_dir: Path | None = None
    ):
        experiments._check_years([data.validation_year])
        started = time.perf_counter()
        self.data, self.plan, self.folder = data, plan, folder
        self.models_dir = models_dir if models_dir is not None else (folder / MODELS_DIR if folder else None)
        self._models: dict[str, tuple[Model | TwoPart, dict]] = {}
        self._trainers: dict[str, Trainer] = {}  # one per family, each keeping its last matrices
        self._rows: TrainingRows | None = None
        self._detail: pl.DataFrame | None = None
        self._reach: pl.DataFrame | None = None  # this loop's Market Reach table, built on first use
        self.results = read_results(folder) or {}
        rows = data.data
        years = sorted(rows.training)
        self.chunks = [rows.training[y] for y in years]
        self.transform = FeatureTransform.fit(scan_chunks(self.chunks), years)
        flows = data.flows()
        is_uk = pl.col("exporter").cast(pl.String) == plan.uk
        validation = drop_excluded(read_chunk(rows.validation))
        validation = pl.concat([validation.filter(~is_uk), self._held_out(validation.filter(is_uk))])
        grid = self._held_out(drop_excluded(read_chunk(rows.grid)).filter(is_uk))
        self.validation = self.transform.apply(naive_potential(validation, flows))
        self.grid = self.transform.apply(naive_potential(grid, flows))
        del flows
        # Hashed once: the round-by-round measure reads the tie key instead of rehashing every round.
        self._grid_keys = with_tie_key(self.grid.select("hs6", "importer", "value_usd"))
        if "naive" not in self.results:
            self.results |= {
                "selection_measure": SELECTION_MEASURE_TEXT,
                "validation_year": data.validation_year,
                "fold": plan.fold,
                "n_folds": plan.n_folds,
                "naive": {
                    **measures(self.grid, self.validation, "naive_usd"),
                    "validation_year": data.validation_year,
                },
                "experiments": {},
            }
        self.results["setup_seconds"] = round(time.perf_counter() - started, 1)
        self._save()

    def _held_out(self, uk_rows: pl.DataFrame) -> pl.DataFrame:
        folded = with_folds(uk_rows, self.plan.n_folds)
        return folded.filter(pl.col("fold") == self.plan.fold).drop("fold")

    # --- Running arms --------------------------------------------------------------------------

    def run_experiment(
        self, name: str, arms: Sequence[Arm], *, base_arm: Arm | None = None, files_kept_if_dropped: bool = True
    ) -> dict:
        """Run each arm against the best so far, decide by the win rule, and return the results.

        With `base_arm`, the arms are compared with that Arm of the best so far, run first on this
        loop's data, instead of with the best so far's recorded measure: a paired comparison when the
        data changed (rebuilt feature files, #127) or every Arm is scored differently (Market Reach).
        The section records it (`base_arm`), the best so far it replaced as base (`previous_best`) and
        the feature set. If no arm wins, the best so far is unchanged. The feature set is served
        (`kept_feature_set`) whatever the decision, as #127's gravity fill was, unless
        `files_kept_if_dropped` is False: then only if an Arm is kept (#164's files exist to hold one feature).
        """
        if name == NOISE_FLOOR:
            raise ValueError(f"{NOISE_FLOOR!r} is the noise floor's name")
        self.noise_floor()
        section = self.results["experiments"].setdefault(name, {"arms": []})
        if "decided" not in section:
            if base_arm is None:
                section["base"] = dict(self.results["best"])
            elif "base_arm" not in section:
                previous = dict(self.results["best"])
                previous_settings = Settings.from_dict(previous["settings"])
                base_settings = base_arm.apply(previous_settings)
                self._check_budget(name, section, [Arm(base_arm.name), *arms], base_settings)
                row = self.run_arm(base_arm, previous_settings)
                section |= {
                    "previous_best": previous, "base_arm": row, "base": _best(name, row),
                    "feature_set": self.data.feature_set,
                }
                if not files_kept_if_dropped:
                    section["files_kept_if_dropped"] = False
                self._save()
                log_progress(f"{base_arm.name}: {json.dumps({k: row[k] for k in (SELECTION_MEASURE, 'rounds', 'seconds')})}")
        base = Settings.from_dict(section["base"]["settings"])
        if "decided" not in section:
            self._check_budget(name, section, arms, base)
        rows = self._run_arms(section, arms, base)
        if "decided" not in section:
            threshold = current_win_threshold(self.results)
            section["arms"], kept = decide(rows, section["base"][SELECTION_MEASURE], threshold)
            section |= {"win_threshold": threshold, "kept": kept, "decided": True}
            if kept is not None:
                row = next(r for r in section["arms"] if r["arm"] == kept)
                self.results["best"] = _best(name, row)
            self._save()
        return self.results

    def noise_floor(self) -> dict:
        """The start configuration with each noise seed; the win threshold; and the best so far."""
        section = self.results.setdefault(NOISE_FLOOR, {"arms": []})
        arms = [Arm(f"seed {seed}", {"seed": seed}) for seed in self.plan.noise_seeds]
        rows = self._run_arms(section, arms, self.plan.start)
        if "win_threshold" not in section:
            values = [row[SELECTION_MEASURE] for row in rows]
            section |= {
                "seeds": list(self.plan.noise_seeds),
                "selection_measures": values,
                "mean": statistics.fmean(values),
                "sd": statistics.stdev(values),
                "win_threshold": win_threshold(values, self.plan.min_win_threshold),
            }
            # Arms run on the first seed, so the best so far is that seed's row: a paired comparison.
            self.results["best"] = _best(NOISE_FLOOR, rows[0])
            self._save()
        return section

    def _check_budget(self, name: str, section: dict, arms: Sequence[Arm], base: Settings) -> None:
        """Refuse, before training anything, an experiment that would train more than MAX_TRAINED_MODELS models.

        Counted: the models its finished arms trained and those its other arms would train
        (a model already cached is free; a two-part model is two).
        """
        models = {row["training_key"]: models_trained_by(row["settings"]) for row in section["arms"] if row.get("trained")}
        done = {row["arm"] for row in section["arms"]}
        for arm in arms:
            if arm.name in done:
                continue
            settings = self._resolved(arm.apply(base))
            key = training_key(settings, self._fingerprint())
            if key not in self._models and not self._model_file(key, settings.family).exists():
                models.setdefault(key, models_trained_by(settings.to_dict()))
        if sum(models.values()) > MAX_TRAINED_MODELS:
            raise ValueError(
                f"Experiment {name!r} would train {sum(models.values())} models; the budget is at most "
                f"{MAX_TRAINED_MODELS} trained models per experiment (spec #65)"
            )

    def _run_arms(self, section: dict, arms: Sequence[Arm], base: Settings) -> list[dict]:
        done = {row["arm"] for row in section["arms"]}
        for arm in arms:
            if arm.name in done:
                continue
            row = self.run_arm(arm, base)
            section["arms"].append(row)
            self._save()
            log_progress(f"{arm.name}: {json.dumps({k: row[k] for k in (SELECTION_MEASURE, 'rounds', 'seconds')})}")
        order = {arm.name: i for i, arm in enumerate(arms)}
        return sorted((row for row in section["arms"] if row["arm"] in order), key=lambda row: order[row["arm"]])

    def run_arm(self, arm: Arm, base: Settings) -> dict:
        """One result row: the arm's settings over `base`, trained (or taken from the cache) and measured."""
        started = time.perf_counter()
        settings = self._resolved(arm.apply(base))
        key, trained, model, meta = self.cached_model(settings)
        model_grid = self._model_potential(model, meta, settings, self.grid)
        extra = {}
        if arm.grid:
            # Choose the scoring setting on the validation year's held-out UK grid only.
            (name, values), = arm.grid.items()
            by_value = []
            for value in values:
                value_settings = replace(settings, **{name: value})
                scored_grid = self.grid.with_columns(export_potential_usd=self._score(model_grid, value_settings, self.grid))
                by_value.append({name: value, **measures(scored_grid, None, "export_potential_usd")})
            chosen = max(by_value, key=lambda v: -2.0 if v[SELECTION_MEASURE] is None else v[SELECTION_MEASURE])
            settings = replace(settings, **{name: chosen[name]})
            extra["grid"] = by_value
        if arm.grid.keys() & {"naive_weight"}:
            extra["rank_average"] = rank_average_measures(self.grid.with_columns(model_usd=model_grid))
            # The model unblended (w = 0), a reference only: the blend's grid leaves 0 out.
            extra["model_alone"] = measures(self.grid.with_columns(model_usd=model_grid), None, "model_usd")
        if settings.two_part:
            extra["parts"] = meta["parts"]
        model_validation = self._model_potential(model, meta, settings, self.validation)
        if settings.market_reach_beta:
            # The same scoring without Market Reach, a reference: what Reach adds on this Arm's model.
            plain = replace(settings, market_reach_beta=0.0)
            extra["without_market_reach"] = measures(
                self.grid.with_columns(export_potential_usd=self._score(model_grid, plain, self.grid)),
                self.validation.with_columns(export_potential_usd=self._score(model_validation, plain, self.validation)),
                "export_potential_usd",
            )
        scored_grid = self.grid.with_columns(export_potential_usd=self._score(model_grid, settings, self.grid))
        scored = self.validation.with_columns(
            export_potential_usd=self._score(model_validation, settings, self.validation)
        )
        return {
            "arm": arm.name,
            "changes": dict(arm.changes),
            "settings": settings.to_dict(),
            "training_key": key,
            "trained": trained,
            **measures(scored_grid, scored, "export_potential_usd"),
            **extra,
            "validation_year": self.data.validation_year,
            **meta,
            "seconds": round(time.perf_counter() - started, 1),
        }

    # --- Models --------------------------------------------------------------------------------

    def export_potential(self, settings: Settings, table: pl.DataFrame) -> pl.Series:
        """Export Potential (USD) of transformed rows (with `naive_usd`) as an arm with `settings` scores them.

        The model is taken from the cache (or trained); e.g. to re-predict after a Scenario
        changes policy features.
        """
        settings = self._resolved(settings)
        model, meta = self._model(settings, training_key(settings, self._fingerprint()))
        return self._score(self._model_potential(model, meta, settings, table), settings, table)

    def cached_model(self, settings: Settings) -> tuple[str, bool, Model | TwoPart, dict]:
        """The model of `settings` from the cache, or trained and cached now.

        Returns its training key, whether this call trained it, the model and its metadata.
        """
        settings = self._resolved(settings)
        key = training_key(settings, self._fingerprint())
        trained = key not in self._models and not self._model_file(key, settings.family).exists()
        model, meta = self._model(settings, key)
        return key, trained, model, meta

    def _resolved(self, settings: Settings) -> Settings:
        trainer_for(settings.family)  # refuses a family with no trainer before anything trains
        zero_rate = settings.zero_rate if settings.zero_rate is not None else self.data.data.built_zero_rate
        non_uk_rate = settings.non_uk_rate
        if non_uk_rate is None:
            non_uk_rate = search.final_non_uk_rate(self.data.data, len(settings.features))
        return replace(settings, zero_rate=zero_rate, non_uk_rate=non_uk_rate)

    def _fingerprint(self) -> dict:
        rows = self.data.data
        # A rebuilt feature set keys its own models; the original's keys are unchanged.
        feature_set = {"feature_set": self.data.feature_set} if self.data.feature_set else {}
        return {
            **feature_set,
            "training_years": sorted(rows.training),
            "validation_year": rows.validation_year,
            "built_zero_rate": rows.built_zero_rate,
            "zero_seed": self.data.zero_seed,
            "fold": self.plan.fold,
            "n_folds": self.plan.n_folds,
            "device": self.plan.device,
        }

    def _model_file(self, key: str, family: str) -> Path:
        return Path(self.models_dir or ".nowhere") / f"{key}{trainer_for(family).suffix}"

    def _model(self, settings: Settings, key: str) -> tuple[Model | TwoPart, dict]:
        if key in self._models:
            return self._models[key]
        trainer = self._trainer(settings.family)
        path = self._model_file(key, settings.family)
        # A two-part model's binary part sits beside its size part, which is saved last.
        any_flow_path = path.with_name(f"{key}.{ANY_FLOW}{path.suffix}")
        if self.models_dir is not None and path.exists():
            model = trainer.load(path)
            if settings.two_part:
                model = TwoPart(any_flow=trainer.load(any_flow_path), size=model)
            meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        else:
            model, meta = self._train_two_part(settings) if settings.two_part else self._train(settings)
            if self.models_dir is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                if isinstance(model, TwoPart):
                    trainer.save(model.any_flow, any_flow_path)
                trainer.save(model.size if isinstance(model, TwoPart) else model, path)
                path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        self._models[key] = (model, meta)
        return model, meta

    def _train_two_part(self, settings: Settings) -> tuple[TwoPart, dict]:
        """Train the binary part, then the size part with the two-part Export Potential measured each round.

        The binary part stops on the selection measure of its probability alone; the size
        part stops on the selection measure of probability × size, the binary part held at
        its chosen round. Both measures are the model alone, never the blend (#67's rule).
        The matrices are built one part at a time, so only one training matrix is on the device.
        """
        if settings.base_margin:
            raise ValueError("A two-part model has no starting point: base_margin and two_part do not combine")
        started = time.perf_counter()
        any_flow, any_flow_meta = self._train(settings, part=ANY_FLOW)
        probability = flow_probability(
            self.grid, any_flow, features=list(settings.features), rounds=any_flow_meta["rounds"],
            device=self.plan.device,
        ).to_numpy()
        trainer = self._trainer(settings.family)
        trainer.release()  # free the binary part's matrices first
        size, size_meta = self._train(settings, part=SIZE, grid_probability=probability)
        trainer.release()
        return TwoPart(any_flow=any_flow, size=size), {
            "rounds": size_meta["rounds"],  # the size part's, chosen on the two-part Export Potential
            "hit_round_cap": any_flow_meta["hit_round_cap"] or size_meta["hit_round_cap"],
            "trained_rounds": size_meta["trained_rounds"],
            "training_rows": any_flow_meta["training_rows"],
            "matrix_seconds": round(any_flow_meta["matrix_seconds"] + size_meta["matrix_seconds"], 1),
            "train_seconds": round(time.perf_counter() - started, 1),
            "parts": {ANY_FLOW: any_flow_meta, SIZE: size_meta},
        }

    def _train(self, settings: Settings, *, part: str | None = None, grid_probability=None) -> tuple[Model, dict]:
        """Train through the family's trainer, stopping on the selection measure of the held-out UK grid.

        `part` trains one part of a two-part model; the size part's measure each round is
        that of `grid_probability` × size.
        """
        measure = self._round_measure
        if grid_probability is not None:

            def measure(predictions) -> float:
                return self._round_measure(grid_probability * predictions)

        job = TrainingJob(
            rows=self._training_rows(settings, part), params=settings.params, seed=settings.seed,
            monotone=settings.monotone, max_rounds=settings.max_rounds, patience=settings.patience,
            measure=measure, device=self.plan.device, threads=self.plan.threads, binary=part == ANY_FLOW,
        )
        return self._trainer(settings.family).train(job)

    def _round_measure(self, predictions) -> float:
        value = selection_measure(self._grid_keys.with_columns(prediction=pl.Series(predictions)), "prediction")
        return NO_MEASURE if value is None else value

    def _trainer(self, family: str) -> Trainer:
        """This loop's trainer of `family`; the other families' kept matrices are freed first."""
        if family not in self._trainers:
            self._trainers[family] = trainer_for(family)
        for other, trainer in self._trainers.items():
            if other != family:
                trainer.release()
        return self._trainers[family]

    def _training_rows(self, settings: Settings, part: str | None = None) -> TrainingRows:
        """The training rows and the held-out grid's, the same object while the rows' settings stay the same.

        For a two-part model's binary part (`ANY_FLOW`) the label is "any Flow" (0 or 1) and
        every row keeps its weight; its size part (`SIZE`) trains on non-zero Flows only.
        """
        key = (
            settings.features, settings.non_uk_rate, settings.zero_rate, settings.seed, settings.base_margin,
            settings.uk_zeros_sampled, part,
        )
        if self._rows is not None and self._rows.key == key:
            return self._rows
        plan, built = self.plan, self.data.data.built_zero_rate
        held_out = outside_fold(plan.uk, plan.n_folds, plan.fold)
        zero_seed = self.data.zero_seed
        world = self._world_products() if settings.base_margin else None
        grid_rows = self._with_product_detail(self.grid, settings)

        def keep(table: pl.DataFrame) -> pl.DataFrame:
            rows = subsample_zeros(
                table, built_rate=built, rate=settings.zero_rate, seed=zero_seed, uk=plan.uk,
                uk_zeros_sampled=settings.uk_zeros_sampled,
            )
            rows = held_out(drop_excluded(rows)).sort(KEYS)  # a fixed order: the same rows, the same model
            rows = self._with_product_detail(rows, settings)
            if part == ANY_FLOW:
                rows = rows.with_columns(_any_flow(pl.col("value_usd")))
            elif part == SIZE:
                rows = rows.filter(pl.col("value_usd") > 0)
            if world is None:
                return rows
            # Before the transform clips the supply and demand features: the naive baseline as built.
            return rows.with_columns(naive_margin(naive_from_features(rows, world)).alias(BASE_MARGIN))

        if settings.base_margin:
            grid_rows = grid_rows.with_columns(naive_margin(self.grid["naive_usd"]).alias(BASE_MARGIN))
        if part == ANY_FLOW:  # a binary label on the grid too; the measure each round reads value_usd itself
            grid_rows = grid_rows.with_columns(_any_flow(pl.col("value_usd")))

        prepare = preparer(
            uk=plan.uk, transform=self.transform, non_uk_rate=settings.non_uk_rate, seed=settings.seed, keep=keep
        )
        self._rows = TrainingRows(
            chunks=self.chunks, prepare=prepare, grid=grid_rows, features=settings.features, key=key
        )
        return self._rows

    def _model_potential(self, model: Model | TwoPart, meta: dict, settings: Settings, table: pl.DataFrame) -> pl.Series:
        """The model's Export Potential in dollars at its chosen round count."""
        table = self._with_product_detail(table, settings)
        if isinstance(model, TwoPart):
            parts = meta["parts"]
            return two_part_potential(
                table, model.any_flow, model.size, features=list(settings.features),
                rounds=(parts[ANY_FLOW]["rounds"], parts[SIZE]["rounds"]), device=self.plan.device,
            )
        margin = naive_margin(table["naive_usd"]) if settings.base_margin else None
        return export_potential(
            table, model, features=list(settings.features), rounds=meta["rounds"], device=self.plan.device,
            base_margin=margin,
        )

    def _with_product_detail(self, table: pl.DataFrame, settings: Settings) -> pl.DataFrame:
        """`table` with the Product detail features its settings use, joined on the Product in the same row order."""
        wanted = [f for f in settings.features if f in PRODUCT_DETAIL_FEATURES]
        if not wanted:
            return table
        if self._detail is None:
            if self.data.product_detail is None:
                raise ValueError("This loop's data has no Product detail features, so no arm can use them")
            self._detail = self.data.product_detail().select(pl.col("hs6").cast(pl.String).alias("_hs6"), *PRODUCT_DETAIL_FEATURES)
        return (
            table.drop(wanted, strict=False)
            .with_columns(_hs6=pl.col("hs6").cast(pl.String))
            .join(self._detail.select("_hs6", *wanted), on="_hs6", how="left", maintain_order="left")
            .drop("_hs6")
        )

    def _world_products(self) -> pl.DataFrame:
        if self.data.world_products is None:
            raise ValueError("This loop's data has no world exports by Product, so no arm can start from the naive baseline")
        return self.data.world_products()

    def _score(self, predicted: pl.Series, settings: Settings, table: pl.DataFrame) -> pl.Series:
        """Export Potential in dollars as the arm scores it: the served blend (`prediction.blend`), the model alone
        at w = 0, then Market Reach where the settings' beta is not 0."""
        potential = blend(predicted.to_numpy(), table["naive_usd"].to_numpy(), settings.naive_weight)
        if settings.market_reach_beta:
            potential = self._with_market_reach(potential, table, settings.market_reach_beta, settings.market_reach_cap)
        return pl.Series("export_potential_usd", potential, dtype=pl.Float64)

    def _with_market_reach(self, potential, table: pl.DataFrame, beta: float, cap: float | None = None):
        """`potential` × exp(beta · offset) of each held-out UK row's Market (at most `cap` where set), rescaled to keep the rows' sum.

        The offsets are this fold's (`market_reach.reach_table`), learned from the year before the
        validation year on the UK pairs outside this fold. Other Exporters' rows get no offset.
        """
        if self._reach is None:
            if self.data.reach_source is None:
                raise ValueError("This loop's data has no UK grid of the year before, so no arm can use Market Reach")
            lag = self.data.validation_year - market_reach.LAG_YEARS
            source = self.data.reach_source().select(
                pl.col("hs6").cast(pl.String), importer=pl.col("importer").cast(pl.String), value_usd="value_usd",
                log_demand_loo=market_reach.DEMAND, year=pl.lit(lag),
            )
            rows = with_folds(source, self.plan.n_folds)
            self._reach = market_reach.reach_table(rows, self.data.validation_year, n_folds=self.plan.n_folds)
        # Every row is held out by this fold; other Exporters' rows take no offset, only c (as #124 measured).
        rows = market_reach.uk_keys(table, fold=pl.lit(self.plan.fold, pl.Int8), uk=self.plan.uk).drop("uk")
        constant = market_reach.rescale_constant(potential, rows, self._reach, beta=beta, cap=cap)
        return market_reach.apply(potential, rows, self._reach, beta=beta, constant=constant, cap=cap)

    def _save(self) -> None:
        if self.folder is not None:
            save_results(self.folder, self.results)


def _any_flow(value: pl.Expr) -> pl.Expr:
    """The binary part's label: 1 where there is a Flow, 0 in a Zero Cell."""
    return (value > 0).cast(pl.Float64).alias("value_usd")


def _best(experiment: str, row: Mapping) -> dict:
    return {
        "experiment": experiment,
        "arm": row["arm"],
        "settings": row["settings"],
        SELECTION_MEASURE: row[SELECTION_MEASURE],
    }


def log_progress(message: str) -> None:
    print(message, flush=True)
