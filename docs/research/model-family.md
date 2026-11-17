# Research: Was XGBoost the best model family for Export Potential?

Ticket: #25 "Model family evidence", child of the wayfinder map #24. It feeds #33 "Model family decision". Researched 2026-11-17.

This note gathers evidence and sets out the trade-offs. **It does not make the decision**, which belongs to #33.

## Bottom line

1. **Of the three boosting libraries, only XGBoost does all four things the project needs on this machine, on the GPU: Tweedie loss, monotone constraints, a streamed quantile matrix and Windows.** LightGBM's CUDA learner has no monotone constraints (issue open since 2021), sends Tweedie back to the CPU, and doesn't build on Windows. CatBoost documents monotone constraints as CPU-only.
2. **The benchmark literature gives no reason to expect another GBDT library to be more accurate.** Large tabular benchmarks find the choice of library or model family matters less than tuning. GBDTs beat neural networks on skewed, heavy-tailed features, which describes trade values.
3. **The strongest competitor in the trade literature is 3-way PPML gravity, not another ML model.** Apfel, Breinlich, Novy, Santos Silva, Zylkin et al. (2025) find 3-way PPML "difficult to beat" on *average* out-of-sample predictions. An ensemble ML model beats it on *individual* Flows, but only when it is given the gravity fixed effects. That predictive power comes from pair fixed effects. A pair held out of the fit (ADR 0001) has no fixed effect to predict with, and a Product-level model would need tens of millions of fixed effects.
4. **ITC's Export Potential Map does not estimate a regression or an ML model.** It rejects a fixed-effects gravity regression at Product level as too large, too noisy and too opaque for advisors. Instead it multiplies supply, demand and ease-of-trade factors computed from shares, with tariff effects set by external GTAP elasticities. Its "unrealized potential" is the ancestor of our Export Gap.
5. **An entity-embedding neural network is the only family that adds information the current model lacks.** v1 has no Exporter, Importer or Product identity features beyond `sector`. But it has no native monotonicity: TF Lattice and certified monotone networks exist, and TensorFlow has had no native-Windows GPU support since 2.10. It explains less well than TreeSHAP and has the weakest benchmark record.
6. **Lean for #33:** keep XGBoost as the production model. The open questions are about the *data budget*: GPU with sampling, or CPU / external memory on all rows. They are not about the family. PPML gravity is worth adding as a benchmark and as a check on the Trade Agreement Scenario Effect, not as a replacement.

## 1. What the problem demands

From `CLAUDE.md`, ADR 0001, ADR 0002, `src/exportspredictor/training.py`, `features.py` and `pipeline.py`:

| Requirement | v1 implementation |
|---|---|
| About 150 M HS6 Flows with heavy zeros, value target | `reg:tweedie`, `tweedie_variance_power` 1.5 (grid 1.3 / 1.5 / 1.7); `tweedie-nloglik@1.5` for selection |
| Policy features must not move Export Potential the wrong way | `monotone_constraints` +1 on all policy features (pair Trade Agreement features plus EU / CPTPP / EFTA bloc membership on each side) |
| Scenarios re-predict with only the policy features changed | One model; tariffs clipped to the training range (ADR 0002) |
| Out-of-fold UK scoring | 5 fold models grouped by Product–Market pair (ADR 0001) |
| GPU, 10 GB VRAM | `tree_method="hist"`, `device="cuda"`, `QuantileDMatrix` built from a chunk iterator. 152 M rows filled VRAM, so a 30% non-UK sample with Horvitz–Thompson weights and 2% zero sampling are used |
| Explanations for Advisors | TreeSHAP from XGBoost (`pred_contribs=True`), additive on the log scale (`explain.py`) |
| Features | 22 numeric features: gravity, leave-one-out supply and demand, `sector`, policy. No Exporter, Importer or Product IDs |
| Results | Within-Product Spearman 0.71 against a naive 0.56; full rebuild about 1 h on the GPU |
| Locked version | `xgboost` 3.4.1 (`uv.lock`); the latest release is 3.4.2 (15 Sep 2026) |

## 2. Gradient-boosted tree libraries: capability matrix

| Capability | XGBoost 3.4 | LightGBM 4.7 | CatBoost 1.2.10 |
|---|---|---|---|
| Tweedie objective | `reg:tweedie`, log link [X1] | `tweedie`, log link [L1] | `Tweedie:variance_power=…` [C1] |
| Tweedie **on GPU** | Yes (`hist` + `device=cuda`) [X1] | **No.** CUDA build logs "Objective tweedie is not implemented in cuda version. Fall back to boosting on CPU." [L3] (Poisson *is* on CUDA) | Yes. Docs: "On GPU, it is recommended to scale labels to range [0,1]" [C1] |
| Monotone constraints | Yes, `hist` / `approx` / `exact`; with `hist`, "may produce unnecessarily shallow trees", so raise `max_bin` [X2] | Yes, three methods. `basic` "over-constrains"; `intermediate` and `advanced` are "much less constraining" [L1]. `intermediate` / `advanced` drop to `basic` if `feature_fraction_bynode` ≠ 1 [L2] | Yes, but **"Supported processing units: CPU"** [C2]. It can't be combined with `Exact` leaf estimation [C3] |
| Monotone **on GPU** | Yes. PR #12809 (merged 6 Nov 2026, after 3.4.2) "Align[s] the GPU monotone constraint implementation to the CPU" [X5], which implies the GPU and CPU implementations differed in detail up to 3.4.x (the PR body says only "Use the same weight pooling. Stage and batch the splits.") | **CUDA: no.** Issue #4781 (open since Nov 2021): implementing it on the new CUDA learner "requires more efforts". The CUDA learner source contains no monotone code [L4]. OpenCL `gpu` learner: yes, because it subclasses `SerialTreeLearner`, but it "only computes histograms on GPUs" [L5] | No [C2] |
| GPU on **Windows** | Yes (current setup) | CUDA: "The CUDA version is not supported on Windows"; OpenCL only [L5] | Yes (CUDA) |
| Out-of-core / larger than device memory | `ExtMemQuantileDMatrix` (3.0+), host-cached. On PCIe, "PCIe bandwidth limits the GPU performance"; CPU external memory "limited by disk IO" [X3] | In-memory `Dataset`; bins stored as `uint8_t` when `max_bin=255` [L1] | In-memory (no external-memory mode documented) |
| Native SHAP | `pred_contribs` (used in v1) | `pred_contrib` | `get_feature_importance(type="ShapValues")` |
| Native categoricals | Yes | Yes (but "cannot be monotonically constrained" on a categorical) [L1] | Yes (target statistics, its main selling point) |

**What this means for this machine.** Today, Tweedie plus monotone plus GPU plus Windows plus streaming works only in XGBoost. Running LightGBM or CatBoost with the same constraints means CPU training. On the CPU, the 32 GB of RAM can probably hold all ~152 M × 22 binned features. At one byte per binned value that is about 3.3 GB, plus label, weight and gradients. This is my arithmetic from [L1], not a measurement. So moving to the CPU could **remove the 30% / 2% sampling**, at an unmeasured cost in wall-clock time. That trade (GPU with sampling or CPU on all rows) is open for XGBoost too: CPU `hist` with a `QuantileDMatrix`, or `ExtMemQuantileDMatrix` on the GPU [X3].

**Accuracy evidence between the GBDT libraries.** No source found compares them on trade data. The general benchmarks say the gaps are small and tuning-dependent: McElfresh et al. (NeurIPS 2023 Datasets & Benchmarks, 19 algorithms, 176 datasets) find the debate "overemphasized", with light GBDT tuning often mattering more than the choice of model [B2]. CatBoost's distinctive strength, ordered target statistics for high-cardinality categoricals, is not used by v1, which has only one categorical (`sector`, encoded as a number).

## 3. PPML structural gravity

**What it is.** Poisson pseudo-maximum likelihood with exporter-time, importer-time and pair fixed effects is the field standard. The WTO/UNCTAD *Advanced Guide* recommends panel data, intervals, intra-national flows, directional time-varying fixed effects, pair fixed effects and PPML (Recommendations 1–6) [G1]. Pair fixed effects "account for the endogeneity of trade policy variables" [G1]: they are what lets gravity claim a policy *effect* rather than an association. PPML handles zeros natively and is the estimator ADR 0002 says economists would normally use.

**Out-of-sample prediction evidence.**
- Apfel, Breinlich, Green, Novy, Santos Silva & Zylkin (arXiv 2509.11271, Sep 2025): 100 largest exporters, 1994–2023, aggregate bilateral trade. They compare traditional, 1-, 2- and 3-way PPML gravity with random forest, gradient boosting, a Poisson-loss NN and a stacked ensemble. "The 3-way gravity model is difficult to beat in terms of out-of-sample average predictive performance". For individual Flows, "the 3-way model can be outperformed by an ensemble machine learning method", but ML *without* the fixed-effect information generally does not beat 3-way. The best was a hybrid, 3-way gravity augmented with NN fitted values [P1].
- Ruzicska, Chariag, Kiss & Kiss (Springer, 2024): gravity against random forest, NN and graph NN on trade and mobility flows. "By most metrics, machine learning models only marginally outperform the gravity equation". All models mainly capture cross-sectional differences rather than changes over time [P2].
- Gopinath, Batarseh, Beckman, Kulkarni & Jeong (*Data & Policy* 2021): seven agricultural commodities. Supervised ML and NNs beat traditional gravity forecasts; NNs fit best over the long term [P3].

**Fit to this project.**
- *Scale.* At HS6 the natural fixed effects are exporter×product×year, importer×product×year and exporter×importer(×product). That means millions to tens of millions of fixed effects. ITC rejects this route for Export Potential for exactly that reason: the model "would be very large", three-dimensional data quality is lower, and "a model estimated with fixed effects however makes it difficult to understand the drivers of the resulting export potential values" [I1]. GPU-capable HDFE Poisson solvers now exist: `ppmlhdfe` uses accelerated IRLS [G2], and pyfixest has GLM Poisson with high-dimensional fixed effects and a torch LSMR demeaner on CUDA [G3]. Neither has documented runs at 150 M rows with fixed effects of this size.
- *Out-of-fold UK scoring (ADR 0001).* A pair fixed effect cannot be estimated for a pair held out of the fit, and that pair fixed effect is where 3-way gravity's predictive advantage comes from [P1]. A 2-way model (no pair fixed effect) can predict held-out pairs, but [P1] finds it is the one ML beats.
- *Inference.* 3-way PPML is consistent with fixed T, but confidence intervals are not centred and cluster-robust standard errors are biased. Corrections exist (`ppml_fe_bias`) [G4].
- *Scenarios.* PPML gives an interpretable, more nearly causal Trade Agreement coefficient (average, or by sector through interactions), and monotonicity can be checked from its sign. It cannot learn the Product × Market interactions that ADR 0002 says drive HS6 Opportunities, unless those are written in by hand.
- *Explainability.* Coefficients are the most familiar form for economists. ITC argues fixed-effects outputs are opaque to advisors [I1].

## 4. ITC Export Potential Map methodology (the reference practice)

From ITC, *Export Potential and Diversification Assessments* (methodology dated 27 Jun 2023) [I1]:
- It starts from a frictionless gravity-like decomposition, $v_{ijk} = \alpha_{ik}\beta_{ij}\gamma_{jk}$. It explicitly declines to estimate this with fixed effects (reasons above), and infers potential "from a multiplicative model based on two-dimensional data".
- $EPI_{ijk} = Supply_{ik} \times Ease_{ij} \times Demand_{ijk}$, normalised so that summing over exporters gives projected imports.
- Supply is the world market share, corrected for the global tariff disadvantage. Demand is projected imports, adjusted for the market tariff advantage and a product-specific distance factor. Ease is actual bilateral trade relative to what market shares would imply.
- Tariff effects use substitution elasticities $\sigma_k$ from GTAP, not estimated ones. The price elasticity of import demand "is assumed to be equal to one for every product". Forward-looking tariffs, GDP and population projections are built in.
- "Unrealized potential" $= EPI_{ijk} - \min(v_{ijk}, EPI_{ijk})$, the ancestor of our Export Gap.
- ML appears only in the literature review: Gupta & Kumar (2021) is cited as a "still nascent" but "promising" literature.

So the reference practice is neither PPML nor ML. It is a transparent, calibrated decomposition. A learned model is a step beyond course and practice, and it needs held-out validation (our Spearman and the Event Backtests) to earn trust.

## 5. Entity-embedding neural network

- **Idea.** Learn dense embeddings of the Exporter, Importer and Product (HS6) categoricals jointly with an MLP. Guo & Berkhahn (2016) report that embeddings help "the neural network to generalize better when the data is sparse" and can be reused as features in other models [N1]. This is the one family that would give the model Product- and country-identity information it lacks now, a learned analogue of fixed effects.
- **Benchmarks.** Grinsztajn, Oyallon & Varoquaux (NeurIPS 2022 Datasets & Benchmarks; 45 datasets): tree-based models "remain state-of-the-art on medium-sized data" [B1]. Their sets are ~10 K rows, far below ours. McElfresh et al. (2023): GBDTs are much better than NNs on "skewed, heavy-tailed" feature distributions [B2], which describes trade values. No benchmark found for the 10⁸-row regime.
- **Monotonicity.** Standard MLPs can't guarantee it. Options include TensorFlow Lattice (calibrated lattices, Deep Lattice Networks), which guarantees monotonicity and can sit as the last layers on top of embeddings [N2]. Certified and constrained monotone networks are a research alternative. Each adds architecture and tuning work that XGBoost's one parameter avoids.
- **Platform.** "TensorFlow 2.10 was the last TensorFlow release that supported GPU on native-Windows"; later versions need WSL2 [N3]. PyTorch runs natively on Windows with CUDA, but TF Lattice is TensorFlow-only.
- **Explainability.** No exact TreeSHAP. Gradient or KernelSHAP-style attributions are approximate and slower.
- **Loss.** A Poisson or Tweedie NLL with log link is easy to write (Apfel et al. used a Poisson-loss NN [P1]).
- **Data.** Minibatch training streams from Parquet, so it is not VRAM-bound in the way the GBDT matrix is. That is a real advantage at 150 M rows.

## 6. Trade-off summary

| | XGBoost (v1) | LightGBM | CatBoost | PPML gravity | Embedding NN |
|---|---|---|---|---|---|
| Tweedie / Poisson, zeros | ✔ GPU | ✔ CPU only for Tweedie (Poisson on CUDA) | ✔ GPU | ✔ Poisson, native | ✔ custom loss |
| Monotone policy features | ✔ GPU and CPU | CPU or OpenCL only | CPU only | Sign of a coefficient (checked, not enforced) | Needs lattice or monotone architecture |
| Scenario re-prediction | ✔ | ✔ | ✔ | ✔ (coefficient × change) | ✔ |
| Product × Market interactions | ✔ learned | ✔ | ✔ | Only if specified | ✔ learned |
| Out-of-fold UK pairs (ADR 0001) | ✔ | ✔ | ✔ | Pair fixed effects unavailable for held-out pairs | ✔ |
| 150 M rows, this machine | GPU needs sampling; CPU or ext-mem could fit all | CPU in-RAM likely fits (est.) | CPU (monotone) | Untested at this fixed-effect scale | Streams minibatches |
| Windows GPU | ✔ | OpenCL only | ✔ (but not with monotone) | torch / pyfixest CUDA | PyTorch ✔, TF ✘ (≥2.11) |
| Advisor explainability | Exact TreeSHAP | Exact TreeSHAP | Exact TreeSHAP | Coefficients (fixed effects opaque, per ITC) | Approximate attributions |
| Evidence of better accuracy | Baseline | None specific to trade | None specific to trade | Best *average* OOS (with pair FE) [P1] | Wins in some forecasting [P3]; loses on heavy tails [B2] |
| Switching cost | 0 | Medium (CPU rewrite of training) | Medium–high | High (new model, ADR 0001 conflict) | High (new stack, monotonicity) |

## 7. Evidence-backed options for #33

These are possibilities only. The decision is #33's.

- **A. Keep XGBoost; revisit the data budget.** Test CPU `hist` or GPU `ExtMemQuantileDMatrix` on all rows against the 30% / 2% sample, then compare Spearman and backtests. Lowest risk; it directly tests whether the sampling cost accuracy.
- **B. Add PPML gravity as a benchmark and Scenario cross-check, not a replacement.** Fit a 2-way or 3-way PPML on a tractable aggregation (e.g. HS2 or HS4, or a Product sample) and compare its Trade Agreement coefficient with the Scenario Effects XGBoost associates with Trade Agreements. This puts the econometric baseline in the notebook (course "go beyond" angle) and fits the associational-language guardrail.
- **C. A LightGBM or CatBoost bake-off on CPU.** Probably small gains (benchmarks [B2]) at a real engineering cost. Worth it only if A shows the GPU memory limit is what is hurting.
- **D. An embedding NN as an experiment, or as an *embedding feature source* for XGBoost** (Guo & Berkhahn's reuse idea [N1]). This is the most novel option and the most work. Monotonicity forces TF Lattice / WSL2 or a custom PyTorch architecture.

## Sources

Library documentation and source code
- [X1] XGBoost parameters: https://xgboost.readthedocs.io/en/stable/parameter.html
- [X2] XGBoost monotonic constraints tutorial: https://xgboost.readthedocs.io/en/stable/tutorials/monotonic.html
- [X3] XGBoost external memory tutorial: https://xgboost.readthedocs.io/en/stable/tutorials/external_memory.html
- [X5] XGBoost PR #12809 "Align the GPU monotone constraint implementation to the CPU" (merged 2026-11-06): https://github.com/dmlc/xgboost/pull/12809
- [L1] LightGBM parameters: https://lightgbm.readthedocs.io/en/latest/Parameters.html
- [L2] LightGBM `src/io/config.cpp` (monotone method fallbacks): https://github.com/lightgbm-org/LightGBM/blob/master/src/io/config.cpp
- [L3] LightGBM `src/objective/objective_function.cpp` (CUDA objective fallbacks): https://github.com/lightgbm-org/LightGBM/blob/master/src/objective/objective_function.cpp
- [L4] LightGBM issue #4781 "[CUDA] Monotonic Constraints on New CUDA Version" (open): https://github.com/lightgbm-org/LightGBM/issues/4781; CUDA learner: https://github.com/lightgbm-org/LightGBM/tree/master/src/treelearner/cuda
- [L5] LightGBM installation guide: https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html
- [C1] CatBoost regression losses: https://catboost.ai/docs/en/concepts/loss-functions-regression
- [C2] CatBoost common training parameters (`monotone_constraints`): https://catboost.ai/docs/en/references/training-parameters/common
- [C3] CatBoost `catboost/private/libs/options/catboost_options.cpp` (monotone validation): https://github.com/catboost/catboost/blob/master/catboost/private/libs/options/catboost_options.cpp
- [N2] TensorFlow Lattice overview: https://www.tensorflow.org/lattice/overview
- [N3] TensorFlow pip install (Windows GPU): https://www.tensorflow.org/install/pip
- [G3] pyfixest README: https://github.com/py-econometrics/pyfixest

Methodology and papers
- [I1] ITC, *Export Potential and Diversification Assessments: A methodology to identify export opportunities* (27 Jun 2023): https://umbraco.exportpotential.intracen.org/media/cklh2pi5/epa-methodology_230627.pdf
- [G1] Yotov, Piermartini, Monteiro & Larch (2016), *An Advanced Guide to Trade Policy Analysis: The Structural Gravity Model*, WTO/UNCTAD, ch. 1 Recommendations 1–6: https://www.wto.org/english/res_e/booksp_e/advancedwtounctad2016_e.pdf
- [G2] Correia, Guimarães & Zylkin (2019), "ppmlhdfe: Fast Poisson Estimation with High-Dimensional Fixed Effects": https://arxiv.org/abs/1903.01690
- [G4] Weidner & Zylkin (2021), "Bias and consistency in three-way gravity models", *J. Int. Econ.* 132, 103513: https://doi.org/10.1016/j.jinteco.2021.103513
- [P1] Apfel, Breinlich, Green, Novy, Santos Silva & Zylkin (2025), "Out-of-sample gravity predictions", arXiv 2509.11271: https://arxiv.org/abs/2509.11271
- [P2] Ruzicska, Chariag, Kiss & Kiss (2024), "Can Machine Learning Beat Gravity in Flow Prediction?", in Matyas (ed.), *The Econometrics of Multi-dimensional Panels*, Springer: https://doi.org/10.1007/978-3-031-49849-7_16
- [P3] Gopinath, Batarseh, Beckman, Kulkarni & Jeong (2021), "International agricultural trade forecasting using machine learning", *Data & Policy* 3: https://doi.org/10.1017/dap.2020.22
- [B1] Grinsztajn, Oyallon & Varoquaux (2022), "Why do tree-based models still outperform deep learning on tabular data?": https://arxiv.org/abs/2207.08815
- [B2] McElfresh et al. (2023), "When Do Neural Nets Outperform Boosted Trees on Tabular Data?", NeurIPS 2023 Datasets & Benchmarks: https://arxiv.org/abs/2305.02997
- [N1] Guo & Berkhahn (2016), "Entity Embeddings of Categorical Variables": https://arxiv.org/abs/1604.06737

Not verified from primary text: CatBoost GPU memory behaviour (the GPU training page doesn't cover it), and wall-clock comparisons between libraries on this hardware (none found; they would need a local run).
