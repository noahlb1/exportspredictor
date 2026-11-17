# Research: compute and coverage facts for the training scope

Ticket: #26 (child of map #24); it feeds #32 "Training scope and sampling". Researched 2026-11-17 on this machine (RTX 3080 10 GB, Ryzen 7 5800X, 32 GB RAM, Windows 11), using XGBoost 3.4.1, LightGBM 4.7.0 and BACI HS12 V202601.

This note gathers facts. It does not choose a scope or a rate. That is #32's job.

## Bottom line

1. **The v1 GPU limit comes from how the matrix is built, not from the GPU itself.** v1 builds its `QuantileDMatrix` from NumPy (host) chunks. Training it on CUDA then peaks at about **150 B per row** of VRAM. Feeding the same chunks as CuPy (device) arrays builds the matrix on the GPU, and the peak drops to **74 B per row**: 33 B for the matrix itself and about 41 B of working memory during training. The fixed cost is about 1.35 GB (the desktop plus the CUDA context). On that path **110 M rows trained at a 9.1 GB peak**. All 152 M rows would need about 12 GB, so they do not fit.
2. **GPU in-core speed is about 2.7–3.4 ns per row per round** (depth 8, Tweedie, monotone constraints). That is 0.29 s per round at 110 M rows. The v1 fold training matches this: 48.6 M rows, 1491 rounds, 5 folds in 1233 s, so 0.165 s per round.
3. **XGBoost GPU external memory works on Windows, but slowly.**
   - All 152 M rows built in 28 s and trained at **5.45 s per round**. That is about 36 ns per row-round, roughly **12× slower** than in-core.
   - It uses about 9.9 GB of VRAM and 10 GB of host RAM.
   - The CUDA async pool, which the docs call crucial for external memory, fails with "Cannot enable `use_cuda_async_pool` on Windows". The PyPI wheel has no RMM.
4. **CPU `hist` in XGBoost is about 50 ns per row-round with 16 threads** (15–18× slower than the GPU). The v1 iterator path holds about **230–270 B per row** of host RAM, which caps it at about 100 M rows in 32 GB. CPU external memory runs at 176 ns per row-round.
5. **LightGBM on CPU is the most memory-frugal option: about 27 B per row**. All 152 M rows would take about 4 GB. Speed is about **34 ns per row-round** at 8 threads, and 16 threads is no faster.
   - The PyPI wheel on Windows has **no GPU learner**. `device_type="gpu"` fails with "GPU Tree Learner was not enabled in this build".
   - The CUDA learner is not supported on Windows.
6. **Top-50 economies**, as Exporter and Importer both (2012–2023 pooled):

   | Ranked by | World trade | UK exports | Non-zero Flows | Relevant zero cells |
   |---|---|---|---|---|
   | GDP | 82.4% | 92.6% | 51.4 M (39% of 131.0 M) | 128 M (11% of 1,203 M) |
   | Imports | 83.0% | 92.9% | 55.1 M | 134 M |

   Restricting only the Importers (Markets) to the top 50 keeps **90–91% of world trade** and 67–69 M non-zero Flows.
7. **Rows at a given rate.** At the v1 zero rate of 2%, a top-50 scope has 54–75 M training rows with no non-UK sampling. That is inside the 110 M GPU budget. The 1-in-6 zero ratio is kept. The full world at 2% zeros is 155 M rows, which needs about a 70% non-UK sample to fit in 110 M. See section 5 for the full table.

## 1. Baseline: what v1 trained on

These are measured from `data/derived/features_V202601/training_2012..2023.parquet` and from the metadata of artefact `20261117T022550Z`.

| Fact | Value |
|---|---|
| Training rows, 2012–2023 | 152,437,263 |
| … of which zero cells (2% sample, weight 50) | 21,434,222 |
| … of which UK rows | 4,148,393 |
| Rows with a missing value (`log_gdp_o` null) | 573,202 |
| Rows actually trained at a 30% non-UK sample | 48.6 M |
| Tuning, 5 configs, early stopping, 2012–2021 | 498 s |
| 5 fold models, 1491 rounds each, depth 8 | 1233 s (0.165 s per round, about 3.4 ns per row-round) |
| Best config | depth 8, eta 0.1, subsample 0.8, colsample 0.8, Tweedie p = 1.3 |

## 2. Measured compute benchmarks

All runs use the v1 model parameters: Tweedie p = 1.3, depth 8, eta 0.1, subsample 0.8, colsample 0.8, `max_bin` 256, and +1 monotone constraints on the 8 policy features. The data is real rows from the feature tables.

GPU memory was polled with `nvidia-smi` every 0.1 s. The "base" figure is about 1.2–1.4 GB used by the desktop before the process starts. Rounds were timed after a warm-up, except where noted. The scripts were throwaway, so this section records their method and results.

### 2.1 XGBoost GPU, in-core

**Host-built `QuantileDMatrix`.** This is the v1 path: NumPy arrays and `device="cuda"` only at train time.

| Rows | Build | s / round | ns / row-round | Peak VRAM over base |
|---|---|---|---|---|
| 5 M | 2.9 s | 0.030 | 6.0 | 121 B/row |
| 10 M | 5.8 s | 0.043 | 4.3 | 162 B/row |
| 20 M | 12.0 s | 0.067 | 3.3 | 151 B/row |
| 30 M | 17.3 s | 0.094 | 3.1 | 148 B/row |

- The VRAM barely moves when the matrix is built, by about 200 MB. The matrix is built on the host and moved to the GPU at the first round, which is where v1 "filled the VRAM at the first round".
- The marginal cost is about 135–155 B per row. Changing `max_bin` to 64, setting `max_depth` to 6, or using `gradient_based` sampling at 0.2 did not change the peak: all stayed at 145–155 B per row.
- At about 8.7 GB free, this path caps out near **55 M rows**. v1's 48.6 M rows fit inside that.

**Device-built `QuantileDMatrix`.** A `DataIter` yields 5 M-row CuPy chunks. Each chunk is freed after it is consumed.

| Rows | Build | s / round | ns / row-round | Matrix | Peak over base | Peak VRAM |
|---|---|---|---|---|---|---|
| 20 M | 1.4 s | 0.066 | 3.3 | 33.0 B/row | 106 B/row (build transient) | 3.7 GB |
| 40 M | 2.9 s | 0.113 | 2.8 | 32.8 B/row | 75 B/row | 4.6 GB |
| 80 M | 9.0 s | 0.236 | 2.9 | 33.0 B/row | 74 B/row | 7.0 GB |
| **110 M** | 9.1 s | **0.295** | 2.7 | 32.8 B/row | 74 B/row | **9.1 GB** |

- **The matrix takes 33 B per row.** This agrees with XGBoost's source: dense ELLPACK uses `ceil(log2(bins + 1))` bits per element when some values are missing. Our data has some missing values (`log_gdp_o`), so it uses 9 bits × 22 features, about 24.75 B, plus 8 B for the float32 label and weight. Removing the missing values, or using `max_bin` ≤ 255, would bring it to 8 bits, about 22 + 8 = 30 B per row. That last figure is inferred from the source and was not measured.
- **Training adds about 41 B per row** (gradients, row partitions, sampling).
- **Variants measured at 40 M rows:**
  - No subsampling: 73 B per row.
  - `gradient_based` sampling at 0.2: 104 B per row, because it keeps a compacted copy, and slower at 3.2 ns per row-round. In-core, it costs memory rather than saving it.
- **Planning figure:** 1.35 GB + 74 B per row, against 10.24 GB. That gives a hard ceiling of about 120 M rows and a comfortable one of about **110 M**. All 152 M rows would need about 12.6 GB.
- Each build reads at most one chunk at a time, so host RAM stays at about one year of rows.
- **The CuPy dependency.** CuPy (`cupy-cuda12x`) was installed only in an ephemeral overlay for these tests (`uv run --with`). It is not a project dependency. A PyTorch or DLPack tensor would serve the same purpose.

### 2.2 XGBoost GPU external memory (`ExtMemQuantileDMatrix`)

| Test | Result |
|---|---|
| 10 M rows, 2 M-row batches, default pool | Works, at 4.6 ns per row-round. Everything fits in VRAM, so this is no faster than in-core. |
| **All 152,437,263 rows**, one year per batch, host cache (`on_host=True`) | Built in **28.5 s**. First round 6.1 s, then **5.45 s per round** (35.7 ns per row-round). Peak VRAM 9.87 GB, peak host RSS 10.0 GB. |
| `use_cuda_async_pool=True` | **Fails:** `Check failed: !use_async_pool: Cannot enable use_cuda_async_pool on Windows.` |
| Warnings printed | "CUDA heterogeneous memory management is not available", and "Neither `use_rmm` nor `use_cuda_async_pool` is enabled. The overhead of iterating through external memory might be significant." |

All 152 M rows therefore train on this machine. At 1500 rounds that is about **2.3 h per model**.

The docs agree with this result:
- In-core "can be an order of magnitude faster than external memory".
- Over PCIe, batch-based training "can be at least five times slower than in-core".
- The async pool or RMM is "crucial", and the async pool says "Windows is not supported yet".

Sources: https://xgboost.readthedocs.io/en/stable/tutorials/external_memory.html and https://xgboost.readthedocs.io/en/release_3.1.0/tutorials/external_memory.html. The v3.0 release notes say the PyPI wheel lacks RMM: https://xgboost.readthedocs.io/en/stable/changes/v3.0.0.html.

The page-concatenation option `extmem_single_page`, which was documented for use with gradient-based subsampling, was **removed in 3.2.0**, so it is not available in our 3.4.1.

### 2.3 XGBoost CPU `hist` (16 threads)

| Test | Rows | Build | s / round | ns / row-round | Host RAM |
|---|---|---|---|---|---|
| In-memory NumPy | 10 M | 5.5 s | 0.47 | 46.6 | +141 B/row after build, +182 B/row peak |
| In-memory NumPy | 20 M | 10.8 s | 0.99 | 49.6 | about +140 B/row peak over the loaded data |
| **v1 iterator** (`training.training_matrix`, years 2021–2023) | 40.7 M | 35.3 s | 2.12 | 52.1 | **232 B/row** after build, **272 B/row** peak in training |
| `ExtMemQuantileDMatrix`, CPU | 10 M | 9.9 s | 1.76 | 176 | n/a |

- In XGBoost's source, the CPU bin index is uint8 for dense data with 256 or fewer bins. Each row also carries a `size_t` row pointer and a `ColumnMatrix` for `hist`.
- The resident size we measured is much larger than that. Part of it is likely allocator retention after the iterator's Polars and NumPy chunks are freed, so RSS overstates the live size. Even so, the RSS is what has to fit in RAM.
- **Planning figure:** at about 272 B per row, about **100 M rows** fit in 28 GB. All 152 M rows would need about 41 GB.
- **Time:** about 5.2 s per round at 100 M rows, which is about 2.2 h per 1500-round model.

### 2.4 LightGBM 4.7.0, CPU

These runs used the same objective and constraints: `num_leaves` 255, `max_depth` 8, bagging 0.8, feature fraction 0.8, `max_bin` 255, and the `basic` monotone method. The data was 10 M rows.

| Threads | Dataset build | s / round | ns / row-round | Dataset | Peak over the loaded data |
|---|---|---|---|---|---|
| 8 | 0.9 s | 0.342 | 34.2 | about 27 B/row | +26 B/row |
| 16 | 0.9 s | 0.358 | 35.8 | about 27 B/row | +28 B/row |

- The bins take 1 byte per feature-value with 256 or fewer bins, and 4 bits with 16 or fewer (`CreateDenseBin` in https://github.com/microsoft/LightGBM/blob/master/src/io/bin.cpp). The Parameters page recommends `num_threads` = the number of physical cores, and our measurement agrees: 16 threads gave no gain. (https://lightgbm.readthedocs.io/en/stable/Parameters.html)
- **Planning figure:** all 152 M rows are about 4.1 GB as a Dataset. The raw float32 matrix during construction is about 13.4 GB, unless it is built from chunks with `lgb.Sequence`.
- **Time:** about 5.2 s per round for 152 M rows, about 2.2 h per 1500-round model. It is not directly comparable to XGBoost: LightGBM grows trees leaf-wise, and this used the `basic` monotone method, which "over-constrains".
- **GPU.** `device_type="gpu"` (OpenCL) fails on the PyPI wheel on this machine with "GPU Tree Learner was not enabled in this build. Please recompile with CMake option -DUSE_GPU=1".
  - The LightGBM docs say the CUDA learner is not supported on Windows ("For macOS and Windows users, the CUDA version is not supported", https://github.com/microsoft/LightGBM/blob/master/python-package/README.rst).
  - So LightGBM is CPU-only here unless built from source.
  - The docs do describe the OpenCL learner as available on Windows when built with `-DUSE_GPU=1`, but this was not tested.

### 2.5 Summary: per-row cost, capacity and time per model

The time per model assumes 1500 rounds, the v1 best of 1491. Each tuning run or fold model is one model; v1 trained 5 + 5.

| Option | Memory per row | Rows that fit | ns / row-round | 1500 rounds at 110 M | 1500 rounds at 152 M |
|---|---|---|---|---|---|
| XGBoost GPU in-core, host-built (v1) | about 150 B VRAM | about 55 M | about 3.1–3.4 | does not fit | does not fit |
| XGBoost GPU in-core, device-built | 74 B VRAM | **about 110 M** | about 2.7–2.9 | **about 7.5 min** | does not fit |
| XGBoost GPU external memory | host cache about 33 B, plus RSS | 152 M (measured) | 35.7 | about 1.6 h | **about 2.3 h** |
| XGBoost CPU `hist`, in-core (v1 iterator) | about 272 B RAM | about 100 M | about 50 | about 2.3 h | does not fit |
| XGBoost CPU external memory | n/a | n/a | 176 | about 8 h | about 11 h |
| LightGBM CPU (8 threads) | about 27 B RAM | 152 M and more | about 34 | about 1.6 h | about 2.2 h |

## 3. Top-50 economy sets

The method was a pooled scan of `data/derived/baci_hs12_V202601/flows_<year>.parquet`, 2012–2023, keeping only non-zero Flows. BACI codes were mapped to ISO3 with `country_map.parquet` (`S19` becomes `TWN`). BACI has 227 economies.

- **By GDP:** mean `gdp_usd` 2012–2023 from `country_year.parquet`, restricted to economies present in BACI.
- **By imports:** total BACI imports 2012–2023.

The two sets share 46 economies, and the UK is in both.

- Both sets: ARE ARG AUS AUT BEL BGD BRA CAN CHE CHL CHN CZE DEU DNK EGY ESP FIN FRA GBR HKG IDN IND IRL ISR ITA JPN KOR MEX MYS NLD NOR PAK PHL POL PRT ROU RUS SAU SGP SWE THA TUR TWN USA VNM ZAF
- Only in the GDP set: COL IRN IRQ NGA
- Only in the imports set: GRC HUN SVK UKR

## 4. Coverage and remaining rows

There are two ways to restrict:
- **both**: Exporter and Importer must both be in the set, and the model only ever sees trade among the top 50.
- **importer**: only the Market must be in the set, and every Exporter is kept.

A **relevant zero cell** is counted within the restricted data, by the rule in `features.training_rows`. Its Exporter ships the Product to some Importer in scope, and its Importer buys it from some Exporter in scope, but not from each other. Counting is exact per Product: Σ_p (|E_p| · |I_p| − |E_p ∩ I_p|) − non-zero.

**2012–2023 pooled**

| Scope | Share of world trade (USD) | Share of UK exports (USD) | Non-zero Flows | Relevant zero cells | UK non-zero | UK relevant zeros |
|---|---|---|---|---|---|---|
| All 227 economies | 100% | 100% | 131.0 M | 1,202.7 M | 4.02 M | 6.23 M |
| Top-50 GDP, both | 82.4% | 92.6% | 51.4 M | 128.1 M | 1.96 M | 0.95 M |
| Top-50 GDP, importer | 90.1% | 92.6% | 67.1 M | 280.4 M | 1.96 M | 0.96 M |
| Top-50 imports, both | 83.0% | 92.9% | 55.1 M | 133.6 M | 2.04 M | 0.90 M |
| Top-50 imports, importer | 90.7% | 92.9% | 69.1 M | 280.8 M | 2.04 M | 0.91 M |

**2023 only**

| Scope | Non-zero | Relevant zeros | World trade | UK exports | Share of non-zero rows | Share of relevant zeros |
|---|---|---|---|---|---|---|
| All | 11.76 M | 106.2 M | 100% | 100% | 100% | 100% |
| GDP, both | 4.41 M | 10.7 M | 81.8% | 92.7% | 37.6% | 10.1% |
| GDP, importer | 5.92 M | 24.3 M | 89.8% | 92.7% | 50.4% | 22.8% |
| Imports, both | 4.76 M | 11.1 M | 82.8% | 93.1% | 40.5% | 10.5% |
| Imports, importer | 6.13 M | 24.3 M | 90.6% | 93.1% | 52.2% | 22.9% |

What this shows:
- A top-50 scope **keeps over 92% of UK export value but under half of the UK's non-zero Flows** (1.96 of 4.02 M). The UK Markets outside the top 50 are many and small.
- A top-50 scope also shrinks the UK scoring grid from about 227 Markets to 49, unless scoring and training scopes differ.
- The zero cells shrink far more than the Flows (to 10–23%), because relevance is quadratic in the number of economies.
- The full world has 1,203 M relevant zeros, or about 100 M a year. That matches the v1 docstring figure of about 95 M.

## 5. Zero and non-UK sampling rates that fit

Rows at zero rate *z* and non-UK rate *r* = UK non-zero + *z* · UK zeros + *r* · (non-UK non-zero + *z* · non-UK zeros). UK rows are always kept in full, as in `training.sample_non_uk`. At *z* = 2% on the full world this formula gives 155.1 M rows. The feature tables actually hold 152.4 M, because a few rows are lost in the feature build, so treat the table as accurate to about ±2%.

The budget columns show the largest non-UK rate *r* that fits each budget:
- 110 M: XGBoost GPU, device-built.
- 100 M: XGBoost CPU in-core.
- 55 M: XGBoost GPU, host-built (v1).

GPU external memory and LightGBM CPU are bounded by time, not memory, at every row count shown.

| Scope | Zero rate | Rows at r = 1 | UK rows | Max r, 110 M | Max r, 100 M | Max r, 55 M |
|---|---|---|---|---|---|---|
| All | 0.02 | 155.1 M | 4.15 M | 0.70 | 0.64 | 0.34 |
| All | 0.05 | 191.1 M | 4.34 M | 0.57 | 0.51 | 0.27 |
| All | 0.10 | 251.3 M | 4.65 M | 0.43 | 0.39 | 0.20 |
| All | 0.25 | 431.7 M | 5.58 M | 0.25 | 0.22 | 0.12 |
| All | 1.00 | 1,333.7 M | 10.25 M | 0.08 | 0.07 | 0.03 |
| GDP, both | 0.02 | 54.0 M | 1.98 M | 1 | 1 | 1 |
| GDP, both | 0.05 | 57.8 M | 2.01 M | 1 | 1 | 0.95 |
| GDP, both | 0.10 | 64.2 M | 2.06 M | 1 | 1 | 0.85 |
| GDP, both | 0.25 | 83.4 M | 2.20 M | 1 | 1 | 0.65 |
| GDP, both | 1.00 | 179.4 M | 2.91 M | 0.61 | 0.55 | 0.30 |
| GDP, importer | 0.02 | 72.7 M | 1.98 M | 1 | 1 | 0.75 |
| GDP, importer | 0.05 | 81.1 M | 2.01 M | 1 | 1 | 0.67 |
| GDP, importer | 0.10 | 95.1 M | 2.06 M | 1 | 1 | 0.57 |
| GDP, importer | 0.25 | 137.2 M | 2.20 M | 0.80 | 0.72 | 0.39 |
| GDP, importer | 1.00 | 347.5 M | 2.92 M | 0.31 | 0.28 | 0.15 |
| Imports, both | 0.02 | 57.7 M | 2.05 M | 1 | 1 | 0.95 |
| Imports, both | 0.05 | 61.7 M | 2.08 M | 1 | 1 | 0.89 |
| Imports, both | 0.10 | 68.4 M | 2.13 M | 1 | 1 | 0.80 |
| Imports, both | 0.25 | 88.5 M | 2.26 M | 1 | 1 | 0.61 |
| Imports, both | 1.00 | 188.7 M | 2.93 M | 0.58 | 0.52 | 0.28 |
| Imports, importer | 0.02 | 74.7 M | 2.05 M | 1 | 1 | 0.73 |
| Imports, importer | 0.05 | 83.1 M | 2.08 M | 1 | 1 | 0.65 |
| Imports, importer | 0.10 | 97.1 M | 2.13 M | 1 | 1 | 0.56 |
| Imports, importer | 0.25 | 139.3 M | 2.26 M | 0.79 | 0.71 | 0.38 |
| Imports, importer | 1.00 | 349.9 M | 2.94 M | 0.31 | 0.28 | 0.15 |

The tuning runs read 2012–2021 (10 of the 12 years), so they use about 80% of these rows. Fold model *k* drops about 20% of UK rows, which is negligible.

**Time, at the measured rates** (1500 rounds per model):

| Scope, rates | Rows | GPU in-core, device-built | GPU external memory | LightGBM CPU | XGBoost CPU |
|---|---|---|---|---|---|
| Top-50, z = 2%, r = 1 | 54–75 M | 4–5 min | n/a (fits in-core) | 46–64 min | 68–94 min |
| Top-50, z = 25%, r = 1 | 83–139 M | 6–7 min where it fits | 1.2–2.1 h | 1.2–2 h | 1.8 h where it fits |
| All, z = 2%, r = 0.7 | 110 M | 7.5 min | n/a | 1.6 h | does not fit |
| All, z = 2%, r = 1 | 155 M | does not fit | 2.3 h | 2.2 h | does not fit |

## 6. Other documented facts

- **Monotone constraints** are supported by all three XGBoost tree methods (https://xgboost.readthedocs.io/en/stable/treemethod.html). They were active in every XGBoost run above, on GPU, CPU and external memory. With `hist`, the docs warn that constraints "may produce unnecessarily shallow trees", and a higher `max_bin` helps (https://xgboost.readthedocs.io/en/stable/tutorials/monotonic.html).
- **LightGBM monotone methods** are `basic`, `intermediate` and `advanced`. `basic` is fastest but over-constrains (https://lightgbm.readthedocs.io/en/stable/Parameters.html).
- **XGBoost on Windows.** The PyPI wheel includes CUDA, and multi-GPU training is Linux-only (https://xgboost.readthedocs.io/en/stable/install.html). Our build reports `USE_CUDA=True` and `USE_RMM=False`.
- **`subsample` with `gradient_based` sampling** "may be set to as low as 0.1 without loss of model accuracy" (https://xgboost.readthedocs.io/en/stable/parameter.html). The memory it saves applies to external memory. In-core on the GPU we measured it using *more* memory (section 2.1).

## Not confirmed

- Training memory at all 152 M rows on GPU in-core was extrapolated from linear fits at 20–110 M rows. It was not run.
- Whether XGBoost CPU RSS of 232–272 B per row is live data or allocator retention. A run with a different allocator, or the Python path without Polars, might measure less.
- LightGBM at more than 10 M rows, and LightGBM built from chunks with `lgb.Sequence`. Both are extrapolated linearly.
- LightGBM's OpenCL GPU learner built from source on Windows.
- Model quality under any of these options. These are compute facts only; #32 decides.
