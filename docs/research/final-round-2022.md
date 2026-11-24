# The final round of experiments on the 2022 validation year

## Wider Exporter scope: feature files

For spec #155, ticket #168, under [ADR 0004](../adr/0004-training-scope-and-sampling.md). v2 trains on the top 100 Exporters by mean GDP; v1 trained on the whole world. The wider scope tests whether that cut caused v2's −0.020 drop from v1. This section covers only the feature files. The experiment that trains on them is #169.

**The feature set is `all_exporters`**: `exporter_count` (#126: the Exporter count as a candidate feature, GDP filled for small Markets) with **every Exporter's rows**, 227 Exporters in place of 100. All Exporters were chosen over a larger top N because they test the hypothesis directly: v1's scope against v2's.

- Built with one command, on the CPU, in 40.4 minutes: `scripts/wt.ps1 python -m exportspredictor.features --set all_exporters --parts 8`. The set's scope comes from `features.FEATURE_SET_SCOPES` (`scope.ALL_EXPORTERS`).
- Peak RAM was **11.6 GB** (the build process's peak working set, sampled every 5 s). Fitting Product proximity alone reached 9.2 GB; building one year in 8 groups of Markets reached 11.6 GB. Other programs held about 22 GB of the 32 GB when it started. The dense Exporter × Importer × Product grid is never formed: Zero Cells come only from Exporters and Importers active in the Product that year, as before.
- The years are 2012–2024 like `exporter_count`, so the pipeline can serve the set if it is kept. The loop never opens 2023 or 2024: `loop.real_data` refuses the test year, and a test runs the real path on this set.
- The files take 7.0 GB on disk, against 5.9 GB for `exporter_count`.

**The top 100's rows are unchanged.** In 2021 and 2022, the rows of the 100 in-scope Exporters are the same rows as `exporter_count`'s, with the same features, and the UK grids are identical. The only difference is in Relatedness: float32 rounding, at most 6.6e−7, because the matrix product runs on a matrix with more Exporter rows. So the wider files equal the current ones plus the other 127 Exporters' rows, and #169's comparison isolates the scope.

### Row counts and sampling rates

Zero Cells of other Exporters are kept at 5%, as in every set (weight 20). Every UK row is kept at weight 1. The files are **not** sampled by Exporter: the loop samples other Exporters' rows at the non-UK rate when it trains (`training.sample_non_uk`, weight 1/rate). That keeps the unsampled files for the GPU external memory Arm, and lets the noise floor's seeds draw different samples.

| | `exporter_count` (top 100) | `all_exporters` | Change |
|---|---|---|---|
| Exporters | 100 | 227 | +127 |
| UK rows, 2012–2021 | 8,445,954 | 8,445,954 | 0 |
| Other Exporters' rows, 2012–2021 | 125,565,413 | 147,263,996 | +17.3% |
| of which Flows | 97,070,202 | 103,682,812 | +6.8% |
| of which Zero Cells (sampled at 5%) | 28,495,211 | 43,581,184 | +52.9% |
| Validation year 2022 (never sampled) | 14,246,975 | 16,765,385 | +17.7% |
| Non-UK rate that fits the VRAM budget | 0.6157 | **0.5079** | |
| Expected rows on the GPU at that rate | 100.0 M | 100.0 M | |

The non-UK rate is `search.final_non_uk_rate`, the rule the loop applies when an Arm leaves `non_uk_rate` unset. It is sized for the pipeline's VRAM budget (`training.row_budget`, 100,032,198 rows with 25 features: 7.5 GiB above an idle desktop). The same rule gives `exporter_count` 0.6157, the rate v2 trains at, which checks the method.

Most of the extra rows are Zero Cells, as ADR 0004 predicted: small Exporters add a Zero Cell for every Importer of each Product they ship. The extra Flows add 7% more Flow rows but carry little trade.

### For the training run (#169)

- Set `EXPORTSPREDICTOR_FEATURE_SET=all_exporters`. An Arm on these files needs `non_uk_rate` at 0.5079 or below (or `None`, which derives it). The best so far's 0.6157 would put about 115.9 M rows on the GPU, over the budget. The loop does not check an explicit rate against the budget (only the pipeline calls `training.check_row_budget`), so the Arm must change it.
- At 0.5079 the training matrix holds about **83.2 M rows** (8.4 M UK plus 74.8 M sampled others), and the whole 2022 year adds 16.8 M, for 100.0 M in all: the same size as v2's. One fold-0 model should take about as long as v2's, roughly 11 minutes on a quiet machine.
- That budget assumes most of the 10 GB is free. When the build ran, the user's API and desktop apps held about 6.8 GB, which leaves too little for 100 M rows. Check `nvidia-smi` before training.
- The comparison is then at a lower non-UK rate (0.51 against 0.62) as well as a wider scope. Keeping the rate fixed would need GPU external memory, which is the spec's idea 6.

These are build records only. Nothing here was trained or scored, and 2023 was not read.
