# The test year is read once per frozen version, and every read is in the ledger

2023 is the test year. It is read **once per frozen version**. A version is frozen when every choice is made: features, sampling, the Model Form, hyperparameters, round count, the Selection Measure and every rule that turns a score into what Advisors see. All of them are chosen on the 2022 validation year and fixed before 2023 is opened. No 2023 result feeds back into that version. If a 2023 figure disappoints, the version is reported as it is; a fix belongs to a new version, chosen on 2022 like any other.

To **read** the test year is to measure a version against its Actual Exports: the headline, its breakdowns, Lift (whose outcome year is 2023) and the v1 comparison. The served fold models train on 2012–2023 and the Gap Period's Export Gaps average 2022–2024; that is the Baseline using 2023's data, not a read, because no test-year figure comes out of it.

"2023 is read once" has always been applied this way: v1, v2 and the Blend each read it once, after their choices were fixed. But it was written as if 2023 could never be read again. This ADR makes the rule say what was done, so a new version gets a fair comparison on the same year without turning 2023 into a second validation year.

A version is identified by its **configuration fingerprint** (`exportspredictor.ledger.fingerprint`): a hash of the metadata that decides the test-year models and how they are scored, such as the features, the Model Form, the hyperparameters, the round count, the sampling and the data source. Version names and timings are left out, so rebuilding the same configuration under a new name is still the same version, and a copied artefact keeps the fingerprint of the one it was copied from.

## The ledger

`docs/test-year-ledger.csv` records every read of the test year: date, version, fingerprint, what was read, the command, whether it was an override, and the record it comes from. It starts with the three reads made before the ledger existed, each citing its source record:

- **v1** (17 Nov 2026, `python -m exportspredictor.evaluation`). The same test-year models re-scored 2023 the same day to add the income breakdown, Lift and the Event Backtests.
- **v2** (18 Nov 2026, `python -m exportspredictor.pipeline`). A throwaway smoke run came before it, and the v1 comparison with its gain ladder after it, on the same test-year models.
- **The Blend** (18 Nov 2026, `python -m exportspredictor.pipeline --event-backtests`).

None of these re-reads changed a choice. They are recorded under their version's entry, not as new versions.

## How it is enforced

Every command that reads the test year records the read in the ledger before it opens the year (`ledger.record_read`). This covers the pipeline's evaluation, `pipeline compare`, `python -m exportspredictor.evaluation` and `python -m exportspredictor.backtests` (Lift's outcome year is 2023). If the ledger already holds the fingerprint, the command refuses. `--read-test-year-again` overrides the refusal, and the override is recorded as a read like any other, marked as an override. The ledger is committed with the work that made the read, so the number of reads is visible in the repo, the notebook and the history.

## Considered Options

- **Never read 2023 again.** This is the old wording taken literally. Every new version would then be compared with v1, v2 and the Blend on 2022, which those versions were chosen on, so the comparison would favour them.
- **Validate on 2023 and test on 2024.** This moves the time split (out of scope for spec #113). 2024 is BACI's provisional year and the Baseline year, so it would be tested while it is still being revised.
- **A rule without a ledger.** The three past reads had to be rebuilt from decision logs, notebooks and git history. A ledger costs one line per read, and a guard in code stops a second read by accident, such as re-running the pipeline after a crash.
- **Fingerprint every metadata key except a list of volatile ones.** A new timing or provenance key would then change the fingerprint and silently allow a second read. Listing the configuration keys (`FINGERPRINT_KEYS`) errs the other way: a setting left off the list makes two configurations look alike, and the second read is refused until the key is added or the override is used.

## Consequences

- **A new setting must join `FINGERPRINT_KEYS`.** A new feature changes `features` and is covered. A new kind of setting, such as a Market Reach offset, is not covered until its metadata key is added.
- **The fingerprint is checked after the fold models are trained.** The round count and the sampling rate it hashes are known only then. Fold models are cached by their settings, so a refused re-run of the same configuration costs minutes, not a retraining.
- **Library calls choose.** `build_artefacts` (the fixture) records nothing unless given a ledger. `build_real_artefacts`, `add_v1_comparison` and the `add_to_latest` commands must be given one, or `None` on purpose, as the tests on synthetic data do.
- **Spec #113's new version reads 2023 once**, after it is frozen, and reports its figures beside v1, v2 and the Blend.
