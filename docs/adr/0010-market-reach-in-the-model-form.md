# Market Reach scales Export Potential inside the Model Form, learned out-of-fold from the year before

The ranking did not know where the UK already sells. The open review of 2022 found that most of the Selection Measure's loss lies in *which* Markets get any UK trade at all. So the served Model Form now carries **Market Reach**: how widely the UK sends its Products to a Market, against how widely the ranking expects it to, learned from the UK's other Products. Export Potential becomes

Export Potential × exp(β · offset) × c,

where the offset is the Market's, β = 4, and c is one constant that keeps the UK's summed Export Potential over the scored grid unchanged. #124 tested it on the 2022 validation year under the tie-proof Selection Measure. The pooled five-fold measure rose from 0.5635 to 0.6041, a gain of +0.0406 against a Win Threshold of 0.008869 ([market-reach-2022.md](../research/market-reach-2022.md)). The form, the β grid, the pseudo-count and the gate were fixed before any result was read.

**The offset (the open review's share form).** For fold *k* and scored year *t*, Market *m*'s offset is learned from the UK grid pairs **outside fold *k*** in year ***t* − 1**:

offset = log((sent + 5) ÷ (imported + 10)) − log((Σ sent + 5) ÷ (Σ imported + 10)).

"Imported" counts the Market's pairs whose Product the Market imports from other Exporters, "sent" those of them with a UK Flow too, and the sums run over every Market. (Sent was first not nested in imported; see the *Amendment* below.) An offset of 0 means "as widely as the UK reaches Markets overall". "Products expected" is the pooled share times "imported". `exportspredictor.market_reach` computes it, and `prediction.export_potential` applies it.

## The hold-out unit is the pair, and the offset is cross-fitted and lagged

[ADR 0001](0001-out-of-fold-uk-training.md) protects one thing: no model reads a UK pair's own Actual Exports, so the Export Gap is not shrunk by memorisation. The hold-out unit is the **Product–Market pair**, not the Market. Market Reach keeps that unit:

- **Cross-fitted.** A pair is scored with its own fold's offsets, which are learned only from pairs in other folds. None of its own Flows, in any year, enters them.
- **Lagged.** Only year *t* − 1 is read, so no UK Flow of the scored year enters any offset. The Baseline year 2024 takes its offsets from 2023, the Gap Period's 2022 from 2021.

The fixture pipeline tests both through the served path. Changing a pair's own Flows, or every UK Flow of the Baseline year, leaves its fold's offsets unchanged.

The offset *does* read UK Flows into the target pair's own Market, in other Products. That is the exception to CLAUDE.md's leave-one-out guardrail ("any feature built from UK data excludes the target pair's own Market"). The guardrail still holds for every model **feature**: Market Reach is post-processing in the Model Form, not a feature. No fold model is trained on it, and it never enters a tree. So the rule that features carry no pair trade in other Products still stands.

## What an Export Gap now means

Before, an Export Gap mixed two things: the UK under-selling one Product in a Market, and the UK under-selling *everything* in that Market. Market Reach moves the Market-wide part into Export Potential. A Market the UK reaches narrowly gets a lower Export Potential for all its Products, and a broadly reached Market a higher one. **An Export Gap is now specific to the Product within its Market.**

The Market-wide part is not hidden. The API serves each Market's reach (UK Products sent, imported, expected, offset) on the Market view and in the pair detail. The app shows it as one sentence in the Leaderboard's Market filter view and in the detail panel: "In 2023 *Market* imported M Products from other Exporters and the UK sent N of them; at the UK's usual reach across Markets, about E were expected." The counts are over every UK grid pair of the reference year, so each fold's counts, which are four-fifths of a Market, are summed and divided by four. The language stays associational: the model associates broad UK reach into a Market with higher Export Potential there. Reach does not cause exports.

## Where it lives

- **The Model Form** (`prediction.ModelForm.market_reach`) records β, the form, the pseudo-count, the lag, the reference year and c. The Market Reach table (fold, Market, Products sent, imported and expected, offset) is a required artefact table, so the **artefact format is 4**. `artefacts.load` refuses format 3 with the usual message.
- **Every Export Potential passes through it**: the Gap Period's yearly scores (each year with its own table and c), the Baseline, the Scenarios, the Blocked Opportunity values and the explanations. Its log multiplier goes into the explanation's base value, since it is the same for every Product of the Market.
- **c is computed once per scored year**, on that year's grid scored without Market Reach. Every Scenario reuses the Baseline year's c, so a Scenario's relative effect is unchanged. The Naive Baseline is unchanged too.
- **Other Exporters' rows get no Market Reach**: no offset and no c. These are the Potential Range's calibration rows ([ADR 0007](0007-potential-range-calibrated-on-other-exporters.md)), and other Exporters' rows in the evaluation. c belongs to the UK grid: on the real Baseline it is about 0.06, because most UK pairs sit in Markets the UK reaches broadly. A first build that scaled the calibration rows by it moved the top band's whole range above Export Potential (offsets +1.9 to +4.1 on the log scale). So the range is calibrated on the Blend as before. It does not include the spread that Market Reach adds to UK pairs. Every build (and `rereach`) records how far the range moves on UK pairs: the share of UK pairs whose Actual Exports lie inside their range, with and without Market Reach (`market_reach_pair_coverage` in the metadata, `uk_pair_coverage` in the evaluation's Potential Range block). More than five points below the 80% target is a known limitation (73.6% on the served Baseline, from 80.9%).
- **The test-year evaluation fits Market Reach again for the test year**, from the year before it, with the test-year models, and reports the same figures without it beside them (`market_reach_effect`): the headline, the old measure and the large-Market measures.
- **Lift is measured on the served form.** Lift's held-out Exporters are not the UK, so each takes its own Market Reach, learned the same way from its own pairs of the year before each Gap Period year (outside each pair's fold, counted by row weight), with its own c. Lift without Market Reach is reported beside it as the kept change's effect, and the pre-registered v1 comparison reads that one.
- **A Scenario's relative effect is unchanged by Market Reach**, which scales a pair's Export Potential with and without a Scenario alike. Its dollar effect is not: it scales with the Market's multiplier, so a Blocked Opportunity value and the $1M floor it must clear follow Market Reach (see the *Amendment*). So the Event Backtests and the PPML benchmark, which compare Scenario Effects, measure the model without it, and the evaluation says so. The v1 comparison scores the Blend without Market Reach, as pre-registered, and its verdict says so.
- **The Configuration Fingerprint** ([ADR 0009](0009-test-year-read-once-per-frozen-version.md)) hashes Market Reach's settings (`market_reach` in the metadata). It leaves out the fitted c, which a rebuild could change in its last digits.
- **The served artefact was not retrained.** `python -m exportspredictor.pipeline rereach` scores the newest version again from its own fold models and writes a format-4 version. Its test-year figures stay those of the Blend without Market Reach, until the next test-year read.

## Considered Options

- **A model feature.** A per-Market reach figure as a feature would let the trees use it, but it would carry UK pair trade in other Products into every fold model. That breaks the leave-one-out rule for features, and the hold-out would then need the Market as its unit. Post-processing keeps the models as they are.
- **The open review's rank form** (observed share against the share the ranking expects). Lagged one year, it needs every pair's out-of-fold Export Potential of the previous year, which is one more scoring pass in serving. The review measured the two forms within 0.002 of each other.
- **β = 1.** This gives +0.030 with almost no change in dollar accuracy. β = 4 widens the 10th–90th spread of the log residual above $10m from 3.09 to 3.74. β = 4 was chosen because the pre-registered rule picks on the Selection Measure, which [ADR 0008](0008-opportunity-ranking-is-the-one-objective.md) makes the one objective. β = 4 is the top of the grid.

## Consequences

- **Dollar accuracy is traded for ranking.** Large pairs' Export Potential spreads further from their Actual Exports. The summed figure is unchanged, so totals and the "about N% of Actual Exports" note do not move.
- **Small Markets the UK barely reaches** take large negative offsets, so their Products drop down the Leaderboard. The Market sentence says why.
- **Offsets lag a year.** A Market that opened or closed to UK goods in the Baseline year shows last year's reach. Sanctioned Markets carry their own caveat whatever their offset.
- **A Gap Period year needs its year before.** The pipeline reads the UK grid of the year before the first Gap Period year, and the evaluation reads that of the year before the test year.

## Amendment: sent is nested in imported, and dollar figures follow Market Reach

**The defect.** The first form counted every UK Flow as sent, whether or not the Market imported the Product from anyone else. For every Market but one the two sets nearly coincide. The Falkland Islands reports no imports of its own, so BACI sees its imports only through Exporters' declarations, and many small UK shipments have no other Exporter beside them. On the served Baseline it had about 920 Products sent against about 300 imported (a "share" of about 3), an offset of 2.06 to 2.13 in every fold (Ireland, the next, 0.92), and a multiplier of about 230×. Its summed Export Potential was $1.95bn against $130M of Actual Exports and $8.45M before Market Reach, and it was the fourth Market by Blocked Opportunity value ($172M, 55 pairs), all of it scaled up by the multiplier.

**The fix.** Sent counts only the pairs that imported also counts, so the share is at most 1 and the offset is bounded. The form is recorded as `nested_share`; β, the pseudo-count and the lag are unchanged. The Market sentence now reads N of M.

**Re-checked on 2022 under the same pre-registered procedure** (`scripts/research/market_reach_2022/run.py`, [market-reach-2022.md](../research/market-reach-2022.md)). β = 4 is chosen again on folds 0–2, and the pooled five-fold Selection Measure is 0.6084 against the Blend's 0.5635, a gain of +0.0449 against a Win Threshold of 0.008869 (the first form gave +0.0406). Every fold gains (+0.044 to +0.045). The Falkland Islands' 2022 offset falls from 2.06 to 0.35; every other Market moves by about 0.01. The test year was not read.

**Dollar figures follow Market Reach.** c keeps only the UK's grand total, so each Market's dollar level moves with its multiplier, and so does every dollar figure built on Export Potential: a dollar Scenario Effect, a Blocked Opportunity value and the $1M floor it must clear. This is kept deliberately (the alternative, a Blocked Opportunity value without Market Reach, would show a pair two Export Potentials on two pages). It means Market Reach decides much of which Markets have Blocked Opportunities: on artefact `20261120T133841Z`, before Market Reach, ADR 0011's rule gave 58 Markets; with the first form, 18. Markets the UK barely reached the year before take large negative offsets and are pushed towards $0 (Algeria, Niue, Saint Helena), which is a known limitation recorded as an open question, not fixed here.

**Rebuilt, not retrained.** `python -m exportspredictor.pipeline rereach` wrote the served version from the saved fold models (`20261122T223312Z`, from `20261122T133316Z`). Under ADR 0011's rule it lists 2,128 Blocked Opportunities, $18.52bn, in 16 Markets, against 2,169, $18.45bn and 18 before; the Falkland Islands' summed Export Potential is $1.1M, against $1.95bn before, and it has none. Gibraltar's one pair also drops below the floor. The Configuration Fingerprint changes with the form, as it should. The evaluation's test-year figures and Lift stay those measured before, until the next frozen read.
