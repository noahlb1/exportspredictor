# UK Exports Predictor

Finds where UK goods exports fall short of what the UK's fundamentals suggest. It also estimates how that changes under trade-policy scenarios, so that government export support teams can target help to industry.

## Language

### Trade

**Flow**:
The annual value, in nominal US dollars, of one Product sent from one Exporter to one Importer in one year.
_Avoid_: trade, shipment, transaction

**Product**:
A single HS6 subheading of the Harmonized System, the most detailed internationally comparable level of goods classification.
_Avoid_: good, commodity, item

**Sector**:
An HS2 chapter, used only to group and summarise Products, never as a modelling unit.
_Avoid_: industry, category

**Exporter**:
The country a Flow originates from.
_Avoid_: origin, reporter, supplier

**Importer**:
The country a Flow is sent to.
_Avoid_: destination, partner

**Market**:
An Importer considered as a destination for UK exports.
_Avoid_: partner, target country

**Zero Cell**:
An Exporter, Importer, Product and year with no Flow, where the Exporter exports that Product to some Importer and the Importer imports it from some Exporter, but not from each other.
_Avoid_: missing flow, null trade, empty pair

**Excluded Product**:
A Product left out of training, scoring and rankings, because its recorded Flows follow market or infrastructure geography the model cannot see, or its values are unreliable. Advisors are shown the list and the reason for each.
_Avoid_: removed product, banned product

**Flagged Product**:
A Product scored and ranked as normal, but shown with a caveat that names its reason: precious stones and jewellery of Sector 71, whose values are volatile and entrepôt-driven; UK exports that swing widely from year to year over the training years (the top fifth by volatility); or a one-off Product sold in few, large contracts.
_Avoid_: warning product, suspect product

**Re-export Hub**:
A Market that re-exports much of what it imports, so part of its demand belongs to other Markets. It is shown with a caveat and can be filtered out.
_Avoid_: entrepôt, transit country

**Sanctioned Market**:
A Market under broad UK trade sanctions in the Baseline year. It is scored and ranked as normal, but shown with a caveat and can be filtered out.
_Avoid_: embargoed market, closed market, blocked market

**Restricted Pair**:
A Product–Market pair whose trade the Market prohibits outright, such as alcoholic beverages into a Market that bans alcohol imports. It is scored and ranked as normal, but shown with a caveat and can be filtered out.
_Avoid_: banned pair, prohibited product

**Caveat Tier**:
A group of Product–Market pairs, keyed on the Market, the Product or the pair, that is scored and ranked as normal but shown with one caveat, its badge and a filter to hide it. Flagged Products, Re-export Hubs, Sanctioned Markets and Restricted Pairs are Caveat Tiers; Excluded Products are not, as they are left out.
_Avoid_: flag type, warning level

**Relatedness**:
How close a Product is to the Products an Exporter already exports competitively, judged by which Products tend to be exported competitively together across the world.
_Avoid_: density, product space, proximity

### Prediction

**Export Potential**:
The value of a Flow that the model expects, given the fundamentals of its Exporter, Importer and Product.
_Avoid_: forecast, expected exports, predicted trade

**Actual Exports**:
The observed value of a UK Flow in a given year.
_Avoid_: real exports, historic trade

**Export Gap**:
The mean, over the Gap Period, of Export Potential minus Actual Exports for one UK Product–Market pair; positive means the UK is underperforming its fundamentals. With Market Reach in the Model Form, it is specific to the Product within its Market; how widely the UK reaches the whole Market shows in the Market's reach.
_Avoid_: untapped potential, headroom, shortfall

**Gap Period**:
The run of recent years whose yearly Export Gaps are averaged into the reported Export Gap. Each year's gap compares that year's Export Potential with that year's Actual Exports, a year with no Flow counting as zero.
_Avoid_: reference period, gap window, averaging window

**Opportunity**:
A UK Product–Market pair with a positive Export Gap, ranked by size of that gap.
_Avoid_: lead, prospect

**Relative Export Gap**:
Export Gap divided by Export Potential: the share of an Opportunity's Export Potential the UK does not yet realise. Every pair with no Actual Exports scores 100%, so it ranks only among pairs above a minimum Export Potential, with ties broken by Export Gap.
_Avoid_: percentage gap, headroom ratio

**Leaderboard**:
Every UK Product–Market pair ranked by Export Gap (or Relative Export Gap), with the Advisor's filters and a summary of the filtered pairs. It is the app's home page; Blocked Opportunities are listed, grouped by Market, on a page of their own.
_Avoid_: top list, dashboard, league table

**Opportunity Type**:
How far the UK already trades an Opportunity: **Expand** (the UK already exports the Product to this Market), **New Market** (the UK exports the Product, but not to this Market) or **New Product** (the UK exports the Product to no Market).
_Avoid_: entry mode, opportunity status

**Potential Range**:
The range within which Actual Exports fall 80% of the time for Product–Market pairs with the same Export Potential. It describes how much comparable pairs vary around their Export Potential, not how precisely Export Potential itself is known.
_Avoid_: confidence interval, error bars, margin

**Uncertain Opportunity**:
An Opportunity whose Actual Exports lie inside its Potential Range, so its Export Gap is within the variation normally seen among comparable pairs. It is still ranked. The term is reported as a share of all Opportunities (on the Model Evaluation page, in the notebook and in the model card), not shown as a badge on each row, because it covers most of them (ADR 0007).
_Avoid_: weak lead, low-confidence opportunity, false positive

**Blocked Opportunity**:
A UK Product–Market pair whose Export Potential rises materially in a Scenario that removes a Barrier. It is measured apart from the Export Gap, because the Barrier already holds Export Potential down: a pair can be an Opportunity, a Blocked Opportunity, both or neither.
_Avoid_: hidden opportunity, latent demand

**Blocked Opportunity value**:
The Scenario Effect the model associates with adding a Trade Agreement to a UK Product–Market pair that has none (zero where one exists). The Blocked Opportunities page lists the Blocked Opportunities grouped by Market, ranked by it, apart from the Leaderboard, in Baseline-year figures; there, Baseline Export Potential plus the value is shown as Export Potential with a Trade Agreement, also Illustrative. Where Baseline-year Actual Exports exceed Export Potential, the value is shown instead as a percentage uplift on Export Potential, so no dollar level with a Trade Agreement sits below what the UK already sells.
_Avoid_: blocked gap, unlock value

**Barrier**:
A trade-policy condition that suppresses Export Potential, namely an Applied Tariff or the absence of a Trade Agreement.
_Avoid_: blocker, obstacle, friction

**Lift**:
How many times more likely Product–Market pairs with large Export Gaps are than random pairs to grow or appear in later years. It describes a tendency, not a forecast.
_Avoid_: hit rate, conversion

**Event Backtest**:
A check of Scenario Effects against a real trade-policy change that formed and took effect within about two years, comparing predicted with observed change against unaffected pairs.
_Avoid_: validation event, natural experiment

**Control Market**:
In an Event Backtest, a similar Market that had no Trade Agreement with the same Exporters throughout the period compared, whose change stands in for what the affected Flows would otherwise have done.
_Avoid_: comparison country, untreated unit

### Model selection

**Selection Measure**:
The one measure every modelling choice is made on: the mean over Products of the rank correlation between Export Potential and Actual Exports across Markets, for held-out UK pairs in the validation year. Ties in Export Potential are broken in a fixed pseudo-random order (a stable hash of the Product and the Market), and tied Actual Exports (the zeros) share their average rank, so tying the bottom of a ranking can never raise it. Every other metric is reported beside it, never chosen on.
_Avoid_: objective, score, headline (the headline is the same measure on the test year)

**Arm**:
One tried setting of an experiment, run against the best Arm so far and kept or dropped by the Win Threshold.
_Avoid_: trial, variant, candidate

**Noise Floor**:
How far the Selection Measure moves when only the random seed changes, measured by training the same Arm with several seeds.
_Avoid_: variance, error bar

**Win Threshold**:
The margin an Arm must beat the best so far by to be kept: twice the Noise Floor's standard deviation, and at least 0.005.
_Avoid_: significance level, tolerance

**Naive Baseline**:
A market-share estimate of a Flow with no model: the Exporter's share of the Product's world exports times the Importer's imports of that Product, each leaving the Flow itself out. Results are compared against it.
_Avoid_: benchmark (the PPML benchmark is another thing), naive model

**Model Form**:
How Export Potential is made from the fold models. The served form is a **Blend**: the model's Export Potential and the Naive Baseline combined on the log scale with fixed weights (the Naive Baseline's weight chosen on the Selection Measure), then scaled by each Market's Market Reach. The Naive Baseline has no policy features, so a Scenario moves only the model's share; Market Reach scales a pair with and without the Scenario alike.
_Avoid_: ensemble, model type

**Market Reach**:
How widely the UK sends its Products to a Market, against how widely the ranking expects it to, learned from the UK's other Products. It scales the Export Potential of every UK pair in the Market alike, learned from other folds' pairs in the year before (ADR 0010).
_Avoid_: market penetration, market share, UK presence

**Configuration Fingerprint**:
A short hash of the settings that decide a version's test-year models and how they are scored (features, Model Form, hyperparameters, round count, sampling, data source). Rebuilding the same configuration under a new version name keeps the fingerprint (ADR 0009).
_Avoid_: version hash, model ID

**Test-year Ledger**:
The record of every read of the test year: date, version, Configuration Fingerprint, what was read, the command, its kind (an evaluation, or a re-score of earlier versions' saved scores under a new measure) and its status (started, then completed once its figures are saved). A fingerprint already completed in it is refused a second read unless overridden, and the override is recorded too (ADR 0009).
_Avoid_: audit log, read log

### Scenarios

**Baseline**:
Export Potential under the trade-policy conditions actually in force.
_Avoid_: status quo, current state

**Scenario**:
A hypothetical change to the UK's trade-policy conditions, such as a trade agreement, a tariff or a trading bloc membership.
_Avoid_: what-if, shock, simulation

**Scenario Effect**:
The change in Export Potential between the Baseline and a Scenario.
_Avoid_: impact, uplift

**Illustrative**:
How every Scenario Effect and Blocked Opportunity value is shown: an upper-end association the model has learned, beside its benchmarks (the PPML benchmark, the model's own Trade Agreement association and the Event Backtest result), never a forecast and never selected on (ADR 0008).
_Avoid_: indicative, estimated, projected

**Trade Agreement**:
A bilateral or plurilateral preferential trade deal in force between two countries.
_Avoid_: FTA (except as a type of Trade Agreement), deal

**Applied Tariff**:
The import duty a Market charges on a Product from a given Exporter: the lower of its preferential rate for that Exporter and its most-favoured-nation rate.
_Avoid_: duty, tariff rate

**Trading Bloc**:
A named group of countries whose membership implies Trade Agreements with every other member, such as the EU or CPTPP.
_Avoid_: union, alliance

**Customs Union**:
A Trading Bloc whose members also share a common external tariff, such as the EU customs union. Membership of the Customs Union and of the wider bloc are tracked separately.
_Avoid_: single market

### People

**Advisor**:
A member of a government export support team who uses the app to find Opportunities and explore Scenarios.
_Avoid_: user, analyst
