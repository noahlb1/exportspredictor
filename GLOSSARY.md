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
A Product scored and ranked as normal, but shown with a caveat that its values are volatile or entrepôt-driven.
_Avoid_: warning product, suspect product

**Re-export Hub**:
A Market that re-exports much of what it imports, so part of its demand belongs to other Markets. It is shown with a caveat and can be filtered out.
_Avoid_: entrepôt, transit country

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
The mean, over the Gap Period, of Export Potential minus Actual Exports for one UK Product–Market pair; positive means the UK is underperforming its fundamentals.
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

**Opportunity Type**:
How far the UK already trades an Opportunity: **Expand** (the UK already exports the Product to this Market), **New Market** (the UK exports the Product, but not to this Market) or **New Product** (the UK exports the Product to no Market).
_Avoid_: entry mode, opportunity status

**Potential Range**:
The range within which Actual Exports fall 80% of the time for Product–Market pairs with the same Export Potential. It describes how much comparable pairs vary around their Export Potential, not how precisely Export Potential itself is known.
_Avoid_: confidence interval, error bars, margin

**Uncertain Opportunity**:
An Opportunity whose Actual Exports lie inside its Potential Range, so its Export Gap is within the variation normally seen among comparable pairs. It is still ranked, with a caveat.
_Avoid_: weak lead, low-confidence opportunity, false positive

**Blocked Opportunity**:
A UK Product–Market pair whose Export Potential rises materially in a Scenario that removes a Barrier. It is distinct from an Opportunity, because the Barrier already holds Export Potential down, so it shows no Export Gap.
_Avoid_: hidden opportunity, latent demand

**Blocked Opportunity value**:
The Scenario Effect the model associates with adding a Trade Agreement to a UK Product–Market pair that has none (zero where one exists). Rankings can be ordered by it instead of by Export Gap.
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
