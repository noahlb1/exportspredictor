# UK Exports Predictor

Finds where UK goods exports fall short of what the UK's fundamentals suggest. It also estimates how that changes under trade-policy scenarios, so that government export support teams can target help to industry.

## Language

### Trade

**Flow**:
The annual value (USD) of one Product sent from one Exporter to one Importer in one year.
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

### Prediction

**Export Potential**:
The value of a Flow that the model expects, given the fundamentals of its Exporter, Importer and Product.
_Avoid_: forecast, expected exports, predicted trade

**Actual Exports**:
The observed value of a UK Flow in a given year.
_Avoid_: real exports, historic trade

**Export Gap**:
Export Potential minus Actual Exports for one UK Product–Market pair; positive means the UK is underperforming its fundamentals.
_Avoid_: untapped potential, headroom, shortfall

**Opportunity**:
A UK Product–Market pair with a positive Export Gap, ranked by size of that gap.
_Avoid_: lead, prospect

**Blocked Opportunity**:
A UK Product–Market pair whose Export Potential rises materially in a Scenario that removes a Barrier. It is distinct from an Opportunity, because the Barrier already holds Export Potential down, so it shows no Export Gap.
_Avoid_: hidden opportunity, latent demand

**Blocked Opportunity value**:
The Scenario Effect the model associates with adding a Trade Agreement to a UK Product–Market pair that has none (zero where one exists). Rankings can be ordered by it instead of by Export Gap.
_Avoid_: blocked gap, unlock value

**Barrier**:
A trade-policy condition that suppresses Export Potential, namely a tariff or the absence of a Trade Agreement.
_Avoid_: blocker, obstacle, friction

**Lift**:
How many times more likely Product–Market pairs with large Export Gaps are than random pairs to grow or appear in later years. It describes a tendency, not a forecast.
_Avoid_: hit rate, conversion

**Event Backtest**:
A check of Scenario Effects against a real trade-policy change that formed and took effect within about two years, comparing predicted with observed change against unaffected pairs.
_Avoid_: validation event, natural experiment

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
