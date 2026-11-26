"""The text of the How the Model Works page: its outline of ideas, each with a one-sentence summary and up to three
layers (what it is, why this project uses it, how it is done here).

`sections(form, non_uk_rate)` words it for a served artefact: its settings come from the package's constants and the
served Model Form, so the page never states a figure the model does not use. The API serves it (`/how-it-works`) and
the app's page (`app/pages/how_the_model_works.py`) lays out whatever is here.

An idea with no layers shows its summary only. Layers are Markdown, so a formula goes in `$$ $$`; avoid a bare dollar
sign, which Markdown reads as the start of a formula.
"""

from dataclasses import dataclass

from exportspredictor.conformal import COVERAGE, HIGH_QUANTILE, LOW_QUANTILE, N_BANDS
from exportspredictor.experiments import VALIDATION_YEAR
from exportspredictor.features import ZERO_SAMPLING_RATE
from exportspredictor.folds import N_FOLDS
from exportspredictor.ledger import TEST_YEAR
from exportspredictor.market_reach import CHOSEN_BETA
from exportspredictor.prediction import ModelForm
from exportspredictor.scope import TOP_EXPORTERS
from exportspredictor.wording import rate_text


@dataclass(frozen=True)
class Idea:
    name: str
    summary: str  # one sentence
    what: str = ""
    why: str = ""
    how: str = ""


@dataclass(frozen=True)
class Section:
    title: str
    ideas: tuple[Idea, ...]


# Each layer's expander label, by field, in the order a reader opens them.
LAYERS = {"what": "What it is", "why": "Why this project uses it", "how": "How it is done here"}

INTRO = "How this model is built, checked and served. Open any idea to see what it is, why it is used and how."

_NUMBER_WORDS = ("no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")


def _count(n: int) -> str:
    """A small count in words ("five"), a larger one in figures."""
    return _NUMBER_WORDS[n] if 0 <= n < len(_NUMBER_WORDS) else str(n)


def sections(form: ModelForm, non_uk_rate: float | None) -> tuple[Section, ...]:
    """The page's sections for the served Model Form `form` and the artefact's non-UK sampling rate. A form without
    Market Reach states the strength the project chose (`market_reach.CHOSEN_BETA`): every built version has it."""
    folds, others = _count(N_FOLDS), _count(N_FOLDS - 1)
    w, model_w = f"{form.naive_weight:g}", f"{1 - form.naive_weight:g}"
    beta = f"{form.market_reach.beta if form.market_reach is not None else CHOSEN_BETA:g}"
    coverage, low, high = f"{COVERAGE:.0%}", f"{LOW_QUANTILE * 100:.0f}th", f"{HIGH_QUANTILE * 100:.0f}th"

    built = Section(
        "How the model is built",
        (
            Idea(
                "The data",
                "The model learns from goods trade between every pair of countries, Product by Product, over a decade.",
                what=(
                    "Flows from BACI, a cleaned version of the UN's trade records: the yearly value, in US dollars, of "
                    "each Product sent from each Exporter to each Importer. A Product is one of about 5,000 HS6 codes, "
                    "the most detailed level of the Harmonized System; a Market is an Importer seen as a destination "
                    "for UK exports. Most combinations never trade: these Zero Cells are rows too, with value 0."
                ),
                why=(
                    "Learning from every country shows what an Exporter with the UK's fundamentals typically sells to "
                    "each Market. A UK-only model could only repeat what the UK already does, and would know nothing "
                    "of Markets it never sells to. Zero Cells teach it where trade does not happen."
                ),
                how=(
                    "- BACI in the Harmonized System's 2012 edition: its longer history holds more Trade Agreements "
                    "coming into force to learn from.\n"
                    f"- Exporters are the {TOP_EXPORTERS} largest economies, the UK always included; every Market is "
                    "kept.\n"
                    "- Every UK row is kept. Other Exporters' rows are sampled to fit the graphics card: "
                    f"{rate_text(ZERO_SAMPLING_RATE)} of their Zero Cells, then {rate_text(non_uk_rate)} of all their "
                    "rows, weighted by the inverse of their sampling rates so totals stay unbiased.\n"
                    "- Products whose trade follows pipelines or bullion markets, such as oil and gold, are left out "
                    "as Excluded Products."
                ),
            ),
            Idea(
                "The features",
                "Each Flow is described by about 25 facts about its Exporter, Market and Product, none of which "
                "reveals the Flow itself.",
                what=(
                    "- **Gravity**: both countries' GDP and population, the distance between them, a shared border, "
                    "language or colonial tie.\n"
                    "- **Supply**: the Exporter's exports of the Product, how specialised it is in it, and its "
                    "Relatedness to what the Exporter already exports well.\n"
                    "- **Demand**: the Market's imports of the Product, the share from high-income Exporters, and how "
                    "much it re-exports.\n"
                    "- **Product**: its Sector.\n"
                    "- **Policy**: a Trade Agreement, a Customs Union, EU, CPTPP or EFTA membership, and the Applied "
                    "Tariff."
                ),
                why=(
                    "A feature built from the Flow being predicted lets the model read the answer: it would copy "
                    "Actual Exports, and every Export Gap would shrink towards zero. Leave-one-out features close that "
                    "path. No feature uses the two countries' trade in other Products either: the model would credit "
                    "that trade with what it now associates with a Trade Agreement, and Scenarios would understate it."
                ),
                how=(
                    "- **Leave-one-out supply**: the Exporter's exports of the Product count every Market except the "
                    "row's own, so a UK row never sees its target Market.\n"
                    "- Demand features leave out the row's own Exporter, so a UK row never sees a UK Flow; Relatedness "
                    "leaves out the Market.\n"
                    "- Value ranges and the Sector encoding are fitted on training years only.\n"
                    "- Training and serving build features with the same code.\n"
                    "- One indirect path remains: other Exporters' demand features include UK sales, which can pull a "
                    "pair's Export Potential slightly towards its Actual Exports."
                ),
            ),
            Idea(
                "Gradient-boosted trees on the GPU",
                "XGBoost adds up many decision trees, each correcting the errors of those before it, trained on a "
                "graphics card.",
                what=(
                    "A decision tree sorts rows by yes/no questions on the features and predicts a value at each leaf. "
                    "Gradient boosting adds trees one at a time, each fitted to what the trees so far get wrong:\n\n"
                    r"$$\log \hat{y}(x) = \sum_{t=1}^{T} \eta \, f_t(x)$$"
                    "\n\nwhere each $f_t$ is a tree and $\\eta$ a small learning rate."
                ),
                why=(
                    "Trees find interactions a regression must be told about, such as a Product that sells only to "
                    "rich, nearby Markets. Trade values run from zero to billions and are mostly zero, so the model "
                    "uses a Tweedie objective, made for many zeros and a long tail. The graphics card trains about 15 "
                    "times faster than the processor, which makes tuning and repeated experiments affordable."
                ),
                how=(
                    "- XGBoost with a Tweedie objective and histogram trees, on an RTX 3080 graphics card with 10 GB of "
                    "memory.\n"
                    "- Rows stream in a year at a time into a compressed matrix built on the card; about 100 million "
                    "rows fit.\n"
                    "- Tree depth, regularisation and the number of trees were tuned on a validation year held back "
                    "from training.\n"
                    f"- {folds.capitalize()} models are trained, so every UK pair is scored by one that never saw it."
                ),
            ),
        ),
    )

    checked = Section(
        "How the model is checked",
        (
            Idea(
                "Out-of-fold scoring",
                "Each UK Product–Market pair is scored by a model that never saw it in training.",
                what=(
                    f"UK Product–Market pairs are split into {folds} groups, or folds, each pair with all its years. "
                    f"{folds.capitalize()} models are trained: each sees the other Exporters' sampled rows and "
                    f"{others} folds of UK pairs, and scores the remaining one. Every UK Export Potential, Export Gap "
                    "and Scenario Effect comes from the model that held that pair out."
                ),
                why=(
                    "A model scored on Flows it trained on can memorise them: Export Potential would copy Actual "
                    "Exports, and every Export Gap would shrink towards zero. Leaving the UK out entirely would lose "
                    "what its trade says about its strengths. Keeping a pair's years in one fold stops one year of a "
                    "pair teaching the model another."
                ),
                how=(
                    "- A pair's fold is a fixed hash of its Product and Market, the same on every run.\n"
                    "- Features built from UK data leave out the pair's own Market.\n"
                    "- A Scenario on a pair re-predicts with the same held-out model."
                ),
            ),
            Idea(
                "The time split",
                "The model is fitted on earlier years and judged on later ones, as it is used.",
                what=(
                    "Training rows come from earlier years and every score from a later year the model never saw: "
                    "choices are made on a validation year, and the final check is the test year after it. Value "
                    "ranges and the Sector encoding are fitted on training years only."
                ),
                why=(
                    "Advisors use the model on years it has not seen. Shuffling years between training and scoring "
                    "would let it learn from the future, such as a Market's later boom, and report a score no real use "
                    "could reach."
                ),
                how=(
                    "- Choices are tuned on models trained on the years before the validation year.\n"
                    "- The test year follows it, and no choice is made on it.\n"
                    "- Once it is read, the served models train on every year up to and including the test year, and "
                    "score the year after, the Baseline year."
                ),
            ),
            Idea(
                "The test year, read once",
                f"Every choice was made on {VALIDATION_YEAR}, and the model was tested once on {TEST_YEAR}.",
            ),
            Idea(
                "The ranking measure",
                "Every choice is made on one measure: how well the model ranks Markets within each Product.",
                what=(
                    "Within each Product, Markets are ranked by Export Potential and by Actual Exports, and the two "
                    "orders compared with a Spearman rank correlation, ρ: 1 if they match, 0 if unrelated. The "
                    "Selection Measure averages it over the P Products, for held-out UK pairs:\n\n"
                    r"$$\text{Selection Measure} = \frac{1}{P} \sum_{p=1}^{P} \rho_p$$"
                ),
                why=(
                    "Advisors use the order: which Markets to look at first for a Product. Choosing on one measure "
                    "stops each choice going to whichever metric looks best. Tweedie deviance, which judges size, is "
                    "reported but never chosen on: choosing on it once lowered the ranking."
                ),
                how=(
                    "- Ties in Export Potential are broken in a fixed pseudo-random order, a hash of Product and "
                    "Market; tied Actual Exports (the zeros) share their average rank. Adding ties can then never raise "
                    "the measure.\n"
                    "- A change is kept only if it beats the best so far by the Win Threshold: twice how far the "
                    "measure moves when only the random seed changes.\n"
                    "- It also chooses the number of trees."
                ),
            ),
            Idea(
                "The blend with the Naive Baseline",
                "Export Potential combines the model with a simple market-share estimate, weighted by what ranks best.",
                what=(
                    "The Naive Baseline is a market-share estimate with no model: the Exporter's share of the "
                    "Product's world exports times the Market's imports of it, each leaving the Flow itself out. Export "
                    "Potential averages it with the model on the log scale, each floored at one US dollar, then scales "
                    "by Market Reach:\n\n"
                    rf"$$\log \text{{Export Potential}} = {model_w} \log \text{{model}} + {w} \log \text{{naive}}$$"
                ),
                why=(
                    "The Naive Baseline is the bar the model must beat, and the blend beats both. On the validation "
                    f"year, the model alone ranked Markets worse than the blend on all {folds} folds: the Naive "
                    "Baseline steadies the order where the model is noisy."
                ),
                how=(
                    f"- The weight {w} was chosen on the Selection Measure, from 0 to 0.9 in steps of 0.1.\n"
                    "- The Naive Baseline has no policy features, so a Scenario moves only the model's share: a "
                    f"Scenario Effect is the model's effect scaled by {model_w} on the log scale.\n"
                    "- The Naive Baseline's own score is reported beside the model's."
                ),
            ),
            Idea(
                "Monotone policy features",
                "A Trade Agreement never lowers Export Potential, and a higher tariff never raises it.",
                what=(
                    "A monotone constraint fixes the direction a feature can move a prediction, all else equal. Adding "
                    "a Trade Agreement, a Customs Union or EU, CPTPP or EFTA membership can only raise Export "
                    "Potential; a higher Applied Tariff can only lower it."
                ),
                why=(
                    "Scenarios change these features and re-predict. Unconstrained, the trees fit noise in them: in a "
                    "test, adding a Trade Agreement moved 39% of pairs the wrong way. The constraint keeps every "
                    "Scenario Effect in the expected direction, but also one-sided, so Scenario Effects are shown as "
                    "illustrative, upper-end associations."
                ),
                how=(
                    "- XGBoost's monotone constraints: up on each agreement and membership flag, down on the log "
                    "tariff.\n"
                    "- Tariffs are clipped to the training range: trees cannot extrapolate, so a very large hike's "
                    "effect flattens.\n"
                    "- A missing tariff stays missing when a Scenario adds an agreement, as the constraint does not "
                    "cover missing values.\n"
                    "- Removing the constraints ranked slightly worse, within noise, so they stay."
                ),
            ),
        ),
    )

    served = Section(
        "What it produces and how it is served",
        (
            Idea(
                "The Potential Range",
                f"Each Export Potential comes with the range Actual Exports of comparable pairs fall in {coverage} of "
                "the time.",
                what=(
                    f"An {coverage} prediction interval for a pair's Actual Exports, built by conformal prediction: "
                    "look at how far real outcomes landed from comparable predictions, and take the "
                    f"{low} and {high} percentiles of those misses. It describes how much comparable pairs vary, not "
                    "how precisely Export Potential is known."
                ),
                why=(
                    "A UK pair's miss is its Export Gap with the sign flipped. Calibrating on UK misses would only "
                    "measure a typical UK gap, and every gap would look normal. So the range is calibrated on other "
                    "Exporters, whose misses are honestly out of sample. An Opportunity whose Actual Exports lie inside "
                    "its range is an Uncertain Opportunity: still ranked, but within normal variation."
                ),
                how=(
                    "- Calibration rows: other Exporters' Flows and Zero Cells in the Baseline year, which no fold "
                    "model trained on.\n"
                    "- Misses on the log scale, so each range is a ratio of Export Potential:\n\n"
                    r"$$\log(1 + \text{Actual Exports}) - \log(1 + \text{Export Potential})$$"
                    "\n\n"
                    f"- Calibrated separately in {N_BANDS} bands by size of Export Potential, the widest split at "
                    "decade edges.\n"
                    "- Coverage is checked on a held-out half of the calibration rows, split by a hash of Exporter, "
                    "Product and Market."
                ),
            ),
            Idea(
                "Market Reach",
                "Export Potential is scaled by how widely the UK already sells across each Market.",
                what=(
                    "For each Market: of the Products it imports from other Exporters, the share the UK sends there "
                    "too, against the UK's share across all Markets. That gives an offset per Market, and"
                    "\n\n"
                    rf"$$\text{{Export Potential}} \times e^{{{beta} \cdot \text{{offset}}}} \times c$$"
                    "\n\nwhere $c$ keeps the UK's total Export Potential unchanged."
                ),
                why=(
                    "Most of the ranking's errors lay in which Markets get any UK trade at all, and the model did not "
                    "know where the UK already sells. With Market Reach, an Export Gap is specific to the Product "
                    "within its Market; the Market-wide part shows as the Market's reach. The model associates broad "
                    "UK reach with higher Export Potential; reach is not shown to drive exports."
                ),
                how=(
                    "- Applied after the model, not as a feature: no fold model trains on it, so every feature stays "
                    "leave-one-out.\n"
                    "- Each pair's offsets come only from pairs in other folds, in the year before, so none of its own "
                    "Flows enters them.\n"
                    f"- The strength, {beta}, was chosen on the ranking measure in the validation year.\n"
                    "- Other Exporters' rows, including the Potential Range's calibration rows, get no Market Reach."
                ),
            ),
            Idea(
                "The gravity-model benchmark",
                "A classic trade-economics regression is fitted beside the model as a check on its Scenario Effects.",
                what=(
                    "A gravity regression, the economist's standard tool for trade, estimated by Poisson "
                    "pseudo-maximum likelihood (PPML):\n\n"
                    r"$$\text{Flow}_{eist} = \exp(a_{est} + b_{ist} + c_{eis} + \beta \cdot \text{Agreement}_{eit})$$"
                    "\n\nfor Exporter $e$, Importer $i$, Sector $s$ and year $t$; $e^{\\beta} - 1$ is the change in "
                    "Flows it associates with a Trade Agreement."
                ),
                why=(
                    "The model learns its Trade Agreement association from patterns, and the monotone constraint "
                    "pushes it one way. The benchmark checks its size against an independent, established method. It "
                    "is not served instead: it works at Sector level, with no HS6 detail, and a held-out pair has no "
                    "pair term to predict with."
                ),
                how=(
                    "- Flows summed to Sector level, at three-year intervals so trade has time to adjust.\n"
                    "- Fixed effects for Exporter-year, Importer-year and each pair, within each Sector, so only pairs "
                    "whose agreement status changes identify the effect.\n"
                    "- Errors clustered by Exporter–Importer pair; fitted once and saved with the model.\n"
                    "- Shown beside every Scenario Effect, with published estimates and the Event Backtest."
                ),
            ),
            Idea(
                "Scenarios as re-prediction",
                "A Scenario changes only the policy features and asks the same model again.",
                what=(
                    "A Scenario is a hypothetical policy change: joining a Trading Bloc, adding or removing Trade "
                    "Agreements, or changing a tariff. The model scores the Baseline year with the change applied, and "
                    "the Scenario Effect is the difference from the Baseline."
                ),
                why=(
                    "Changing only policy features keeps every other fact fixed, so the effect isolates what the model "
                    "associates with the policy. It is an association learned from past trade, not a measured result "
                    "of any deal, so every Scenario Effect is labelled Illustrative and shown beside its benchmarks."
                ),
                how=(
                    "- Copies the affected Markets' UK rows, changes only the policy columns, and checks nothing else "
                    "moved.\n"
                    "- Each pair is re-predicted by its own fold model, through the same prediction function.\n"
                    "- Adding a Trade Agreement sets a known tariff to 0; tariffs are clipped to the training range.\n"
                    "- The Naive Baseline has no policy features, so only the model's share of the Blend moves.\n"
                    "- Blocked Opportunity values are this Scenario, precomputed for every pair without an agreement."
                ),
            ),
            Idea(
                "From data to app",
                "Polars builds the data, FastAPI serves the model and Streamlit shows it, each in a container.",
                what=(
                    "Four layers: a Python package that builds features, trains and predicts; saved artefacts (the "
                    "fold models and precomputed tables); a FastAPI service that loads them; and a Streamlit app that "
                    "Advisors use."
                ),
                why=(
                    "Training and serving import the same package, so features are computed identically in both; a "
                    "copy in the API would drift. The app talks to the model only through the API, so it holds no "
                    "model logic. Containers and locked dependencies let anyone run the same stack with one command."
                ),
                how=(
                    "- Polars scans Parquet files lazily, so the full trade data never sits in memory at once.\n"
                    "- The API loads the newest artefact at startup, serves rankings from precomputed tables and "
                    "re-predicts Scenarios on the processor.\n"
                    "- The app calls the API over HTTP, which also serves this page's text.\n"
                    "- One container image serves both, started together by a compose file.\n"
                    "- uv manages Python and pins every dependency in a lock file."
                ),
            ),
        ),
    )
    return built, checked, served
