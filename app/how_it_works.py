"""The text of the How the model works page: its outline of ideas, each with a one-sentence summary and up to three
layers (what it is, why this project uses it, how it is done here).

An idea with no layers yet shows its summary only. To write one, fill its `what`, `why` and `how` in place: the page
(`app/pages/how_the_model_works.py`) lays out whatever is here. Layers are Markdown, so a formula goes in `$$ $$`;
avoid a bare dollar sign, which Markdown reads as the start of a formula.
"""

from dataclasses import dataclass


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

BUILT = Section(
    "How the model is built",
    (
        Idea(
            "The data",
            "The model learns from goods trade between every pair of countries, Product by Product, over a decade.",
            what=(
                "Flows from BACI, a cleaned version of the UN's trade records: the yearly value, in US dollars, of "
                "each Product sent from each Exporter to each Importer. A Product is one of about 5,000 HS6 codes, "
                "the most detailed level of the Harmonized System; a Market is an Importer seen as a destination for "
                "UK exports. Most combinations never trade: these Zero Cells are rows too, with value 0."
            ),
            why=(
                "Learning from every country shows what an Exporter with the UK's fundamentals typically sells to "
                "each Market. A UK-only model could only repeat what the UK already does, and would know nothing of "
                "Markets it never sells to. Zero Cells teach it where trade does not happen."
            ),
            how=(
                "- BACI in the Harmonized System's 2012 edition: its longer history holds more Trade Agreements "
                "coming into force for the model to learn from.\n"
                "- Exporters are the 100 largest economies, the UK always included; every Market is kept.\n"
                "- Every UK row is kept. Other Exporters' rows are sampled to fit the graphics card (5% of their Zero "
                "Cells), each weighted by the inverse of its sampling rate so totals stay unbiased.\n"
                "- Products whose trade follows pipelines or bullion markets, such as oil and gold, are left out as "
                "Excluded Products."
            ),
        ),
        Idea(
            "The features",
            "Each Flow is described by about 25 facts about its Exporter, Market and Product, none of which reveals "
            "the Flow itself.",
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
                "A feature built from the Flow being predicted lets the model read the answer: it would copy Actual "
                "Exports, and every Export Gap would shrink towards zero. Leave-one-out features close that path. "
                "No feature uses the two countries' trade in other Products either: it would soak up the effect of a "
                "Trade Agreement, and Scenarios would understate it."
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
                "Trees find interactions a regression must be told about, such as a Product that sells only to rich, "
                "nearby Markets. Trade values run from zero to billions and are mostly zero, so the model uses a "
                "Tweedie objective, made for many zeros and a long tail. The graphics card trains about 15 times "
                "faster than the processor, which makes tuning and repeated experiments affordable."
            ),
            how=(
                "- XGBoost with a Tweedie objective and histogram trees, on an RTX 3080 graphics card with 10 GB of "
                "memory.\n"
                "- Rows stream in a year at a time into a compressed matrix built on the card; about 100 million "
                "rows fit.\n"
                "- Tree depth, regularisation and the number of trees were tuned on a validation year held back "
                "from training.\n"
                "- Five models are trained, so every UK pair is scored by one that never saw it."
            ),
        ),
    ),
)

CHECKED = Section(
    "How the model is checked",
    (
        Idea("Out-of-fold scoring", "Each UK Product–Market pair is scored by a model that never saw it in training."),
        Idea("The time split", "The model is fitted on earlier years and judged on later ones, as it is used."),
        Idea("The test year, read once", "The last year is opened once, after every choice is fixed, so nothing is "
             "tuned to it."),
        Idea("The ranking measure", "Every choice is made on one measure: how well the model ranks Markets within "
             "each Product."),
        Idea("The blend with the Naive Baseline", "Export Potential combines the model with a simple market-share "
             "estimate, weighted by what ranks best."),
        Idea("Monotone policy features", "A Trade Agreement never lowers Export Potential, and a higher tariff never "
             "raises it."),
    ),
)

SERVED = Section(
    "What it produces and how it is served",
    (
        Idea(
            "The Potential Range",
            "Each Export Potential comes with the range Actual Exports of comparable pairs fall in 80% of the time.",
            what=(
                "An 80% prediction interval for a pair's Actual Exports, built by conformal prediction: look at how "
                "far real outcomes landed from comparable predictions, and take the 10th and 90th percentiles of "
                "those misses. It describes how much comparable pairs vary, not how precisely Export Potential is "
                "known."
            ),
            why=(
                "A UK pair's miss is its Export Gap with the sign flipped. Calibrating on UK misses would only "
                "measure a typical UK gap, and every gap would look normal. So the range is calibrated on other "
                "Exporters, whose misses are honestly out of sample. An Opportunity whose Actual Exports lie inside "
                "its range is an Uncertain Opportunity: still ranked, but within normal variation."
            ),
            how=(
                "- Calibration rows: other Exporters' Flows and Zero Cells in the Baseline year, which no fold model "
                "trained on.\n"
                "- Misses on the log scale, so each range is a ratio of Export Potential:\n\n"
                r"$$\log(1 + \text{Actual Exports}) - \log(1 + \text{Export Potential})$$"
                "\n\n"
                "- Calibrated separately in 20 bands by size of Export Potential, the widest split at decade edges.\n"
                "- Coverage is checked on a held-out half of the calibration rows, split by Exporter."
            ),
        ),
        Idea(
            "Market Reach",
            "Export Potential is scaled by how widely the UK already sells across each Market.",
            what=(
                "For each Market: of the Products it imports from other Exporters, the share the UK sends there too, "
                "against the UK's share across all Markets. That gives an offset per Market, and"
                "\n\n"
                r"$$\text{Export Potential} \times e^{4 \cdot \text{offset}} \times c$$"
                "\n\nwhere $c$ keeps the UK's total Export Potential unchanged."
            ),
            why=(
                "Most of the ranking's errors lay in which Markets get any UK trade at all, and the model did not "
                "know where the UK already sells. With Market Reach, an Export Gap is specific to the Product within "
                "its Market; the Market-wide part shows as the Market's reach. The model associates broad UK reach "
                "with higher Export Potential; reach is not shown to drive exports."
            ),
            how=(
                "- Applied after the model, not as a feature: no fold model trains on it, so every feature stays "
                "leave-one-out.\n"
                "- Each pair's offsets come only from pairs in other folds, in the year before, so none of its own "
                "Flows enters them.\n"
                "- The strength, 4, was chosen on the ranking measure in the validation year.\n"
                "- Other Exporters' rows, including the Potential Range's calibration rows, get no Market Reach."
            ),
        ),
        Idea(
            "The gravity-model benchmark",
            "A classic trade-economics regression is fitted beside the model as a check on its Scenario Effects.",
            what=(
                "A gravity regression, the economist's standard tool for trade, estimated by Poisson pseudo-maximum "
                "likelihood (PPML):\n\n"
                r"$$\text{Flow}_{eist} = \exp(a_{est} + b_{ist} + c_{eis} + \beta \cdot \text{Agreement}_{eit})$$"
                "\n\nfor Exporter $e$, Importer $i$, Sector $s$ and year $t$; $e^{\\beta} - 1$ is the change in "
                "Flows it associates with a Trade Agreement."
            ),
            why=(
                "The model learns its Trade Agreement association from patterns, and the monotone constraint pushes "
                "it one way. The benchmark checks its size against an independent, established method. It is not "
                "served instead: it works at Sector level, with no HS6 detail, and a held-out pair has no pair term "
                "to predict with."
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
                "associates with the policy. It is an association learned from past trade, not a measured result of "
                "any deal, so every Scenario Effect is labelled Illustrative and shown beside its benchmarks."
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
                "Four layers: a Python package that builds features, trains and predicts; saved artefacts (the fold "
                "models and precomputed tables); a FastAPI service that loads them; and a Streamlit app that "
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
                "- The app calls the API over HTTP.\n"
                "- One container image serves both, started together by a compose file.\n"
                "- uv manages Python and pins every dependency in a lock file."
            ),
        ),
    ),
)

SECTIONS = (BUILT, CHECKED, SERVED)
