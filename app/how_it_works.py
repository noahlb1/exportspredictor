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
        Idea(
            "Out-of-fold scoring",
            "Each UK Product–Market pair is scored by a model that never saw it in training.",
            what=(
                "UK Product–Market pairs are split into five groups, or folds, each pair with all its years. Five "
                "models are trained: each sees every other Exporter's Flows and four folds of UK pairs, and scores "
                "the fifth. Every UK Export Potential, Export Gap and Scenario Effect comes from the model that held "
                "that pair out."
            ),
            why=(
                "A model scored on Flows it trained on can memorise them: Export Potential would copy Actual "
                "Exports, and every Export Gap would shrink towards zero. Leaving the UK out entirely would lose what "
                "its trade says about its strengths. Keeping a pair's years in one fold stops one year of a pair "
                "teaching the model another."
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
                "choices are made on a validation year, and the final check is the test year after it. Value ranges "
                "and the Sector encoding are fitted on training years only."
            ),
            why=(
                "Advisors use the model on years it has not seen. Shuffling years between training and scoring "
                "would let it learn from the future, such as a Market's later boom, and report a score no real use "
                "could reach."
            ),
            how=(
                "- Choices are tuned on models trained on the years before the validation year.\n"
                "- The test year follows it, and no choice is made on it.\n"
                "- The served models then train on every year up to the test year and score the latest one, the "
                "Baseline year."
            ),
        ),
        Idea(
            "The test year, read once",
            "The test year is opened once, after every choice is fixed, so nothing is tuned to it.",
            what=(
                "A version is frozen when every choice, from features to the number of trees, is fixed on the "
                "validation year. Only then is it measured on the test year, once. Its result is reported as it "
                "is; a fix belongs to a new version."
            ),
            why=(
                "Each look at a test score tempts a tweak, and a test year tweaked against becomes a second "
                "validation year whose score flatters the model. Read once, it stays an honest measure of a year "
                "the model was not tuned to."
            ),
            how=(
                "- A version is identified by its Configuration Fingerprint, a hash of the settings that decide its "
                "models.\n"
                "- Every read is a line in the Test-year Ledger, written before the year is opened.\n"
                "- A fingerprint already read is refused; an override is possible, and is recorded as a read too."
            ),
        ),
        Idea(
            "The ranking measure",
            "Every choice is made on one measure: how well the model ranks Markets within each Product.",
            what=(
                "Within each Product, Markets are ranked by Export Potential and by Actual Exports, and the two "
                "orders compared with a Spearman rank correlation, ρ: 1 if they match, 0 if unrelated. The Selection "
                "Measure averages it over the P Products, for held-out UK pairs:\n\n"
                r"$$\text{Selection Measure} = \frac{1}{P} \sum_{p=1}^{P} \rho_p$$"
            ),
            why=(
                "Advisors use the order: which Markets to look at first for a Product. Choosing on one measure "
                "stops each choice going to whichever metric looks best. Tweedie deviance, which judges size, is "
                "reported but never chosen on: choosing on it once lowered the ranking."
            ),
            how=(
                "- Ties in Export Potential are broken in a fixed pseudo-random order, a hash of Product and Market; "
                "tied Actual Exports (the zeros) share their average rank. Adding ties can then never raise the "
                "measure.\n"
                "- A change is kept only if it beats the best so far by the Win Threshold: twice how far the measure "
                "moves when only the random seed changes.\n"
                "- It also chooses the number of trees."
            ),
        ),
        Idea(
            "The blend with the Naive Baseline",
            "Export Potential combines the model with a simple market-share estimate, weighted by what ranks best.",
            what=(
                "The Naive Baseline is a market-share estimate with no model: the Exporter's share of the Product's "
                "world exports times the Market's imports of it, each leaving the Flow itself out. Export Potential "
                "averages it with the model on the log scale, each floored at one US dollar, then scales by Market "
                "Reach:\n\n"
                r"$$\log \text{Export Potential} = 0.7 \log \text{model} + 0.3 \log \text{naive}$$"
            ),
            why=(
                "The Naive Baseline is the bar the model must beat, and the blend beats both. On the validation "
                "year, the model alone ranked Markets worse than the blend on all five folds: the Naive Baseline "
                "steadies the order where the model is noisy."
            ),
            how=(
                "- The weight 0.3 was chosen on the Selection Measure, from 0 to 0.9 in steps of 0.1.\n"
                "- The Naive Baseline has no policy features, so a Scenario moves only the model's share: a Scenario "
                "Effect is the model's effect scaled by 0.7 on the log scale.\n"
                "- The Naive Baseline's own score is reported beside the model's."
            ),
        ),
        Idea(
            "Monotone policy features",
            "A Trade Agreement never lowers Export Potential, and a higher tariff never raises it.",
            what=(
                "A monotone constraint fixes the direction a feature can move a prediction, all else equal. Adding "
                "a Trade Agreement, a Customs Union or EU, CPTPP or EFTA membership can only raise Export Potential; "
                "a higher Applied Tariff can only lower it."
            ),
            why=(
                "Scenarios change these features and re-predict. Unconstrained, the trees fit noise in them: in a "
                "test, adding a Trade Agreement moved 39% of pairs the wrong way. The constraint keeps every "
                "Scenario Effect in the expected direction, but also one-sided, so Scenario Effects are shown as "
                "illustrative, upper-end associations."
            ),
            how=(
                "- XGBoost's monotone constraints: up on each agreement and membership flag, down on the log tariff.\n"
                "- Tariffs are clipped to the training range: trees cannot extrapolate, so a very large hike's effect "
                "flattens.\n"
                "- A missing tariff stays missing when a Scenario adds an agreement, as the constraint does not cover "
                "missing values.\n"
                "- Removing the constraints ranked slightly worse, within noise, so they stay."
            ),
        ),
    ),
)

SERVED = Section(
    "What it produces and how it is served",
    (
        Idea("The Potential Range", "Each Export Potential comes with the range Actual Exports of comparable pairs "
             "fall in 80% of the time."),
        Idea("Market Reach", "Export Potential is scaled by how widely the UK already sells across each Market."),
        Idea("The gravity-model benchmark", "A classic trade-economics regression is fitted beside the model as a "
             "check on its Scenario Effects."),
        Idea("Scenarios as re-prediction", "A Scenario changes only the policy features and asks the same model "
             "again."),
        Idea("From data to app", "Polars builds the data, FastAPI serves the model and Streamlit shows it, each in a "
             "container."),
    ),
)

SECTIONS = (BUILT, CHECKED, SERVED)
