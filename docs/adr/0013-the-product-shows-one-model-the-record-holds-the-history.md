# The product shows one model; the record holds the history

Decided in spec [#209](https://github.com/noahlb1/exportspredictor/issues/209), built from [#210](https://github.com/noahlb1/exportspredictor/issues/210).

The shipped surfaces read like a lab notebook: the README led with a table of five versions, and a reader could not tell which model was served. So the project keeps two kinds of surface.

- **Product surfaces** present the served model: what it is, how it was chosen, its 2023 result beside its benchmarks, and its limits. They are the README (outside its reproduction section), the main notebook, every app page, the model card, the API responses and the GitHub release page.
- **The record** holds the chronology: every version, every Arm, every read of 2023, the gain ladder. It is the ADRs, `docs/research/`, the Test-year Ledger, the Model History notebook and the README's reproduction section.

Design evidence (the Selection Measure, the alternatives considered on 2022, the five-fold confirmation, the search, ablation, learning curve, constraint test and benchmarks) stays in the product, framed as why the model is this way.

## The naming rule

Product text names earlier configurations by role ("the model alone", "the starting configuration"), never by version name. Version names mean nothing to an Advisor, and they invite comparisons with models that are not shipped.

## The guard

`tests/test_product_surface.py` fails when a history term appears on an enrolled product surface. The terms are defined once, in `exportspredictor.product_surface`: version names (`v1`, `v2`, also inside identifiers like `beats_v2`) and phrases such as "earlier version", "frozen version", "first form" and "Arm against". Release tags pass: `v1.0` and the tag the fetch points at (`release.TAG`). Each surface is enrolled by the ticket that cleans it.

### Considered options

- **Keep the version comparisons on the product, captioned.** Rejected: the reader still meets the history before the product.
- **Delete the history.** Rejected: the record is how a reader checks the 2023 result was not tuned.
- **Split product and record, with a test.** Chosen: the test keeps later edits from bringing version names back.
