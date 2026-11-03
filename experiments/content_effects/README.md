# content_effects

**Question.** When familiar items are shown with relations that are congruent with, reversed
from, or shuffled relative to their real-world ordering, do direct answers follow the prior
or the context, and does step-by-step reasoning overcome the prior?

**Design.** The main item set of each of the five dimensions (plus the constant-ratio city
item set `cities`), the 10 adjacent relations shown either congruent, reversed or shuffled;
answers are scored against the ordering shown. Uninformative control: 10 true relations among 11 other
items of the same dimension. Every context condition uses the instruction that ends with
"Base your answer only on the comparisons given — not on outside knowledge." Budget ladders
over the congruent, reversed and shuffled contexts, as in `symbolic_distance`.

**Tests** (`experiment.yaml`): `content` (conditions congruent / reversed / shuffled),
`uninformative` (direct); `budget_natural`, `budget_reversed`, `budget_shuffled` (ladders).
22 models, the paper's five dimensions plus the constant-ratio city item set
(`cities`; the paper's City population is `citieslinear`).

**Feeds.** The LLM panels of the figure "Content effects in transitive inference" (direct
accuracy by distance under the four contexts; accuracy by reasoning budget, the budget required,
and each context's accuracy at 1,024 tokens) and their appendix per-model versions ("LLM content
effects", "Recovery with reasoning" and "LLM reasoning requirements under content conditions",
one figure per model family). `query_labels.parquet` records whether a shuffled query's shown
order agrees with the real one (agree fraction per distance in `MANIFEST.yaml`).

**Tables.** `results/content_effects/tables/direct.parquet` (380,160 rows),
`ladders.parquet` (3,136,320 rows), `query_labels.parquet` (18,000 rows), `MANIFEST.yaml`.

**Provenance.** The same software stack and GPU as `symbolic_distance`.
