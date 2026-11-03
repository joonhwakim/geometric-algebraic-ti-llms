# symbolic_distance

**Question.** How does accuracy change with the symbolic distance between the queried items
when the ordering is known from the weights (IWL test: familiar items, no relations shown) and
when it is known only from the context (ICL test: novel items, all adjacent relations shown),
and how much reasoning does each test need?

**Design.** Orderings of 11 items; every pair of the 9 queryable ranks asked once as a true and
once as a false statement; figures use the innermost 7 ranks (distances 1–6). IWL test: five
real-world dimensions (country population, US state population, city population, animal body
mass, country surface area), the main item set plus replicate item sets built to the same spacing
from disjoint items. ICL test: 11 given names ordered by height, and 11 invented city names
under the population instruction; 10 random premise orders per problem. Budget ladders:
one 1,024-token greedy chain per trial, scored at every prefix budget
(32, 64, 128, 256, 384, 512, 640, 768, 896, 1,024); budget 0 is the direct answer.

**Tests** (`experiment.yaml`): `iwl`, `icl`, `icl_cities` (direct); `budget_iwl`, `budget_icl`,
`budget_icl_cities` (ladders). 22 models (`configs/models/`).

**Feeds.** The LLM panels of the figure "Symbolic distance effects (SDE)" (accuracy and
reasoning budget by distance on the IWL and ICL tests, each test's accuracy at 1,024 tokens, and
the ICL test's best accuracy so far by budget), their appendix per-model and per-dimension
versions ("LLM symbolic-distance results" and "LLM reasoning requirements", one figure per model
family), the appendix figure "LLM ICL accuracy by reasoning budget", the appendix table "Main
LLM item sets", and one verbatim stored ICL prompt in the format the appendix prints
(`results/symbolic_distance/tables/prompt_examples.json`).

**Tables.** `results/symbolic_distance/tables/direct.parquet` (53,856 rows),
`ladders.parquet` (592,416 rows), `MANIFEST.yaml`.

**Provenance of the published runs.** NVIDIA RTX PRO 6000 Blackwell Max-Q, bf16, greedy,
ladders generated once at 1,024 tokens in batches of 16. The
runs are not in the repository (per-trial JSONL with prompts and chains); the tables
were exported from them with `geo-alg-ti-llms export --experiment symbolic_distance`.
