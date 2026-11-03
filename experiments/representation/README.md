# representation

**Question.** Does Qwen3-32B answer directly from a rank direction in its representation of
familiar items, and does reasoning free the answer from it? (The rank-direction bars of the
mechanisms figure's LLM panel; the probe by layer, the edits by distance, the writes and the
bars on the other two dimensions in the appendix figure "LLM mechanism results: Qwen3-32B".)

**Design.** For country population, city population and US state population: item
representations at the last subtoken of every mention of the dimension's familiar items (the
fit set: every item with at least 1.2 M annual English Wikipedia page views, every state for
US states, ranked by value) at
every hidden state (token embeddings and the 64 blocks); a rank direction per hidden state from
the LEACE closed form (Ledoit–Wolf covariance, standardized rank); held-out probe accuracy by
layer. Interventions, applied at the last subtoken of every item mention in the prompt, the
chain and the answer pass: erase (gain 0), invert (−1), write toward a random ordering with |Spearman ρ| ≤ .2 at
strengths .25 to 1, and a random-direction control. Scored on the IWL test (direct), the
reversed context (direct and after 1,024 reasoning tokens) and the ICL test (after 1,024
tokens); the context conditions use the context-only instruction.

**Tests.** Each re-scores stored trials of a behavioral experiment:

- `iwl`: every trial of `symbolic_distance/iwl` (the main item set and its replicates), budget 0;
- `content_reversed`: `content_effects/content`, reversed assignment, premise orders 0–2, budget 0;
- `cot_reversed`: the reasoning-pass prompts of `content_effects/budget_reversed`, premise
  orders 0–2, budget 1,024;
- `cot_icl`: the reasoning-pass prompts of `symbolic_distance/budget_icl` (the given names),
  premise orders 0–2, budget 1,024.

**Settings.**

- Fit: each item is read at one query position A, three query positions B and three premise
  lines (prompt seed 42), at the last subtoken of the mention; the covariance is Ledoit–Wolf
  shrinkage with eigenvalues floored at 1e-8 of the largest; the rank is standardized over the
  fit set, ascending value.
- Probe: held-out pairwise ordering accuracy over items, five folds (seed 42); a ridge probe
  with alpha 10 for comparison.
- Edits: at every hidden state (indices 0–64: the token embeddings and the output of each
  block, the last before the final normalization), where the fit reads it; the random
  direction is a unit Gaussian direction seeded 1000 + layer (numpy `default_rng`), under the
  same covariance transform; the written ordering is a random order of the trial's 11 items
  with |Spearman ρ| ≤ .2 to the real ordering, seeded by the CRC-32 of the trial's set and
  trial ids.
- Chains: 1,024 tokens, read out at 256, 512 and 1,024 (the table keeps 0 and 1,024); batch 16.

**Declaration.** `intervention.yaml` (not a behavioral test file): the dimensions, the
conditions of each test and where the stored runs live. Stages:
`geo-alg-ti-llms represent fit | probe | intervene` (`scripts/slurm/interventions.sh`).

**Tables.** `results/representation/tables/trials.parquet` (9,450 rows: one per scored trial
and intervention condition), `probe.parquet` (195 rows: three dimensions × 65 hidden states),
`MANIFEST.yaml` (records the panel values recomputed from the table).
`summary.csv` and `fig_app_representation_data.csv` under the run directories are files of the
published runs that this repository does not write; the export compares its table against them
when present and records their absence otherwise.

The runs used the same software stack as the behavioral runs. The paper's panels pool
distances 2–6 of the innermost ranks.
