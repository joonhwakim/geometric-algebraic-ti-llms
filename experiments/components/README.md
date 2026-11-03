# components

**Question.** Do specific attention heads that read the context relations during reasoning
carry the algebraic strategy? (The head-ablation bars of the mechanisms figure's LLM panel
and of the appendix figure "LLM mechanism results: Qwen3-32B".)

**Design.** Qwen3-32B, the same three dimensions and the same four tests as `representation`
(see its README for the stored trials each test re-scores). Every one of the 4,096 heads is
scored on 252 reasoning chains (reversed context + ICL test) by its attention mass from the
tokens of every composed inference in the chain onto the context relations mentioning either
item (`components.score_heads`). The 32 highest-scoring heads are ablated together by
replacing each head's output with its mean over all positions; the control ablates 32 heads
drawn at random from outside the top 64 (`components.run_ablation`).

**Settings.**

- Scoring: the 1,024-token chains of `cot_reversed` and `cot_icl` (premise orders 0–2, the
  innermost seven ranks), eager attention, at most 1,200 tokens. A composed inference is a
  relational statement in the chain whose two items are two or more context ranks apart (a
  chained statement such as "A > B > C" is read as its first relation only); a head's score is
  its attention mass on the relevant premises at those tokens, averaged over tokens and chains;
  restatements of a shown relation and every other chain token are the controls.
- Ablation: each ablated head's output is replaced by its mean vector from the scoring pass
  (one mean per head over all positions and chains), at every position of the prompt, every
  generated token and the answer pass.
- Head sets: top-k and random-k for k in 8, 16, 32 and 64; random sets are drawn without
  replacement from the 4,032 heads outside the top 64, seeded 100 k + seed (numpy
  `default_rng`); the sets of 16 and 32 heads (seed 0) were ablated on every test and those
  of 8 on the two direct tests; k = 64 was drawn but not ablated. `heads.parquet` also lists
  a second random draw (seed 1) from the published runs that was not used; the code draws
  only seed 0.
- Chains: 1,024 tokens, read out at 256, 512 and 1,024 (the table keeps 0 and 1,024); batch 16.

**Declaration.** `intervention.yaml`: the dimensions, the head sets scored on each test and
where the stored runs live. Stages: `geo-alg-ti-llms components score | ablate`
(`scripts/slurm/interventions.sh`).

**Tables.** `results/components/tables/trials.parquet` (8,778 rows), `heads.parquet`
(1,080 rows: every head set drawn), `MANIFEST.yaml`.
`summary.csv` and `fig_app_representation_data.csv` under the run directories are files of the
published runs that this repository does not write; the export compares its table against them
when present and records their absence otherwise.

The runs used the same software stack as the behavioral runs. The paper's panels pool
distances 2–6 of the innermost ranks.
