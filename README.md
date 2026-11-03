# geometric-algebraic-ti-llms

Large language model experiments for the paper *Geometric and algebraic
inference in humans and neural networks*.

## What this is

The transitive-inference tasks the paper gave 22 open-weight language models,
the code that scored them (direct answers and chain-of-thought at ten
reasoning budgets), the two mechanistic interventions on Qwen3-32B, and the
tidy tables of every trial the paper reports, committed under `results/`.
Everything the paper reports about LLMs is computed from those tables; the
per-trial runs themselves (prompts and reasoning chains, 28.2 GB) are not
distributed. The preliminary prompt-design sweep and the item-spacing pilot
the methods describe are not included either; their outcomes are the prompt
format of the declarations and the item sets' provenance records.

## Install

`uv sync` installs the locked analysis dependencies (matplotlib, numpy,
pandas, pyarrow, pyyaml) for reading the tables, drawing the figures,
recomputing the paper's numbers and running the tests. Running a model needs
the inference extra and a GPU: see `REPRODUCE.md`.

## Quickstart

`uv run geo-alg-ti-llms jobs --experiment symbolic_distance` lists the runs
the experiment declares; `uv run geo-alg-ti-llms run-direct --experiment
symbolic_distance --item-set countries --test iwl --model qwen3-32b --out
runs/iwl --dry-run` prints the trial grid without loading a model;
`uv run geo-alg-ti-llms smoke` runs the whole pipeline (trials, prompts,
scoring, stored rows, export, manifest) on a stub model in seconds;
`uv run geo-alg-ti-llms figure --name symbolic_distance` redraws a figure from
the committed tables into `figures/out/`; `uv run geo-alg-ti-llms numbers`
recomputes every number the paper prints about the models into
`results/numbers.csv`; `uv run pytest` checks the package, every committed
table against its manifest, every figure and every number.

## Repository layout

`src/geo_alg_ti_llms/` the package (`items`, `trials`, `prompts`, `models`,
`scoring`, `results`, `run`: one run; `experiments`: the declarations;
`tables`, `traces`, `interp`: the exports; `represent`, `components`: the
interventions; `analysis`: the estimators; `roster`: the models and
dimensions as the figures name them; `plots/`: the panel functions;
`numbers`: the paper's numbers; `cli`); `configs/models/` one YAML per model;
`data/items/` the item sets and their provenance; `experiments/` one
declaration and a README per experiment; `results/<experiment>/tables/` the
committed tables with a sha256 manifest; `results/numbers.csv` every number
the paper prints about the models, printed against recomputed; `figures/`
one script per figure (`fig_<topic>.py`); `scripts/slurm/` the cluster
launcher; `tests/`.

## Experiments

- `symbolic_distance` — the IWL test (familiar items, no relations shown) and
  the ICL test (novel items, all adjacent relations shown), direct and at ten
  reasoning budgets.
- `content_effects` — congruent, reversed and shuffled contexts over familiar
  items, an uninformative-context control, and reasoning-budget ladders.
- `prior_strength` — the same direct tests on 19 item sets of other value
  spacings.
- `reasoning_traces` — inferences composed in the chains, and the four
  example traces.
- `representation` — the rank direction in Qwen3-32B: probe by layer, erase,
  invert, overwrite.
- `components` — the context-reading heads of Qwen3-32B: scoring and mean
  ablation.

The paper's five dimensions are `countries`, `citieslinear`, `animals`, `usstates`,
`countryarea`; `cities`/`cities_rep2` are the constant-ratio item sets: the behavioral tables carry
them, and of the paper's analyses only `representation` and `components` use them.

## Tables

Each `results/<experiment>/tables/` holds one row per scored trial (direct
answers) or per trial and reasoning budget (ladders): the model, test,
condition, item set, queried ranks and distance, the gold and predicted
truth values, the log-probabilities of True and False, and for reasoning rows
the chain length and whether it was cut at the budget. In the paper's words,
`no_context` is the IWL test, `icl_people` and `icl_cities` are the ICL test's
given names and invented city names, `unrelated_context` is the uninformative
context, and `budget_natural` is the congruent context's ladder. `MANIFEST.yaml`
records every table's rows, columns, sha256, the models, tests and conditions it
carries, per trial set row counts, and the invariants asserted at export.
`uv run geo-alg-ti-llms export --experiment <name> --runs <dir>` regenerates a
table from a runs directory in the published layout.

## Figures

Panel functions live in `geo_alg_ti_llms.plots`; `figures/fig_<topic>.py`
redraws each paper figure's LLM panels from the committed tables.

## Reproducing the paper

See `REPRODUCE.md`.

## Related repositories

Code for the paper's other experiments:

- Transformers: https://github.com/jlrussin/geometric-algebraic-ti-transformers
- Humans: https://github.com/jlrussin/geometric-algebraic-ti-humans

## Citation

Russin, J., Kim, J., Pavlick, E., & Frank, M. J. (2026). Geometric and algebraic
inference in humans and neural networks. arXiv:2611.00387.
https://arxiv.org/abs/2611.00387

See `CITATION.cff`.

## License

MIT, for the code. The source values in `data/items/sources/` remain under their
sources' terms (`data/items/sources/PROVENANCE.md`).

**Development notes.** This code was developed with substantial assistance
from AI coding tools (Claude Code), used for drafting code, tests,
documentation, and analysis scripts. The human authors designed the
experiments, specified and reviewed all code, and are responsible for its
correctness.
