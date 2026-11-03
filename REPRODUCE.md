# Reproducing the LLM results

## Environment

The analysis side (tables, tests, exports) needs Python 3.11 and the
analysis dependencies locked in `uv.lock`. Install
[uv](https://docs.astral.sh/uv/) (for example `pip install uv`), then `uv sync`
in the checkout creates `.venv` with them. The
commands below are written as the installed `geo-alg-ti-llms` executable: run
them inside the environment (`source .venv/bin/activate`) or prefix them with
`uv run`.

The published runs used one software stack on one GPU type: NVIDIA RTX PRO
6000 Blackwell Max-Q (96 GB), CUDA 12.8, torch 2.8.0, transformers 5.6.2,
tokenizers 0.22.2, accelerate 1.13.0, numpy 2.4.4, every model in bfloat16
(gpt-oss-20b dequantized from its released MXFP4 weights), greedy decoding.
`uv sync --extra inference` installs the inference stack at those versions
(numpy at the locked 2.4.6 on Python 3.11); the torch wheel must match your CUDA. Model weights come
from the Hugging Face Hub (`HF_HOME`) at the commit each `configs/models/<model>.yaml` pins
(`revision:`); Llama and Gemma weights are gated.

| Model | Hugging Face ID | Revision | Weights | License | Access |
|---|---|---|---|---|---|
| deepseek-r1-distill-llama-8b | deepseek-ai/DeepSeek-R1-Distill-Llama-8B | 6a6f4aa | bf16 | MIT (base model under the Llama 3.1 license) | open |
| deepseek-r1-distill-qwen-14b | deepseek-ai/DeepSeek-R1-Distill-Qwen-14B | 1df8507 | bf16 | MIT (base model under Apache-2.0) | open |
| deepseek-r1-distill-qwen-32b | deepseek-ai/DeepSeek-R1-Distill-Qwen-32B | 711ad2e | bf16 | MIT (base model under Apache-2.0) | open |
| gemma-2-9b-it | google/gemma-2-9b-it | 11c9b30 | bf16 | Gemma terms of use | gated |
| glm-4-32b-0414 | zai-org/GLM-4-32B-0414 | 077b5c2 | bf16 | MIT | open |
| glm-4-9b-0414 | zai-org/GLM-4-9B-0414 | 645b848 | bf16 | MIT | open |
| gpt-oss-20b | openai/gpt-oss-20b | 6cee5e8 | bf16, dequantized from the released MXFP4 weights | Apache-2.0 | open |
| llama-3.1-70b-instruct | meta-llama/Llama-3.1-70B-Instruct | 1605565 | bf16, two GPUs | Llama 3.1 Community License | gated |
| llama-3.1-8b-instruct | meta-llama/Llama-3.1-8B-Instruct | 0e9e39f | bf16 | Llama 3.1 Community License | gated |
| llama-3.2-1b | meta-llama/Llama-3.2-1B-Instruct | 9213176 | bf16 | Llama 3.2 Community License | gated |
| llama-3.2-3b | meta-llama/Llama-3.2-3B-Instruct | 0cb88a4 | bf16 | Llama 3.2 Community License | gated |
| mistral-7b-instruct-v0.3 | mistralai/Mistral-7B-Instruct-v0.3 | c170c70 | bf16 | Apache-2.0 | open |
| olmo-2-32b-instruct | allenai/OLMo-2-0325-32B-Instruct | b960243 | bf16 | Apache-2.0 | open |
| olmo-2-7b-instruct | allenai/OLMo-2-1124-7B-Instruct | 470b1fb | bf16 | Apache-2.0 | open |
| phi-4-reasoning | microsoft/Phi-4-reasoning | 1de18ec | bf16 | MIT | open |
| qwen2.5-32b-instruct | Qwen/Qwen2.5-32B-Instruct | 5ede1c9 | bf16 | Apache-2.0 | open |
| qwen2.5-7b-instruct | Qwen/Qwen2.5-7B-Instruct | a09a354 | bf16 | Apache-2.0 | open |
| qwen3-1.7b | Qwen/Qwen3-1.7B | 70d244c | bf16 | Apache-2.0 | open |
| qwen3-4b | Qwen/Qwen3-4B | 1cfa9a7 | bf16 | Apache-2.0 | open |
| qwen3-8b | Qwen/Qwen3-8B | b968826 | bf16 | Apache-2.0 | open |
| qwen3-14b | Qwen/Qwen3-14B | 40c0698 | bf16 | Apache-2.0 | open |
| qwen3-32b | Qwen/Qwen3-32B | 9216db5 | bf16 | Apache-2.0 | open |

Greedy reasoning chains are reproducible bitwise only on the same GPU type and
software versions; direct answers and accuracies agree within binomial noise
across GPU types.

## Smoke test

    geo-alg-ti-llms smoke

Runs the whole pipeline on a stub model in seconds (no GPU, no torch): the IWL and ICL
tests and a two-budget IWL ladder on country population, written in the published layout
under a temporary directory, exported to tables and checked against their manifest.

## Running the tasks

A run is one (experiment, item set, test, model):

    geo-alg-ti-llms run-direct --experiment symbolic_distance --item-set countries --test iwl --model qwen3-32b --out <dir>
    geo-alg-ti-llms run-ladder --experiment content_effects --item-set usstates --test budget_reversed --model qwen3-32b --out <dir> --gen-batch-size 16

`run-direct` writes one JSONL file per distance (every queryable pair, both
polarities, every premise order) with one row per trial: the prompt, the
log-probabilities of True and False at the answer position, the prediction and
its correctness. `run-ladder` writes one file per distance and budget: every
trial's 1,024-token greedy chain is generated once, and each budget scores
its first B tokens followed by "Therefore the answer is:" (budget 0 is the
direct pass, repeated in every trial set). `--dry-run` prints the trial grid without
loading a model. `jobs --experiment <name>` lists every (item set, test) of an
experiment; `prior_strength` uses set keys such as `countries_scale2`.

Native-thinking models (Qwen3, DeepSeek-R1 distills, Phi-4-reasoning, gpt-oss)
reason in their own thinking block with no trigger, and have it closed on the
direct pass; the other models are prompted with "Let's think step by step."
(`configs/models/<model>.yaml`, `thinking_mode`).

## Running on a cluster

Copy `scripts/slurm/env.example.sh` to `.env` and fill in the account,
partition, results root and cache. `scripts/slurm/submit.sh <experiment>
[models ...]` submits one job per (item set, test, model), writing the published
layout `<RUNS_ROOT>/<nxhop | cot_sweep>/final_<set key>/<test>/<model>/*.jsonl`
(`nxhop` holds the direct tests, `cot_sweep` the ladders; the directory names
are the published runs' and are listed by `jobs`). Sizes: a 32B ladder (720
trials, one 1,024-token chain each, batches of 16) takes up to 16 h on one
96 GB GPU; a direct test minutes. Llama-3.1-70B-Instruct is split over two
GPUs (`n_gpus` in its config). The published behavioral runs come to 2,200
jobs (100 per model) and the 28.2 GB of JSONL the tables are exported from.

The interventions (`scripts/slurm/interventions.sh`) run on Qwen3-32B after
the behavioral runs they re-score exist, one job per stage and dimension
(`DIM=countries | cities | usstates`). The job writes its log under
`slurm_logs/`, which must exist before submission:

    mkdir -p slurm_logs; source .env
    S="sbatch --account=$SLURM_ACCOUNT --partition=$SLURM_PARTITION ${SLURM_QOS:+--qos=$SLURM_QOS} ${SLURM_CONSTRAINT:+--constraint=$SLURM_CONSTRAINT}"
    STAGE=fit       DIM=countries $S scripts/slurm/interventions.sh
    STAGE=intervene DIM=countries $S --dependency=afterok:<fit job> scripts/slurm/interventions.sh
    STAGE=score     DIM=countries $S scripts/slurm/interventions.sh
    STAGE=ablate    DIM=countries $S --dependency=afterok:<score job> scripts/slurm/interventions.sh

## Exporting the tables

    export GEO_ALG_TI_LLMS_PUBLISHED_RUNS=<RUNS_ROOT>
    geo-alg-ti-llms export --experiment symbolic_distance
    geo-alg-ti-llms export --experiment symbolic_distance --what prompt-examples
    geo-alg-ti-llms export --experiment content_effects
    geo-alg-ti-llms export --experiment prior_strength
    geo-alg-ti-llms export --experiment content_effects --what labels --check-ladders
    geo-alg-ti-llms export --experiment reasoning_traces --what steps
    geo-alg-ti-llms export --experiment reasoning_traces --what examples
    geo-alg-ti-llms export --experiment representation
    geo-alg-ti-llms export --experiment components

Each export asserts the invariants of the runs it reads (every test present
for every model, complete ladders, the queried pair of every reasoning row
equal to its direct twin's, the per trial set trial counts) and writes the table and
its manifest under `results/<experiment>/tables/`. The labels come after both
trial tables they label (content effects and prior strength). The committed tables were
exported from the published runs in October 2026.

Two preliminary steps the paper's methods describe are not part of this
repository: the sweep over prompt designs that chose the prompt format, and the
pilot on Qwen3-32B that set the spacing of the familiar items. Their outcomes are
here: the prompt format is the one every declaration uses, and each item set's
provenance record (`data/items/sources/<item set>_provenance.json`) holds the
values and the spacing the pilot chose.

## Numbers

    geo-alg-ti-llms numbers

recomputes every number the paper prints about the language models from the
committed tables and writes `results/numbers.csv`, one row per number: `id`,
`site` (the section or caption of the paper), `printed`, `computed`, `source`
(the value of `geo_alg_ti_llms.numbers.values` it comes from) and `match`
(whether `computed`, written at the printed precision, equals `printed`). A
claim the paper states as a bound is printed as `>= x` or `< x`. It exits with
an error if a number differs. It takes about 30 s on one CPU core.

## Figures

    geo-alg-ti-llms figure --name <topic>

(or `python figures/fig_<topic>.py`) redraws the LLM panels of one paper
figure from the committed tables and writes a PDF at print size plus a PNG
preview to `figures/out/`. A per-model figure is drawn as its four pages, one
per model family of its caption title (`--page qwen3`: Qwen3;
`deepseek_phi_gptoss`: DeepSeek, Phi, and gpt-oss; `qwen25_glm_gemma_mistral`:
Qwen2.5, GLM, Gemma, and Mistral; `llama_olmo`: Llama and OLMo);
`example_traces` writes the four traces as text. The figures
with reasoning budgets bootstrap every distance 1,000 times over trials (seed
42), under a minute per page; the others take seconds.

| Topic (`fig_<topic>.py`) | Figure | Tables |
|---|---|---|
| `symbolic_distance` | Symbolic distance effects (SDE): one model's accuracy and reasoning budget by distance, each test's accuracy at 1,024 tokens beside its line, and the ICL test's best accuracy so far by budget; the roster scatters | `symbolic_distance/direct`, `ladders` |
| `content_effects` | Content effects in transitive inference: one model's accuracy and reasoning budget by distance, each context's accuracy at 1,024 tokens beside its line, and the congruent and reversed contexts' best accuracy so far by budget; the roster scatters | `content_effects/direct`, `ladders`; `symbolic_distance/direct`; `prior_strength/direct` |
| `mechanisms` | Mechanisms underlying the geometric and algebraic strategies: the intervention bars | `representation/trials`; `components/trials` |
| `symbolic_distance_by_model` | LLM symbolic-distance results: *family* models (appendix): accuracy by distance, every model and dimension | `symbolic_distance/direct` |
| `reasoning_budget_by_model` | LLM reasoning requirements: *family* models (appendix): reasoning budget by distance, every model and dimension | `symbolic_distance/ladders` |
| `icl_accuracy_by_budget` | LLM ICL accuracy by reasoning budget (appendix): one panel per model | `symbolic_distance/ladders` |
| `content_effects_by_model` | LLM content effects: *family* models (appendix): every model and dimension | `content_effects/direct`; `symbolic_distance/direct` |
| `recovery_by_model` | Recovery with reasoning: *family* models (appendix): every model and dimension | `content_effects/ladders` |
| `reasoning_budget_contexts_by_model` | LLM reasoning requirements under content conditions: *family* models (appendix): every model and dimension | `content_effects/ladders` |
| `representation` | LLM mechanism results: Qwen3-32B (appendix): the mechanism results in detail | `representation/probe`, `trials`; `components/trials` |
| `composed_inferences` | Composed inferences in LLM reasoning chains (appendix) | `reasoning_traces/chain_steps` |
| `example_traces` | Example reasoning traces (appendix): the four traces | `reasoning_traces/trace_examples.json` |

The paper's five dimensions are `countries`, `citieslinear`, `animals`, `usstates`,
`countryarea`; `cities`/`cities_rep2` are the constant-ratio item sets: the behavioral tables carry
them, and of the paper's analyses only `representation` and `components` use them. The ICL test
was run with the country population runs, so its rows carry `dim = countries`; a per-model figure
draws it in every column of a row.

Analyses (`geo_alg_ti_llms.analysis`): accuracy per distance with 68 % Wilson
intervals; the reasoning budget required at a distance is the smallest budget
whose accuracy reaches 90 % of the maximum over budgets, with no budget when
the maximum's 95 % Wilson lower bound is at or below .5, and a band of plus and
minus its bootstrap standard deviation over trials; the best accuracy so far at
a budget is the highest accuracy at that budget or below, with the Wilson
interval of the first budget that reached it; a model is left out of the
IWL-test scatters when its IWL accuracy over the first two item sets of every
dimension (378 trials) is not above chance by an exact one-sided binomial
test; cross-model slopes are least squares on distance, and a slope within
1e-12 of zero is zero.

## Tests

`uv run pytest` runs the CPU tests: trial generation and prompt rendering
(fingerprints of published trial sets), scoring arithmetic on a stub model, the
export invariants on synthetic rows, the LEACE closed form, the estimators and
panel functions on synthetic rows, every figure script on the committed tables,
`results/numbers.csv` against a fresh recomputation, and every committed table
against its manifest. Tests that need the published runs
(`GEO_ALG_TI_LLMS_PUBLISHED_RUNS`) or the inference extra (torch, scipy) are
skipped when they are absent.

## What has been checked without a GPU

The analysis side was re-run from the published runs on CPU: every table and
manifest exported in the order above is byte-identical to the committed one (apart from the
export date), `results/numbers.csv` matches every printed number, and every figure page is
byte-identical as a PNG and, apart from its creation date, as a PDF drawn with
`PYTHONHASHSEED=0` (without it, the font subsets embedded in some PDF pages differ between runs).
The inference side is checked without model weights by `tests/test_published_runs.py`: trial sets regenerated from the declarations
match sha256 fingerprints taken from the published runs, and, where the runs are available, the
regenerated trials and prompts reproduce the stored rows of a sample of trial files (with the
inference extra, where a model's tokenizer loads, the full prompts rendered with its chat
template; otherwise, the instruction and the user turn). Not checked: the log-probabilities, the answers and the
generated chains themselves, which need the GPU runs described above.

## Expected runtime and storage

Analysis side, measured on one CPU core:

| Step | Time | Storage |
|---|---|---|
| `uv sync` (analysis dependencies) | under a minute (13 s with a warm cache) | 0.35 GB environment |
| `geo-alg-ti-llms smoke` | 2 s | a few MB under the system temporary directory |
| `uv run pytest` | about 2.5 minutes | none |
| `geo-alg-ti-llms numbers` | about 30 s | `results/numbers.csv` |
| `geo-alg-ti-llms figure` | seconds per figure; under a minute per page with reasoning budgets (1,000 bootstrap draws) | a PDF and a PNG per page under `figures/out/` |

The committed tables and manifests take 59 MB; the checkout about 61 MB.

Model side, from the published runs (NVIDIA RTX PRO 6000 Blackwell Max-Q, 96 GB):

| Step | Time | Storage |
|---|---|---|
| A direct test, one model and item set | minutes | 0.1 MB (IWL test) to 5 MB (content conditions) of JSONL |
| A reasoning-budget ladder of a 32B model (720 trials, one 1,024-token chain each, batches of 16) | up to 16 h on one GPU | about 50 MB of JSONL |
| Every behavioral run (2,200 jobs) | | 28.2 GB of JSONL |
| The rank-direction fit and interventions on Qwen3-32B, one dimension | 4 to 5.5 h on one GPU | about 470 MB |
| A head-scoring or ablation job on Qwen3-32B, one dimension | 10 minutes to 1.5 h on one GPU | about 15 MB per dimension |
| Exporting every table from the runs | about 30 minutes on 8 CPU cores, 64 GB memory | the committed tables |
| The 22 models' weights in bfloat16 | | about 730 GB under `HF_HOME` |

