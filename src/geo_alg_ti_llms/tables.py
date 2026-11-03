"""The stored per-trial runs → the committed tidy tables, with manifests.

A published run is one directory of JSONL files per (dimension, test,
model): ``<runs>/<tree>/final_<dim>/<published_as>/<model>/*.jsonl``, where the
tree is ``nxhop`` for a direct test and ``cot_sweep`` for a reasoning-budget
ladder. :func:`export_trials` reads the tests one experiment declares and writes,
under ``results/<experiment>/tables/``:

* ``direct.parquet``   one row per scored trial of every direct test (budget 0);
* ``ladders.parquet``  one row per trial × reasoning budget of every ladder test,
  budget 0 the direct pass collapsed to one row per trial (every budget
  re-scores the same direct pass);
* ``pools.parquet``    (prior strength) one row per item-spacing set;
* ``query_labels.parquet`` (content effects) every shuffled trial's label: does
  the queried pair's shown order agree with the real one (:func:`export_query_labels`);
* ``prompt_examples.json`` (symbolic distance) one verbatim stored prompt per
  condition for the two main-text models (:func:`export_prompt_examples`);
* ``MANIFEST.yaml``    per table: rows, sha256, columns, the tests, conditions
  and models carried, per trial set row counts and digests, the row guards asserted.

Prompt and trace text are dropped; everything a figure or a statistic needs is
kept. A ladder row's queried pair is read from the question of the user turn,
located before the reasoning trace (a trace may restate the question, with a
wrong item), and asserted against the trial's direct twin, the generator's own
record of it. Every invariant is asserted; a failure names the offending test.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from geo_alg_ti_llms import experiments
from geo_alg_ti_llms.models import DEFAULT_MODELS_DIR, load_model_config

PACKAGE_VERSION = "0.1.0"
RUNS_ENV = "GEO_ALG_TI_LLMS_PUBLISHED_RUNS"
"""Environment variable naming the root of the published runs."""
RUNS_ROOT_PLACEHOLDER = f"<{RUNS_ENV}>"
"""What the manifests record as ``runs_root``: the variable, not the exporting machine's path."""

_REPO = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR: Path = _REPO / "results"
DEFAULT_ITEMS_DIR: Path = _REPO / "data" / "items"

PUBLISHED_TREES = {"direct": "nxhop", "ladder": "cot_sweep"}
"""Results subtree of each runner in the published runs."""

N_ITEMS = 11
MASTER_SEED = 42
INNERMOST_RANKS = 7
DROP_LO = (N_ITEMS - INNERMOST_RANKS) // 2          # 2
DROP_HI = N_ITEMS - 1 - DROP_LO                     # 8
QUERYABLE_PAIRS = 2 * sum(N_ITEMS - 2 - d for d in range(1, N_ITEMS - 2))   # 72
DIRECT_DUPLICATE_TOLERANCE = 0.01                   # bf16 tie-flip tolerance
SPACING_SENS_D = 6              # the distance the spacing sets' value ratio is calibrated at

CTXONLY = "_ctxonly"
CTXONLY_SENTENCE = "Base your answer only on the comparisons given — not on outside knowledge."
# Every prompt asks its question in this wording (the clause the queried pair is read
# from) and carries the instruction's closing sentence; a prompt without either was
# built or stored wrongly.
QUERY_PHRASE = "true or false that"
INSTRUCTION_PHRASE = "Answer whether the statement is true or false."
# Byte-level BPE stand-ins for a space and a newline. A reasoning trace that carries
# them was decoded token by token without the tokenizer's byte map, so the scored
# prompt did not hold the text the model generated.
BPE_MARKERS = ("Ġ", "Ċ")    # Ġ (space), Ċ (newline)
REASONING_KIND = {"none": "native", "passive": "prompted"}

_QUERY_RE = re.compile(r"true or false that\s+(.+?)\s*[\?\.]", re.IGNORECASE | re.DOTALL)
_BUDGET_RE = re.compile(r"^b(\d+)$")

COLUMNS = [
    "model", "reasoning_kind", "test", "condition", "item_pool", "dim",
    "d", "signed_distance", "rank_low", "rank_high", "innermost", "premise_order",
    "trial_id", "trial_key", "mode", "budget", "gold_true", "predicted_true", "is_correct",
    "logprob_true", "logprob_false", "logit_gap", "cot_n_tokens", "cot_finish_reason",
    "truncated",
]
DTYPES = {
    "model": "string", "reasoning_kind": "string", "test": "string", "condition": "string",
    "item_pool": "string", "dim": "string", "d": "int8", "signed_distance": "int8",
    "rank_low": "int8", "rank_high": "int8", "innermost": "bool", "premise_order": "int8",
    "trial_id": "int16", "trial_key": "string", "mode": "string", "budget": "int16",
    "gold_true": "bool", "predicted_true": "bool", "is_correct": "bool",
    "logprob_true": "float32", "logprob_false": "float32", "logit_gap": "float32",
    "cot_n_tokens": "float32", "cot_finish_reason": "string", "truncated": "bool",
}
SPACING_COLUMNS = ["pool_kind", "value_ratio", "spacing"]
SPACING_DTYPES = {"pool_kind": "string", "value_ratio": "float32", "spacing": "string"}
_SORT = ["model", "test", "dim", "budget", "source_file", "source_line"]
TRIAL_KEY_FORMAT = "<dim>:<surface_form_mode>|<assignment_strategy>|<symbolic_distance>|<trial_id>"
INNERMOST_RULE = (f"rank_low >= {DROP_LO} and rank_high <= {DROP_HI} "
                  f"(innermost {INNERMOST_RANKS} of {N_ITEMS}; distances 1-6)")
ROW_ORDER = ("within (model, test, dim, budget): the stored files in name order, then the "
             "rows in file order")
TEXT_COLUMNS_DROPPED = ["prompt", "cot_trace", "cot_stage1_prompt",
                        "extra (rank_to_item, actual_items, exemplar_items)", "prompt_spec",
                        "cell_id", "timestamp_iso", "duration_seconds"]
ROW_GUARDS = [
    "N == 11 and master_seed == 42 on every row",
    "the file's hop, symbolic_distance, extra.hop and prompt_spec.hop agree",
    "item_pool and surface_form_mode are the test's; the assignment is one the test declares",
    "prompt_spec.instructions_v1_id ends in '_ctxonly' exactly when the test's instruction is "
    "the context-only one, and the context-only sentence is in the prompt exactly then",
    f"every prompt contains {QUERY_PHRASE!r} and {INSTRUCTION_PHRASE!r}",
    "an IWL test row (surface zero_context) shows no '<item> > <item>' or '<item> < <item>' "
    "relation before the question, in its prompt or, for a reasoning row, its reasoning-pass "
    "prompt",
    "rank_high - rank_low == symbolic_distance (a ladder row's ranks read from the question "
    "before the trace)",
    "a reasoning row's prompt_spec.max_new_tokens is its file's budget; its trigger is the "
    "model's (none on native thinking models, passive otherwise)",
    "no reasoning trace carries byte-level BPE markers (U+0120, U+010A)",
    "every ladder row's queried ranks equal its direct twin's; every budget-0 row equals the "
    "twin in gold, answer, correctness and logit gap (1e-6) and takes its premise order index",
    "every trial of a ladder carries every budget; every direct trial of a condition with a "
    "ladder is in it",
    "(model, test, dim, budget, trial_key) is unique",
    "rows per (model, test, dim) = assignments x premise orders x 72 queryable pairs "
    "(x 11 budgets for a ladder); innermost trials per (dim, condition, distance) = "
    "2 x premise orders x (7 - distance)",
]


def fail(msg: str) -> None:
    raise AssertionError(msg)


def runs_root(runs: str | Path | None = None) -> Path:
    """The published runs: the argument, else ``$GEO_ALG_TI_LLMS_PUBLISHED_RUNS``."""
    root = runs or os.environ.get(RUNS_ENV)
    if not root:
        raise ValueError(f"no runs directory: pass --runs or set {RUNS_ENV}")
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"runs directory {root} does not exist")
    return root


def roster(models_dir: str | Path = DEFAULT_MODELS_DIR) -> list[str]:
    """Every model of ``configs/models``, by short name."""
    return sorted(p.stem for p in Path(models_dir).glob("*.yaml"))


def reasoning_trigger(model: str, exp: dict | None = None, models_dir: str | Path = DEFAULT_MODELS_DIR) -> str:
    """The trigger id a model's reasoning pass was run with (``none`` / ``passive``): the
    experiment's ``triggers:`` block when ``exp`` is given, else the declaration defaults."""
    cfg = load_model_config(Path(models_dir) / f"{model}.yaml")
    return experiments.trigger_for(exp or {}, cfg)


# --- the tests of an experiment ---------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TestSpec:
    """One test of an experiment as the stored runs lay it out."""

    test: str
    runner: str                        # direct | ladder
    published_as: str                  # the test's directory name in the published runs
    conditions: dict[str, str]         # assignment strategy -> condition name
    surface: str
    ctxonly: bool                      # the instruction is the context-only one
    n_premise_orders: int
    item_sets: dict[str, tuple[str, str]]  # dim (set key) -> (dimension, item set)

    @property
    def tree(self) -> str:
        return PUBLISHED_TREES[self.runner]

    @property
    def rows_per_set(self) -> int:
        """Scored trials per (model, item set): assignments x premise orders x queryable pairs."""
        return len(self.conditions) * self.n_premise_orders * QUERYABLE_PAIRS


def test_specs(exp: dict) -> dict[str, TestSpec]:
    """Every test of an experiment declaration, in declaration order."""
    out = {}
    for test, spec in exp["tests"].items():
        item_sets = experiments.test_set_keys(exp, test)
        q = exp["dimensions"][next(iter(item_sets.values()))[0]]
        instruction = experiments._test_instruction(spec, q)
        out[test] = TestSpec(
            test=test, runner=spec["runner"], published_as=spec.get("published_as", test),
            conditions=experiments._conditions(spec), surface=spec["surface"],
            ctxonly=instruction.endswith(CTXONLY),
            n_premise_orders=int(spec.get("n_premise_orders",
                                             exp["geometry"]["n_premise_orders"])),
            item_sets=item_sets,
        )
    return out


def twin_test(specs: dict[str, TestSpec], ladder: str) -> str:
    """The direct test whose rows are a ladder's own direct pass: same item sets, surface,
    instruction, and conditions (a one-condition ladder of a three-condition test)."""
    lad = specs[ladder]
    twins = [a for a, s in specs.items()
             if s.runner == "direct" and s.surface == lad.surface and s.ctxonly == lad.ctxonly
             and s.item_sets == lad.item_sets and set(lad.conditions.items()) <= set(s.conditions.items())]
    if len(twins) != 1:
        fail(f"ladder test {ladder!r} has {len(twins)} direct twins, expected one")
    return twins[0]


def dims_of(exp: dict) -> dict[str, tuple[str, str]]:
    """dim -> (dimension, item set) over every test of an experiment."""
    out: dict[str, tuple[str, str]] = {}
    for test in exp["tests"]:
        for dim, (dimension, item_set) in experiments.test_set_keys(exp, test).items():
            if item_set.endswith("_disjoint") or item_set.startswith("novel_"):
                continue                     # the test's own item set, not the dim's
            out.setdefault(dim, (dimension, item_set))
    return out


def test_dir(runs: Path, spec: TestSpec, dim: str, model: str) -> Path:
    return runs / spec.tree / f"final_{dim}" / spec.published_as / model


# --- reading one test directory ----------------------------------------------------------


def _query_ranks(prompt: str, rank_to_item: list[str]) -> tuple[int, int] | None:
    """(rank_low, rank_high) of the queried pair, or None if unrecoverable.

    The query is the LAST 'true or false that X <rel> Y' clause in the prompt
    (premises, when present, never use that wording).
    """
    ms = list(_QUERY_RE.finditer(prompt))
    if not ms:
        return None
    clause = ms[-1].group(1)
    hits = []
    for it in rank_to_item:
        m = re.search(rf"(?<!\w){re.escape(it)}(?!\w)", clause)
        if m:
            hits.append((m.start(), it))
    hits.sort()
    seen, ordered = set(), []
    for _, it in hits:
        if it not in seen:
            seen.add(it)
            ordered.append(it)
    if len(ordered) != 2:
        return None
    a, b = (rank_to_item.index(ordered[0]), rank_to_item.index(ordered[1]))
    return (min(a, b), max(a, b))


def _budget_of(stem: str) -> int | None:
    """'b256__N=11__hop=3__surface=fixed_gt' -> 256; 'main__...' -> None."""
    m = _BUDGET_RE.match(stem.split("__")[0])
    return int(m.group(1)) if m else None


def ladder_query_ranks(r: dict, where: str) -> tuple[int, int]:
    """(rank_low, rank_high) of a ladder row's question, read before the reasoning trace.

    The question is the one "true or false that ..." clause of the reasoning-pass prompt (the
    prompt the trace was generated from; premises never use the wording). The scored
    prompt is not always the reasoning pass + trace byte for byte -- some chat templates re-render
    the
    turns -- so the boundary is located instead: the reasoning-pass prompt from its clause to its
    end (the question, then the opening of the assistant turn) must appear verbatim in the
    scored prompt, and the scored prompt's FIRST clause must start there. Everything after
    it is the trace and the scoring suffix. A direct row has no trace: exactly one clause.
    """
    prompt, r2i = r["prompt"], r["extra"]["rank_to_item"]
    clauses = list(_QUERY_RE.finditer(prompt))
    if not clauses:
        fail(f"{where}: no query clause in the prompt")
    if r["mode"] == "cot":
        reasoning_prompt = r["cot_stage1_prompt"]
        head = list(_QUERY_RE.finditer(reasoning_prompt))
        if len(head) != 1:
            fail(f"{where}: {len(head)} query clauses in the reasoning-pass prompt, expected one")
        tail = reasoning_prompt[head[0].start():]
        at = prompt.find(tail)
        if at < 0 or clauses[0].start() != at:
            fail(f"{where}: the scored prompt's first query clause is not the reasoning-pass question "
                 "(the question is not located before the trace)")
        before_trace = prompt[:at + len(tail)]
    else:
        if len(clauses) != 1:
            fail(f"{where}: {len(clauses)} query clauses in a direct prompt, expected one")
        before_trace = prompt
    pair = _query_ranks(before_trace, r2i)       # the one clause before the trace
    if pair is None:
        fail(f"{where}: the question's two items are not in rank_to_item")
    return pair


def read_test(runs: Path, spec: TestSpec, dim: str, model: str,
             budgets: tuple[int, ...] = ()) -> tuple[list[dict], set[str]]:
    """Every row of one test directory as a slim dict; also the trigger ids seen."""
    runner, condition_of = spec.tree, spec.conditions
    set_exp, surface_exp = spec.item_sets[dim][1], spec.surface
    test, is_ctxonly = spec.test, spec.ctxonly
    d = test_dir(runs, spec, dim, model)
    files = sorted(d.glob("*.jsonl"))
    if not files:
        fail(f"{dim}/{test}/{model}: no jsonl files under {d}")
    rows, triggers = [], set()
    for f in files:
        budget = _budget_of(f.stem)
        if runner == "cot_sweep" and budget not in budgets:
            fail(f"{f}: ladder file without a known budget prefix")
        if runner == "nxhop" and budget is not None:
            fail(f"{f}: direct file carries a budget prefix")
        hop_token = int(f.stem.split("hop=")[1].split("__")[0])
        for line_no, line in enumerate(open(f, encoding="utf-8")):
            if not line.strip():
                continue
            r = json.loads(line)
            ex = r["extra"]
            spec_ = r["prompt_spec"]
            if r["N"] != N_ITEMS or r["master_seed"] != MASTER_SEED:
                fail(f"{f}: N={r['N']} master_seed={r['master_seed']}")
            sd = int(r["symbolic_distance"])
            if sd != hop_token or ex["hop"] != sd or spec_["hop"] != sd:
                fail(f"{f}: hop tokens disagree ({sd}, {hop_token}, {ex['hop']}, {spec_['hop']})")
            assign = ex["assignment_strategy"]
            surface = ex["surface_form_mode"]
            item_set = r["item_pool"]
            if item_set != set_exp or surface != surface_exp:
                fail(f"{f}: item set/surface {item_set}/{surface}, expected {set_exp}/{surface_exp}")
            cond = condition_of.get(assign)
            if cond is None:
                fail(f"{f}: assignment {assign!r}, expected one of {sorted(condition_of)}")
            # The instruction the trial was asked under: its id and its text must both say
            # context-only exactly when the test does.
            prompt = r["prompt"]
            instr = spec_["instructions_v1_id"]
            if instr.endswith(CTXONLY) != is_ctxonly:
                fail(f"{f}:{line_no}: instruction id {instr!r} under test {test!r}")
            if (CTXONLY_SENTENCE in prompt) != is_ctxonly:
                fail(f"{f}:{line_no}: the context-only sentence is "
                     f"{'missing from' if is_ctxonly else 'present in'} a prompt of test {test!r}")
            for phrase in (QUERY_PHRASE, INSTRUCTION_PHRASE):
                if phrase not in prompt:
                    fail(f"{f}:{line_no}: the prompt does not contain {phrase!r}")
            mode = r["mode"]
            if mode not in ("direct", "cot"):
                fail(f"{f}: mode {mode!r}")
            if surface == "zero_context":
                # The IWL test shows the query alone: no relation between two of the trial's
                # items precedes the question in the prompt answered or reasoned from.
                shown = r["cot_stage1_prompt"] if mode == "cot" else prompt
                question = _QUERY_RE.search(shown)
                before = shown[:question.start()] if question else ""
                if question is None or item_pattern(ex["rank_to_item"]).search(before):
                    fail(f"{f}:{line_no}: a relation precedes the question in an IWL test prompt")
            if mode == "cot":
                triggers.add(spec_["trigger_id"])
                if int(spec_["max_new_tokens"]) != budget:
                    fail(f"{f}: max_new_tokens {spec_['max_new_tokens']} != file budget {budget}")
                trace = r["cot_trace"] or ""
                if any(m in trace for m in BPE_MARKERS):
                    fail(f"{f}:{line_no}: the reasoning trace carries byte-level BPE markers "
                         f"(a trace decoded without the tokenizer's byte map)")
            if runner == "nxhop":
                rl, rh = int(ex["rank_low"]), int(ex["rank_high"])
                premise_order = int(ex["permutation_idx"])
            else:
                rl, rh = ladder_query_ranks(r, f"{f}:{line_no}")
                premise_order = -1                                # filled from the twin below
            if rh - rl != sd:
                fail(f"{f}:{line_no}: rank_high - rank_low = {rh - rl} != symbolic_distance {sd}")
            finish = r["cot_finish_reason"]
            rows.append(dict(
                model=model, reasoning_kind="", test=test, condition=cond, item_pool=item_set, dim=dim,
                d=sd, signed_distance=int(r["signed_distance"]), rank_low=rl, rank_high=rh,
                innermost=bool(rl >= DROP_LO and rh <= DROP_HI), premise_order=premise_order,
                trial_id=int(r["trial_id"]),
                trial_key=f"{dim}:{surface}|{assign}|{sd}|{r['trial_id']}",
                mode=mode, budget=(0 if mode == "direct" else int(budget)),
                gold_true=bool(r["correct"]), predicted_true=bool(r["predicted"]),
                is_correct=bool(r["is_correct"]), logprob_true=float(r["logprob_yes"]),
                logprob_false=float(r["logprob_no"]), logit_gap=float(r["logit_gap"]),
                cot_n_tokens=(np.nan if r["cot_n_tokens"] is None else float(r["cot_n_tokens"])),
                cot_finish_reason=finish, truncated=bool(finish == "length"),
                source_file=str(f.relative_to(runs)), source_line=line_no,
            ))
    return rows, triggers


# --- the ladder collapse and the row checks ---------------------------------------------


def collapse_ladder_direct(rows: list[dict], label: str, budgets: tuple[int, ...]) -> tuple[list[dict], dict]:
    """One budget-0 row per trial from the identical direct copies of every budget."""
    direct = [r for r in rows if r["mode"] == "direct"]
    cot = [r for r in rows if r["mode"] == "cot"]
    by_key: dict[str, list[dict]] = defaultdict(list)
    for r in direct:                                     # rows arrive in sorted-file order
        by_key[r["trial_key"]].append(r)
    copies = Counter(len(v) for v in by_key.values())
    if set(copies) != {len(budgets)}:
        fail(f"{label}: direct copies per trial {dict(copies)}, expected {len(budgets)}")
    n_bad = sum(1 for v in by_key.values() if len({r["is_correct"] for r in v}) > 1)
    frac_bad = n_bad / len(by_key)
    if frac_bad > DIRECT_DUPLICATE_TOLERANCE:
        fail(f"{label}: {n_bad}/{len(by_key)} trials disagree across budgets")
    kept = [v[0] for v in by_key.values()]
    for r in kept:
        r["budget"] = 0
    return kept + cot, {"n_trials": len(by_key), "n_direct_disagreements": n_bad}


def attach_twins(ladder_rows: list[dict], direct_rows: list[dict], label: str) -> dict:
    """Check every ladder trial against its direct twin and copy premise_order over.

    Every ladder row's queried ranks (read from the question before the trace) must equal
    the twin's, the generator's own record of the pair; the budget-0 row must equal the
    twin in answer, correctness, and logit gap too.
    """
    twin_by_key = {}
    for r in direct_rows:
        k = (r["model"], r["condition"], r["item_pool"], r["trial_key"])
        if k in twin_by_key:
            fail(f"{label}: duplicate direct trial key {k}")
        twin_by_key[k] = r
    seen = set()
    max_gap = 0.0
    for r in ladder_rows:
        k = (r["model"], r["condition"], r["item_pool"], r["trial_key"])
        twin = twin_by_key.get(k)
        if twin is None:
            fail(f"{label}: ladder trial {k} has no direct twin")
        r["premise_order"] = twin["premise_order"]
        if (r["rank_low"], r["rank_high"]) != (twin["rank_low"], twin["rank_high"]):
            fail(f"{label}: ladder ranks {(r['rank_low'], r['rank_high'])} != the direct twin's "
                 f"{(twin['rank_low'], twin['rank_high'])} at {k}, budget {r['budget']} "
                 f"({r['source_file']}:{r['source_line']})")
        if r["budget"] == 0:
            seen.add(k)
            same = (r["gold_true"] == twin["gold_true"] and r["predicted_true"] == twin["predicted_true"]
                    and r["is_correct"] == twin["is_correct"] and r["rank_low"] == twin["rank_low"]
                    and r["rank_high"] == twin["rank_high"])
            gap = abs(r["logit_gap"] - twin["logit_gap"])
            max_gap = max(max_gap, gap)
            if not same or gap > 1e-6:
                fail(f"{label}: ladder budget-0 row differs from its direct twin at {k}")
    # Every direct trial of a condition that HAS a ladder must appear in it (the
    # unrelated-context test was never run as a ladder, so it is exempt).
    ladder_conditions = {r["condition"] for r in ladder_rows}
    missing = {k for k in twin_by_key if k[1] in ladder_conditions} - seen
    if missing:
        fail(f"{label}: {len(missing)} direct trials have no ladder counterpart")
    return {"n_twins": len(seen), "max_abs_logit_gap_delta": max_gap,
            "n_ladder_rows_ranks_checked": len(ladder_rows),
            "conditions_without_ladder": sorted({r["condition"] for r in direct_rows} - ladder_conditions)}


def check_complete_ladders(ladder_rows: list[dict], label: str, full_ladder: tuple[int, ...]) -> None:
    """Every trial of every ladder carries every budget.

    Keyed by (model, test, condition, trial_key): two ladders of one experiment can share
    trial keys, so a key without the test and condition would let one ladder's budgets
    fill the other's gaps.
    """
    budgets_by_key: dict[tuple, set[int]] = defaultdict(set)
    for r in ladder_rows:
        budgets_by_key[(r["model"], r["test"], r["condition"], r["trial_key"])].add(r["budget"])
    bad = [k for k, b in budgets_by_key.items() if b != set(full_ladder)]
    if bad:
        fail(f"{label}: {len(bad)} trials with an incomplete ladder, e.g. {bad[0]}")


def check_unique_keys(rows: list[dict], label: str) -> None:
    c = Counter((r["model"], r["test"], r["dim"], r["budget"], r["trial_key"]) for r in rows)
    dup = [k for k, n in c.items() if n > 1]
    if dup:
        fail(f"{label}: {len(dup)} duplicated (test, budget, trial_key), e.g. {dup[0]}")


def check_trial_set_counts(df: pd.DataFrame, specs: dict[str, TestSpec], n_budgets: int, label: str) -> None:
    """Every (model, test, dim) carries the full grid of the test's trials, and every (model,
    dim, condition, distance) the expected innermost trials of the direct pass."""
    for (model, test, dim), n in df.groupby(["model", "test", "dim"], observed=True).size().items():
        want = specs[test].rows_per_set * (n_budgets if specs[test].runner == "ladder" else 1)
        if int(n) != want:
            fail(f"{label}: {model}/{dim}/{test}: {n} rows, expected {want}")
    inner = df[df["innermost"] & (df["budget"] == 0) & (df["d"] <= INNERMOST_RANKS - 1)]
    orders = {(a, c): s.n_premise_orders for a, s in specs.items() for c in s.conditions.values()}
    for (model, test, dim, cond, d), n in inner.groupby(
            ["model", "test", "dim", "condition", "d"], observed=True).size().items():
        want = 2 * orders[(test, cond)] * (INNERMOST_RANKS - d)
        if int(n) != want:
            fail(f"{label}: {model}/{dim}/{test}/{cond} at distance {d}: {n} innermost trials, "
                 f"expected {want}")


# --- frames, digests, manifests ---------------------------------------------------------


def to_frame(rows: list[dict], extra_columns: list[str] = ()) -> pd.DataFrame:
    """Typed frame of slim rows in the committed row order (sort columns kept)."""
    columns = [*COLUMNS, *extra_columns, "source_file", "source_line"]
    dtypes = {**DTYPES, **{c: SPACING_DTYPES[c] for c in extra_columns},
              "source_file": "string", "source_line": "int32"}
    df = pd.DataFrame(rows, columns=columns)
    if (df["premise_order"] < 0).any():
        fail("premise_order left unfilled on some ladder rows")
    df["cot_finish_reason"] = df["cot_finish_reason"].astype(object).where(df["cot_finish_reason"].notna(), None)
    df = df.astype(dtypes)
    return df.sort_values(_SORT, kind="stable").reset_index(drop=True)


def _cast(df: pd.DataFrame) -> pd.DataFrame:
    """The committed dtypes (a cached frame comes back from parquet with object strings)."""
    dtypes = {**DTYPES, **SPACING_DTYPES, "source_file": "string", "source_line": "int32"}
    return df.astype({c: dtypes[c] for c in df.columns if c in dtypes})


def finish_frame(df: pd.DataFrame) -> pd.DataFrame:
    """The committed table: sorted, then without the sort-only source columns."""
    df = _cast(df).sort_values(_SORT, kind="stable").reset_index(drop=True)
    return df.drop(columns=["source_file", "source_line"])


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def row_digest(df: pd.DataFrame) -> str:
    """sha256 over a frame's rows in order (NA as null, floats as doubles)."""
    h = hashlib.sha256()
    for row in df.itertuples(index=False, name=None):
        h.update(json.dumps([None if pd.isna(v) else (float(v) if isinstance(v, (float, np.floating))
                                                       else (int(v) if isinstance(v, (np.integer,)) else
                                                             (bool(v) if isinstance(v, (np.bool_,)) else v)))
                             for v in row], ensure_ascii=False).encode())
        h.update(b"\n")
    return h.hexdigest()


class Flow(list):
    """A list the manifest writes on one line (``[rows, sha256]`` of a trial set)."""


yaml.SafeDumper.add_representer(
    Flow, lambda d, data: d.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True))


class _ManifestLoader(yaml.SafeLoader):
    """safe_load that reads a one-line list back as a ``Flow``, so a manifest keeps its
    layout when a later export of another table rewrites it."""


_ManifestLoader.add_constructor(
    "tag:yaml.org,2002:seq",
    lambda loader, node: (Flow if node.flow_style else list)(loader.construct_sequence(node)))


def trial_set_digests(df: pd.DataFrame) -> dict[str, list]:
    """``"model|test|dim" -> [rows, sha256]`` of every trial set, each hashed over its own rows."""
    out = {}
    for (model, test, dim), g in df.groupby(["model", "test", "dim"], observed=True, sort=True):
        out[f"{model}|{test}|{dim}"] = Flow([int(len(g)), row_digest(g)])
    return out


def write_parquet(df: pd.DataFrame, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="zstd")
    return {"path": path.name, "rows": int(len(df)), "sha256": sha256(path),
            "bytes": int(path.stat().st_size)}


def innermost_counts(df: pd.DataFrame) -> dict[str, int]:
    """Innermost direct trials per (dim, condition, distance), asserted equal across models."""
    inner = df[df["innermost"] & (df["budget"] == 0)]
    per = inner.groupby(["model", "dim", "condition", "d"], observed=True).size()
    out: dict[str, int] = {}
    for (model, dim, cond, d), n in per.items():
        key = f"{dim}|{cond}|d={int(d)}"
        if out.setdefault(key, int(n)) != int(n):
            fail(f"innermost trials at {key} differ between models ({out[key]} vs {n} for {model})")
    return out


def table_entry(name: str, df: pd.DataFrame, written: dict, note: str, **more) -> dict:
    """One ``tables`` entry of a manifest."""
    return {
        "table": name, **written,
        "columns": {c: str(df[c].dtype) for c in df.columns},
        "tests": sorted(df["test"].unique()) if "test" in df else [],
        "conditions": sorted(df["condition"].unique()) if "condition" in df else [],
        "models": sorted(df["model"].unique()) if "model" in df else [],
        "dims": sorted(df["dim"].unique()) if "dim" in df else [],
        **more,
        "note": note,
    }


def load_manifest(path: Path) -> dict:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return yaml.load(f, Loader=_ManifestLoader) or {}
    return {}


def update_manifest(path: Path, header: dict, entries: list[dict]) -> dict:
    """Rewrite the header fields and replace the entries of the tables named."""
    manifest = load_manifest(path)
    tables = [t for t in manifest.get("tables", []) if t["table"] not in {e["table"] for e in entries}]
    tables += entries
    out = {"package_version": PACKAGE_VERSION, **{k: v for k, v in manifest.items()
                                                  if k not in ("package_version", "tables")},
           **header, "tables": sorted(tables, key=lambda t: t["table"])}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(out, sort_keys=False, width=100, allow_unicode=True),
                    encoding="utf-8")
    return out


def _header(exp: dict, specs: dict[str, TestSpec], models: list[str],
            models_dir: str | Path = DEFAULT_MODELS_DIR) -> dict:
    dims = dims_of(exp)
    dimensions = {q: v["name"] for q, v in exp["dimensions"].items()}
    glossary = CONDITION_GLOSSARY.get(exp["name"], {})
    header = {
        "experiment": exp["name"],
        "exported": date.today().isoformat(),
        "runs_root": RUNS_ROOT_PLACEHOLDER,
        "runs_layout": "<runs_root>/<nxhop | cot_sweep>/final_<dim>/<published_as>/<model>/*.jsonl "
                       "(nxhop: the direct tests; cot_sweep: the ladders)",
        "models": models,
        "reasoning_kind": {m: REASONING_KIND[reasoning_trigger(m, exp, models_dir)] for m in models},
        "dimensions": dimensions,
        "dims": {dim: {"dimension": q, "item_set": item_set} for dim, (q, item_set) in dims.items()},
        "tests": {a: {"runner": s.runner, "published_as": s.published_as,
                     "conditions": dict(s.conditions), "surface": s.surface,
                     "context_only_instruction": s.ctxonly,
                     "premise_orders": s.n_premise_orders,
                     "rows_per_model_and_set": s.rows_per_set
                     * (len(exp["budgets"]) + 1 if s.runner == "ladder" else 1),
                     "item_sets": {dim: item_set for dim, (_q, item_set) in s.item_sets.items()}}
                 for a, s in specs.items()},
        "conditions": {c: glossary[c] for c in sorted({c for s in specs.values() for c in s.conditions.values()})
                       if c in glossary},
        "innermost_ranks": INNERMOST_RULE,
        "trial_key_format": TRIAL_KEY_FORMAT,
        "row_order": ROW_ORDER,
        "text_columns_dropped": TEXT_COLUMNS_DROPPED,
        "row_guards": ROW_GUARDS,
    }
    if "budgets" in exp:
        header["ladder_budgets"] = [0, *exp["budgets"]]
    if exp["name"] == "content_effects":
        header["instruction"] = (
            "every context condition uses the context-only instruction: the system "
            f"instruction ends with the sentence {CTXONLY_SENTENCE!r}; the test and condition "
            "names carry no suffix because the original-instruction runs are not exported")
    return header


CONDITION_GLOSSARY = {
    "symbolic_distance": {
        "no_context": "familiar items of the dimension, query alone (iwl / budget_iwl; the "
                      "main item set and its replicate item sets built to the same spacing from "
                      "disjoint items)",
        "icl_people": "novel given names ordered by height, all adjacent relations shown "
                      "(icl / budget_icl; run with country population)",
        "icl_cities": "invented city names under the population instruction, all adjacent "
                      "relations shown (icl_cities / budget_icl_cities; run with country population)",
    },
    "content_effects": {
        "congruent": "familiar items, relations in the true order (content natural / budget_natural)",
        "reversed": "familiar items, relations exactly reversed (content reversed / budget_reversed)",
        "shuffled": "familiar items, random assignment of items to ranks (content random / "
                    "budget_shuffled)",
        "unrelated_context": "familiar items queried after ten true relations among eleven other "
                             "items of the same dimension (uninformative; no ladder)",
    },
    "prior_strength": {
        "no_context": "familiar items of this spacing set, query alone (iwl)",
        "congruent": "familiar items, relations in the true order (content natural)",
        "reversed": "familiar items, relations exactly reversed (content reversed)",
        "shuffled": "familiar items, random assignment of items to ranks (content random)",
    },
}
TABLE_NOTES = {
    "direct": ("One row per scored trial of every direct test (budget 0): the answer is the "
               "comparison of the True and False token families at the position after the "
               "answer cue, with no reasoning tokens generated."),
    "ladders": ("One row per scored trial x reasoning budget of every ladder test. Budget 0 is "
                "the direct pass, collapsed to one row per trial from the identical copies the "
                "budgets each re-score, and equal to the trial's row in direct.parquet. "
                "Every budget is a prefix of one 1,024-token greedy trace."),
    "pools": ("One row per item-spacing set: the same eleven-item geometry as the main item sets, "
              "with the items chosen so that the value ratio between the ends of a distance-6 pair "
              "is value_ratio (the tag scale3p16 is 3.16); spacing and step_ratio from the item "
              "set's provenance record, whose sha256 is recorded."),
}


# --- the spacing sets -------------------------------------------------------------------


def tag_ratio(tag: str) -> float:
    """``"scale3p16"`` -> 3.16, the value ratio between the ends of a distance-6 pair."""
    if not tag.startswith("scale"):
        fail(f"{tag!r} is not a spacing tag")
    return float(tag[len("scale"):].replace("p", "."))


def set_provenance(item_set: str, items_dir: Path = DEFAULT_ITEMS_DIR) -> tuple[Path, dict]:
    """One item set's provenance record, as written when the item set was built."""
    path = Path(items_dir) / "sources" / f"{item_set}_provenance.json"
    if not path.exists():
        fail(f"{item_set}: no provenance file at {path}")
    return path, json.loads(path.read_text(encoding="utf-8"))


def spacing_sets(exp: dict, items_dir: Path = DEFAULT_ITEMS_DIR) -> list[dict]:
    """One record per item-spacing set: its value ratio (from the tag, checked against
    the provenance record), spacing, step ratio and provenance file."""
    item_sets = []
    for dim, (dimension, item_set) in dims_of(exp).items():
        tag = dim[len(dimension) + 1:]
        prov_path, prov = set_provenance(item_set, items_dir)
        ratio = tag_ratio(tag)
        if abs(float(prov["sens_ratio"]) - ratio) > 1e-9:
            fail(f"{item_set}: provenance value ratio {prov['sens_ratio']} != tag {ratio}")
        if int(prov["sens_d"]) != SPACING_SENS_D:
            fail(f"{item_set}: provenance sens_d {prov['sens_d']}, expected {SPACING_SENS_D}")
        item_sets.append({"dim": dim, "dimension": exp["dimensions"][dimension]["name"],
                          "item_pool": item_set, "tag": tag, "value_ratio": ratio,
                          "spacing": str(prov["spacing"]), "step_ratio": float(prov["step_ratio"]),
                          "provenance_file": str(prov_path.relative_to(_REPO))
                          if prov_path.is_relative_to(_REPO) else str(prov_path),
                          "provenance_sha256": sha256(prov_path)})
    return item_sets


def attach_spacing(df: pd.DataFrame, item_sets: list[dict]) -> pd.DataFrame:
    by_dim = {p["dim"]: p for p in item_sets}
    named = df["dim"].map(lambda d: by_dim[d]["item_pool"]).astype("string")
    if not (df["item_pool"] == named).all():
        fail("a row's item_pool is not the item set its spacing set key names")
    df["pool_kind"] = pd.Series("spacing", index=df.index, dtype="string")
    df["value_ratio"] = df["dim"].map(lambda d: by_dim[d]["value_ratio"]).astype("float32")
    df["spacing"] = df["dim"].map(lambda d: by_dim[d]["spacing"]).astype("string")
    return df


# --- the trial tables of one experiment --------------------------------------------------


def read_model(runs: Path, exp: dict, specs: dict[str, TestSpec], model: str,
               tests: list[str], log=print,
               models_dir: str | Path = DEFAULT_MODELS_DIR) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Every row of one model over the tests asked for (a ladder's direct twin is read for
    the check even when it is not asked for), as a direct frame and a ladder frame."""
    budgets = tuple(int(b) for b in exp.get("budgets", ()))
    full_ladder = (0, *budgets)
    twins = {a: twin_test(specs, a) for a in tests if specs[a].runner == "ladder"}
    read = list(dict.fromkeys([*tests, *twins.values()]))
    triggers: set[str] = set()
    direct_rows: list[dict] = []
    ladder_rows: list[dict] = []
    per_test: dict[str, dict] = {}
    for test in read:
        spec = specs[test]
        for dim in spec.item_sets:
            label = f"{model}/{dim}/{test}"
            if not test_dir(runs, spec, dim, model).exists():
                fail(f"{label}: directory missing ({test_dir(runs, spec, dim, model)})")
            rows, trig = read_test(runs, spec, dim, model, budgets)
            triggers |= trig
            if spec.runner == "direct":
                direct_rows += rows
                per_test[label] = {"rows": len(rows), "files": len({r["source_file"] for r in rows})}
            else:
                rows, note = collapse_ladder_direct(rows, label, budgets)
                ladder_rows += rows
                per_test[label] = {"rows": len(rows), "files": len({r["source_file"] for r in rows}), **note}
            log(f"  {label}: {per_test[label]}")
    if len(triggers) > 1:
        fail(f"{model}: trigger ids {sorted(triggers)}, expected exactly one")
    # The reasoning pass is either the model's own thinking block (trigger id "none")
    # or the prompted trigger; the model config says which, and the stored id must agree.
    expected = reasoning_trigger(model, exp, models_dir)
    if triggers and next(iter(triggers)) != expected:
        fail(f"{model}: trigger id {sorted(triggers)}, expected {expected!r} for this model")
    kind = REASONING_KIND[expected]
    for r in direct_rows + ladder_rows:
        r["reasoning_kind"] = kind
    twin_note = attach_twins(ladder_rows, direct_rows, model) if ladder_rows else {}
    if ladder_rows:
        log(f"  {model}: reasoning_kind={kind}; twins {twin_note}")
        check_complete_ladders(ladder_rows, model, full_ladder)
    check_unique_keys(direct_rows + ladder_rows, model)
    direct_rows = [r for r in direct_rows if r["test"] in tests]
    return to_frame(direct_rows), to_frame(ladder_rows), {"tests": per_test, "twins": twin_note}


CACHE_FILES = ("direct.parquet", "ladders.parquet", "notes.json")   # one model's work_dir cache
PER_TEST_NOTE = (
    "one entry per <model>/<dim>/<test> directory read (a ladder's direct twin is read for its "
    "checks even when not exported): rows kept and files read; for a ladder also the trials "
    "and the trials whose identical direct copies disagree across the budgets "
    "(n_direct_disagreements, a bf16 tie flip; the export fails above "
    f"{DIRECT_DUPLICATE_TOLERANCE:.0%} of a directory's trials)")


def export_trials(exp: dict, runs: Path, out_dir: Path, models: list[str] | None = None,
                  tests: list[str] | None = None, work_dir: Path | None = None,
                  items_dir: Path = DEFAULT_ITEMS_DIR, log=print,
                  models_dir: str | Path = DEFAULT_MODELS_DIR) -> dict:
    """``direct.parquet``, ``ladders.parquet`` (and ``pools.parquet``) + the manifest."""
    specs = test_specs(exp)
    models = list(models or roster(models_dir))
    tests = list(tests or specs)
    unknown = sorted(set(tests) - set(specs))
    if unknown:
        raise ValueError(f"experiment {exp['name']!r} has no tests {unknown}")
    extra = SPACING_COLUMNS if exp["name"] == "prior_strength" else []
    item_sets = spacing_sets(exp, items_dir) if extra else []
    direct_frames, ladder_frames = [], []
    per_test: dict[str, dict] = {}       # <model>/<dim>/<test> -> read_model's notes on that directory
    for model in models:
        cache = (work_dir / exp["name"] / model) if work_dir else None
        # What a cached model was read for (in its notes.json): a cache read for other tests or
        # another runs directory is refused, never reused.
        read_for = {"experiment": exp["name"], "model": model, "tests": tests,
                    "runs": str(Path(runs).resolve())}
        if cache and all((cache / f).exists() for f in CACHE_FILES):
            with (cache / "notes.json").open("r", encoding="utf-8") as f:
                notes = json.load(f)
            if notes.get("read_for") != read_for:
                raise ValueError(f"{cache} was read for {notes.get('read_for')}, not for {read_for}; "
                                 "use another --work-dir")
            log(f"{model}: cached")
            d, l = pd.read_parquet(cache / "direct.parquet"), pd.read_parquet(cache / "ladders.parquet")
        else:
            log(f"{model}:")
            d, l, notes = read_model(runs, exp, specs, model, tests, log, models_dir)
            if extra:
                d, l = attach_spacing(d, item_sets), attach_spacing(l, item_sets)
            if cache:
                cache.mkdir(parents=True, exist_ok=True)
                d.to_parquet(cache / "direct.parquet", index=False)
                l.to_parquet(cache / "ladders.parquet", index=False)
                with (cache / "notes.json").open("w", encoding="utf-8") as f:
                    json.dump({**notes, "read_for": read_for}, f, indent=1)
        per_test.update(notes["tests"])
        direct_frames.append(d)
        ladder_frames.append(l)
    n_budgets = len(exp.get("budgets", ())) + 1
    direct = finish_frame(pd.concat(direct_frames, ignore_index=True))
    ladders = finish_frame(pd.concat(ladder_frames, ignore_index=True))
    for name, df in (("direct", direct), ("ladders", ladders)):
        if len(df):
            if df["budget"].nunique() != (1 if name == "direct" else n_budgets):
                fail(f"{name}: budgets {sorted(df['budget'].unique())}")
            check_trial_set_counts(df, specs, n_budgets, name)
            # Every model must contribute the same tests: an experiment either ran a test
            # across the roster or not at all.
            test_sets = {frozenset(g["test"].unique()) for _, g in df.groupby("model", observed=True)}
            if len(test_sets) != 1 or df["model"].nunique() != len(models):
                fail(f"{name}: the models disagree on which tests they ran")
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, df in (("direct", direct), ("ladders", ladders)):
        if not len(df):
            continue
        written = write_parquet(df, out_dir / f"{name}.parquet")
        entries.append(table_entry(name, df, written, TABLE_NOTES[name],
                                   innermost_direct_trials_per_dim_condition_distance=innermost_counts(df),
                                   trial_sets=trial_set_digests(df)))
        log(f"wrote {out_dir / f'{name}.parquet'} ({written['bytes'] / 1e6:.2f} MB, {written['rows']:,} rows)")
    if item_sets:
        pdf = pd.DataFrame(item_sets)
        written = write_parquet(pdf, out_dir / "pools.parquet")
        entries.append({"table": "pools", **written,
                        "columns": {c: str(pdf[c].dtype) for c in pdf.columns},
                        "item_sets": item_sets,
                        "note": TABLE_NOTES["pools"]})
    header = _header(exp, specs, models, models_dir)
    header["checks"] = {"per_test_directories_note": PER_TEST_NOTE, "per_test_directories": per_test}
    manifest = update_manifest(out_dir / "MANIFEST.yaml", header, entries)
    return manifest


# --- the shuffled trials' query labels ---------------------------------------------------

AGREE, CONFLICT = "agree", "conflict"
SHUFFLED = "random"
SURFACE = "fixed_gt"
LABEL_COLUMNS = ["trial_key", "dim", "d", "trial_id", "rank_low", "rank_high",
                 "query_congruence", "knowledge_distance"]
LABEL_DTYPES = {"trial_key": "string", "dim": "string", "d": "int8", "trial_id": "int16",
                "rank_low": "int8", "rank_high": "int8", "query_congruence": "string",
                "knowledge_distance": "int8"}
# The five item sets the paper's figures draw (the evenly spaced city item set, not the other one).
DRAWN_DIMS = ("countries", "citieslinear", "animals", "usstates", "countryarea")
_ASSIGN_IN_NAME = re.compile(r"__assign=([a-z]+)$")
_OPERATOR = re.compile(r"\s+([<>])\s+")


LABEL_ROW_GUARDS = [
    "N, master seed, hop tokens, item set, surface and the file's assignment",
    "the instruction id and the context-only sentence match the test",
    "actual_items equals the item set's provenance order (values strictly rising)",
    "exactly one question clause, naming the pair at rank_low/rank_high (a ladder's "
    "question read before the trace)",
    "the question's truth in the ordering shown equals the stored gold answer",
    "the premises before the question are exactly the ten adjacent relations of rank_to_item",
]
LABELS_NOTE = ("One row per shuffled trial (assignment 'random') of the content test, on the six "
               "main item sets and the nineteen item-spacing sets. The label is a property of the "
               "trial, shared by every model. Congruent trials always agree and reversed trials "
               "always conflict, at knowledge_distance == d; they are not listed. No model output "
               "is used.")


def real_order(item_set: str, items_dir: Path = DEFAULT_ITEMS_DIR) -> list[str]:
    """The item set's items lowest first, from its provenance record (values strictly rising)."""
    path = Path(items_dir) / "sources" / f"{item_set}_provenance.json"
    if not path.exists():
        fail(f"{item_set}: no provenance record at {path}")
    items = json.loads(path.read_text(encoding="utf-8"))["items"]
    values = [float(it["value"]) for it in items]
    if any(b <= a for a, b in zip(values, values[1:], strict=False)):
        fail(f"{item_set}: provenance values do not rise strictly")
    names = [it["name"] for it in items]
    if len(names) != N_ITEMS or len(set(names)) != N_ITEMS:
        fail(f"{item_set}: {len(names)} provenance items, expected {N_ITEMS} distinct")
    return names


def item_pattern(items) -> re.Pattern:
    """'<item> > <item>' with whole-item boundaries (longest names first)."""
    alt = "|".join(re.escape(it) for it in sorted(items, key=len, reverse=True))
    return re.compile(rf"(?<![\w])({alt})\s+([<>])\s+({alt})(?![\w])")


def label_of(rti, real, rl: int, rh: int) -> tuple[str, int]:
    """(agree | conflict, |distance in the real ordering|) of the pair at ranks rl < rh."""
    signed = real.index(rti[rh]) - real.index(rti[rl])
    return (AGREE if signed > 0 else CONFLICT), abs(signed)


def check_row(r: dict, where: str, *, dim: str, item_set: str, ctxonly: bool, assign: str, hop: int,
              real: list[str], pattern: re.Pattern, ladder: bool = False) -> tuple:
    """Every guard on one stored row; returns the trial's signature."""
    ex, spec = r["extra"], r["prompt_spec"]
    if r["N"] != N_ITEMS or r["master_seed"] != MASTER_SEED:
        fail(f"{where}: N={r['N']} master_seed={r['master_seed']}")
    sd = int(r["symbolic_distance"])
    if sd != hop or ex["hop"] != sd or spec["hop"] != sd:
        fail(f"{where}: hop tokens disagree ({sd}, {hop}, {ex['hop']}, {spec['hop']})")
    if r["item_pool"] != item_set or ex["surface_form_mode"] != SURFACE:
        fail(f"{where}: item set/surface {r['item_pool']}/{ex['surface_form_mode']}")
    if ex["assignment_strategy"] != assign or r["assignment_strategy"] != assign:
        fail(f"{where}: assignment {ex['assignment_strategy']!r}, the file says {assign!r}")
    ctx = ctxonly
    if spec["instructions_v1_id"].endswith(CTXONLY) != ctx:
        fail(f"{where}: instruction id {spec['instructions_v1_id']!r} under a "
             f"{'context-only' if ctx else 'original'}-instruction test")
    prompt = r["prompt"]
    if (CTXONLY_SENTENCE in prompt) != ctx:
        fail(f"{where}: the context-only sentence does not match the test")
    rti = tuple(ex["rank_to_item"])
    # A ladder row stores no actual_items; its ordering is compared with its direct twin's.
    if not ladder and list(ex["actual_items"]) != real:
        fail(f"{where}: actual_items is not the provenance record's order")
    if sorted(rti) != sorted(real):
        fail(f"{where}: rank_to_item is not a premise order of the item set")
    if ladder:
        rl, rh = ladder_query_ranks(r, where)        # the question, read before the trace
        question = (r["cot_stage1_prompt"] if r["mode"] == "cot" else prompt)
    else:
        rl, rh = int(ex["rank_low"]), int(ex["rank_high"])
        question = prompt
    if rh - rl != sd or rl < 1 or rh > N_ITEMS - 2:
        fail(f"{where}: queried ranks {rl}-{rh} at distance {sd}")
    clauses = list(_QUERY_RE.finditer(question))
    if len(clauses) != 1:
        fail(f"{where}: {len(clauses)} question clauses, expected one")
    # The clause is "<item> <op> <item>" with whole item names (an item name may contain
    # another, as "Virginia" does "West Virginia" in an item-spacing set).
    parts = _OPERATOR.split(clauses[0].group(1).strip())
    if len(parts) != 3 or parts[0] not in rti or parts[2] not in rti:
        fail(f"{where}: cannot read the question {clauses[0].group(1)!r}")
    a, op, b = parts
    if sorted((rti.index(a), rti.index(b))) != [rl, rh]:
        fail(f"{where}: the question does not name the pair at ranks {rl}, {rh}")
    true_shown = (rti.index(a) > rti.index(b)) if op == ">" else (rti.index(a) < rti.index(b))
    if bool(r["correct"]) != true_shown:
        fail(f"{where}: gold {r['correct']} but the question is {true_shown} in the ordering shown")
    # The premises: everything before the question, as '<item> > <item>' relations.
    before = question[:clauses[0].start()]
    shown_rel = {(m.group(1), m.group(2), m.group(3)) for m in pattern.finditer(before)}
    adjacent = {(rti[i + 1], ">", rti[i]) for i in range(N_ITEMS - 1)}
    if shown_rel != adjacent:
        fail(f"{where}: the premises are not the adjacent relations of rank_to_item "
             f"({len(shown_rel ^ adjacent)} differ)")
    premise_order = int(ex["permutation_idx"]) if not ladder else None
    return (rti, rl, rh, (a, op, b), bool(r["correct"]), premise_order)


def read_direct_labels(runs: Path, spec: TestSpec, dim: str, model: str,
                       items_dir: Path = DEFAULT_ITEMS_DIR) -> dict:
    """One (model, dim) directory of the content test: every trial's signature and label."""
    item_set = spec.item_sets[dim][1]
    real = real_order(item_set, items_dir)
    pattern = item_pattern(real)
    d = test_dir(runs, spec, dim, model)
    files = sorted(d.glob("*.jsonl"))
    if len(files) != len(spec.conditions) * (N_ITEMS - 3):
        fail(f"{d}: {len(files)} files, expected {len(spec.conditions) * (N_ITEMS - 3)}")
    out = {}
    counts = Counter()
    for f in files:
        if _budget_of(f.stem) is not None:
            fail(f"{f}: a direct file with a budget prefix")
        m = _ASSIGN_IN_NAME.search(f.stem)
        assign = m.group(1) if m else SHUFFLED
        hop = int(f.stem.split("hop=")[1].split("__")[0])
        seen = set()
        for line_no, line in enumerate(open(f, encoding="utf-8")):
            if not line.strip():
                continue
            r = json.loads(line)
            where = f"{f.relative_to(runs)}:{line_no}"
            sig = check_row(r, where, dim=dim, item_set=item_set, ctxonly=spec.ctxonly, assign=assign,
                            hop=hop, real=real, pattern=pattern)
            tid = int(r["trial_id"])
            if tid in seen:
                fail(f"{where}: trial id {tid} repeated in one file")
            seen.add(tid)
            rti, rl, rh = sig[0], sig[1], sig[2]
            label, kd = label_of(rti, real, rl, rh)
            if assign == "natural":
                if list(rti) != real or label != AGREE or kd != hop:
                    fail(f"{where}: a congruent trial is {label} at real distance {kd}")
            elif assign == "reversed":
                if list(rti) != real[::-1] or label != CONFLICT or kd != hop:
                    fail(f"{where}: a reversed trial is {label} at real distance {kd}")
            counts[(assign, label)] += 1
            out[(dim, assign, hop, tid)] = (*sig, label, kd)
    return {"trials": out, "counts": counts, "files": len(files)}


def read_ladder_labels(runs: Path, spec: TestSpec, dim: str, model: str,
                       items_dir: Path = DEFAULT_ITEMS_DIR) -> dict:
    """One shuffled ladder directory on disk: every row's ordering and question."""
    item_set = spec.item_sets[dim][1]
    real = real_order(item_set, items_dir)
    pattern = item_pattern(real)
    d = test_dir(runs, spec, dim, model)
    out, n_rows, files = {}, 0, sorted(d.glob("*.jsonl"))
    for f in files:
        if _budget_of(f.stem) is None:
            fail(f"{f}: a ladder file without a budget prefix")
        hop = int(f.stem.split("hop=")[1].split("__")[0])
        for line_no, line in enumerate(open(f, encoding="utf-8")):
            if not line.strip():
                continue
            r = json.loads(line)
            where = f"{f.relative_to(runs)}:{line_no}"
            sig = check_row(r, where, dim=dim, item_set=item_set, ctxonly=spec.ctxonly, assign=SHUFFLED,
                            hop=hop, real=real, pattern=pattern, ladder=True)
            key = (dim, SHUFFLED, hop, int(r["trial_id"]))
            prev = out.setdefault(key, sig[:5])
            if prev != sig[:5]:
                fail(f"{where}: the ladder's budgets disagree on trial {key}")
            n_rows += 1
    return {"trials": out, "rows": n_rows, "files": len(files)}


def labels_frame(reference: dict[tuple, tuple]) -> pd.DataFrame:
    """The label table from the shuffled trials' signatures."""
    rows = []
    for (dim, assign, hop, tid), sig in reference.items():
        if assign != SHUFFLED:
            continue
        _rti, rl, rh, _q, _gold, _perm, label, kd = sig
        rows.append({"trial_key": f"{dim}:{SURFACE}|{assign}|{hop}|{tid}", "dim": dim, "d": hop,
                     "trial_id": tid, "rank_low": rl, "rank_high": rh,
                     "query_congruence": label, "knowledge_distance": kd})
    labels = (pd.DataFrame(rows, columns=LABEL_COLUMNS).astype(LABEL_DTYPES)
              .sort_values(["dim", "d", "trial_id"], kind="stable").reset_index(drop=True))
    if labels["trial_key"].duplicated().any():
        fail("a trial key carries more than one label")
    return labels


def coverage(labels: pd.DataFrame, tables: dict[str, Path]) -> dict:
    """Every shuffled row of every trial table has exactly one matching label."""
    by_key = labels.set_index("trial_key")
    used: set = set()
    out = {}
    for name, path in tables.items():
        if not path.exists():
            fail(f"{path}: a trial table is missing (export the trials first)")
        df = pd.read_parquet(path, columns=["condition", "test", "trial_key", "dim", "d",
                                            "trial_id", "rank_low", "rank_high"])
        sh = df[df["condition"] == "shuffled"]
        if sh.empty:
            out[name] = {"shuffled_rows": 0}
            continue
        joined = sh.join(by_key, on="trial_key", rsuffix="_label")
        if joined["query_congruence"].isna().any():
            fail(f"{name}: {int(joined['query_congruence'].isna().sum())} shuffled rows have no label")
        if len(joined) != len(sh):
            fail(f"{name}: a trial key matches more than one label")
        for col in ("dim", "d", "trial_id", "rank_low", "rank_high"):
            if not (joined[col].astype(str) == joined[f"{col}_label"].astype(str)).all():
                fail(f"{name}: {col} of a shuffled row differs from its label's")
        used |= set(sh["trial_key"])
        out[name] = {
            "shuffled_rows": int(len(sh)),
            "labeled": int(joined["query_congruence"].notna().sum()),
            "rows_by_test": {a: int(n) for a, n in sh.groupby("test", observed=True).size().items()},
            "agree_fraction": round(float((joined["query_congruence"] == AGREE).mean()), 6),
        }
    orphans = set(labels["trial_key"]) - used
    if orphans:
        fail(f"{len(orphans)} labels match no exported row, e.g. {sorted(orphans)[:2]}")
    return out


def agree_by_distance(labels: pd.DataFrame, dims) -> dict:
    sub = labels[labels["dim"].isin(list(dims)) & (labels["rank_low"] >= DROP_LO)
                 & (labels["rank_high"] <= DROP_HI)]
    g = sub.groupby("d")["query_congruence"].agg(n="size", agree=lambda s: int((s == AGREE).sum()))
    return {int(d): {"n": int(r.n), "agree": int(r.agree), "agree_fraction": round(r.agree / r.n, 6)}
            for d, r in g.iterrows()}


def export_query_labels(exp: dict, spacing_exp: dict, runs: Path, out_dir: Path,
                        results_dir: Path = DEFAULT_RESULTS_DIR, models: list[str] | None = None,
                        check_ladders: bool = False, items_dir: Path = DEFAULT_ITEMS_DIR,
                        log=print) -> dict:
    """``query_labels.parquet``: every shuffled trial of the content test on the main item sets
    and the item-spacing sets, labeled agree / conflict with its knowledge distance.

    The label depends on the trial only, so every model's directory must show the same
    trial (ordering, queried ranks, question, gold answer, premise order index); the
    shuffled ladders are labeled through their direct twins (``check_ladders`` compares
    their rows with the twins' as well).
    """
    models = list(models or roster())
    content = {a: s for a, s in test_specs(exp).items()
               if s.runner == "direct" and SHUFFLED in s.conditions}
    content.update({f"spacing:{a}": s for a, s in test_specs(spacing_exp).items()
                    if s.runner == "direct" and SHUFFLED in s.conditions})
    if len(content) != 2:
        fail(f"expected one shuffled direct test per experiment, got {sorted(content)}")
    ladders = {a: s for a, s in test_specs(exp).items()
               if s.runner == "ladder" and SHUFFLED in s.conditions}
    tasks = [(model, dim, spec) for spec in content.values() for dim in spec.item_sets for model in models]
    for model, dim, spec in tasks:
        if not test_dir(runs, spec, dim, model).is_dir():
            fail(f"{dim}/{spec.test}/{model}: directory missing (every content directory must be on disk)")
    set_of = {dim: spec.item_sets[dim][1] for spec in content.values() for dim in spec.item_sets}

    reference: dict[tuple, tuple] = {}
    first_from: dict[tuple, tuple] = {}
    compared = Counter()
    counts = Counter()
    n_files = 0
    for model, dim, spec in tasks:
        res = read_direct_labels(runs, spec, dim, model, items_dir)
        n_files += res["files"]
        counts.update(res["counts"])
        for key, sig in res["trials"].items():
            prev = reference.get(key)
            if prev is None:
                reference[key] = sig
                first_from[key] = (model, spec.test)
            elif prev != sig:
                fail(f"{key}: {model}/{spec.test} shows a different trial than "
                     f"{first_from[key][0]}/{first_from[key][1]} ({prev[:4]} vs {sig[:4]})")
            else:
                compared[key[1]] += 1
        log(f"  {model}/{dim}: {len(res['trials'])} trials")
    ladder_rows = ladder_files = 0
    ladder_trials = Counter()
    ladder_tasks, ladders_absent = [], []
    if check_ladders:
        for spec in ladders.values():
            for dim in spec.item_sets:
                for model in models:
                    if test_dir(runs, spec, dim, model).is_dir():
                        ladder_tasks.append((model, dim, spec))
                    else:
                        ladders_absent.append(f"{dim}/{spec.test}/{model}")
        for model, dim, spec in ladder_tasks:
            res = read_ladder_labels(runs, spec, dim, model, items_dir)
            ladder_rows += res["rows"]
            ladder_files += res["files"]
            for key, sig in res["trials"].items():
                twin = reference.get(key)
                if twin is None:
                    fail(f"{model}/{spec.test} {key}: a ladder trial with no direct twin")
                if twin[:4] != sig[:4] or twin[4] != sig[4]:
                    fail(f"{model}/{spec.test} {key}: the ladder shows a different ordering or "
                         "question than its direct twin")
                ladder_trials[f"{dim}/{spec.test}/{model}"] += 1
            log(f"  ladder {model}/{dim}/{spec.test}: {res['rows']} rows")

    # One instance per (dim, assignment, d, trial id) per model directory.
    per_key = len(models)
    by_assign = Counter(k[1] for k in reference)
    for assign, n in by_assign.items():
        if compared[assign] != n * (per_key - 1):
            fail(f"{assign}: {compared[assign]} cross-file comparisons, expected {n * (per_key - 1)}")
    labels = labels_frame(reference)
    orderings = {dim: len({sig[0] for (dd, a, _h, _t), sig in reference.items()
                           if dd == dim and a == SHUFFLED}) for dim in set_of}

    tables = {"content_effects/direct": results_dir / exp["name"] / "tables" / "direct.parquet",
              "content_effects/ladders": results_dir / exp["name"] / "tables" / "ladders.parquet",
              "prior_strength/direct": results_dir / spacing_exp["name"] / "tables" / "direct.parquet"}
    cover = coverage(labels, tables)

    out_dir.mkdir(parents=True, exist_ok=True)
    written = write_parquet(labels, out_dir / "query_labels.parquet")
    per_dim = (labels.groupby("dim")["query_congruence"]
               .agg(n="size", agree=lambda s: int((s == AGREE).sum())))
    inner = labels[(labels["rank_low"] >= DROP_LO) & (labels["rank_high"] <= DROP_HI)]
    by_dim_d = {f"{dim}|d={int(d)}": Flow([int(len(g)), int((g["query_congruence"] == AGREE).sum())])
                for (dim, d), g in inner.groupby(["dim", "d"])}
    entry = {
        "table": "query_labels", **written,
        "columns": {c: str(labels[c].dtype) for c in labels.columns},
        "trial_key_format": TRIAL_KEY_FORMAT,
        "row_order": "dim, d, trial_id",
        "labels": {
            AGREE: ("the queried pair is shown in its real order: the item at rank_high of the "
                    "ordering shown (extra.rank_to_item) is also the higher item of the real "
                    "ordering (extra.actual_items, lowest first)"),
            CONFLICT: "the queried pair is shown in the reverse of its real order",
            "knowledge_distance": "the absolute difference of the pair's two ranks in the real ordering",
        },
        "dims": {dim: {"item_set": item_set, "trials": int(per_dim.at[dim, "n"]),
                       "agree": int(per_dim.at[dim, "agree"]),
                       "distinct_orderings_shown": orderings[dim]}
                 for dim, item_set in set_of.items()},
        "innermost_ranks": INNERMOST_RULE,
        "innermost_trials_by_dim_and_distance": {"format": "[n, agree]", **by_dim_d},
        "innermost_agree_by_distance_drawn_dims": {"dims": list(DRAWN_DIMS),
                                                   "by_distance": agree_by_distance(labels, DRAWN_DIMS)},
        "checks": {
            "direct_files_read": n_files,
            "direct_directories": len(tasks),
            "models": models,
            "trials_by_assignment": {a: int(n) for a, n in sorted(by_assign.items())},
            "cross_file_comparisons_by_assignment": {a: int(n) for a, n in sorted(compared.items())},
            "rule_shared": ("every (dim, assignment, d, trial_id) shows the same ordering, queried "
                            "ranks, question, gold answer and premise order index in all "
                            f"{len(models)} models"),
            "congruent_trials_agree": int(counts[("natural", AGREE)]),
            "congruent_trials_conflict": int(counts[("natural", CONFLICT)]),
            "reversed_trials_conflict": int(counts[("reversed", CONFLICT)]),
            "reversed_trials_agree": int(counts[("reversed", AGREE)]),
            "congruent_reversed_rule": ("every congruent trial agrees and every reversed trial "
                                        "conflicts, at knowledge_distance == d (counts are rows: "
                                        "models x trials)"),
            "shuffled_rows_read": int(counts[(SHUFFLED, AGREE)] + counts[(SHUFFLED, CONFLICT)]),
            "ladders_cross_checked": bool(check_ladders),
            "ladder_directories_on_disk": sorted(f"{t[1]}/{t[2].test}/{t[0]}" for t in ladder_tasks),
            "ladder_rows_cross_checked": int(ladder_rows),
            "ladder_files_cross_checked": int(ladder_files),
            "ladder_trials_cross_checked": dict(sorted(ladder_trials.items())),
            "ladder_directories_not_on_disk": len(ladders_absent),
            "row_guards": LABEL_ROW_GUARDS,
        },
        "coverage_of_trial_tables": cover,
        "ladders": ("Ladder rows (budget_shuffled) carry their direct twin's trial_key and are "
                    "labeled through it: the trials export asserts every ladder row's queried "
                    "ranks against the twin's (attach_twins)."),
        "note": LABELS_NOTE,
    }
    manifest = update_manifest(out_dir / "MANIFEST.yaml", {"exported": date.today().isoformat(),
                                                           "runs_root": RUNS_ROOT_PLACEHOLDER}, [entry])
    log(f"wrote {out_dir / 'query_labels.parquet'} ({len(labels):,} labels, "
        f"agree {float((labels.query_congruence == AGREE).mean()):.3f})")
    return manifest


# --- verbatim prompt examples --------------------------------------------------------------

MAIN_MODEL = "qwen3-32b"
PROMPTED_MODEL = "qwen2.5-32b-instruct"
EXAMPLE_DIMENSION = "countries"
# The distance and polarity of the shared example query: an interior pair (both ranks
# inside the innermost seven) that the true ordering answers "true", so the congruent
# context agrees with the world and the reversed context contradicts it.
TARGET_DISTANCE = 3
TARGET_GOLD = True
# The reasoning examples: one budget of one ladder test per model. 128 tokens is the
# shortest rung whose trace is long enough to show the shape of a chain and still fits a
# figure panel when elided.
REASONING_BUDGET = 128
# The five familiar-item contexts, in the order the content figure draws them, as
# (condition name, experiment, test, assignment strategy).
EXAMPLE_CONTEXTS = (
    ("no_context", "symbolic_distance", "iwl", "natural"),
    ("congruent", "content_effects", "content", "natural"),
    ("reversed", "content_effects", "content", "reversed"),
    ("shuffled", "content_effects", "content", "random"),
    ("unrelated_context", "content_effects", "uninformative", "natural"),
)
_TURN_RE = re.compile(r"<\|im_start\|>(\w+)\n(.*?)(?:<\|im_end\|>\n|\Z)", re.DOTALL)
_EXAMPLE_QUERY_RE = re.compile(r"Is it true or false that\s+(.+?)\s*\?")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def query_items(prompt: str) -> tuple[str, str] | None:
    """The two item names of the query clause, in the order the question states them."""
    ms = _EXAMPLE_QUERY_RE.findall(prompt)
    if not ms:
        return None
    parts = re.split(r"\s+[<>]\s+", ms[-1].strip())
    return (parts[0], parts[1]) if len(parts) == 2 else None


def split_turns(prompt: str) -> dict:
    """The chat template's turns, plus whatever the assistant turn was prefilled with.

    Every example here comes from a Qwen chat template, whose turns are delimited by
    ``<|im_start|>role`` / ``<|im_end|>``; the assistant turn is left open and may carry
    a prefilled thinking block.
    """
    turns = {role: body for role, body in _TURN_RE.findall(prompt)}
    user = turns.get("user", "")
    query_match = list(_EXAMPLE_QUERY_RE.finditer(user))
    query = user[query_match[-1].start():].strip() if query_match else user.strip()
    premises = user[:query_match[-1].start()].strip() if query_match else ""
    return {
        "system": turns.get("system", "").strip(),
        "premises": [line for line in premises.splitlines() if line.strip()],
        "query": query,
        "assistant_prefill": turns.get("assistant", ""),
    }


def _example_rows(runs: Path, spec: TestSpec, model: str, *, dim: str = EXAMPLE_DIMENSION,
                  budget_prefix: str = "", assign: str | None = None):
    """Every stored row of one test directory at the target distance, with its file and line."""
    d = test_dir(runs, spec, dim, model)
    pattern = f"{budget_prefix}*hop={TARGET_DISTANCE}__*.jsonl" if budget_prefix \
        else f"*hop={TARGET_DISTANCE}__*.jsonl"
    for f in sorted(d.glob(pattern)):
        if assign is not None and f"assign={assign}" not in f.stem and "assign=" in f.stem:
            continue
        for line_no, line in enumerate(f.open(encoding="utf-8")):
            if line.strip():
                yield f, line_no, json.loads(line)


def example(runs: Path, spec: TestSpec, condition: str, assign: str | None,
            want: tuple[str, str] | None, *, model: str = MAIN_MODEL,
            dim: str = EXAMPLE_DIMENSION) -> dict | None:
    """One direct trial of a test, asking about ``want`` if given, as a figure record."""
    for f, line_no, r in _example_rows(runs, spec, model, dim=dim, assign=assign):
        if r["mode"] != "direct":
            continue
        ex = r["extra"]
        if assign is not None and ex["assignment_strategy"] != assign:
            continue
        pair = query_items(r["prompt"])
        if pair is None or (want is not None and pair != want):
            continue
        if want is None:
            rl, rh = int(ex.get("rank_low", -1)), int(ex.get("rank_high", -1))
            if not (rl >= DROP_LO and rh <= DROP_HI) or bool(r["correct"]) is not TARGET_GOLD:
                continue
        return {
            "condition": condition,
            "test": spec.test,
            "model": r["model"],
            "model_key": model,
            "item_pool": r["item_pool"],
            "instruction_id": r["prompt_spec"]["instructions_v1_id"],
            "assignment_strategy": ex["assignment_strategy"],
            "surface_form_mode": ex["surface_form_mode"],
            "symbolic_distance": int(r["symbolic_distance"]),
            "rank_low": int(ex["rank_low"]) if ex.get("rank_low") is not None else None,
            "rank_high": int(ex["rank_high"]) if ex.get("rank_high") is not None else None,
            "gold_true": bool(r["correct"]),
            "query_items": list(query_items(r["prompt"])),
            "context_ordering": list(ex["rank_to_item"]),
            "prompt": r["prompt"],
            "prompt_sha256": sha256_text(r["prompt"]),
            "parts": split_turns(r["prompt"]),
            "source_file": str(f.relative_to(runs)),
            "source_line": line_no,
            **({"trial_id": int(r["trial_id"])}),
        }
    return None


def shared_query_examples(runs: Path, contexts: list[tuple[str, TestSpec, str]]) -> tuple[list[dict], tuple[str, str]]:
    """The five contexts, all asking one question that every test happens to have asked.

    Candidates come from the congruent test (a fixed assignment, so its query pairs are
    real country pairs at the requested distance and polarity in the true ordering); the
    first candidate every other test also asked is taken.
    """
    congruent = next(spec for cond, spec, assign in contexts if cond == "congruent")
    candidates: list[tuple[str, str]] = []
    for f, _line, r in _example_rows(runs, congruent, MAIN_MODEL, assign="natural"):
        if r["mode"] != "direct" or r["extra"]["assignment_strategy"] != "natural":
            continue
        if bool(r["correct"]) is not TARGET_GOLD:
            continue
        ex = r["extra"]
        rl, rh = int(ex["rank_low"]), int(ex["rank_high"])
        if not (rl >= DROP_LO and rh <= DROP_HI):
            continue
        pair = query_items(r["prompt"])
        if pair and pair not in candidates:
            candidates.append(pair)
    for pair in candidates:
        found = [example(runs, spec, cond, assign, pair) for cond, spec, assign in contexts]
        if all(f is not None for f in found):
            return found, pair
    raise AssertionError(
        f"no query pair at distance {TARGET_DISTANCE} with gold {TARGET_GOLD} was asked "
        f"by all five tests; tried {len(candidates)} candidates")


def reasoning_example(runs: Path, spec: TestSpec, model: str, condition: str) -> dict:
    """One reasoning trial: the open prompt, the trace as generated, the scoring suffix."""
    for f, line_no, r in _example_rows(runs, spec, model, budget_prefix=f"b{REASONING_BUDGET}__"):
        if r["mode"] != "cot":
            continue
        reasoning_prompt, trace, full = r["cot_stage1_prompt"], r["cot_trace"], r["prompt"]
        if not full.startswith(reasoning_prompt) or full[len(reasoning_prompt):len(reasoning_prompt) + len(trace)] != trace:
            raise AssertionError(f"{f}: the scored prompt is not the reasoning pass + trace + suffix")
        suffix = full[len(reasoning_prompt) + len(trace):]
        return {
            "condition": condition,
            "test": spec.test,
            "model": r["model"],
            "model_key": model,
            "reasoning_kind": "native" if r["prompt_spec"]["trigger_id"] == "none" else "prompted",
            "trigger_id": r["prompt_spec"]["trigger_id"],
            "budget": REASONING_BUDGET,
            "item_pool": r["item_pool"],
            "instruction_id": r["prompt_spec"]["instructions_v1_id"],
            "symbolic_distance": int(r["symbolic_distance"]),
            "gold_true": bool(r["correct"]),
            "query_items": list(query_items(reasoning_prompt) or ()),
            "reasoning_prompt": reasoning_prompt,
            "trace": trace,
            "scoring_suffix": suffix,
            "cot_n_tokens": r["cot_n_tokens"],
            "cot_finish_reason": r["cot_finish_reason"],
            "truncated": r["cot_finish_reason"] == "length",
            "parts": split_turns(reasoning_prompt),
            "prompt_sha256": sha256_text(full),
            "source_file": str(f.relative_to(runs)),
            "source_line": line_no,
        }
    raise AssertionError(f"no reasoning row found in {spec.test}/{model} at budget {REASONING_BUDGET}")


def export_prompt_examples(runs: Path, out_dir: Path,
                           experiments_dir: Path = experiments.DEFAULT_EXPERIMENTS_DIR,
                           log=print) -> dict:
    """``prompt_examples.json``: verbatim prompts: the ICL test prompt the appendix prints, and
    the other contexts and reasoning passes.

    Nothing is written by hand: every string is copied out of a stored trial, with the
    file and line it came from and a sha256 of the prompt. One complete in-context prompt
    (novel names ordered by height, split into the chat template's turns); the user turn
    of each of the five familiar-item contexts, all asking THE SAME question about the same
    two countries (the context tests under the context-only instruction); and the two ways
    a reasoning pass was run (the model's own thinking block, Qwen3; the prompted trigger,
    Qwen2.5), each with the trace as generated at one budget and the scoring suffix.
    """
    specs = {name: test_specs(experiments.load_experiment(name, experiments_dir))
             for name in ("symbolic_distance", "content_effects")}
    contexts = [(cond, specs[e][test], assign) for cond, e, test, assign in EXAMPLE_CONTEXTS]
    icl = example(runs, specs["symbolic_distance"]["icl"], "icl_people", "random", None)
    if icl is None:
        raise AssertionError("no in-context example at the requested distance and polarity")
    found, pair = shared_query_examples(runs, contexts)
    for ex in found:
        ctx = specs["content_effects"].get(ex["test"], specs["symbolic_distance"].get(ex["test"]))
        if ex["instruction_id"].endswith(CTXONLY) != ctx.ctxonly \
                or (CTXONLY_SENTENCE in ex["parts"]["system"]) != ctx.ctxonly:
            raise AssertionError(f"{ex['condition']}: the instruction does not match the test")
    reasoning = [
        reasoning_example(runs, specs["symbolic_distance"]["budget_icl"], MAIN_MODEL, "icl_people"),
        reasoning_example(runs, specs["content_effects"]["budget_reversed"], PROMPTED_MODEL, "reversed"),
    ]
    payload = {
        "in_context_full": icl,
        "familiar_contexts": found,
        "context_only_sentence": CTXONLY_SENTENCE,
        "shared_query_items": list(pair),
        "reasoning": reasoning,
        "scoring": {
            "answer_cue": reasoning[0]["scoring_suffix"].strip(),
            "rule": ("the answer is scored by comparing the total probability of the True "
                     "token family with that of the False token family at the position "
                     "after the answer cue; a direct answer is the same comparison with "
                     "no reasoning tokens generated"),
            "direct_budget": 0,
        },
        "geometry": {
            "n_items": N_ITEMS,
            "queryable_ranks": [1, N_ITEMS - 2],
            "innermost_ranks": [DROP_LO, DROP_HI],
            "note": ("the two ends of the ordering are never queried, and the figures "
                     f"keep the innermost seven ranks ({DROP_LO}-{DROP_HI}), giving "
                     "symbolic distances 1-6"),
            "rank_direction": ("rank 0 is the LOWEST item on the dimension and rank "
                               f"{N_ITEMS - 1} the highest, so `context_ordering` reads "
                               "from smallest to largest; a query is answered against the "
                               "ordering shown in context, which is why the reversed and "
                               "shuffled examples answer the same question differently"),
        },
        "condition_of_assignment": dict(specs["content_effects"]["content"].conditions),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "prompt_examples.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    entry = {
        "table": "prompt_examples", "path": out_json.name, "sha256": sha256(out_json),
        "bytes": int(out_json.stat().st_size),
        "shared_query_items": list(pair),
        "target_distance": TARGET_DISTANCE,
        "target_gold_true": TARGET_GOLD,
        "reasoning_budget": REASONING_BUDGET,
        "examples": [
            {"key": "in_context_full", "condition": icl["condition"], "model": icl["model"],
             "source_file": icl["source_file"], "source_line": icl["source_line"],
             "prompt_sha256": icl["prompt_sha256"]},
            *[{"key": f"familiar_contexts[{i}]", "condition": c["condition"], "test": c["test"],
               "model": c["model"], "source_file": c["source_file"],
               "source_line": c["source_line"], "prompt_sha256": c["prompt_sha256"]}
              for i, c in enumerate(found)],
            *[{"key": f"reasoning[{i}]", "condition": c["condition"], "test": c["test"],
               "model": c["model"], "source_file": c["source_file"],
               "source_line": c["source_line"], "prompt_sha256": c["prompt_sha256"]}
              for i, c in enumerate(reasoning)],
        ],
        "note": ("Every string is verbatim from a stored trial; a figure must not edit them "
                 "beyond eliding a reasoning trace, and the sha256 of each prompt lets the "
                 "figure's text be checked against the stored runs. The context examples "
                 "are the context-only-instruction runs."),
    }
    manifest = update_manifest(out_dir / "MANIFEST.yaml", {"exported": date.today().isoformat(),
                                                           "runs_root": RUNS_ROOT_PLACEHOLDER}, [entry])
    log(f"wrote {out_json}; shared query: {pair[0]} > {pair[1]} (distance {TARGET_DISTANCE})")
    for c in found:
        log(f"  {c['condition']:20s} {len(c['parts']['premises'])} premises  gold={c['gold_true']}  "
            f"{c['source_file']}:{c['source_line']}")
    for c in reasoning:
        log(f"  reasoning {c['model_key']:22s} {c['reasoning_kind']:8s} {c['cot_n_tokens']} tokens, "
            f"finish={c['cot_finish_reason']}")
    return manifest


# --- reading the committed tables ----------------------------------------------------------


def table_path(experiment: str, name: str, results_dir: Path = DEFAULT_RESULTS_DIR) -> Path:
    return Path(results_dir) / experiment / "tables" / f"{name}.parquet"


def read_table(experiment: str, name: str, results_dir: Path = DEFAULT_RESULTS_DIR,
               columns: list[str] | None = None) -> pd.DataFrame:
    """One committed table (``direct``, ``ladders``, ``pools``, ``query_labels``)."""
    return pd.read_parquet(table_path(experiment, name, results_dir), columns=columns)


def verify_manifest(tables_dir: Path) -> list[str]:
    """Every table of a manifest is present with the recorded sha256 and size; returns
    the problems found (empty when all is well)."""
    manifest = load_manifest(Path(tables_dir) / "MANIFEST.yaml")
    problems = []
    for t in manifest.get("tables", []):
        path = Path(tables_dir) / t["path"]
        if not path.exists():
            problems.append(f"{t['table']}: {path} missing")
            continue
        if sha256(path) != t["sha256"]:
            problems.append(f"{t['table']}: sha256 differs from the manifest")
        if "rows" in t and path.suffix == ".parquet":
            import pyarrow.parquet as pq
            n = pq.read_metadata(path).num_rows
            if n != t["rows"]:
                problems.append(f"{t['table']}: {n} rows, manifest says {t['rows']}")
    return problems
