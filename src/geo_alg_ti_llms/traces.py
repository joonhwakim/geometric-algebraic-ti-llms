"""Reasoning traces: the inferences a chain has composed at every budget, and
four verbatim example chains.

An inference combines two relations that share an item; a query at symbolic
distance d needs d - 1 of them. For every reasoning trial of two ladders -- the
ICL test (given names ordered by height) and the reversed context (US state
population, context-only instruction), on Qwen3-32B and Qwen2.5-32B-Instruct --
:func:`export_chain_steps` reads the first B tokens of the stored chain at every
budget B and counts how many of the d - 1 inferences the text has composed::

    links      the ten adjacent relations shown in the context (always available),
               plus every relation in the chain whose direction agrees with the
               ordering shown; a run of three or more items joined by one operator
               and agreeing with the ordering at every link ("A > B > C") links
               every pair of its items
    composed   (d - 1) - (the fewest links that connect the two queried items - 1)

So a chain with no relations has composed 0, one that contains the queried
relation in its true direction (or a run containing both items) has composed
d - 1, and pieces in different places combine ("A > C" early and "C > E" later
count as two inferences toward a query about A and E, one short of the three it
needs). Relations in the chain that go against the ordering add nothing; a
mention of the queried pair within 70 characters after a question cue
("statement", "whether", "determine", ...) is the model restating the question,
not deriving it; a direct answer (budget 0) has composed 0.

The text is read with the relation reader (:func:`normalize_tracked`,
:func:`_chains` and their tables): LaTeX, markdown and HTML operators are
normalized, item names and aliases become
placeholders, verbal comparatives ("has a larger population than", "is smaller
than", "is above", ...) become > or <, and a relation is a run of items joined
by operators with only whitespace between them.

:func:`export_trace_examples` copies four chains verbatim for the appendix: the
reversed context on US state population at distance four within the innermost
ranks, scored at the 1,024-token budget, one correct and one incorrect trial per
model. Within each model and outcome only chains that ended within the budget
are eligible, a correct one must also have composed all three inferences, and
the example is the eligible trial whose chain length is nearest the eligible
trials' median (ties to the lower trial id).

Written under ``results/reasoning_traces/tables/``:

* ``chain_steps.parquet``   one row per reasoning trial x budget of the two ladders;
* ``trace_examples.json``   the four trials: prompt, chain, scoring suffix, scored answer;
* ``MANIFEST.yaml``         per table: rows, sha256, columns, the selection rule,
  per trial set digests, the checks run and the self-test sentences.
"""

from __future__ import annotations

import hashlib
import json
import re
import statistics
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from geo_alg_ti_llms import experiments, tables
from geo_alg_ti_llms.tables import (
    CTXONLY_SENTENCE,
    DEFAULT_RESULTS_DIR,
    DROP_HI,
    DROP_LO,
    INNERMOST_RULE,
    REASONING_KIND,
    TRIAL_KEY_FORMAT,
    Flow,
    fail,
    row_digest,
    sha256,
)

EXPERIMENT = "reasoning_traces"
MODELS = ("qwen3-32b", "qwen2.5-32b-instruct")
# The two ladders, as (experiment, test, dim): the ICL-test ladder of the
# symbolic-distance experiment and the reversed-context ladder of the content-effects
# experiment (context-only instruction). Nothing was run for this analysis; it re-reads
# those runs, so there is no experiment declaration of its own.
LADDERS = (("symbolic_distance", "budget_icl", "countries"),
           ("content_effects", "budget_reversed", "usstates"))
TRIALS_PER_LADDER = 720
CUE_WINDOW = 70                 # characters before a run searched for a question cue

COLUMNS = ["model", "condition", "dim", "d", "rank_low", "rank_high", "innermost",
           "trial_key", "budget", "is_correct", "inferences_needed", "inferences_composed"]
DTYPES = {"model": "string", "condition": "string", "dim": "string",
          "d": "int8", "rank_low": "int8", "rank_high": "int8", "innermost": "bool",
          "trial_key": "string", "budget": "int16", "is_correct": "bool",
          "inferences_needed": "int8", "inferences_composed": "int8"}
ROW_ORDER = "condition, model, d, trial_key, budget"

# The example traces: the reversed ladder above, at this distance and budget.
EXAMPLE_LADDER = LADDERS[1]
EXAMPLE_DISTANCE = 4
EXAMPLE_BUDGET = 1024
OUTCOMES = (True, False)                 # correct first, then incorrect
TRIGGER = "Let's think step by step."
_EXAMPLE_QUERY_RE = re.compile(r"Is it true or false that\s+(.+?)\s*\?")


# --- the relation reader ----------------------------------------------------------------

ALIASES = {"United States": ["United States of America", "United States", "U.S.A.", "U.S.",
                             "USA", "US", "America"]}
_RESTATE_CUES = ("statement", "whether", " if ", "question", "determine", "check", "asked",
                 "claim", "figure out", "true or false", "verify", "conclusion", "answer",
                 "need to", "evaluate", "assess", "compare")

_GT_VERBAL = re.compile(
    r"\b(?:is|are|was|were|has|have|had|being|be)?\s*(?:\w+ly\s+)?(?:a\s+|an\s+|the\s+)?"
    r"(?:larger|bigger|greater|higher|taller|heavier|more\s+populous|more\s+populated|more)"
    r"(?:\s+(?:population|populated|people|populous))?\s+than\b", re.IGNORECASE)
_LT_VERBAL = re.compile(
    r"\b(?:is|are|was|were|has|have|had|being|be)?\s*(?:\w+ly\s+)?(?:a\s+|an\s+|the\s+)?"
    r"(?:smaller|lower|lesser|less|fewer|shorter|lighter|less\s+populous|less\s+populated)"
    r"(?:\s+(?:population|populated|people|populous))?\s+than\b", re.IGNORECASE)
_GT_POS = re.compile(r"\b(?:is|are|was|were|ranks?|comes?|sits?|lies?|placed)?\s*(?:\w+ly\s+)?"
                     r"(?:above|ahead\s+of|higher\s+than|over|before)\b", re.IGNORECASE)
_LT_POS = re.compile(r"\b(?:is|are|was|were|ranks?|comes?|sits?|lies?|placed)?\s*(?:\w+ly\s+)?"
                     r"(?:below|behind|beneath|under|lower\s+than|after)\b", re.IGNORECASE)
_TOK = re.compile(r"\u27e6(\d+)\u27e7|[<>]")


class _Tracked:
    """A string plus, for every character of it, the index in the ORIGINAL text it came
    from (``map[len(text)]`` is the original length). Each ``sub`` rewrites the text and
    the map together, replacement characters spreading linearly over the span they
    replace, so a position in the normalized text maps back to the raw trace exactly.
    The map, not the character fraction, is what recovers a raw position: normalization
    shortens the text (verbal comparatives -> ' > ', multi-word names -> placeholders),
    so a proportional estimate would land up to a few tokens off."""

    def __init__(self, text: str):
        self.text = text
        self.map = list(range(len(text) + 1))

    def sub(self, pattern, repl) -> "_Tracked":
        pat = re.compile(pattern) if isinstance(pattern, str) else pattern
        out, omap, last = [], [], 0
        for m in pat.finditer(self.text):
            s_, e_ = m.span()
            if s_ == e_:
                continue
            out.append(self.text[last:s_]); omap.extend(self.map[last:s_])
            r = m.expand(repl) if isinstance(repl, str) else repl(m)
            rs, re_ = self.map[s_], self.map[e_]
            L = len(r)
            omap.extend(rs + int((re_ - rs) * i / max(1, L)) for i in range(L))
            out.append(r); last = e_
        out.append(self.text[last:]); omap.extend(self.map[last:])
        self.text = "".join(out); self.map = omap
        return self

    def apply(self, subs) -> "_Tracked":
        for pat, repl in subs:
            self.sub(pat, repl)
        return self


# markup normalization, in this order (each entry = one regex substitution)
_MARKUP_SUBS = [
    (re.compile(r"</?think>"), " "),
    (re.compile(r"\\text\s*\{([^}]*)\}"), r"\1"),
    (re.compile(r"\\(?:mathrm|textbf|mathbf)\s*\{([^}]*)\}"), r"\1"),
    (re.compile(re.escape("\\gt")), ">"), (re.compile(re.escape("\\lt")), "<"),
    (re.compile(re.escape("&gt;")), ">"), (re.compile(re.escape("&lt;")), "<"),
    (re.compile(r"\\[\[\]\(\)]"), " "),
    (re.compile(re.escape("**")), " "), (re.compile(re.escape("`")), " "), (re.compile(re.escape("$")), " "),
    (re.compile("\u2265"), ">"), (re.compile("\u2264"), "<"),
    (re.compile("\u226b"), ">"), (re.compile("\u226a"), "<"),
    (re.compile("\u2192"), " -> "),
]


def _placeholder_subs(items: list[str]) -> list:
    """Item aliases -> ' ⟦k⟧ ', longest alias first so "United States of America" wins
    over "United States" / "US"."""
    subs = []
    for k, it in enumerate(items):
        for al in ALIASES.get(it, [it]):
            subs.append((al, k))
    subs.sort(key=lambda s_: -len(s_[0]))
    out = []
    for al, k in subs:
        if al.isupper() and len(al) <= 4:
            pat = re.compile(rf"(?<![A-Za-z])(?:the\s+)?{re.escape(al)}(?![A-Za-z])")
        else:
            pat = re.compile(rf"(?<!\w)(?:[Tt]he\s+)?{re.escape(al)}(?!\w)")
        out.append((pat, f" \u27e6{k}\u27e7 "))
    return out


_OPERATOR_SUBS = [(_GT_VERBAL, " > "), (_LT_VERBAL, " < "), (_GT_POS, " > "), (_LT_POS, " < ")]


def normalize_tracked(text: str, items: list[str]) -> _Tracked:
    """The full normalization (markup -> placeholders -> operators) with the offset map."""
    return _Tracked(text).apply(_MARKUP_SUBS).apply(_placeholder_subs(items)).apply(_OPERATOR_SUBS)


def _chains(norm: str):
    """Yield (items, ops, start, end) for every maximal run item (op item)+ where only
    whitespace separates consecutive tokens."""
    toks = [(m.start(), m.end(), m.group(0), m.group(1)) for m in _TOK.finditer(norm)]
    i = 0
    while i < len(toks):
        s, e, g, k = toks[i]
        if k is None:
            i += 1
            continue
        items, ops, start, end = [int(k)], [], s, e
        j = i + 1
        while j + 1 < len(toks):
            s1, e1, g1, k1 = toks[j]
            s2, e2, g2, k2 = toks[j + 1]
            if k1 is None and k2 is not None and norm[end:s1].strip() == "" \
                    and norm[e1:s2].strip() == "":
                ops.append(g1)
                items.append(int(k2))
                end = e2
                j += 2
            else:
                break
        if ops:
            yield items, ops, start, end
        i = j if ops else i + 1


# --- the composed-inference counter ------------------------------------------------------


def composed_inferences(trace: str, items: list[str], rank_low: int, rank_high: int) -> int:
    """Inferences toward the pair (rank_low, rank_high) that the text has composed (module doc).

    ``items`` is the ordering shown, rank 0 lowest (``extra.rank_to_item``); the
    relation reader's placeholders index it, so a placeholder's index is the item's rank.
    """
    need = rank_high - rank_low - 1
    if need <= 0 or not trace:
        return 0
    norm = normalize_tracked(trace, items).text
    query = {rank_low, rank_high}
    edges: set[tuple[int, int]] = set()
    for its, ops, start, _end in _chains(norm):
        links = [(a, b) if op == ">" else (b, a) for a, op, b in zip(its, ops, its[1:])]
        cued = any(c in norm[max(0, start - CUE_WINDOW):start].lower() for c in _RESTATE_CUES)
        for g, l in links:
            if g > l and not (cued and {g, l} == query):
                edges.add((g, l))
        if len(its) >= 3 and len(set(ops)) == 1 and all(g > l for g, l in links):
            run = sorted(its, reverse=True)
            edges.update((run[i], run[j]) for i in range(len(run)) for j in range(i + 1, len(run)))
    # Fewest links from the higher queried item down to the lower one; links only point
    # downward, so one pass in rank order suffices. The premises give hops[v] <= hi - v.
    hops = {rank_high: 0}
    for v in range(rank_high - 1, rank_low - 1, -1):
        hops[v] = min([hops[v + 1] + 1] + [hops[u] + 1 for (u, w) in edges
                                           if w == v and v < u <= rank_high])
    return need - (hops[rank_low] - 1)


# Self-test: the ordering shown (rank 0 lowest) and (text, rank_low, rank_high, expected count).
SELF_TEST_ITEMS = ["United States", "Brazil", "Mexico", "Germany", "Spain", "Ukraine",
                   "Netherlands", "Belgium", "Singapore", "Panama", "Jamaica"]
SELF_TEST = (
    ("", 4, 7, 0),
    ("Belgium > Netherlands, Netherlands > Ukraine, Ukraine > Spain, Belgium > Spain", 4, 7, 2),
    ("Belgium > Netherlands > Ukraine > Spain", 4, 7, 2),
    ("Spain < Ukraine < Netherlands < Belgium", 4, 7, 2),
    ("Belgium is larger than Ukraine.", 4, 7, 1),
    ("**Belgium** has a larger population than Ukraine.", 4, 7, 1),
    ("To determine if Belgium > Spain: Belgium > Netherlands, Netherlands > Ukraine.", 4, 7, 0),
    ("Spain > Belgium", 4, 7, 0),
    ("Jamaica > Panama > Singapore", 4, 7, 0),
    ("So Belgium > Ukraine. Later, Ukraine > Germany.", 3, 7, 2),
    ("Belgium > Ukraine > Netherlands", 4, 7, 1),     # a wrong link voids the run, not its right link
)


def self_test() -> list[dict]:
    """Every self-test sentence gives its stated count (fatal otherwise)."""
    out = []
    for text, lo, hi, want in SELF_TEST:
        got = composed_inferences(text, SELF_TEST_ITEMS, lo, hi)
        if got != want:
            fail(f"self-test: {text!r} (ranks {lo}-{hi}) gives {got}, expected {want}")
        out.append({"text": text, "query": f"{SELF_TEST_ITEMS[hi]} vs {SELF_TEST_ITEMS[lo]}",
                    "needed": hi - lo - 1, "composed": got})
    return out


# --- reading one ladder ------------------------------------------------------------------


def dir_digest(runs: Path, d: Path) -> dict:
    """The files of one stored directory: count and a digest over their names and contents."""
    h = hashlib.sha256()
    files = sorted(d.glob("*.jsonl"))
    for f in files:
        h.update(f"{f.name}\t{sha256(f)}\n".encode())
    return {"dir": str(d.relative_to(runs)), "n_files": len(files), "sha256": h.hexdigest()}


def ladder_budgets(exp: dict) -> tuple[int, ...]:
    return tuple(int(b) for b in exp["budgets"])


def read_ladder(runs: Path, exp: dict, test: str, dim: str, model: str,
                log=print) -> tuple[list[dict], dict]:
    """One ladder directory as slim rows (budget 0 once per trial, then every reasoning
    row), checked as the trial export checks it: the direct copies of every budget
    collapse to one row, every row's queried pair equals its direct twin's, every trial
    carries every budget, the trigger is the model's."""
    specs = tables.test_specs(exp)
    spec, twin = specs[test], specs[tables.twin_test(specs, test)]
    budgets = ladder_budgets(exp)
    label = f"{model}/{dim}/{test}"
    if not tables.test_dir(runs, spec, dim, model).exists():
        fail(f"{label}: directory missing ({tables.test_dir(runs, spec, dim, model)})")
    rows, triggers = tables.read_test(runs, spec, dim, model, budgets)
    rows, info = tables.collapse_ladder_direct(rows, label, budgets)
    direct, _ = tables.read_test(runs, twin, dim, model, budgets)
    expected = tables.reasoning_trigger(model, exp)
    if triggers != {expected}:
        fail(f"{label}: trigger ids {sorted(triggers)}, expected {expected!r} for this model")
    for r in rows + direct:
        r["reasoning_kind"] = REASONING_KIND[expected]
    info["twins"] = tables.attach_twins(rows, direct, label)
    tables.check_complete_ladders(rows, label, (0, *budgets))
    tables.check_unique_keys(rows, label)
    log(f"  {label}: {info}")
    return rows, info


def count_ladder(runs: Path, rows: list[dict]) -> None:
    """Fill ``inferences_needed`` / ``inferences_composed`` on every row of one ladder from
    the stored chains (a direct answer counts 0)."""
    lines: dict[str, list[str]] = {}
    for r in rows:
        if r["mode"] == "direct":
            r["inferences_composed"] = 0
            continue
        src = r["source_file"]
        if src not in lines:
            lines[src] = list(open(runs / src, encoding="utf-8"))       # split as read_test splits
        raw = json.loads(lines[src][r["source_line"]])
        if int(raw["trial_id"]) != r["trial_id"] or int(raw["symbolic_distance"]) != r["d"]:
            fail(f"{src}:{r['source_line']}: the stored row is not the ladder row's trial")
        r["inferences_composed"] = composed_inferences(raw["cot_trace"] or "",
                                                       list(raw["extra"]["rank_to_item"]),
                                                       r["rank_low"], r["rank_high"])
    for r in rows:
        r["inferences_needed"] = r["d"] - 1


def check_against_committed(raw: pd.DataFrame, experiment: str, test: str,
                            results_dir: Path = DEFAULT_RESULTS_DIR) -> dict:
    """Every row joins one row of the experiment's committed ladder table with the same
    queried ranks, correctness and logit gap."""
    path = tables.table_path(experiment, "ladders", results_dir)
    if not path.exists():
        fail(f"{path}: the ladder table is missing (export the trials of {experiment} first)")
    keys = ["model", "condition", "dim", "trial_key", "budget"]
    staged = pd.read_parquet(path, columns=[*keys, "test", "rank_low", "rank_high", "innermost",
                                            "is_correct", "logit_gap"])
    staged = staged[staged["test"] == test].drop(columns=["test"])
    staged = staged.merge(raw[keys].astype(staged[keys].dtypes.to_dict()).drop_duplicates(),
                          on=keys, how="inner")
    if staged.duplicated(keys).any():
        fail(f"{path.name}: a (model, condition, dim, trial_key, budget) occurs more than once")
    joined = (raw.astype({k: staged[k].dtype for k in keys})
              .merge(staged, on=keys, how="left", suffixes=("", "_staged"), indicator=True))
    if (joined["_merge"] != "both").any():
        fail(f"{experiment}/ladders: {int((joined['_merge'] != 'both').sum())} rows have no committed twin")
    for col in ("rank_low", "rank_high", "innermost", "is_correct"):
        bad = joined[col].astype(str) != joined[f"{col}_staged"].astype(str)
        if bad.any():
            fail(f"{experiment}/ladders: {int(bad.sum())} rows differ from the committed table in {col}")
    gap = np.abs(joined["logit_gap"].astype("float32") - joined["logit_gap_staged"].astype("float32"))
    if (gap > 1e-6).any():
        fail(f"{experiment}/ladders: {int((gap > 1e-6).sum())} rows differ from the committed table in logit_gap")
    return {"table": f"{experiment}/ladders", "sha256": sha256(path), "rows_matched": int(len(joined))}


# --- chain_steps.parquet ------------------------------------------------------------------


def _header(runs: Path, ladders: dict, budgets: tuple[int, ...]) -> dict:
    return {
        "experiment": EXPERIMENT,
        "exported": date.today().isoformat(),
        "runs_root": tables.RUNS_ROOT_PLACEHOLDER,
        "runs_layout": "<runs_root>/cot_sweep/final_<dim>/<published_as>/<model>/b<budget>__*.jsonl "
                       "(the ladders of the experiments named under derived_from)",
        "models": list(MODELS),
        "reasoning_kind": {m: REASONING_KIND[tables.reasoning_trigger(m)] for m in MODELS},
        "derived_from": ladders,
        "ladder_budgets": [0, *budgets],
        "innermost_ranks": INNERMOST_RULE,
        "trial_key_format": TRIAL_KEY_FORMAT,
    }


def _ladder_record(exp: dict, test: str, dim: str, condition: str) -> dict:
    spec = tables.test_specs(exp)[test]
    dimension, item_set = spec.item_sets[dim]
    return {"experiment": exp["name"], "dim": dim, "condition": condition,
            "dimension": exp["dimensions"][dimension]["name"], "item_set": item_set,
            "published_as": spec.published_as, "context_only_instruction": spec.ctxonly}


def export_chain_steps(runs: Path, out_dir: Path, results_dir: Path = DEFAULT_RESULTS_DIR,
                       experiments_dir: Path = experiments.DEFAULT_EXPERIMENTS_DIR,
                       log=print) -> dict:
    """``chain_steps.parquet`` + its manifest entry: one row per reasoning trial and
    budget of the two ladders, with the inferences needed and composed."""
    tests = self_test()
    frames, dirs, collapse, matched, ladders, budgets = [], [], {}, [], {}, None
    for experiment, test, dim in LADDERS:
        exp = experiments.load_experiment(experiment, experiments_dir)
        if budgets is None:
            budgets = ladder_budgets(exp)
        elif ladder_budgets(exp) != budgets:
            fail(f"{experiment}: ladder budgets differ from {LADDERS[0][0]}'s")
        spec = tables.test_specs(exp)[test]
        rows_here = []
        for model in MODELS:
            log(f"{model}/{dim}/{test}:")
            rows, info = read_ladder(runs, exp, test, dim, model, log)
            count_ladder(runs, rows)
            collapse[f"{dim}/{test}/{model}"] = info
            dirs.append(dir_digest(runs, tables.test_dir(runs, spec, dim, model)))
            rows_here += rows
        raw = pd.DataFrame(rows_here)
        matched.append(check_against_committed(raw, experiment, test, results_dir))
        condition = next(iter(spec.conditions.values()))
        if set(raw["condition"]) != {condition}:
            fail(f"{test}: conditions {sorted(set(raw['condition']))}, expected {condition!r}")
        ladders[test] = _ladder_record(exp, test, dim, condition)
        frames.append(raw[COLUMNS].astype(DTYPES))
    full_ladder = (0, *budgets)
    df = (pd.concat(frames, ignore_index=True)
          .sort_values(["condition", "model", "d", "trial_key", "budget"], kind="stable")
          .reset_index(drop=True))

    expected = len(MODELS) * len(LADDERS) * TRIALS_PER_LADDER * len(full_ladder)
    if len(df) != expected:
        fail(f"{len(df)} rows, expected {expected}")
    if df.duplicated(["model", "condition", "trial_key", "budget"]).any():
        fail("a (model, condition, trial_key, budget) occurs more than once")
    per_trial = df.groupby(["model", "condition", "trial_key"])["budget"].apply(lambda s: tuple(sorted(s)))
    if set(per_trial) != {full_ladder}:
        fail("some trial lacks a budget of the ladder")
    if (df.loc[df["budget"] == 0, "inferences_composed"] != 0).any():
        fail("a direct answer (budget 0) has a nonzero count")
    if ((df["inferences_composed"] < 0) | (df["inferences_composed"] > df["inferences_needed"])).any():
        fail("a count lies outside 0 .. d - 1")

    # Not fatal: prefixes grow with the budget, but a run cut at one budget can turn inconsistent
    # once the next tokens extend it, so a count can drop between budgets.
    wide = df.pivot_table(index=["model", "condition", "trial_key"], columns="budget",
                          values="inferences_composed")
    drops = int((wide.diff(axis=1) < 0).any(axis=1).sum())

    inner = df[df["innermost"] & (df["d"] >= 2)]
    at_top = {f"{m} | {c} | d={int(d)}": {"trials": int(len(g)),
                                         "mean_at_1024": round(float(g["inferences_composed"].mean()), 3),
                                         "needed": int(d) - 1}
              for (m, c, d), g in inner[inner["budget"] == full_ladder[-1]].groupby(["model", "condition", "d"])}

    out_dir.mkdir(parents=True, exist_ok=True)
    written = tables.write_parquet(df, out_dir / "chain_steps.parquet")
    trial_sets = {f"{m}|{c}|{dim}": Flow([int(len(g)), row_digest(g)])
             for (m, c, dim), g in df.groupby(["model", "condition", "dim"], sort=True)}
    entry = {
        "table": "chain_steps", **written,
        "columns": {c: str(df[c].dtype) for c in df.columns},
        "conditions": sorted(df["condition"].unique()),
        "models": sorted(df["model"].unique()),
        "dims": sorted(df["dim"].unique()),
        "row_order": ROW_ORDER,
        "definition": {
            "inferences_needed": "d - 1, the inferences (compositions of adjacent relations) a "
                                 "distance-d query needs",
            "inferences_composed": ("(d - 1) - (fewest links connecting the two queried items - 1), where "
                                    "the links are the ten adjacent relations of the context plus every "
                                    "relation in the first B tokens of the chain that runs in the direction of "
                                    "the ordering shown (a run of three or more items agreeing at every link "
                                    f"links all its pairs); a mention of the queried pair within {CUE_WINDOW} "
                                    "characters after one of the question cues is ignored; budget 0 (direct "
                                    "answer) is 0"),
            "reader": "geo_alg_ti_llms.traces (normalize_tracked, _chains, composed_inferences)",
            "question_cues": list(_RESTATE_CUES),
        },
        "innermost_mean_at_1024_by_model_condition_distance": at_top,
        "trial_sets": trial_sets,
        "checks": {
            "self_test": tests,
            "committed_tables_matched": matched,
            "raw_directories": dirs,
            "ladder_direct_collapse": collapse,
            "rows_expected": expected,
            "trials_with_a_count_dropping_between_budgets": drops,
            "rules": [
                "every ladder row is read and checked as the trial export reads it (collapse, "
                "direct twin, complete ladder, unique keys, the model's trigger)",
                "every row joins one committed ladder row (model, condition, dim, trial_key, budget) "
                "of its experiment with the same rank_low, rank_high, innermost, is_correct and "
                "logit_gap",
                "every stored trace row names the ladder row's trial id and distance",
                "every trial has all eleven budgets; budget 0 counts 0; every count lies in 0 .. d - 1",
            ],
        },
        "note": ("One row per reasoning trial and budget of the two ladders. Rows at every distance "
                 "and rank are kept; the paper's panels use the innermost ranks at distances 2-6 "
                 "(distance 1 needs no inference). Each count is read from the first B tokens of "
                 "the trial's single 1,024-token greedy chain (cot_trace of the b<B> file); it "
                 "reads the text only and says nothing about how the answer was scored. Answers "
                 "and correctness are the stored ones, unchanged."),
    }
    manifest = tables.update_manifest(out_dir / "MANIFEST.yaml", _header(runs, ladders, budgets), [entry])
    log(f"self-test: {len(tests)} sentences pass")
    for c in matched:
        log(f"matched {c['rows_matched']:,} rows of {c['table']}")
    log(f"wrote {out_dir / 'chain_steps.parquet'} ({len(df):,} rows); {drops} trials have a count "
        f"dropping between budgets")
    for k, v in at_top.items():
        log(f"  {k}: {v['mean_at_1024']} of {v['needed']} ({v['trials']} trials)")
    return manifest


# --- trace_examples.json ------------------------------------------------------------------


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def composed_at_budget(chains: pd.DataFrame, model: str, trial_key: str, budget: int) -> pd.Series:
    """The trial's chain_steps row at one budget."""
    row = chains[(chains["model"] == model) & (chains["trial_key"] == trial_key)
                 & (chains["budget"] == budget)]
    if len(row) != 1:
        fail(f"{model} {trial_key}: {len(row)} chain_steps rows at budget {budget}")
    return row.iloc[0]


def pick(group: list[dict]) -> tuple[dict, float]:
    """The trial whose chain length is nearest the group's median (ties: lower trial id)."""
    median = float(statistics.median(r["cot_n_tokens"] for r in group))
    return min(group, key=lambda r: (abs(r["cot_n_tokens"] - median), r["trial_id"])), median


def example(runs: Path, r: dict, model: str, prompted: bool, median: float, n_eligible: int,
            count: pd.Series, dimension: str) -> dict:
    """One trial as a verbatim record: reasoning-pass prompt, chain, scoring suffix, scored
    answer."""
    where = f"{r['source_file']}:{r['source_line']}"
    raw = json.loads(list(open(runs / r["source_file"], encoding="utf-8"))[r["source_line"]])
    if int(raw["trial_id"]) != r["trial_id"] or raw["mode"] != "cot":
        fail(f"{where}: the stored row is not the ladder row's trial")
    reasoning_prompt, trace, scored = raw["cot_stage1_prompt"], raw["cot_trace"], raw["prompt"]
    if not scored.startswith(reasoning_prompt + trace):
        fail(f"{where}: the scored prompt is not the reasoning-pass prompt followed by the chain")
    suffix = scored[len(reasoning_prompt) + len(trace):]
    if reasoning_prompt + trace + suffix != scored or "Therefore the answer is:" not in suffix:
        fail(f"{where}: unexpected scoring suffix {suffix!r}")
    if CTXONLY_SENTENCE not in reasoning_prompt:
        fail(f"{where}: the reasoning-pass prompt lacks the context-only sentence")
    user_turn = reasoning_prompt.split("<|im_start|>user", 1)[1].split("<|im_end|>", 1)[0]
    if prompted != user_turn.rstrip().endswith(TRIGGER):
        fail(f"{where}: the step-by-step trigger is {'missing' if prompted else 'present'}")
    queries = _EXAMPLE_QUERY_RE.findall(reasoning_prompt)
    if len(queries) != 1:
        fail(f"{where}: {len(queries)} query clauses in the reasoning-pass prompt")
    left, right = (s.strip() for s in queries[0].split(">"))
    r2i = list(raw["extra"]["rank_to_item"])
    if {left, right} != {r2i[r["rank_low"]], r2i[r["rank_high"]]}:
        fail(f"{where}: the queried items are not the trial's queried ranks")
    composed = int(count["inferences_composed"])
    if composed != composed_inferences(trace, r2i, r["rank_low"], r["rank_high"]):
        fail(f"{where}: the chain_steps count is not the count of the stored chain")
    return {
        "model": model,
        "model_hf": raw["model"],
        "test": r["test"],
        "condition": r["condition"],
        "dimension": dimension,
        "trial_key": r["trial_key"],
        "trial_id": r["trial_id"],
        "d": r["d"],
        "rank_low": r["rank_low"],
        "rank_high": r["rank_high"],
        "context_ordering": r2i,
        "query": f"{left} > {right}",
        "query_true_in_ordering_shown": r["gold_true"],
        "scored_answer_true": r["predicted_true"],
        "is_correct": r["is_correct"],
        "logprob_true": r["logprob_true"],
        "logprob_false": r["logprob_false"],
        "cot_n_tokens": int(r["cot_n_tokens"]),
        "cot_finish_reason": r["cot_finish_reason"],
        "inferences_needed": int(count["inferences_needed"]),
        "inferences_composed": composed,
        "group_eligible_trials": n_eligible,
        "group_median_tokens": median,
        "source_file": r["source_file"],
        "source_line": r["source_line"],
        "sha256_scored_prompt": sha256_text(scored),
        "sha256_trace": sha256_text(trace),
        "non_ascii": sorted({c for c in reasoning_prompt + trace + suffix if ord(c) > 127}),
        "reasoning_prompt": reasoning_prompt,
        "trace": trace,
        "scoring_suffix": suffix,
    }


def export_trace_examples(runs: Path, out_dir: Path, results_dir: Path = DEFAULT_RESULTS_DIR,
                          experiments_dir: Path = experiments.DEFAULT_EXPERIMENTS_DIR,
                          log=print) -> dict:
    """``trace_examples.json`` + its manifest entry: the four example chains, verbatim.

    Reads ``chain_steps.parquet`` from ``out_dir`` (export the chain steps first) for the
    correct examples' count criterion.
    """
    chains_path = out_dir / "chain_steps.parquet"
    if not chains_path.exists():
        fail(f"{chains_path}: missing (export the chain steps first)")
    chains = pd.read_parquet(chains_path)
    experiment, test, dim = EXAMPLE_LADDER
    exp = experiments.load_experiment(experiment, experiments_dir)
    spec = tables.test_specs(exp)[test]
    dimension = exp["dimensions"][spec.item_sets[dim][0]]["name"]
    picked, examples, groups = [], [], {}
    for model in MODELS:
        log(f"{model}/{dim}/{test}:")
        rows, _ = read_ladder(runs, exp, test, dim, model, log)
        prompted = REASONING_KIND[tables.reasoning_trigger(model, exp)] == "prompted"
        top = [r for r in rows if r["mode"] == "cot" and r["budget"] == EXAMPLE_BUDGET
               and r["innermost"] and r["d"] == EXAMPLE_DISTANCE]
        for correct in OUTCOMES:
            outcome = "correct" if correct else "incorrect"
            finished = [r for r in top if r["is_correct"] == correct and not r["truncated"]]
            group = [r for r in finished if not correct
                     or int(composed_at_budget(chains, model, r["trial_key"], EXAMPLE_BUDGET)
                            ["inferences_composed"]) == r["d"] - 1]
            groups[f"{model} | {outcome}"] = {
                "trials": sum(r["is_correct"] == correct for r in top),
                "finished_within_budget": len(finished),
                "eligible": len(group)}
            if not group:
                fail(f"{model} {outcome}: no eligible trial at distance {EXAMPLE_DISTANCE}")
            r, median = pick(group)
            picked.append(r)
            count = composed_at_budget(chains, model, r["trial_key"], EXAMPLE_BUDGET)
            examples.append({"id": f"{model}-{outcome}",
                             **example(runs, r, model, prompted, median, len(group), count, dimension)})
    matched = check_against_committed(pd.DataFrame(picked), experiment, test, results_dir)

    selection = {
        "ladder": f"{dim}/{test} ({dimension}, reversed context, context-only instruction; "
                  f"experiment {experiment}, stored as {spec.published_as})",
        "budget": EXAMPLE_BUDGET,
        "distance": EXAMPLE_DISTANCE,
        "ranks": f"innermost (rank_low >= {DROP_LO} and rank_high <= {DROP_HI})",
        "rule": (f"within each model and outcome (correct, incorrect at the {EXAMPLE_BUDGET:,}-token "
                 "budget), the chains that ended within the budget are eligible, and a correct "
                 f"example must also have composed all d - 1 inferences by {EXAMPLE_BUDGET:,} tokens; "
                 "the example is the eligible trial whose chain length (tokens) is nearest the "
                 "eligible trials' median, ties to the lower trial id"),
        "groups": groups,
    }
    payload = {"selection": selection, "examples": examples}
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "trace_examples.json"
    out_json.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    entry = {
        "table": "trace_examples", "path": out_json.name, "sha256": sha256(out_json),
        "bytes": int(out_json.stat().st_size),
        "n_examples": len(examples),
        "selection": selection,
        "examples": [{k: e[k] for k in ("id", "trial_key", "query", "query_true_in_ordering_shown",
                                        "scored_answer_true", "is_correct", "cot_n_tokens",
                                        "cot_finish_reason", "inferences_composed", "inferences_needed",
                                        "source_file", "source_line", "sha256_scored_prompt",
                                        "sha256_trace")}
                     for e in examples],
        "checks": {
            "committed_tables_matched": [matched],
            "chain_steps": {"path": chains_path.name, "sha256": sha256(chains_path)},
            "rules": [
                "every example joins one committed ladder row (model, condition, dim, trial_key, "
                "budget) with the same rank_low, rank_high, innermost, is_correct and logit_gap",
                "the stored scored prompt is the reasoning-pass prompt, the chain and the scoring suffix, "
                "byte for byte",
                "the reasoning-pass prompt carries the context-only sentence; only a prompted-reasoning "
                "model's ends its user turn with the step-by-step trigger",
                "the queried statement's items are the trial's queried ranks",
                "the example's count at the full budget is the count of its stored chain",
            ],
        },
        "note": ("Four reasoning trials of the reversed context on US state population under the "
                 "context-only instruction (distance 4, 1,024-token budget), one correct and one "
                 "incorrect for each model, chosen by the rule in 'selection': the reasoning-pass prompt, "
                 "the chain and the scoring suffix verbatim, with the stored answer. Strings are "
                 "verbatim; a figure or appendix must not edit them beyond eliding a chain."),
    }
    manifest = tables.update_manifest(out_dir / "MANIFEST.yaml", {"exported": date.today().isoformat(),
                                                                  "runs_root": tables.RUNS_ROOT_PLACEHOLDER}, [entry])
    log(f"matched {matched['rows_matched']} rows of {matched['table']}")
    for k, v in groups.items():
        log(f"  {k}: {v['finished_within_budget']} of {v['trials']} trials ended within the budget; "
            f"{v['eligible']} eligible")
    for e in examples:
        log(f"  {e['id']}: {e['trial_key']} | {e['query']} "
            f"(true: {e['query_true_in_ordering_shown']}; answered true: {e['scored_answer_true']}) "
            f"| {e['cot_n_tokens']} tokens (median {e['group_median_tokens']}) "
            f"| composed {e['inferences_composed']} of {e['inferences_needed']} | non-ASCII {e['non_ascii']}")
    log(f"wrote {out_json}")
    return manifest


# --- reading the committed tables ----------------------------------------------------------


def read_chain_steps(results_dir: Path = DEFAULT_RESULTS_DIR) -> pd.DataFrame:
    return tables.read_table(EXPERIMENT, "chain_steps", results_dir)


def read_trace_examples(results_dir: Path = DEFAULT_RESULTS_DIR) -> dict:
    return json.loads((Path(results_dir) / EXPERIMENT / "tables" / "trace_examples.json").read_text(encoding="utf-8"))
