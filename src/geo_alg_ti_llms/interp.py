"""The stored intervention runs → the committed tables of the representation and
components experiments, with manifests.

Both experiments score Qwen3-32B on three dimensions (country, city and
US state population) under the context-only instruction, on the same four
conditions: the IWL test (no context, familiar items), the reversed context
(direct answer), the reversed context after a 1,024-token reasoning chain and
the ICL test (given names) after a chain. A run is one directory of
``<test>__<condition>.jsonl`` files per dimension, declared in
``experiments/<experiment>/intervention.yaml``:

* representation: ``rank_intervention/<dim>/<model>/`` holds the fit
  (``rank_axis.npz``, ``probe_by_layer.csv``, ``fit_manifest.json``) and the
  intervention directory (conditions ``intact``, ``erase``, ``invert``,
  ``rand_erase``, ``write_<eta>``, ``write_rand``);
* components: ``head_ablation/<dim>/<model>/`` holds the head scores and sets
  (``selected_heads.json``) and the ablation directory (``top<k>``,
  ``rand<k>_s0``).

:func:`export_interp` writes, under ``results/<experiment>/tables/``, one row
per scored trial read-out of the innermost 7 ranks at budgets 0 and 1,024
(``trials.parquet``), the probe by layer (``probe.parquet``) or the head sets
(``heads.parquet``), and ``MANIFEST.yaml``: the tests and their conditions, the job
records, per trial set digests, and every number of the
paper's panels recomputed from the table and checked against the summaries
stored with the runs.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

from geo_alg_ti_llms import tables
from geo_alg_ti_llms.experiments import DEFAULT_EXPERIMENTS_DIR
from geo_alg_ti_llms.tables import DROP_HI, DROP_LO, INNERMOST_RANKS, N_ITEMS, fail

EXPERIMENTS = ("representation", "components")
DIRECT_TESTS = ("iwl", "content_reversed")
CHAIN_TESTS = ("cot_reversed", "cot_icl")
TEST_ASSIGNMENT = {"iwl": "natural", "content_reversed": "reversed", "cot_reversed": "reversed",
                  "cot_icl": "random"}
NOVEL_SET = "novel_names_11"
CHAIN_BUDGETS = (256, 512, 1024)

TRIAL_COLUMNS = ["dim", "test", "condition", "budget", "pool", "premise_order", "trial_id",
                 "rank_low", "rank_high", "d", "is_correct", "agree_world", "agree_written"]
TRIAL_DTYPES = {"dim": "string", "test": "string", "condition": "string",
                "budget": "int16", "pool": "string", "premise_order": "int8", "trial_id": "int16",
                "rank_low": "int8", "rank_high": "int8", "d": "int8", "eta": "Float32",
                "k": "Int8", "is_correct": "bool", "agree_world": "boolean",
                "agree_written": "bool"}
INTERVENTION_COLUMN = {"representation": "eta", "components": "k"}
PROBE_SOURCE_COLUMNS = ["layer", "n_occ", "acc_axis_heldout", "acc_ridge_heldout", "alpha",
                        "beta", "r_a_z", "a_std"]
PROBE_COLUMNS = ["dim", *PROBE_SOURCE_COLUMNS, "n_items", "n_occurrences"]
PROBE_DTYPES = {"dim": "string", "layer": "int8", "n_occ": "int32",
                "acc_axis_heldout": "float64", "acc_ridge_heldout": "float64",
                "alpha": "float64", "beta": "float64", "r_a_z": "float64", "a_std": "float64",
                "n_items": "int16", "n_occurrences": "int32"}
HEADS_COLUMNS = ["dim", "set", "rank", "layer", "head"]
HEADS_DTYPES = {"dim": "string", "set": "string", "rank": "int8", "layer": "int8",
                "head": "int8"}
# Fields of a raw record compared across conditions (the trial, never the model's answer).
TRIAL_SIGNATURE = ("pool", "premise_order", "trial_id", "d", "rank_low", "rank_high", "A", "B",
                   "correct", "world_truth", "written_truth")
# Panel C of the stored figure data: series -> (test, budget); the plotted measure is agree_written.
PANEL_C_SERIES = {"direct": ("content_reversed", 0), "cot": ("cot_reversed", 1024)}
PANEL_C_ETA = {"intact": "0", "write_0.25": "0.25", "write_0.5": "0.5", "write_0.75": "0.75",
               "write_1.0": "1.0", "write_rand": "control"}
PANELS_OF = {"representation": ("A", "B", "C", "E"), "components": ("D", "E")}
SOURCE_OF = {"representation": "leace", "components": "heads"}
SE_TOLERANCE = 1e-12
COUNT_TOLERANCE = 1e-9        # value * n within this of an integer counts as k/n
WILSON_Z = 1.96
WILSON_TOLERANCE = 1e-12
PROBE_TOLERANCE = 1e-15

# The intervention conditions and the trial-table columns, as the manifest records them.
CONDITION_GLOSSARY = {
    "representation": {
        "intact": "no edit (a' = a)",
        "erase": "the coordinate on the rank direction set to zero (a' = 0)",
        "invert": "the coordinate negated (a' = -a)",
        "rand_erase": "the coordinate on a random unit direction set to zero (control for erase)",
        "write_<eta>": ("a' = (1 - eta) a + eta (alpha + beta z_new), z_new a random ordering of the "
                        "trial's items with |Spearman rho| <= 0.2 to the real one"),
        "write_rand": "the write at eta = 1 along the random direction (control for the writes)",
    },
    "components": {
        "intact": "no head ablated",
        "top<k>": "the k highest-scoring heads mean-ablated",
        "rand<k>_s0": "k random heads outside the top 64 mean-ablated (seed 100 k + 0)",
    },
}
_COLUMNS = {
    "dim": "dimension: countries | cities | usstates",
    "test": ("iwl = IWL test (no context, familiar items); content_reversed = reversed context, "
             "direct answer; cot_reversed = reversed context after a reasoning chain; cot_icl = ICL "
             "test (given names) after a reasoning chain"),
    "condition": None,
    "budget": "reasoning tokens; 0 = direct answer; reasoning tests kept at 1024 only",
    "pool": "item set; the IWL test pools the main item set and its replicates",
    "premise_order": "premise order index within the item set",
    "trial_id": "trial id within an item set and distance",
    "rank_low": "lower queried rank in the ordering shown",
    "rank_high": "higher queried rank",
    "d": "symbolic distance (rank_high - rank_low)",
    "intervention": None,
    "is_correct": "the answer equals the gold answer in the ordering shown (accuracy)",
    "agree_world": "the answer equals the truth in the real ordering; null for the given names",
    "agree_written": None,
}
COLUMN_DEFINITIONS = {
    experiment: {(INTERVENTION_COLUMN[experiment] if c == "intervention" else c): (text or own[c])
                 for c, text in _COLUMNS.items()}
    for experiment, own in {
        "representation": {
            "condition": "the intervention (see condition_glossary); intact = no edit",
            "intervention": "write strength (write conditions; write_rand carries 1.0); null elsewhere",
            "agree_written": ("the answer equals the truth in the written ordering (the same "
                              "ordering under every condition)")},
        "components": {
            "condition": "the head set ablated (see condition_glossary); intact = none",
            "intervention": "number of heads ablated (0 = intact)",
            "agree_written": ("the answer equals the truth in the written ordering of the "
                              "representation experiment's writes (the same ordering under every "
                              "condition)")},
    }.items()
}


# --- the declaration -------------------------------------------------------------------


def declaration_path(experiment: str, experiments_dir: str | Path = DEFAULT_EXPERIMENTS_DIR) -> Path:
    return Path(experiments_dir) / experiment / "intervention.yaml"


def load_declaration(experiment: str, experiments_dir: str | Path = DEFAULT_EXPERIMENTS_DIR) -> dict:
    """``experiments/<experiment>/intervention.yaml``, validated."""
    if experiment not in EXPERIMENTS:
        raise ValueError(f"not an intervention experiment: {experiment!r} (one of {EXPERIMENTS})")
    path = declaration_path(experiment, experiments_dir)
    with path.open("r", encoding="utf-8") as f:
        decl = yaml.safe_load(f)
    if decl.get("name") != experiment:
        raise ValueError(f"{path}: name field {decl.get('name')!r} != directory {experiment!r}")
    for key in ("model", "instruction", "dimensions", "conditions", "headline_conditions",
                "published_as", "budgets_kept"):
        if key not in decl:
            raise ValueError(f"{path}: missing {key!r}")
    for test in tests_of(decl):
        if test not in TEST_ASSIGNMENT:
            raise ValueError(f"{path}: unknown test {test!r}")
    return decl


def tests_of(decl: dict) -> list[str]:
    """The scored tests of a declaration, in order: the keys of its ``conditions``."""
    return list(decl["conditions"])


def expected_eta(experiment: str, condition: str):
    if experiment != "representation" or not condition.startswith("write_"):
        return None
    return 1.0 if condition == "write_rand" else float(condition.removeprefix("write_"))


def expected_k(experiment: str, condition: str):
    if experiment != "components":
        return None
    if condition == "intact":
        return 0
    return int(condition.removeprefix("top").removeprefix("rand").split("_")[0])


# --- reading the runs ------------------------------------------------------------------


def run_dir(runs: Path, decl: dict, dim: str) -> Path:
    return Path(runs) / decl["published_as"]["run"].format(dim=dim, model=decl["model"])


def trial_dir(runs: Path, decl: dict, dim: str) -> Path:
    return run_dir(runs, decl, dim) / decl["published_as"]["trials"]


def need(path: Path) -> Path:
    if not path.exists():
        fail(f"{path}: missing from the published runs")
    return path


def read_csv(path: Path) -> pd.DataFrame:
    """The stored CSVs, parsed so that every float round-trips exactly."""
    return pd.read_csv(path, float_precision="round_trip")


def main_set(decl: dict, dim: str) -> str:
    return decl["dimensions"][dim]["item_set"]


def read_file(runs: Path, decl: dict, dim: str, test: str, condition: str) -> list[dict]:
    """One stored trial file, every record guarded."""
    experiment = decl["name"]
    path = need(trial_dir(runs, decl, dim) / f"{test}__{condition}.jsonl")
    rows, seen = [], set()
    for line_no, line in enumerate(open(path, encoding="utf-8")):
        if not line.strip():
            continue
        r = json.loads(line)
        where = f"{path.relative_to(runs)}:{line_no}"
        if r["arm"] != test or r["condition"] != condition:
            fail(f"{where}: test/condition {r['arm']}/{r['condition']}, the file says {test}/{condition}")
        if r["assignment"] != TEST_ASSIGNMENT[test]:
            fail(f"{where}: assignment {r['assignment']!r} under test {test}")
        item_set = r["pool"]
        if test == "cot_icl":
            if item_set != NOVEL_SET:
                fail(f"{where}: item set {item_set!r}, expected {NOVEL_SET}")
        elif test == "iwl":
            if item_set != main_set(decl, dim) and not item_set.startswith(f"{main_set(decl, dim)}_rep"):
                fail(f"{where}: item set {item_set!r} is not {main_set(decl, dim)} or one of its replicates")
        elif item_set != main_set(decl, dim):
            fail(f"{where}: item set {item_set!r}, expected {main_set(decl, dim)}")
        if r["rank_high"] - r["rank_low"] != r["d"] or r["d"] < 1:
            fail(f"{where}: ranks {r['rank_low']}-{r['rank_high']} at distance {r['d']}")
        if test in DIRECT_TESTS and r["budget"] != 0:
            fail(f"{where}: a direct test at budget {r['budget']}")
        if test in CHAIN_TESTS and r["budget"] not in CHAIN_BUDGETS:
            fail(f"{where}: a reasoning test at budget {r['budget']}")
        if r.get("eta") != expected_eta(experiment, condition):
            fail(f"{where}: eta {r.get('eta')!r} under {experiment}/{condition}")
        if r.get("k") != expected_k(experiment, condition):
            fail(f"{where}: k {r.get('k')!r} under {experiment}/{condition}")
        if r["is_correct"] != (r["predicted"] == r["correct"]):
            fail(f"{where}: is_correct disagrees with predicted and correct")
        if r["agree_written"] != (r["predicted"] == r["written_truth"]):
            fail(f"{where}: agree_written disagrees with predicted and written_truth")
        if (r["world_truth"] is None) != (item_set == NOVEL_SET):
            fail(f"{where}: world_truth {r['world_truth']!r} on item set {item_set}")
        if r["world_truth"] is None:
            if r["agree_world"] is not None:
                fail(f"{where}: agree_world set without a real ordering")
        elif r["agree_world"] != (r["predicted"] == r["world_truth"]):
            fail(f"{where}: agree_world disagrees with predicted and world_truth")
        key = (item_set, r["d"], r["trial_id"], r["budget"])
        if key in seen:
            fail(f"{where}: trial {key} repeated in one file")
        seen.add(key)
        rows.append({"dim": dim, "test": r["arm"], **{f: r[f] for f in (
            "condition", "budget", "pool")}, "premise_order": r["perm"], **{f: r[f] for f in (
            "trial_id", "rank_low", "rank_high", "d", "is_correct", "agree_world", "agree_written",
            "A", "B", "correct", "world_truth", "written_truth")}, "eta": r.get("eta"), "k": r.get("k")})
    if not rows:
        fail(f"{path}: empty")
    return rows


def check_pairing(raw: pd.DataFrame) -> dict:
    """Every condition of a test scores the same trials; chains have 3 readouts."""
    out = {}
    for (dim, test, budget), g in raw.groupby(["dim", "test", "budget"], sort=False):
        sigs = {c: sorted(map(tuple, s[list(TRIAL_SIGNATURE)].astype(str).values))
                for c, s in g.groupby("condition", sort=False)}
        first = next(iter(sigs.values()))
        for c, s in sigs.items():
            if s != first:
                fail(f"{dim}/{test} budget {budget}: condition {c} scores different trials")
        out[f"{dim}|{test}|{budget}"] = {"trials": len(first), "conditions": len(sigs)}
    chains = raw[raw["test"].isin(CHAIN_TESTS)]
    per_trial = chains.groupby(["dim", "test", "condition", "pool", "premise_order", "d",
                                "trial_id"])["budget"].apply(lambda s: tuple(sorted(s)))
    bad = per_trial[per_trial != CHAIN_BUDGETS]
    if len(bad):
        fail(f"{len(bad)} reasoning trials without exactly the readouts {CHAIN_BUDGETS}, "
             f"e.g. {bad.index[0]}")
    return out


def innermost(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["rank_low"] >= DROP_LO) & (df["rank_high"] <= DROP_HI)]


def order_rows(df: pd.DataFrame, decl: dict) -> pd.DataFrame:
    dims = list(decl["dimensions"])
    tests = list(tests_of(decl))
    rank = {"dim": {v: i for i, v in enumerate(dims)}, "test": {v: i for i, v in enumerate(tests)}}
    keys = pd.DataFrame({c: df[c].map(m) for c, m in rank.items()})
    keys["condition"] = [decl["conditions"][a].index(c) for a, c in
                         zip(df["test"], df["condition"], strict=True)]
    for c in ("budget", "pool", "premise_order", "d", "trial_id"):
        keys[c] = df[c].values
    order = keys.sort_values(list(keys.columns), kind="stable").index
    return df.loc[order].reset_index(drop=True)


def read_trials(runs: Path, decl: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Every record of the declared files (guarded), the innermost rows at the kept budgets in the
    committed order and columns, and the checks' outcomes."""
    raw_rows, files = [], []
    for dim in decl["dimensions"]:
        for test in tests_of(decl):
            for condition in decl["conditions"][test]:
                files.append(trial_dir(runs, decl, dim) / f"{test}__{condition}.jsonl")
                raw_rows += read_file(runs, decl, dim, test, condition)
    raw = pd.DataFrame(raw_rows)
    pairing = check_pairing(raw)
    chains_outside = int(len(raw[raw["test"].isin(CHAIN_TESTS)])
                         - len(innermost(raw[raw["test"].isin(CHAIN_TESTS)])))
    raw_inner = innermost(raw)
    iwl_sets = {dim: sorted(raw[(raw["dim"] == dim) & (raw["test"] == "iwl")]["pool"].unique())
                 for dim in decl["dimensions"]}
    kept = raw_inner[raw_inner["budget"].isin(decl["budgets_kept"])]
    column = INTERVENTION_COLUMN[decl["name"]]
    columns = TRIAL_COLUMNS[:10] + [column] + TRIAL_COLUMNS[10:]
    trials = order_rows(kept, decl)[columns].astype({c: TRIAL_DTYPES[c] for c in columns})
    sizes = trials.groupby(["dim", "test", "budget", "condition"], sort=False).size()
    per_test = sizes.groupby(level=["dim", "test", "budget"], sort=False).agg(["min", "max", "size"])
    if (per_test["min"] != per_test["max"]).any():
        fail("the conditions of a test keep different numbers of innermost trials")
    if sorted(trials["d"].unique()) != list(range(1, INNERMOST_RANKS)):
        fail(f"innermost distances {sorted(trials['d'].unique())}")
    checks = {
        "trial_files_read": len(files),
        "records_read": int(len(raw)),
        "same_trials_in_every_condition": pairing,
        "reasoning_trials_have_three_readouts": True,
        "reasoning_records_outside_innermost": chains_outside,
        "innermost_trials_per_condition": {f"{dim}|{test}|budget={budget}": int(r["max"])
                                           for (dim, test, budget), r in per_test.iterrows()},
        "iwl_sets": iwl_sets,
    }
    return raw_inner, trials, checks


def probe_table(runs: Path, decl: dict) -> tuple[pd.DataFrame, dict]:
    frames, fits = [], {}
    for dim in decl["dimensions"]:
        csv_path = need(run_dir(runs, decl, dim) / "probe_by_layer.csv")
        fit_path = need(run_dir(runs, decl, dim) / "fit_manifest.json")
        df = read_csv(csv_path)
        if list(df.columns) != PROBE_SOURCE_COLUMNS:
            fail(f"{csv_path}: columns {list(df.columns)}")
        if list(df["layer"]) != list(range(len(df))):
            fail(f"{csv_path}: layers are not 0..{len(df) - 1}")
        fit = json.loads(fit_path.read_text(encoding="utf-8"))
        if fit["dim"] != dim:
            fail(f"{fit_path}: dim {fit['dim']}")
        fits[dim] = fit
        frames.append(df.assign(dim=dim, n_items=fit["n_items"], n_occurrences=fit["n_occurrences"]))
    probe = pd.concat(frames, ignore_index=True)[PROBE_COLUMNS].astype(PROBE_DTYPES)
    return probe, fits


def heads_table(runs: Path, decl: dict) -> tuple[pd.DataFrame, dict]:
    rows, meta = [], {}
    for dim in decl["dimensions"]:
        path = need(run_dir(runs, decl, dim) / "selected_heads.json")
        sel = json.loads(path.read_text(encoding="utf-8"))
        n_layers, n_heads = int(sel["L"]), int(sel["H"])
        for name, heads in sel["selected"].items():
            if len(set(map(tuple, heads))) != len(heads):
                fail(f"{path}: set {name} lists a head twice")
            for i, (layer, head) in enumerate(heads, start=1):
                if not (0 <= layer < n_layers and 0 <= head < n_heads):
                    fail(f"{path}: set {name} head ({layer}, {head}) out of range")
                rows.append({"dim": dim, "set": name, "rank": i, "layer": layer, "head": head})
        tops = sorted((k for k in sel["selected"] if k.startswith("top")),
                      key=lambda k: int(k.removeprefix("top")))
        nested = all(sel["selected"][b][:len(sel["selected"][a])] == sel["selected"][a]
                     for a, b in zip(tops, tops[1:], strict=False))
        meta[dim] = {**{k: v for k, v in sel.items() if k != "selected"},
                     "sets": {k: len(v) for k, v in sel["selected"].items()},
                     "top_sets_are_prefixes_of_each_other": bool(nested)}
    heads = pd.DataFrame(rows, columns=HEADS_COLUMNS).astype(HEADS_DTYPES)
    return heads, meta


# --- the numbers of the paper's panels ------------------------------------------------


def counts_of(frame: pd.DataFrame, measure: str) -> tuple[int, int]:
    """(k, n) of one measure over a frame, nulls dropped (agree_world on invented names)."""
    vals = frame[measure].dropna().astype(bool)
    return int(vals.sum()), int(len(vals))


def acc(k: int, n: int) -> float:
    return k / n


def wilson(k: int, n: int, z: float = WILSON_Z) -> tuple[float, float]:
    p = k / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return center - half, center + half


def se(p: float, n: int) -> float:
    return math.sqrt(p * (1 - p) / n)


def count_of(trials: pd.DataFrame, dim: str, test: str, condition: str, budget: int,
             measure: str = "is_correct", d: int | None = None) -> tuple[int, int]:
    sub = trials[(trials["dim"] == dim) & (trials["test"] == test)
                 & (trials["condition"] == condition) & (trials["budget"] == budget)]
    if d is not None:
        sub = sub[sub["d"] == d]
    k, n = counts_of(sub, measure)
    if not n:
        fail(f"no rows for {dim}/{test}/{condition}/{budget}/{measure}/d={d}")
    return k, n


NOT_PRESENT = "stored summaries not present"
"""Recorded by :func:`compare_summary` / :func:`compare_figure` when the run directory carries no
``summary.csv`` / ``fig_app_representation_data.csv``: these are files of the published runs that
this repository's stages do not write, so a re-run has nothing to compare against."""


def stored_count(value: float, n: int) -> int | None:
    """The k of a value reported as k/n (None when value * n is not an integer)."""
    k = round(value * n)
    return k if abs(value * n - k) < COUNT_TOLERANCE else None


def compare_summary(runs: Path, decl: dict, dim: str, trials: pd.DataFrame, raw_inner: pd.DataFrame) -> dict:
    """Every trial set of the run's ``summary.csv`` that the export covers against the table (kept
    budgets) or the raw records (budgets 256 and 512)."""
    path = run_dir(runs, decl, dim) / decl["published_as"]["summary"]
    if not path.is_file():
        return {"status": NOT_PRESENT, "mismatches": []}
    stored = read_csv(path)
    exported = {(test, c) for test in tests_of(decl) for c in decl["conditions"][test]}
    counts, mismatches, wilson_err = Counter(), [], 0.0
    for row in stored.itertuples(index=False):
        if (row.arm, row.condition) not in exported:          # the stored csv names the test "arm"
            counts["test_or_condition_not_exported"] += 1
            continue
        source = trials if row.budget in decl["budgets_kept"] else raw_inner
        sub = source[(source["dim"] == dim) & (source["test"] == row.arm)
                     & (source["condition"] == row.condition) & (source["budget"] == row.budget)]
        d = str(getattr(row, "d", "all"))
        if d != "all":
            sub = sub[sub["d"] == int(d)]
        k, n = counts_of(sub, row.measure)
        ok = n == row.n and n > 0 and acc(k, n) == row.acc
        where = "table" if row.budget in decl["budgets_kept"] else "raw_256_512"
        counts[f"compared_{where}"] += 1
        if not ok:
            mismatches.append({"test": row.arm, "condition": row.condition,
                               "budget": int(row.budget), "d": d, "measure": row.measure,
                               "stored": [float(row.acc), int(row.n)],
                               "ours": [acc(k, n) if n else None, n]})
            continue
        lo, hi = wilson(k, n)
        wilson_err = max(wilson_err, abs(lo - row.lo), abs(hi - row.hi))
    if wilson_err > WILSON_TOLERANCE:
        fail(f"{dim}: a Wilson bound differs from z = {WILSON_Z} by {wilson_err:g}")
    return {"rows": int(len(stored)), **dict(counts), "max_abs_wilson_error": float(wilson_err),
            "mismatches": mismatches}


def panel_numbers(trials: pd.DataFrame, probe: pd.DataFrame | None, decl: dict, dim: str) -> list[dict]:
    """The values of the paper's panels for one dimension, computed from the table: probe accuracy
    by layer (A), the IWL test by distance under the rank-direction edits (B), agreement with the
    written ordering by write strength (C), the change of IWL-direct and ICL-chain accuracy per head
    set (D), and accuracy on the four conditions under the headline interventions (E)."""
    out = []
    experiment = decl["name"]
    source = SOURCE_OF[experiment]
    if "A" in PANELS_OF[experiment]:
        for r in probe[probe["dim"] == dim].itertuples(index=False):
            out.append(dict(panel="A", layer=int(r.layer), value=float(r.acc_axis_heldout), n=int(r.n_occ)))
    if "B" in PANELS_OF[experiment]:
        for cond in ("intact", "invert", "erase", "rand_erase"):
            for d in range(1, 7):
                k, n = count_of(trials, dim, "iwl", cond, 0, "is_correct", d)
                out.append(dict(panel="B", condition=cond, d=d, value=acc(k, n), n=n, se=se(acc(k, n), n)))
    if "C" in PANELS_OF[experiment]:
        for key, (test, b) in PANEL_C_SERIES.items():
            for cond, eta in PANEL_C_ETA.items():
                if cond not in decl["conditions"][test]:
                    continue
                k, n = count_of(trials, dim, test, cond, b, "agree_written")
                out.append(dict(panel="C", series=key, eta=eta, condition=cond, value=acc(k, n), n=n, se=se(acc(k, n), n)))
    if "D" in PANELS_OF[experiment]:
        i0, n0 = count_of(trials, dim, "iwl", "intact", 0)
        c0, nc0 = count_of(trials, dim, "cot_icl", "intact", 1024)
        for cond in [c for c in decl["conditions"]["cot_icl"] if c != "intact"]:
            ki, ni = count_of(trials, dim, "iwl", cond, 0)
            ci, nci = count_of(trials, dim, "cot_icl", cond, 1024)
            out.append(dict(panel="D", condition=cond, d_iwl_direct=acc(ki, ni) - acc(i0, n0),
                            d_icl_chain=acc(ci, nci) - acc(c0, nc0), n_direct=ni, n_chain=nci))
    if "E" in PANELS_OF[experiment]:
        tests = [("iwl", 0), ("content_reversed", 0), ("cot_reversed", 1024), ("cot_icl", 1024)]
        for cond in decl["headline_conditions"]:
            for test, b in tests:
                k, n = count_of(trials, dim, test, cond, b)
                out.append(dict(panel="E", test=test, budget=b, condition=cond, source=source,
                                value=acc(k, n), n=n, se=se(acc(k, n), n)))
    return out


def compare_figure(runs: Path, decl: dict, dim: str, ours: list[dict]) -> dict:
    """Every value of the experiment's panels in the stored ``fig_app_representation_data.csv``
    against :func:`panel_numbers`: B, C and E are accuracies k/n and D differences (k1 - k0)/n,
    each required to be the same count over the same n (the stored script's arithmetic leaves some
    values one binary digit off Python's k/n, which is recorded); A within PROBE_TOLERANCE."""
    path = Path(runs) / decl["published_as"]["figure_data"].format(dim=dim, model=decl["model"])
    if not path.is_file():
        return {"status": NOT_PRESENT, "mismatches": []}
    fig = read_csv(path)
    experiment = decl["name"]
    source = SOURCE_OF[experiment]
    counts, mismatches, se_err = Counter(), [], 0.0
    float_diff = {"points_not_bit_identical": 0, "max_abs_float_difference": 0.0}

    def key_of(r: dict) -> tuple:
        p = r["panel"]
        if p == "A":
            return ("A", r["layer"])
        if p == "B":
            return ("B", r["condition"], r["d"])
        if p == "C":
            return ("C", r["series"], r["condition"])
        if p == "D":
            return ("D", r["condition"])
        return ("E", r["test"], r["budget"], r["condition"], r["source"])

    mine = {key_of(r): r for r in ours}

    def check(panel: str, label: str, stored: list, ours_v: list, n: int) -> None:
        counts[panel] += 1
        ok = all(stored_count(h, n) is not None and stored_count(h, n) == stored_count(o, n)
                 for h, o in zip(stored, ours_v, strict=True))
        diff = max(abs(h - o) for h, o in zip(stored, ours_v, strict=True))
        if diff:
            float_diff["points_not_bit_identical"] += 1
            float_diff["max_abs_float_difference"] = max(float_diff["max_abs_float_difference"], diff)
        if not ok:
            mismatches.append({"panel": panel, "point": label, "n": n, "stored": stored, "ours": ours_v})

    for row in fig.itertuples(index=False):
        p = row.panel
        if p not in PANELS_OF[experiment] or (p == "E" and row.source != source):
            counts["not_this_experiment"] += 1
            continue
        if p == "A":
            if row.quantity != dim:
                counts["other_dimension"] += 1
                continue
            r = mine.get(("A", int(row.layer)))
            if r is None:
                fail(f"{dim} probe accuracy: no probe row for layer {row.layer}")
            counts["A"] += 1
            diff = abs(row.value - r["value"])
            if diff:
                float_diff["points_not_bit_identical"] += 1
                float_diff["max_abs_float_difference"] = max(float_diff["max_abs_float_difference"], diff)
            if diff > PROBE_TOLERANCE or int(row.n) != r["n"]:
                mismatches.append({"panel": "A", "point": f"layer {int(row.layer)}",
                                   "stored": [row.value, int(row.n)], "ours": [r["value"], r["n"]]})
            continue
        if p == "D":
            r = mine.get(("D", row.condition))
            if r is None:
                counts["set_not_exported"] += 1
                continue
            if int(row.n_direct) != r["n_direct"] or int(row.n_chain) != r["n_chain"]:
                fail(f"{dim} accuracy change {row.condition}: n differs ({row.n_direct}, {row.n_chain} "
                     f"vs {r['n_direct']}, {r['n_chain']})")
            check("D", f"iwl {row.condition} - intact, budget 0", [row.d_iwl_direct], [r["d_iwl_direct"]], r["n_direct"])
            check("D", f"cot_icl {row.condition} - intact, budget 1024", [row.d_icl_chain], [r["d_icl_chain"]], r["n_chain"])
            continue
        if p == "B":
            r = mine[("B", row.condition, int(row.d))]
            label = f"iwl {row.condition} d={int(row.d)}"
        elif p == "C":
            if str(row.eta) != PANEL_C_ETA[row.condition]:
                fail(f"{dim} write strength: eta {row.eta!r} for {row.condition}")
            r = mine[("C", row.series, row.condition)]
            label = f"{PANEL_C_SERIES[row.series][0]} {row.condition} budget {PANEL_C_SERIES[row.series][1]} agree_written"
        else:
            r = mine.get(("E", row.test, int(row.budget), row.condition, row.source))
            if r is None:
                counts["trial_set_not_exported"] += 1
                continue
            label = f"{row.test} {row.condition} budget {int(row.budget)}"
        if int(row.n) != r["n"]:
            mismatches.append({"panel": p, "point": label, "stored_n": int(row.n), "ours_n": r["n"]})
            counts[p] += 1
            continue
        check(p, label, [row.value], [r["value"]], r["n"])
        se_err = max(se_err, abs(r["se"] - row.se))
    if se_err > SE_TOLERANCE:
        fail(f"{dim}: a standard error differs from sqrt(p(1-p)/n) by {se_err:g}")
    return {"values_by_panel": dict(sorted(counts.items())), **float_diff,
            "max_abs_se_error": float(se_err), "mismatches": mismatches}


def relative_to_runs(value, tree: str):
    """A stored path (``<anything>/<tree>/<path>``) as ``<tree>/<path>``, relative to the runs
    root whatever its local name; other values unchanged, through nested dicts and lists."""
    if isinstance(value, dict):
        return {k: relative_to_runs(v, tree) for k, v in value.items()}
    if isinstance(value, list):
        return [relative_to_runs(v, tree) for v in value]
    if isinstance(value, str) and "/" in value:
        parts = value.split("/")
        if tree in parts[:-1]:
            return "/".join(parts[parts.index(tree):])
    return value


def job_records(runs: Path, decl: dict, dim: str) -> dict:
    """The run manifest of the intervention directory (its paths relative to the runs root)
    and every event of its history file."""
    d = trial_dir(runs, decl, dim)
    manifest = relative_to_runs(json.loads(need(d / "run_manifest.json").read_text(encoding="utf-8")),
                                decl["published_as"]["run"].split("/")[0])
    history = [json.loads(line) for line in need(d / "run_manifest_history.jsonl").read_text(encoding="utf-8").splitlines()
               if line.strip()]

    def slim(m: dict) -> dict:
        out = {k: v for k, v in m.items() if k != "layers"}
        if "layers" in m:
            layers = m["layers"]
            if layers != list(range(len(layers))):
                fail(f"{d}: edited layers are not 0..{len(layers) - 1}")
            out["layers"] = f"0-{len(layers) - 1} ({len(layers)} layers)"
        return out

    jobs: dict = {}
    for event in history:
        job = event.get("job", "not recorded")
        j = jobs.setdefault(job, {"job": job, "tests": event["arms"], "conditions": event["conditions"],
                                  "events": []})
        j["events"].append({k: event[k] for k in ("event", "started", "finished") if k in event})
    return {"run_manifest_json": slim(manifest), "history_jobs": list(jobs.values())}


# --- writing ---------------------------------------------------------------------------


def trial_set_digests(df: pd.DataFrame) -> dict[str, list]:
    """``"dim|test|condition" -> [rows, sha256]`` of every trial set, each hashed over its own
    rows."""
    out = {}
    for (dim, test, cond), g in df.groupby(["dim", "test", "condition"], observed=True, sort=True):
        out[f"{dim}|{test}|{cond}"] = tables.Flow([int(len(g)), tables.row_digest(g)])
    return out


def manifest_header(runs: Path, decl: dict, checks: dict, jobs: dict, comparisons: dict) -> dict:
    experiment = decl["name"]
    return {
        "experiment": experiment,
        "exported": date.today().isoformat(),
        "runs_root": tables.RUNS_ROOT_PLACEHOLDER,
        "runs_layout": decl["published_as"],
        "model": decl["model"],
        "models": [decl["model"]],
        "instruction": decl["instruction"],
        "dimensions": {q: v["name"] for q, v in decl["dimensions"].items()},
        "dims": {dim: {"dimension": q["name"], "item_set": q["item_set"],
                       "iwl_sets": list(checks["iwl_sets"][dim])} for dim, q in decl["dimensions"].items()},
        "tests": tests_of(decl),
        "conditions": decl["conditions"],
        "condition_glossary": CONDITION_GLOSSARY[experiment],
        "budgets_kept": list(decl["budgets_kept"]),
        "budgets_dropped": [b for b in CHAIN_BUDGETS if b not in decl["budgets_kept"]],
        "innermost_ranks": (f"rank_low >= {DROP_LO} and rank_high <= {DROP_HI} (innermost "
                            f"{INNERMOST_RANKS} of {N_ITEMS}; distances 1-{INNERMOST_RANKS - 1})"),
        "row_order": "dim (as declared), test (as declared), condition (as declared), budget, pool, premise_order, d, trial_id",
        "trial_key": "(dim, test, condition, budget, pool, d, trial_id) is unique",
        "text_columns_dropped": ["prompt", "trace_text", "items", "A", "B", "cell_id", "lp_yes", "lp_no",
                                 "predicted", "correct", "world_truth", "written_truth", "stored_predicted",
                                 "layers", "n_edits", "trace_n_tokens", "cut", "stage2_source"],
        "record_guards": [
            "test and condition equal the file name's; assignment and item set fit the test",
            "d == rank_high - rank_low; direct tests at budget 0, reasoning tests at 256/512/1024",
            f"{INTERVENTION_COLUMN[experiment]} as the condition names it",
            "is_correct, agree_written and agree_world equal the agreement of predicted with "
            "correct, written_truth and world_truth (agree_world null exactly for the given names)",
            "no (pool, d, trial_id, budget) repeated in a file",
            "every condition of a test scores the same trials; every reasoning trial has its three readouts",
        ],
        "checks": checks,
        "jobs": jobs,
        "reproduces_stored_summaries": comparisons,
    }


def export_interp(runs: Path, out_dir: Path, experiment: str,
                  experiments_dir: str | Path = DEFAULT_EXPERIMENTS_DIR) -> dict:
    """Write ``trials.parquet`` and ``probe.parquet`` (representation) or ``heads.parquet``
    (components) with ``MANIFEST.yaml`` under ``out_dir``; returns the manifest."""
    runs = Path(runs); out_dir = Path(out_dir)
    decl = load_declaration(experiment, experiments_dir)
    raw_inner, trials, checks = read_trials(runs, decl)
    probe, fits, heads, heads_meta = None, None, None, None
    if experiment == "representation":
        probe, fits = probe_table(runs, decl)
    else:
        heads, heads_meta = heads_table(runs, decl)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {"trials": tables.write_parquet(trials, out_dir / "trials.parquet")}
    if probe is not None:
        written["probe"] = tables.write_parquet(probe, out_dir / "probe.parquet")
    if heads is not None:
        written["heads"] = tables.write_parquet(heads, out_dir / "heads.parquet")
    back = pd.read_parquet(out_dir / "trials.parquet")
    for c in back.columns:
        if TRIAL_DTYPES[c] != "string" and str(back[c].dtype) != TRIAL_DTYPES[c]:
            fail(f"trials.parquet: column {c} reads back as {back[c].dtype}, declared {TRIAL_DTYPES[c]}")
    probe_back = pd.read_parquet(out_dir / "probe.parquet") if probe is not None else None
    numbers, comparisons, jobs = {}, {}, {}
    for dim in decl["dimensions"]:
        numbers[dim] = panel_numbers(back, probe_back, decl, dim)
        comparisons[f"{dim}|summary_csv"] = compare_summary(runs, decl, dim, back, raw_inner)
        comparisons[f"{dim}|figure_data_csv"] = compare_figure(runs, decl, dim, numbers[dim])
        jobs[dim] = job_records(runs, decl, dim)
    bad = sum(len(v["mismatches"]) for v in comparisons.values())
    if bad:
        print(yaml.safe_dump(comparisons, sort_keys=False, width=100))
        fail(f"{bad} stored numbers are not reproduced (listed above)")
    comparisons = {"rules": {
        "summary_csv": ("acc = k/n over the innermost trials of (test, condition, budget[, d]), nulls dropped; "
                        "exact float equality of acc and n with every covered trial set; lo and hi equal the Wilson "
                        f"interval at z = {WILSON_Z} within {WILSON_TOLERANCE:g}; budgets 256 and 512 are "
                        "checked against the raw records"),
        "figure_data_csv": ("B, C, E: the stored value is k/n with the recomputed k and n; D: the stored "
                            "difference is (k1 - k0)/n with the recomputed counts; A: equal to the probe "
                            f"table within {PROBE_TOLERANCE:g}; standard errors sqrt(p(1-p)/n) within "
                            f"{SE_TOLERANCE:g}"),
    }, **comparisons}
    entries = [tables.table_entry(
        "trials", trials, written["trials"],
        note=("one row per scored trial read-out: innermost 7 ranks, budgets "
              f"{list(decl['budgets_kept'])}; is_correct = the answer equals the gold answer in the ordering "
              "shown, agree_world = equals the truth in the real ordering (null for the given names), "
              "agree_written = equals the truth in the written ordering a write moves toward"),
        column_definitions=COLUMN_DEFINITIONS[experiment],
        trial_sets=trial_set_digests(trials),
        panel_numbers=numbers,
    )]
    if probe is not None:
        entries.append(tables.table_entry(
            "probe", probe, written["probe"],
            note="probe_by_layer.csv of every dimension stacked: held-out pairwise ordering accuracy of the "
                 "axis coordinate (acc_axis_heldout) and of a ridge probe (acc_ridge_heldout), 5-fold over "
                 "items, per hidden state (layer 0 = token embeddings); alpha/beta the OLS of the coordinate "
                 "on the standardized rank; n_items / n_occurrences from fit_manifest.json",
            fit_manifests=fits,
        ))
    if heads is not None:
        entries.append(tables.table_entry(
            "heads", heads, written["heads"],
            note="every head set of selected_heads.json: top<k> = the k highest-scoring heads in rank "
                 "order; rand<k>_s<seed> = k random heads outside the top 64 (seed 100 k + seed), in rank "
                 "order; layer and head 0-based",
            selected_heads=heads_meta,
        ))
    header = manifest_header(runs, decl, checks, jobs, comparisons)
    manifest = tables.update_manifest(out_dir / "MANIFEST.yaml", header, entries)
    print(f"read {checks['trial_files_read']} trial files ({checks['records_read']:,} records)")
    for t in manifest["tables"]:
        print(f"wrote {out_dir / t['path']} ({t['rows']:,} rows)")
    for key, v in comparisons.items():
        if key == "rules":
            continue
        print(f"  {key}: " + ", ".join(f"{k}={v[k]}" for k in v if k != "mismatches") + f"; mismatches {len(v['mismatches'])}")
    return manifest


# --- reading the committed tables -------------------------------------------------------


def read_trials_table(experiment: str, results_dir: Path = tables.DEFAULT_RESULTS_DIR) -> pd.DataFrame:
    return tables.read_table(experiment, "trials", results_dir)


def read_probe(results_dir: Path = tables.DEFAULT_RESULTS_DIR) -> pd.DataFrame:
    return tables.read_table("representation", "probe", results_dir)


def read_heads(results_dir: Path = tables.DEFAULT_RESULTS_DIR) -> pd.DataFrame:
    return tables.read_table("components", "heads", results_dir)


__all__ = ["EXPERIMENTS", "load_declaration", "relative_to_runs", "export_interp", "panel_numbers", "count_of", "wilson", "se",
           "read_trials_table", "read_probe", "read_heads"]
