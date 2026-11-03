"""The tidy-table export: its row checks on synthetic rows, the committed manifests, and
(where the published runs are present) a small re-export against the committed trial sets."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from geo_alg_ti_llms import experiments, interp, tables, traces
from geo_alg_ti_llms.tables import (
    DROP_HI,
    DROP_LO,
    _budget_of,
    _query_ranks,
    attach_twins,
    trial_set_digests,
    check_complete_ladders,
    check_unique_keys,
    collapse_ladder_direct,
    label_of,
    ladder_query_ranks,
    row_digest,
    twin_test,
)

_ROOT = Path(__file__).resolve().parent.parent
BUDGETS = (32, 64, 128)
ITEMS = ["Ant", "Bee", "Cat", "Dog", "Elk", "Fox", "Gnu", "Hen", "Ibex", "Jay", "Kiwi"]


def _row(trial_id, budget=0, mode="direct", rl=2, rh=5, is_correct=True, test="budget_x",
         cond="no_context", dim="countries", premise_order=0, gap=1.0):
    return dict(model="m", reasoning_kind="", test=test, condition=cond, item_pool="pool", dim=dim,
                d=rh - rl, signed_distance=rh - rl, rank_low=rl, rank_high=rh,
                innermost=bool(rl >= DROP_LO and rh <= DROP_HI), premise_order=premise_order,
                trial_id=trial_id, trial_key=f"{dim}:zero_context|natural|{rh - rl}|{trial_id}",
                mode=mode, budget=budget, gold_true=True, predicted_true=is_correct,
                is_correct=is_correct, logprob_true=-0.1, logprob_false=-2.0, logit_gap=gap,
                cot_n_tokens=float("nan"), cot_finish_reason=None, truncated=False,
                source_file=f"b{budget}__x.jsonl", source_line=trial_id)


def _ladder(trial_ids, flip=()):
    rows = []
    for b in BUDGETS:
        for t in trial_ids:
            rows.append(_row(t, budget=b, mode="direct", premise_order=-1,
                             is_correct=(b == BUDGETS[0]) if t in flip else True))
        for t in trial_ids:
            rows.append(_row(t, budget=b, mode="cot", premise_order=-1))
    return rows


# --- the stored-file and prompt parsers ---------------------------------------------------

def test_budget_of_a_trial_set_file_stem():
    assert _budget_of("b256__N=11__hop=3__surface=fixed_gt") == 256
    assert _budget_of("N=11__hop=3__surface=fixed_gt") is None


def test_query_ranks_reads_the_last_clause():
    prompt = "Fox > Elk\nIs it true or false that Cat > Gnu?"
    assert _query_ranks(prompt, ITEMS) == (2, 6)
    assert _query_ranks("no question here", ITEMS) is None
    assert _query_ranks("Is it true or false that Cat > Zebra?", ITEMS) is None


def test_ladder_query_ranks_reads_the_question_before_the_trace():
    reasoning_prompt = "<s>Is it true or false that Gnu > Cat?\nassistant\n"
    trace = "Let me restate: is it true or false that Gnu > Kiwi? ... Gnu > Hen > Cat."
    r = {"mode": "cot", "prompt": reasoning_prompt + trace + "\nTherefore the answer is:",
         "cot_stage1_prompt": reasoning_prompt, "extra": {"rank_to_item": ITEMS}}
    assert ladder_query_ranks(r, "x") == (2, 6)        # not the trace's (6, 10)
    assert _query_ranks(r["prompt"], ITEMS) == (6, 10)  # what a last-clause parser would read
    direct = {"mode": "direct", "prompt": reasoning_prompt, "extra": {"rank_to_item": ITEMS}}
    assert ladder_query_ranks(direct, "x") == (2, 6)
    with pytest.raises(AssertionError, match="expected one"):
        ladder_query_ranks({**direct, "prompt": reasoning_prompt + reasoning_prompt}, "x")
    with pytest.raises(AssertionError, match="not located before the trace"):
        ladder_query_ranks({**r, "cot_stage1_prompt": "Is it true or false that Gnu > Kiwi?"}, "x")


# --- the ladder collapse and the row checks ----------------------------------------------

def test_collapse_ladder_direct_keeps_one_budget_0_row_per_trial():
    rows = _ladder([1, 2, 3])
    out, note = collapse_ladder_direct(rows, "t", BUDGETS)
    assert note == {"n_trials": 3, "n_direct_disagreements": 0}
    direct = [r for r in out if r["mode"] == "direct"]
    assert [r["trial_id"] for r in direct] == [1, 2, 3]
    assert all(r["budget"] == 0 and r["source_file"] == f"b{BUDGETS[0]}__x.jsonl" for r in direct)
    assert sorted({r["budget"] for r in out if r["mode"] == "cot"}) == list(BUDGETS)
    assert len(out) == 3 + 3 * len(BUDGETS)


def test_collapse_ladder_direct_tolerates_rare_tie_flips_only():
    out, note = collapse_ladder_direct(_ladder(list(range(200)), flip=(7,)), "t", BUDGETS)
    assert note["n_direct_disagreements"] == 1
    with pytest.raises(AssertionError, match="disagree across budgets"):
        collapse_ladder_direct(_ladder([1, 2, 3], flip=(1,)), "t", BUDGETS)
    with pytest.raises(AssertionError, match="direct copies per trial"):
        collapse_ladder_direct(_ladder([1, 2]) + [_row(3, budget=32, premise_order=-1)], "t", BUDGETS)


def test_attach_twins_copies_the_premise_order_and_checks_the_pair():
    ladder, _ = collapse_ladder_direct(_ladder([1, 2]), "t", BUDGETS)
    direct = [_row(1, test="x", premise_order=4), _row(2, test="x", premise_order=5)]
    note = attach_twins(ladder, direct, "t")
    assert note["n_twins"] == 2 and note["max_abs_logit_gap_delta"] == 0.0
    assert note["n_ladder_rows_ranks_checked"] == len(ladder)
    assert {r["trial_id"]: r["premise_order"] for r in ladder} == {1: 4, 2: 5}
    with pytest.raises(AssertionError, match="has no direct twin"):
        attach_twins(ladder, direct[:1], "t")
    bad = [dict(r, rank_low=3, rank_high=6) if r["trial_id"] == 2 and r["budget"] == 64 else r for r in ladder]
    with pytest.raises(AssertionError, match="ladder ranks"):
        attach_twins(bad, direct, "t")
    with pytest.raises(AssertionError, match="differs from its direct twin"):
        attach_twins(ladder, [direct[0], dict(direct[1], logit_gap=1.5)], "t")
    with pytest.raises(AssertionError, match="no ladder counterpart"):
        attach_twins(ladder, direct + [_row(3, test="x")], "t")
    # a condition without a ladder is exempt
    note = attach_twins(ladder, direct + [_row(3, test="u", cond="unrelated_context")], "t")
    assert note["conditions_without_ladder"] == ["unrelated_context"]


def test_check_complete_ladders_and_unique_keys():
    ladder, _ = collapse_ladder_direct(_ladder([1, 2]), "t", BUDGETS)
    check_complete_ladders(ladder, "t", (0, *BUDGETS))
    with pytest.raises(AssertionError, match="incomplete ladder"):
        check_complete_ladders([r for r in ladder if not (r["trial_id"] == 2 and r["budget"] == 64)],
                               "t", (0, *BUDGETS))
    check_unique_keys(ladder, "t")
    with pytest.raises(AssertionError, match="duplicated"):
        check_unique_keys(ladder + [ladder[0]], "t")


def test_label_of():
    real = ITEMS
    shown = ["Cat", "Ant", "Gnu", "Bee", "Elk", "Dog", "Kiwi", "Fox", "Hen", "Ibex", "Jay"]
    assert label_of(shown, real, 1, 2) == ("agree", 6)        # Ant < Gnu in reality (6 apart)
    assert label_of(shown, real, 0, 1) == ("conflict", 2)     # Cat shown below Ant
    assert label_of(real, real, 3, 7) == ("agree", 4)
    assert label_of(real[::-1], real, 3, 7) == ("conflict", 4)


def test_row_digest_is_order_and_value_sensitive():
    df = pd.DataFrame([_row(1), _row(2)]).astype({"trial_key": "string"})
    assert row_digest(df) == row_digest(df.copy())
    assert row_digest(df) != row_digest(df.iloc[::-1])
    assert row_digest(df) != row_digest(df.assign(logit_gap=[1.0, 2.0]))
    trial_sets = trial_set_digests(df.assign(test=["a", "b"]))
    assert set(trial_sets) == {"m|a|countries", "m|b|countries"} and trial_sets["m|a|countries"][0] == 1


# --- the declarations ---------------------------------------------------------------------

def test_test_specs_and_twins_from_the_declarations():
    exp = experiments.load_experiment("content_effects")
    specs = tables.test_specs(exp)
    assert specs["content"].rows_per_set == 2160 and specs["uninformative"].rows_per_set == 720
    assert specs["content"].ctxonly and specs["content"].published_as == "content_ctxonly"
    assert {a: twin_test(specs, a) for a in ("budget_natural", "budget_reversed", "budget_shuffled")} == \
        {"budget_natural": "content", "budget_reversed": "content", "budget_shuffled": "content"}
    exp = experiments.load_experiment("symbolic_distance")
    specs = tables.test_specs(exp)
    assert specs["iwl"].rows_per_set == 72 and len(specs["iwl"].item_sets) == 14
    assert not specs["iwl"].ctxonly
    assert twin_test(specs, "budget_icl_cities") == "icl_cities" and twin_test(specs, "budget_iwl") == "iwl"
    assert sorted(tables.dims_of(exp)) == sorted(specs["iwl"].item_sets)


# --- the committed tables -----------------------------------------------------------------

# The trial tables of the declared experiments (the reasoning-trace tables, derived from two
# of their ladders, are checked in test_traces.py).
MANIFESTS = [_ROOT / "results" / e / "tables" / "MANIFEST.yaml" for e in experiments.list_experiments()]


@pytest.mark.parametrize("manifest", MANIFESTS, ids=[m.parent.parent.name for m in MANIFESTS])
def test_committed_tables_match_their_manifest(manifest):
    assert tables.verify_manifest(manifest.parent) == []
    m = tables.load_manifest(manifest)
    assert m["package_version"] == tables.PACKAGE_VERSION
    names = {t["table"] for t in m["tables"]}
    assert "direct" in names
    direct = next(t for t in m["tables"] if t["table"] == "direct")
    assert list(direct["columns"])[:len(tables.COLUMNS)] == tables.COLUMNS
    assert sum(n for n, _ in direct["trial_sets"].values()) == direct["rows"]
    # The per-directory notes of the export cover every table row exactly once.
    specs = tables.test_specs(experiments.load_experiment(m["experiment"]))
    rows = {"direct": 0, "ladders": 0}
    for label, note in m["checks"]["per_test_directories"].items():
        test = label.split("/")[2]
        rows["ladders" if specs[test].runner == "ladder" else "direct"] += note["rows"]
        assert note.get("n_direct_disagreements", 0) <= tables.DIRECT_DUPLICATE_TOLERANCE * note.get("n_trials", 0)
    assert {t["table"]: t["rows"] for t in m["tables"] if t["table"] in rows} == {k: n for k, n in rows.items() if n}


def test_results_directories_are_the_declared_experiments_plus_the_traces_and_interventions():
    found = sorted(p.parent.parent.name for p in (_ROOT / "results").glob("*/tables/MANIFEST.yaml"))
    assert found == sorted([*experiments.list_experiments(), traces.EXPERIMENT, *interp.EXPERIMENTS])


class _UniqueKeys(tables._ManifestLoader):
    """The manifest loader, refusing a key that appears twice in one mapping."""


def _unique_mapping(loader, node, deep=False):
    keys = [loader.construct_object(k, deep=deep) for k, _ in node.value]
    dupes = sorted({k for k in keys if keys.count(k) > 1})
    if dupes:
        raise ValueError(f"duplicate keys {dupes}")
    return loader.construct_mapping(node, deep=deep)


_UniqueKeys.add_constructor("tag:yaml.org,2002:map", _unique_mapping)
ALL_MANIFESTS = sorted((_ROOT / "results").glob("*/tables/MANIFEST.yaml"))


@pytest.mark.parametrize("manifest", ALL_MANIFESTS, ids=[m.parent.parent.name for m in ALL_MANIFESTS])
def test_manifests_have_no_repeated_keys_and_no_research_paths(manifest):
    text = manifest.read_text(encoding="utf-8")
    import yaml
    yaml.load(text, Loader=_UniqueKeys)
    assert "data/results" not in text


@pytest.mark.parametrize("manifest", MANIFESTS, ids=[m.parent.parent.name for m in MANIFESTS])
def test_manifest_header_is_what_the_export_writes(manifest):
    """The header of a trial manifest is the export's, from the declaration and the code, with
    no key a later rename left behind."""
    m = tables.load_manifest(manifest)
    exp = experiments.load_experiment(m["experiment"])
    header = tables._header(exp, tables.test_specs(exp), m["models"])
    assert {k: m[k] for k in header if k != "exported"} == {k: v for k, v in header.items() if k != "exported"}
    assert set(m) == {"package_version", *header, "checks", "tables"}
    assert m["checks"]["per_test_directories_note"] == tables.PER_TEST_NOTE


def test_pools_table_is_the_spacing_sets_of_the_declaration():
    exp = experiments.load_experiment("prior_strength")
    committed = tables.read_table("prior_strength", "pools")
    pd.testing.assert_frame_equal(committed, pd.DataFrame(tables.spacing_sets(exp)))
    m = tables.load_manifest(_ROOT / "results/prior_strength/tables/MANIFEST.yaml")
    entry = next(t for t in m["tables"] if t["table"] == "pools")
    assert entry["item_sets"] == tables.spacing_sets(exp) and entry["note"] == tables.TABLE_NOTES["pools"]


# --- the manifest keeps its layout across rewrites -------------------------------------------

def test_update_manifest_keeps_one_line_lists_when_another_table_is_added(tmp_path):
    path = tmp_path / "MANIFEST.yaml"
    first = [{"table": "a", "path": "a.parquet", "trial_sets": {"m|x|d": tables.Flow([2, "ab"])}}]
    tables.update_manifest(path, {"experiment": "e"}, first)
    text = path.read_text(encoding="utf-8")
    assert "    m|x|d: [2, ab]\n" in text
    tables.update_manifest(path, {"experiment": "e"}, [{"table": "b", "path": "b.parquet", "n": [1, 2]}])
    again = path.read_text(encoding="utf-8")
    assert "    m|x|d: [2, ab]\n" in again and "  n:\n  - 1\n  - 2\n" in again
    assert tables.load_manifest(path)["tables"][0]["trial_sets"]["m|x|d"] == [2, "ab"]


# --- the trial export on synthetic rows ---------------------------------------------------

def _direct_set(model="qwen3-32b", dim="countries"):
    """The 72 rows of one iwl trial set: both orders of every pair of the nine non-terminal
    ranks."""
    rows, t = [], 0
    for rl in range(1, 10):
        for rh in range(rl + 1, 10):
            for _ in range(2):
                rows.append({**_row(t, rl=rl, rh=rh, test="iwl", dim=dim), "model": model})
                t += 1
    return rows


def test_export_trials_records_the_per_test_directory_notes(monkeypatch, tmp_path):
    exp = experiments.load_experiment("symbolic_distance")
    notes = {"qwen3-32b/countries/iwl": {"rows": 72, "files": 8}}
    read = []

    def stub(runs, exp_, specs, model, tests, log=print, models_dir=None):
        read.append(model)
        return tables.to_frame(_direct_set(model)), tables.to_frame([]), {"tests": dict(notes), "twins": {}}

    monkeypatch.setattr(tables, "read_model", stub)
    out, kw = tmp_path / "tables", dict(models=["qwen3-32b"], tests=["iwl"], work_dir=tmp_path / "work",
                                        log=lambda *_: None)
    manifest = tables.export_trials(exp, tmp_path / "runs", out, **kw)
    assert manifest["checks"]["per_test_directories"] == notes
    assert tables.load_manifest(out / "MANIFEST.yaml")["checks"]["per_test_directories"] == notes
    assert tables.verify_manifest(out) == []
    assert pd.read_parquet(out / "direct.parquet").shape[0] == 72
    # A second pass serves the model from work_dir, notes included, without reading the runs.
    again = tables.export_trials(exp, tmp_path / "runs", out, **kw)
    assert read == ["qwen3-32b"] and again["checks"]["per_test_directories"] == notes
    assert tables.verify_manifest(out) == []


def test_export_trials_refuses_a_cache_read_for_other_tests_or_runs(monkeypatch, tmp_path):
    exp = experiments.load_experiment("symbolic_distance")

    def stub(runs, exp_, specs, model, tests, log=print, models_dir=None):
        return tables.to_frame(_direct_set(model)), tables.to_frame([]), {"tests": {}, "twins": {}}

    monkeypatch.setattr(tables, "read_model", stub)
    kw = dict(models=["qwen3-32b"], work_dir=tmp_path / "work", log=lambda *_: None)
    tables.export_trials(exp, tmp_path / "runs", tmp_path / "a", tests=["iwl"], **kw)
    tables.export_trials(exp, tmp_path / "runs", tmp_path / "b", tests=["iwl"], **kw)
    assert (tmp_path / "a" / "direct.parquet").read_bytes() == (tmp_path / "b" / "direct.parquet").read_bytes()
    for runs, tests in ((tmp_path / "runs", ["iwl", "icl"]), (tmp_path / "other", ["iwl"])):
        with pytest.raises(ValueError, match="use another --work-dir"):
            tables.export_trials(exp, runs, tmp_path / "c", tests=tests, **kw)
    assert not (tmp_path / "c").exists()


# --- a small re-export against the committed trial sets (published runs only) --------------------

PUBLISHED = os.environ.get(tables.RUNS_ENV)


@pytest.mark.skipif(not PUBLISHED, reason=f"{tables.RUNS_ENV} not set")
def test_reexport_of_one_test_reproduces_the_committed_trial_sets(tmp_path):
    models = ["qwen3-32b", "llama-3.2-1b"]
    exp = experiments.load_experiment("symbolic_distance")
    manifest = tables.export_trials(exp, Path(PUBLISHED), tmp_path, models=models, tests=["iwl"],
                                    log=lambda *_: None)
    new = pd.read_parquet(tmp_path / "direct.parquet")
    assert len(new) == 2 * 14 * 72 and sorted(new["model"].unique()) == sorted(models)
    committed = tables.load_manifest(_ROOT / "results/symbolic_distance/tables/MANIFEST.yaml")
    trial_sets = next(t for t in committed["tables"] if t["table"] == "direct")["trial_sets"]
    mine = next(t for t in manifest["tables"] if t["table"] == "direct")["trial_sets"]
    assert mine == {k: v for k, v in trial_sets.items() if k.split("|")[0] in models and k.split("|")[1] == "iwl"}
    assert mine == trial_set_digests(new)
    full = tables.read_table("symbolic_distance", "direct")
    sub = full[full["model"].isin(models) & (full["test"] == "iwl")].reset_index(drop=True)
    pd.testing.assert_frame_equal(sub, new)
