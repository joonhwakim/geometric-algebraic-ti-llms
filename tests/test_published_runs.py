"""The declarations regenerate the published trial sets byte for byte.

Two layers. The fingerprint tests always run: a trial set is regenerated from its
experiment declaration and the sha256 over ``(trial_id, rank_to_item,
signed_distance, correct, messages)`` of every trial is compared with the value
recorded here, which was verified against the stored rows of the published
runs (trial order, ``extra.rank_to_item``, ``extra.actual_items``, rank pair,
gold, the system and user text of the prompt, and -- through the models'
chat templates -- the full ``prompt`` / ``cot_stage1_prompt`` strings).

The second layer re-does that comparison against the stored JSONL rows and
runs only where the published results tree is available
(``GEO_ALG_TI_LLMS_PUBLISHED_RUNS=<RUNS_ROOT>``). Its exact comparison of the
prompt strings renders them with each model's tokenizer at its pinned revision
and is skipped without the inference extra. The same tree gives the trials the
interventions score, which must be the trials of the committed tables.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
from collections import Counter
from pathlib import Path

import pytest

from geo_alg_ti_llms import experiments, interp, represent, run, tables
from geo_alg_ti_llms.models import load_model_config, load_tokenizer, model_config_path
from geo_alg_ti_llms.prompts import (
    INSTRUCTIONS_V1, TEMPLATES, build_grid_messages, build_reasoning_messages, render_grid_premise,
)
from geo_alg_ti_llms.results import read_jsonl
from geo_alg_ti_llms.scoring import apply_chat_template_cot, build_direct_prompt


def _direct_set(exp_name, key, test, suffix, model="qwen3-32b"):
    exp = experiments.load_experiment(exp_name)
    spec = experiments.resolve(exp, key, test, load_model_config(model_config_path(model)))
    trial_sets = {tag: (s, item_set, trials) for tag, s, item_set, trials in run.direct_trial_sets(spec.geometry)}
    return trial_sets[suffix]


def _ladder_set(exp_name, key, test, model, suffix):
    exp = experiments.load_experiment(exp_name)
    spec = experiments.resolve(exp, key, test, load_model_config(model_config_path(model)))
    trial_sets = {sf: (b, c, item_set, trials) for sf, b, c, item_set, trials in run.ladder_trial_sets(spec.geometry, spec.ladder)}
    return trial_sets[suffix]


def _fingerprint(trials, messages_of) -> str:
    h = hashlib.sha256()
    for t in trials:
        h.update(json.dumps([t.trial_id, list(t.rank_to_item), t.query.signed_distance,
                             t.query.correct, messages_of(t)], ensure_ascii=False).encode())
    return h.hexdigest()


DIRECT_FINGERPRINTS = [
    # (experiment, set key, test, trial set file stem, n trials, sha256)
    ("content_effects", "countries", "content", "N=11__hop=3__surface=fixed_gt", 120,
     "90254c1aeecb32adee3d7822753181e0d28966d9d7ea920f60cf55e5bf55d78d"),
    ("content_effects", "countries", "content", "N=11__hop=3__surface=fixed_gt__assign=reversed", 120,
     "5551c5ed96087f9f4560a4c9ee3107b3cd0d8f2e28335c8b779616b81c1db5b8"),
    ("symbolic_distance", "countries", "iwl", "N=11__hop=3__surface=zero_context__assign=natural", 12,
     "f1260c4c3978258936709a0bad5b1464936e2586f4d19b04f74e920e1c084832"),
    ("content_effects", "countries", "uninformative", "N=11__hop=3__surface=disjoint_context__assign=natural", 120,
     "14e9d4ee1c332701ed50a7eb2be37cbdda3f65588daf54ce8c0beac47259eb28"),
    ("symbolic_distance", "countries", "icl", "N=11__hop=3__surface=fixed_gt", 120,
     "6090aa624dd24299b59b1b32a040fc3f421fcdc7ab113ee062591e36cc1bc44e"),
    ("content_effects", "usstates", "content", "N=11__hop=1__surface=fixed_gt__assign=natural", 160,
     "cb0b98c9713255d71c7a3c34d5923508e8a4394073f1c721d91aea121c119665"),
    ("prior_strength", "countries_scale2", "content", "N=11__hop=2__surface=fixed_gt__assign=natural", 140,
     "098325f0fc849ad2525818f48534e027124fcbd70698963eda4efd5ea1ce133e"),
]
LADDER_FINGERPRINTS = [
    # (experiment, set key, test, model, trial set suffix, n trials, sha256)
    ("content_effects", "countries", "budget_reversed", "qwen3-32b",
     "N=11__hop=4__surface=fixed_gt__assign=reversed", 100,
     "e00e0d45d6ab4bf5e19d1c6d335659c2f5a4f400e76e1d72db80e6f761135a12"),
    ("content_effects", "countries", "budget_natural", "llama-3.1-8b-instruct",
     "N=11__hop=2__surface=fixed_gt__assign=natural", 140,
     "a9d824f680fe135379f7a44b80e612ce4004f5ccfdcb26bb175f4f1175f96a6b"),
    ("symbolic_distance", "countries", "budget_iwl", "qwen3-32b",
     "N=11__hop=3__surface=zero_context__assign=natural", 12,
     "f1260c4c3978258936709a0bad5b1464936e2586f4d19b04f74e920e1c084832"),
    ("symbolic_distance", "countries", "budget_iwl", "qwen2.5-32b-instruct",
     "N=11__hop=3__surface=zero_context__assign=natural", 12,
     "ea2c6514d04ad9c069869267331d13f6a32bf107111d7429ab2fe6c9cdc2ac79"),
]


@pytest.mark.parametrize("exp_name,key,test,suffix,n,digest", DIRECT_FINGERPRINTS)
def test_direct_trial_set_fingerprint(exp_name, key, test, suffix, n, digest):
    spec, item_set, trials = _direct_set(exp_name, key, test, suffix)
    assert len(trials) == n
    assert _fingerprint(trials, lambda t: build_grid_messages(t, spec)) == digest


@pytest.mark.parametrize("exp_name,key,test,model,suffix,n,digest", LADDER_FINGERPRINTS)
def test_ladder_trial_set_fingerprint(exp_name, key, test, model, suffix, n, digest):
    base, cot, item_set, trials = _ladder_set(exp_name, key, test, model, suffix)
    assert len(trials) == n
    assert _fingerprint(trials, lambda t: build_reasoning_messages(t, cot)) == digest


def test_direct_trial_set_files_match_the_published_names():
    exp = experiments.load_experiment("content_effects")
    spec = experiments.resolve(exp, "countries", "content")
    tags = [tag for tag, *_ in run.direct_trial_sets(spec.geometry)]
    assert tags[:3] == ["N=11__hop=1__surface=fixed_gt__assign=natural",
                        "N=11__hop=1__surface=fixed_gt__assign=reversed",
                        "N=11__hop=1__surface=fixed_gt"]
    assert len(tags) == 8 * 3                      # hops 1..8 are feasible at N=11
    exp = experiments.load_experiment("symbolic_distance")
    spec = experiments.resolve(exp, "countries", "budget_iwl",
                               load_model_config(model_config_path("qwen3-32b")))
    suffixes = [s for s, *_ in run.ladder_trial_sets(spec.geometry, spec.ladder)]
    assert suffixes[0] == "N=11__hop=1__surface=zero_context__assign=natural"
    assert [f"b{b}__{suffixes[0]}.jsonl" for b in (32, 1024)] == [
        "b32__N=11__hop=1__surface=zero_context__assign=natural.jsonl",
        "b1024__N=11__hop=1__surface=zero_context__assign=natural.jsonl"]


# --- against the stored rows (only where the published tree is present) ------

PUBLISHED = os.environ.get("GEO_ALG_TI_LLMS_PUBLISHED_RUNS")
PUBLISHED_TESTS = {
    "iwl": ("symbolic_distance", "iwl"), "icl": ("symbolic_distance", "icl"),
    "icl_cities": ("symbolic_distance", "icl_cities"),
    "budget_iwl": ("symbolic_distance", "budget_iwl"),
    "budget_icl": ("symbolic_distance", "budget_icl"),
    "budget_icl_cities": ("symbolic_distance", "budget_icl_cities"),
    "content_ctxonly": ("content_effects", "content"),
    "uninformative_ctxonly": ("content_effects", "uninformative"),
    "budget_natural_ctxonly": ("content_effects", "budget_natural"),
    "budget_reversed_ctxonly": ("content_effects", "budget_reversed"),
    "budget_shuffled_ctxonly": ("content_effects", "budget_shuffled"),
}
SPACING_TESTS = {"iwl": ("prior_strength", "iwl"), "content_ctxonly": ("prior_strength", "content")}
_SET_RE = re.compile(r"^(?:(?P<budget>b\d+)__)?(?P<suffix>N=\d+__hop=\d+__surface=\w+?(?:__assign=\w+)?)\.jsonl$")
STORED_FILES = [
    "nxhop/final_countries/content_ctxonly/qwen3-32b/N=11__hop=3__surface=fixed_gt.jsonl",
    "nxhop/final_countries/iwl/qwen3-32b/N=11__hop=2__surface=zero_context__assign=natural.jsonl",
    "nxhop/final_countries/uninformative_ctxonly/qwen3-32b/N=11__hop=6__surface=disjoint_context__assign=natural.jsonl",
    "nxhop/final_countries/icl/qwen3-32b/N=11__hop=5__surface=fixed_gt.jsonl",
    "nxhop/final_countries/icl_cities/qwen3-32b/N=11__hop=1__surface=fixed_gt.jsonl",
    "nxhop/final_usstates/content_ctxonly/gemma-2-9b-it/N=11__hop=3__surface=fixed_gt__assign=natural.jsonl",
    "nxhop/final_countries_scale2/content_ctxonly/qwen2.5-7b-instruct/N=11__hop=2__surface=fixed_gt__assign=natural.jsonl",
    "cot_sweep/final_countries/budget_reversed_ctxonly/qwen3-32b/b1024__N=11__hop=4__surface=fixed_gt__assign=reversed.jsonl",
    "cot_sweep/final_countries/budget_iwl/qwen3-32b/b32__N=11__hop=3__surface=zero_context__assign=natural.jsonl",
    "cot_sweep/final_countries/budget_iwl/qwen2.5-32b-instruct/b32__N=11__hop=3__surface=zero_context__assign=natural.jsonl",
]


def _locate(path: Path):
    model = path.parent.name
    test_dir = path.parent.parent.name
    key = path.parent.parent.parent.name[len("final_"):]
    exp_name, test = (SPACING_TESTS if "_scale" in key else PUBLISHED_TESTS)[test_dir]
    m = _SET_RE.match(path.name)
    return exp_name, test, key, model, m.group("suffix"), m.group("budget")


def _stored(rel):
    """One stored file's settings, regenerated trial set and stored rows (direct, reasoning)."""
    path = Path(PUBLISHED) / rel
    if not path.is_file():
        pytest.skip(f"{path} absent")
    exp_name, test, key, model, suffix, budget = _locate(path)
    cfg = load_model_config(model_config_path(model))
    exp = experiments.load_experiment(exp_name)
    spec = experiments.resolve(exp, key, test, cfg)
    if spec.runner == "direct":
        trial_sets = {tag: (s, item_set, trials) for tag, s, item_set, trials in run.direct_trial_sets(spec.geometry)}
        base, item_set, trials = trial_sets[suffix]
        cot = None
    else:
        trial_sets = {sf: (b, c, item_set, trials) for sf, b, c, item_set, trials in run.ladder_trial_sets(spec.geometry, spec.ladder)}
        base, cot, item_set, trials = trial_sets[suffix]
    rows = list(read_jsonl(path))
    direct_rows = [r for r in rows if r["mode"] == "direct"]
    cot_rows = [r for r in rows if r["mode"] == "cot"]
    assert len(direct_rows) == len(trials)
    assert not cot_rows or len(cot_rows) == len(trials)
    return cfg, budget, base, cot, item_set, trials, direct_rows, cot_rows


def _tokenizer(cfg):
    """The model's tokenizer at its pinned revision; the test is skipped when none loads."""
    if importlib.util.find_spec("transformers") is None:
        pytest.skip("needs the inference extra")
    try:
        return load_tokenizer(cfg.name, cfg.revision)[0]
    except OSError as error:            # not cached and not downloadable (offline or gated)
        pytest.skip(f"no tokenizer for {cfg.name} at revision {cfg.revision}: {error}")


@pytest.mark.skipif(not PUBLISHED, reason="GEO_ALG_TI_LLMS_PUBLISHED_RUNS not set")
@pytest.mark.parametrize("rel", STORED_FILES)
def test_stored_rows_regenerate(rel):
    cfg, budget, base, cot, item_set, trials, direct_rows, cot_rows = _stored(rel)
    for r, t in zip(direct_rows, trials):
        assert (r["trial_id"], r["cell_id"], r["master_seed"]) == (t.trial_id, t.cell_id, t.master_seed)
        assert r["extra"]["rank_to_item"] == list(t.rank_to_item)
        assert (r["signed_distance"], r["correct"]) == (t.query.signed_distance, t.query.correct)
        if "actual_items" in r["extra"]:
            assert r["extra"]["actual_items"] == list(item_set.items)
            assert (r["extra"]["rank_high"], r["extra"]["rank_low"]) == (t.extra["rank_high"], t.extra["rank_low"])
        msgs = build_grid_messages(t, base)
        assert INSTRUCTIONS_V1[base.instructions_v1_id] in r["prompt"]
        assert msgs[-1]["content"] in r["prompt"]
    for r, t in zip(cot_rows, trials):
        msgs = build_reasoning_messages(t, cot)
        assert r["prompt_spec"]["trigger_id"] == cot.trigger_id
        assert r["prompt_spec"]["max_new_tokens"] == int(budget[1:])
        assert msgs[-1]["content"] in r["cot_stage1_prompt"]
        if t.extra and t.extra.get("surface_form_mode") == "zero_context":
            # the IWL test's reasoning pass shows none of the trial's hidden premises: with
            # the query (and trigger) taken out, no premise appears anywhere in the prompt
            shown = r["cot_stage1_prompt"].replace(msgs[-1]["content"], "")
            template = TEMPLATES[cot.base.template_id]
            assert not any(render_grid_premise(p, template) in shown for p in t.context_pairs)


@pytest.mark.skipif(not PUBLISHED, reason="GEO_ALG_TI_LLMS_PUBLISHED_RUNS not set")
@pytest.mark.parametrize("rel", STORED_FILES)
def test_stored_prompts_are_the_chat_template_renderings(rel):
    cfg, budget, base, cot, item_set, trials, direct_rows, cot_rows = _stored(rel)
    tok = _tokenizer(cfg)
    for r, t in zip(direct_rows, trials):
        assert r["prompt"] == build_direct_prompt(t, base, tok, cfg, cfg.chat_template)
    for r, t in zip(cot_rows, trials):
        assert r["cot_stage1_prompt"] == apply_chat_template_cot(
            tok, build_reasoning_messages(t, cot), add_generation_prompt=True,
            enable_thinking=(True if cfg.thinking_mode else None))


# --- the stored trials the interventions score (only where the published tree is present) ---

@pytest.mark.skipif(not PUBLISHED, reason="GEO_ALG_TI_LLMS_PUBLISHED_RUNS not set")
@pytest.mark.parametrize("dim", list(represent.DIM_TAG))
@pytest.mark.parametrize("test", ["cot_reversed", "cot_icl"])
def test_intervention_chain_trials_are_the_published_ones(test, dim):
    runs = Path(PUBLISHED)
    if not (runs / "nxhop" / f"final_{represent.DIM_TAG[dim][0]}").is_dir():
        pytest.skip(f"no runs of {dim}")
    trials = represent.load_trials(runs, dim, "qwen3-32b", test, n_orders=3)
    assert len(trials) == 126
    assert Counter(t["perm"] for t in trials) == {0: 42, 1: 42, 2: 42}
    keys = sorted((t["d"], t["perm"], t["trial_id"], t["rank_low"], t["rank_high"]) for t in trials)
    for experiment in interp.EXPERIMENTS:
        table = tables.read_table(experiment, "trials")
        rows = table[(table["dim"] == dim) & (table["test"] == test) & (table["condition"] == "intact")]
        stored = zip(rows["d"], rows["premise_order"], rows["trial_id"], rows["rank_low"], rows["rank_high"])
        assert keys == sorted(tuple(int(v) for v in k) for k in stored)
