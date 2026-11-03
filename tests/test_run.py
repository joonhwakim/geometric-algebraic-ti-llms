"""The runners end to end on a stub model: file names and row schema of the published runs."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest

import geo_alg_ti_llms.run as runmod
from geo_alg_ti_llms import experiments
from geo_alg_ti_llms.models import load_model_config, model_config_path
from geo_alg_ti_llms.results import read_jsonl
from geo_alg_ti_llms.run import run_direct, run_ladder
from geo_alg_ti_llms.stub import StubModel, StubTokenizer
from geo_alg_ti_llms.stub import generate as _stub_generate
from geo_alg_ti_llms.stub import score as _stub_score

QWEN3 = load_model_config(model_config_path("qwen3-32b"))
GEMMA = load_model_config(model_config_path("gemma-2-9b-it"))

# Key lists of the stored rows (content_ctxonly direct trial set; budget_reversed_ctxonly ladder
# trial set).
DIRECT_PROMPT_SPEC = ["template_id", "instructions_v1_id", "query_wrapper_id", "separator_repr",
                      "answer_family", "phase", "N", "hop", "surface_form_mode", "assignment_strategy"]
DIRECT_EXTRA = ["phase", "N", "hop", "permutation_idx", "rank_high", "rank_low", "surface_form_mode",
                "assignment_strategy", "answer_cue", "actual_items", "rank_to_item"]
LADDER_NONE_PROMPT_SPEC = ["cot_cell_id", "template_id", "instructions_v1_id", "query_wrapper_id",
                           "separator_repr", "answer_family", "phase", "context_N", "hop",
                           "surface_form_mode", "assignment_strategy", "condition"]
LADDER_NONE_EXTRA = ["condition", "context_N", "hop", "surface_form_mode", "assignment_strategy", "rank_to_item"]
LADDER_COT_PROMPT_SPEC = ["cot_cell_id", "template_id", "instructions_v1_id", "query_wrapper_id",
                          "separator_repr", "answer_family", "phase", "context_N", "hop",
                          "surface_form_mode", "assignment_strategy", "n_exemplars", "trigger_id",
                          "reproduce", "direction", "query_position", "novel_item_exemplars",
                          "exemplar_pool", "exemplar_N", "exemplar_hop", "max_new_tokens", "condition"]
LADDER_COT_EXTRA = ["condition", "context_N", "hop", "surface_form_mode", "assignment_strategy",
                    "exemplar_items", "rank_to_item", "ladder_once", "stage2_source", "stage2_text_faithful"]



@pytest.fixture
def stubbed(monkeypatch):
    monkeypatch.setattr(runmod, "load_model_and_tokenizer", lambda cfg: (StubModel(), StubTokenizer()))
    monkeypatch.setattr(runmod, "score_yes_no", _stub_score)
    monkeypatch.setattr(runmod, "score_yes_no_ids", lambda model, ids, cands: (-0.2, -1.2, len(ids)))
    monkeypatch.setattr(runmod, "generate_new_token_ids", _stub_generate)


def _small(geometry: dict, hops=(7, 8)) -> dict:
    g = json.loads(json.dumps(geometry))
    g["grid"]["hop_values"] = list(hops)
    return g


def test_run_direct_writes_the_published_files(stubbed, tmp_path):
    exp = experiments.load_experiment("content_effects")
    spec = experiments.resolve(exp, "countries", "content", QWEN3)
    assert run_direct(_small(spec.geometry), QWEN3, tmp_path) == 0
    names = sorted(p.name for p in tmp_path.glob("*.jsonl"))
    assert names == [
        "N=11__hop=7__surface=fixed_gt.jsonl", "N=11__hop=7__surface=fixed_gt__assign=natural.jsonl",
        "N=11__hop=7__surface=fixed_gt__assign=reversed.jsonl",
        "N=11__hop=8__surface=fixed_gt.jsonl", "N=11__hop=8__surface=fixed_gt__assign=natural.jsonl",
        "N=11__hop=8__surface=fixed_gt__assign=reversed.jsonl",
    ]
    rows = list(read_jsonl(tmp_path / "N=11__hop=7__surface=fixed_gt__assign=reversed.jsonl"))
    assert len(rows) == 2 * (11 - 2 - 7) * 10 == 40
    r = rows[0]
    assert list(r["prompt_spec"]) == DIRECT_PROMPT_SPEC
    assert list(r["extra"]) == DIRECT_EXTRA
    assert r["mode"] == "direct" and r["model"] == "Qwen/Qwen3-32B"
    assert r["cell_id"] == "phaseD|N=11|pool=countries_by_population_11|hop=7|surface=fixed_gt|assign=reversed"
    assert r["prompt_spec"]["instructions_v1_id"] == "iwl_population_T2_ctxonly"
    assert r["extra"]["answer_cue"] is None
    assert r["extra"]["rank_to_item"] == list(reversed(r["extra"]["actual_items"]))
    assert r["prompt"].endswith("<|assistant|>\n<think>\n\n</think>\n\n")      # thinking suppressed
    assert r["predicted"] == (r["logprob_yes"] > r["logprob_no"])
    assert r["is_correct"] == (r["predicted"] == r["correct"])
    assert math.isclose(r["logit_gap"], r["logprob_yes"] - r["logprob_no"])
    assert r["prompt_n_tokens"] == len(r["prompt"])


def test_run_direct_zero_context_shows_no_premises(stubbed, tmp_path):
    exp = experiments.load_experiment("symbolic_distance")
    spec = experiments.resolve(exp, "countries", "iwl", GEMMA)
    run_direct(_small(spec.geometry, hops=(8,)), GEMMA, tmp_path)
    rows = list(read_jsonl(tmp_path / "N=11__hop=8__surface=zero_context__assign=natural.jsonl"))
    assert len(rows) == 2
    assert ">" not in rows[0]["prompt"].split("<|user|>")[1].split("Is it true")[0]
    assert rows[0]["prompt"].endswith("<|assistant|>\n")        # no thinking toggle on Gemma


def test_run_direct_dry_run_loads_no_model(monkeypatch, tmp_path):
    def boom(cfg):
        raise AssertionError("model loaded")
    monkeypatch.setattr(runmod, "load_model_and_tokenizer", boom)
    exp = experiments.load_experiment("symbolic_distance")
    spec = experiments.resolve(exp, "countries", "icl", QWEN3)
    assert run_direct(spec.geometry, QWEN3, tmp_path, dry_run=True) == 0
    assert not list(tmp_path.glob("*.jsonl"))


@pytest.mark.parametrize("exp_name,test", [("content_effects", "content"), ("symbolic_distance", "icl"),
                                           ("symbolic_distance", "iwl")])
def test_run_direct_dry_run_projects_every_trial(exp_name, test, tmp_path, capsys):
    spec = experiments.resolve(experiments.load_experiment(exp_name), "countries", test, QWEN3)
    assert run_direct(spec.geometry, QWEN3, tmp_path, dry_run=True) == 0
    total = int(re.search(r"projected total trials: (\d+)", capsys.readouterr().out).group(1))
    assert total == sum(len(trials) for *_, trials in runmod.direct_trial_sets(spec.geometry))


def test_run_ladder_writes_one_file_per_budget(stubbed, tmp_path):
    exp = experiments.load_experiment("content_effects")
    spec = experiments.resolve(exp, "countries", "budget_reversed", GEMMA)
    assert run_ladder(_small(spec.geometry, hops=(8,)), spec.ladder, GEMMA, tmp_path,
                      gen_batch_size=16, budget_ids=["b32", "b1024"]) == 0
    names = sorted(p.name for p in tmp_path.glob("*.jsonl"))
    assert names == ["b1024__N=11__hop=8__surface=fixed_gt__assign=reversed.jsonl",
                     "b32__N=11__hop=8__surface=fixed_gt__assign=reversed.jsonl"]
    rows = list(read_jsonl(tmp_path / "b32__N=11__hop=8__surface=fixed_gt__assign=reversed.jsonl"))
    assert [r["mode"] for r in rows] == ["direct", "cot"] * 20   # 2*(11-2-8)*10 trials
    none, cot = rows[0], rows[1]
    assert list(none["prompt_spec"]) == LADDER_NONE_PROMPT_SPEC
    assert list(none["extra"]) == LADDER_NONE_EXTRA
    assert list(cot["prompt_spec"]) == LADDER_COT_PROMPT_SPEC
    assert list(cot["extra"]) == LADDER_COT_EXTRA
    assert none["trial_id"] == cot["trial_id"]
    assert none["prompt_spec"]["cot_cell_id"] == "b32" and none["prompt_spec"]["condition"] == "none"
    assert cot["prompt_spec"]["condition"] == "cot" and cot["prompt_spec"]["max_new_tokens"] == 32
    assert cot["prompt_spec"]["trigger_id"] == "passive" and cot["prompt_spec"]["n_exemplars"] == 0
    assert cot["prompt_spec"]["exemplar_pool"] == "countries_by_population_11"
    assert cot["cot_stage1_prompt"].endswith("Let's think step by step.<|end|>\n<|assistant|>\n")
    assert cot["cot_n_tokens"] == 32 and cot["cot_finish_reason"] == "length"
    assert cot["prompt"].endswith("\nTherefore the answer is:")
    assert cot["prompt"].startswith(cot["cot_stage1_prompt"])
    assert cot["extra"]["ladder_once"] == {"generated_at_budget": 1024, "gen_batch_size": 16,
                                           "stage2": "text-if-faithful-else-ids", "eos_ids": [7]}
    assert cot["extra"]["stage2_source"] == "text" and cot["extra"]["stage2_text_faithful"] is True
    # the budget-1024 chain ran to its EOS
    rows1024 = list(read_jsonl(tmp_path / "b1024__N=11__hop=8__surface=fixed_gt__assign=reversed.jsonl"))
    assert rows1024[1]["cot_finish_reason"] == "eos"
    assert rows1024[1]["cot_trace"].endswith("x")


def test_run_ladder_native_thinking_model_leaves_the_turn_open(stubbed, tmp_path):
    exp = experiments.load_experiment("symbolic_distance")
    spec = experiments.resolve(exp, "countries", "budget_icl", QWEN3)
    run_ladder(_small(spec.geometry, hops=(8,)), spec.ladder, QWEN3, tmp_path, budget_ids=["b64"])
    rows = list(read_jsonl(tmp_path / "b64__N=11__hop=8__surface=fixed_gt.jsonl"))
    none, cot = rows[0], rows[1]
    assert cot["prompt_spec"]["trigger_id"] == "none"
    assert cot["cot_stage1_prompt"].endswith("<|assistant|>\n")
    assert none["prompt"].endswith("<|assistant|>\n<think>\n\n</think>\n\n")
    assert none["prompt_spec"]["instructions_v1_id"] == "iwl_height_T2"


def test_ladder_budgets_must_differ_only_in_max_new_tokens():
    with pytest.raises(ValueError, match="max_new_tokens alone"):
        runmod._ladder_budgets_compatible({"a": {"max_new_tokens": 1, "trigger_id": "none"},
                                         "b": {"max_new_tokens": 2, "trigger_id": "passive"}})
    assert runmod._ladder_budgets_compatible({"b64": {"max_new_tokens": 64}, "b32": {"max_new_tokens": 32}}) \
        == ["b32", "b64"]


def test_valid_grid_skips_infeasible_hops():
    kept, skipped = runmod._valid_grid([11], list(range(1, 11)), 11)
    assert [h for _, h in kept] == list(range(1, 9))
    assert skipped == [(11, 9), (11, 10)]


def test_score_yes_no_ids_on_a_stub_model():
    torch = pytest.importorskip("torch")
    from geo_alg_ti_llms.scoring import CandidateIds, score_yes_no_ids

    class Out:
        def __init__(self, logits):
            self.logits = logits

    class M:
        device = "cpu"

        def __call__(self, input_ids, attention_mask):
            logits = torch.zeros(1, input_ids.shape[1], 10)
            logits[0, -1, 1] = 2.0      # "True" id 1 favored
            return Out(logits)

    lp_yes, lp_no, n = score_yes_no_ids(M(), [5, 6, 7], CandidateIds(yes=(1,), no=(2,), yes_strs=(), no_strs=()))
    assert n == 3 and lp_yes > lp_no
    assert math.isclose(lp_yes - lp_no, 2.0, abs_tol=1e-5)
