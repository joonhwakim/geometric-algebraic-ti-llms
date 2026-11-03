"""Tests for TrialResult and the JSONL files."""

from __future__ import annotations

import json

from geo_alg_ti_llms.results import TrialResult, read_jsonl, read_results, write_jsonl, write_results


def _direct() -> TrialResult:
    return TrialResult(
        trial_id=0, cell_id="x", master_seed=42,
        N=11, item_pool="x", assignment_strategy="random",
        signed_distance=2, symbolic_distance=2, correct=True,
        model="m", prompt_spec={"instructions_v1_id": "iwl_population_T2_ctxonly"},
        mode="direct",
        predicted=True, is_correct=True,
        logprob_yes=-0.1, logprob_no=-2.0, logit_gap=1.9,
        prompt="Base your answer only on the comparisons given — not on outside knowledge.",
        extra={"rank_to_item": ["a", "b"]},
    )


def test_trial_result_round_trip():
    r = _direct()
    d = r.to_dict()
    json.dumps(d)
    assert TrialResult.from_dict(d) == r


def test_trial_result_field_names_are_the_published_ones():
    assert list(_direct().to_dict()) == [
        "trial_id", "cell_id", "master_seed", "N", "item_pool", "assignment_strategy",
        "signed_distance", "symbolic_distance", "correct", "model", "prompt_spec", "mode",
        "predicted", "is_correct", "logprob_yes", "logprob_no", "logit_gap",
        "cot_trace", "cot_n_tokens", "cot_finish_reason", "prompt", "prompt_n_tokens",
        "cot_stage1_prompt", "cot_stage1_n_tokens", "timestamp_iso", "duration_seconds", "extra",
    ]


def test_trial_result_cot_fields_optional():
    r = TrialResult(
        trial_id=1, cell_id="x", master_seed=0,
        N=3, item_pool="x", assignment_strategy="random",
        signed_distance=1, symbolic_distance=1, correct=False,
        model="m", prompt_spec={"trigger_id": "passive"},
        mode="cot",
        predicted=False, is_correct=True,
        logprob_yes=-2.0, logprob_no=-0.1, logit_gap=-1.9,
        cot_trace="some thinking", cot_n_tokens=42, cot_finish_reason="eos",
    )
    assert TrialResult.from_dict(r.to_dict()) == r


def test_jsonl_round_trip_keeps_utf8(tmp_path):
    p = tmp_path / "sub" / "rows.jsonl"
    n = write_results(p, [_direct()])
    assert n == 1
    raw = p.read_text(encoding="utf-8")
    assert raw.endswith("\n") and raw.count("\n") == 1
    assert "—" in raw and "\\u2014" not in raw        # ensure_ascii=False, as the published files
    assert raw.startswith('{"trial_id": 0, "cell_id": "x", ')
    assert list(read_results(p)) == [_direct()]


def test_read_jsonl_skips_blank_lines(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")
    assert list(read_jsonl(p)) == [{"a": 1}, {"a": 2}]
    assert write_jsonl(p, []) == 0
