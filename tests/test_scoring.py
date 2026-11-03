"""Tests for the scoring module that run without a model.

- candidate-id resolution (single-token forms, de-duplication)
- the chat-template helpers (date pin, system-turn merge, enable_thinking)
- ``build_direct_prompt`` plumbing for every model kind
- the answer-pass input rule, on a stub tokenizer
The logit math is covered when torch is installed (``importorskip``).
"""

from __future__ import annotations

import pytest

import geo_alg_ti_llms.scoring as sc
from geo_alg_ti_llms.scoring import (
    DEFAULT_FALSE_CANDIDATES,
    DEFAULT_TRUE_CANDIDATES,
    CandidateIds,
    apply_chat_template_cot,
    apply_chat_template_safe,
    build_direct_prompt,
    build_answer_pass_ids,
    eos_token_ids,
    merge_system_into_user,
    resolve_candidate_ids,
    resolve_family_candidates,
    resolve_forced_choice_candidates,
    answer_pass_input,
    strip_trace_ids,
    truncate_new_ids,
)


# --- candidate id resolution -------------------------------------------------


def test_resolve_candidates_keeps_only_single_token(mock_tokenizer):
    cands = resolve_candidate_ids(
        mock_tokenizer,
        yes_candidates=("True", " True", "two words"),  # last is multi-token in mock
        no_candidates=("False", " False"),
    )
    assert "two words" not in cands.yes_strs
    assert "True" in cands.yes_strs and " True" in cands.yes_strs
    for ids in (cands.yes, cands.no):
        for tid in ids:
            assert isinstance(tid, int)


def test_resolve_candidates_raises_when_no_singles(mock_tokenizer):
    with pytest.raises(ValueError, match="No single-token Yes"):
        resolve_candidate_ids(
            mock_tokenizer,
            yes_candidates=("two words", "another phrase"),
            no_candidates=("False",),
        )


def test_resolve_family_true_false(mock_tokenizer):
    cands = resolve_family_candidates("true_false", mock_tokenizer)
    assert cands.yes_strs == DEFAULT_TRUE_CANDIDATES
    assert cands.no_strs == DEFAULT_FALSE_CANDIDATES
    assert resolve_family_candidates("item_names", mock_tokenizer) is None
    with pytest.raises(ValueError, match="answer family"):
        resolve_family_candidates("gt_lt", mock_tokenizer)


def test_forced_choice_candidates(mock_tokenizer):
    a_ids, b_ids = resolve_forced_choice_candidates(mock_tokenizer, "Belgium", "two words")
    assert len(a_ids) == 2 and b_ids == ()


def test_candidate_ids_are_deduplicated_by_token_id():
    """Several surface forms can encode to ONE token id; scoring must not count it twice."""
    class CollidingTok:
        """Strips a leading space, so 'True' and ' True' share an id."""
        def encode(self, s, add_special_tokens=False):
            table = {"True": [1], "true": [2], "False": [3], "false": [4]}
            return table.get(s.strip(), [99])

    c = resolve_candidate_ids(CollidingTok(), yes_candidates=DEFAULT_TRUE_CANDIDATES,
                              no_candidates=DEFAULT_FALSE_CANDIDATES)
    assert c.yes == (1, 2) and c.no == (3, 4)
    assert len(c.yes_strs) == len(c.yes) and len(c.no_strs) == len(c.no)


def test_dedup_does_not_change_the_logit_gap():
    """The fix must not move any prediction: both families inflate identically."""
    torch = pytest.importorskip("torch")
    logits = torch.randn(50, generator=torch.Generator().manual_seed(0))
    dup = CandidateIds(yes=(1, 1, 2, 2), no=(3, 3, 4, 4), yes_strs=(), no_strs=())
    ded = CandidateIds(yes=(1, 2), no=(3, 4), yes_strs=(), no_strs=())
    y1, n1 = sc._yes_no_from_logits(logits, dup)
    y2, n2 = sc._yes_no_from_logits(logits, ded)
    assert abs((y1 - n1) - (y2 - n2)) < 1e-6
    assert y2 < 0 and n2 < 0
    assert abs((y1 - y2) - torch.log(torch.tensor(2.0)).item()) < 1e-5


# --- eos ids -----------------------------------------------------------------


def test_eos_token_ids_merges_generation_config_and_tokenizer():
    class GC:
        eos_token_id = [151645, 151643]

    class M:
        generation_config = GC()

    class T:
        eos_token_id = 151643

    assert eos_token_ids(M(), T()) == (151645, 151643)

    class GC1:
        eos_token_id = 2

    class M1:
        generation_config = GC1()

    class T1:
        eos_token_id = 7

    assert eos_token_ids(M1(), T1()) == (2, 7)


# --- chat-template helpers ---------------------------------------------------


class _ThinkingTokenizer:
    """Stub whose template accepts enable_thinking and records the call."""

    name_or_path = "stub/thinking"
    chat_template = "plain"

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt,
                            enable_thinking=False):
        self.calls.append({"enable_thinking": enable_thinking})
        return f"<think={enable_thinking}>"


class _LegacyTokenizer:
    """Stub whose template does NOT accept enable_thinking (raises TypeError)."""

    name_or_path = "stub/no_system_role"
    chat_template = "plain"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        return "<plain>"


def test_apply_chat_template_cot_enable_thinking_true():
    tok = _ThinkingTokenizer()
    assert apply_chat_template_cot(tok, [{"role": "user", "content": "x"}], enable_thinking=True) == "<think=True>"
    assert tok.calls[-1]["enable_thinking"] is True


def test_apply_chat_template_cot_enable_thinking_false_suppresses():
    """Explicit False must be passed through (Qwen3 direct path)."""
    tok = _ThinkingTokenizer()
    assert apply_chat_template_cot(tok, [{"role": "user", "content": "x"}], enable_thinking=False) == "<think=False>"
    assert tok.calls[-1]["enable_thinking"] is False


def test_apply_chat_template_cot_none_omits_flag():
    assert apply_chat_template_cot(_LegacyTokenizer(), [{"role": "user", "content": "x"}],
                                   enable_thinking=None) == "<plain>"


def test_apply_chat_template_cot_folds_the_system_turn_when_the_template_rejects_it():
    assert apply_chat_template_cot(_LegacyTokenizer(), [{"role": "user", "content": "x"}],
                                   enable_thinking=True) == "<plain>"


def test_merge_system_into_user():
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    assert merge_system_into_user(msgs) == [{"role": "user", "content": "S\n\nU"}]
    assert merge_system_into_user([{"role": "user", "content": "U"}]) == [{"role": "user", "content": "U"}]
    with pytest.raises(ValueError, match="user turn"):
        merge_system_into_user([{"role": "system", "content": "S"}])


class _RejectsSystem:
    """Like Gemma-2: raises on a system role."""
    name_or_path = "fake/rejects-system"
    chat_template = "x"

    def apply_chat_template(self, messages, **kw):
        if messages[0]["role"] == "system":
            raise ValueError("System role not supported")
        return "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)


class _DropsSystem:
    """Like Phi-4-reasoning: accepts a system role and replaces it with its own text."""
    name_or_path = "fake/drops-system"
    chat_template = "x"

    def apply_chat_template(self, messages, **kw):
        body = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages if m["role"] != "system")
        return "<system>You are a fixed assistant.</system>" + body


def test_rejected_system_is_merged_into_user():
    msgs = [{"role": "system", "content": "In the following question, > means bigger."},
            {"role": "user", "content": "Is it true that A > B?"}]
    out = apply_chat_template_safe(_RejectsSystem(), msgs)
    assert out == "<user>In the following question, > means bigger.\n\nIs it true that A > B?</user>"


def test_dropped_system_is_merged_into_user():
    msgs = [{"role": "system", "content": "In the following question, > means bigger."},
            {"role": "user", "content": "Is it true that A > B?"}]
    out = apply_chat_template_safe(_DropsSystem(), msgs)
    assert "In the following question, > means bigger." in out
    assert out.count("Is it true that A > B?") == 1


def test_date_is_pinned_for_templates_that_stamp_it():
    class Llama:
        name_or_path = "fake/llama"
        chat_template = "{{ date_string }}"
        seen = None

        def apply_chat_template(self, messages, **kw):
            self.seen = kw
            return "x"

    class GptOss:
        name_or_path = "fake/gpt-oss"
        chat_template = "{{ strftime_now('%Y-%m-%d') }}"
        seen = None

        def apply_chat_template(self, messages, **kw):
            self.seen = kw
            return "x"

    ll = Llama()
    apply_chat_template_safe(ll, [{"role": "user", "content": "x"}], tokenize=False)
    assert ll.seen["date_string"] == sc.PINNED_DATE_STRING == "26 Jul 2024"
    oss = GptOss()
    apply_chat_template_safe(oss, [{"role": "user", "content": "x"}], tokenize=False)
    assert oss.seen["strftime_now"]("%Y-%m-%d") == sc.PINNED_STRFTIME_DATE == "2026-09-13"


# --- build_direct_prompt: the scored position for every model kind ------------


class _Cfg:
    def __init__(self, suffix=None, thinking=False, suppress=None):
        self.chat_generation_suffix = suffix
        self.thinking_mode = thinking
        self.thinking_suppress_prefix = suppress


def test_direct_prompt_suppresses_thinking_for_hybrid_reasoning_models(monkeypatch):
    """A hybrid reasoning model (Qwen3) defaults thinking ON, so the direct pass
    must pass enable_thinking=False or the scored token is the start of a trace."""
    class Tok:
        name_or_path = "stub"
        chat_template = "plain"
        seen = None

        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kw):
            self.seen = kw.get("enable_thinking", "ABSENT")
            return "PROMPT"

    monkeypatch.setattr(sc, "build_grid_messages", lambda t, s: [{"role": "user", "content": "x"}])
    for thinking, expected in [(True, False), (False, "ABSENT")]:
        tok = Tok()
        build_direct_prompt(object(), object(), tok, _Cfg(thinking=thinking), True)
        assert tok.seen == expected


def test_direct_prompt_prefills_closed_think_for_toggleless_reasoning_models(monkeypatch):
    """DeepSeek-R1-distill / Phi-4 / gpt-oss have no toggle; their configs carry a
    suppression prefix that is prefilled as assistant content, after GLM's suffix."""
    class Tok:
        name_or_path = "stub"
        chat_template = "plain"

        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kw):
            if kw:
                raise TypeError("template does not accept enable_thinking")
            return "<|user|>Q<|assistant|>"

    monkeypatch.setattr(sc, "build_grid_messages", lambda t, s: [{"role": "user", "content": "Q"}])
    assert build_direct_prompt(object(), object(), Tok(), _Cfg(thinking=True, suppress="<think>\n\n</think>\n\n"), True) \
        == "<|user|>Q<|assistant|><think>\n\n</think>\n\n"
    assert build_direct_prompt(object(), object(), Tok(), _Cfg(), True) == "<|user|>Q<|assistant|>"
    assert build_direct_prompt(object(), object(), Tok(), _Cfg(suffix="\n"), True) == "<|user|>Q<|assistant|>\n"
    assert build_direct_prompt(object(), object(), Tok(), _Cfg(suffix="\n", thinking=True, suppress="<t></t>"), True) \
        == "<|user|>Q<|assistant|>\n<t></t>"


def test_direct_prompt_raw_text_without_chat_template(monkeypatch):
    monkeypatch.setattr(sc, "build_grid_user_text", lambda t, s: "RAW")
    assert build_direct_prompt(object(), object(), object(), _Cfg(), False) == "RAW"


# --- answer-pass input rule ------------------------------------------------------


class _IdTok:
    """Byte-faithful stub: each character is one id; ' ' and '\\n' are whitespace tokens."""
    all_special_ids = [1000]

    def encode(self, s, add_special_tokens=False):
        return [ord(c) for c in s]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids if not (skip_special_tokens and i in self.all_special_ids))


def test_strip_trace_ids_drops_eos_specials_and_trailing_whitespace():
    tok = _IdTok()
    ids = [ord(c) for c in "a b \n"] + [1000, 7]
    assert strip_trace_ids(tok, ids, eos_ids=(7,)) == [ord("a"), ord(" "), ord("b")]


def test_stage2_text_path_when_faithful():
    tok = _IdTok()
    reasoning_prompt = "S1\n"
    s1_ids = tok.encode(reasoning_prompt)
    trace_ids = tok.encode("x > y\n") + [7]
    answer_text = reasoning_prompt + "x > y" + "\nTherefore the answer is:"
    ids, source, faithful = answer_pass_input(tok, reasoning_prompt, s1_ids, trace_ids, answer_text,
                                         eos_ids=(7,), cue="Therefore the answer is:")
    assert source == "text" and faithful
    assert ids == tok.encode(answer_text)


def test_stage2_ids_path_when_text_drifts():
    tok = _IdTok()
    reasoning_prompt = "S1\n"
    s1_ids = tok.encode(reasoning_prompt)
    trace_ids = tok.encode("x > y")
    drifted = reasoning_prompt + "X > y" + "\nTherefore the answer is:"   # the re-rendered text differs
    ids, source, faithful = answer_pass_input(tok, reasoning_prompt, s1_ids, trace_ids, drifted,
                                         eos_ids=(7,), cue="Therefore the answer is:")
    assert source == "ids" and not faithful
    assert ids == build_answer_pass_ids(tok, s1_ids, trace_ids, eos_ids=(7,), cue="Therefore the answer is:")
    assert ids == s1_ids + trace_ids + tok.encode("\nTherefore the answer is:")


def test_truncate_new_ids_finish_reasons():
    tok = _IdTok()
    full = tok.encode("abc</think>") + [7]
    trace, n, finish = truncate_new_ids(full, budget=2, thinking_mode=True, tokenizer=tok, eos_ids=(7,))
    assert (trace, n, finish) == ("ab", 2, "length")
    trace, n, finish = truncate_new_ids(full[:-1], budget=100, thinking_mode=True, tokenizer=tok, eos_ids=(7,))
    assert (trace, n, finish) == ("abc</think>", 11, "stop")
    trace, n, finish = truncate_new_ids(full, budget=100, thinking_mode=True, tokenizer=tok, eos_ids=(7,))
    assert finish == "eos"
    trace, n, finish = truncate_new_ids(tok.encode("ab"), budget=100, thinking_mode=False, tokenizer=tok, eos_ids=(7,))
    assert finish == "unknown"


def test_true_new_length_stops_at_eos_or_stop_string():
    tok = _IdTok()
    ids = tok.encode("ab</think>cd")
    assert sc._true_new_length(ids, eos_ids=(7,), stops=("</think>",), tokenizer=tok) == 10
    assert sc._true_new_length(ids + [7, 7], eos_ids=(7,), stops=(), tokenizer=tok) == len(ids) + 1
    assert sc._true_new_length(ids, eos_ids=(7,), stops=(), tokenizer=tok) == len(ids)
