"""Tests for the prompt format: registries, rendering, and the reasoning-pass variants."""

from __future__ import annotations

import pytest

from geo_alg_ti_llms.prompts import (
    ANSWER_CUES,
    GRID_COT_TRIGGERS,
    INSTRUCTIONS_V1,
    QUERY_WRAPPERS,
    TEMPLATES,
    WRAPPER_ANSWER_FAMILY,
    CotSweepSpec,
    GridPromptSpec,
    build_grid_messages,
    build_grid_user_text,
    build_reasoning_messages,
    build_reasoning_user_body,
    format_answer_pass,
    render_grid_context,
    render_grid_premise,
    render_grid_query,
)
from geo_alg_ti_llms.trials import ContextPair, Query, Trial

CTXONLY_SENTENCE = "Base your answer only on the comparisons given — not on outside knowledge."


@pytest.fixture
def trial_apple_book():
    """A 3-item trial: book < ball < apple. Query asks if apple > book (True, d=2).

    The two context pairs are deliberately constructed with opposite surface
    forms so that both premise renderings are exercised at once.
    """
    return Trial(
        trial_id=0, cell_id="test", master_seed=42, seed=0, N=3,
        item_pool="test_pool", assignment_strategy="random",
        rank_to_item=("book", "ball", "apple"),
        context_pairs=(
            ContextPair(higher="ball", lower="book", surface_form="high_first"),
            ContextPair(higher="apple", lower="ball", surface_form="low_first"),
        ),
        query=Query(a="apple", b="book", correct=True, symbolic_distance=2, signed_distance=2),
    )


@pytest.fixture
def spec_population():
    return GridPromptSpec(
        template_id="T2", instructions_v1_id="iwl_population_T2",
        query_wrapper_id="is_it_true_or_false", separator="\n",
    )


# ---------- registries ----------


def test_registries_hold_exactly_the_paper_format():
    assert set(TEMPLATES) == {"T2"}
    assert set(QUERY_WRAPPERS) == {"is_it_true_or_false"}
    assert set(WRAPPER_ANSWER_FAMILY) == set(QUERY_WRAPPERS)
    assert set(GRID_COT_TRIGGERS) == {"none", "passive"}
    assert GRID_COT_TRIGGERS["none"] == ""
    assert GRID_COT_TRIGGERS["passive"] == "Let's think step by step."
    assert ANSWER_CUES["extract"] == "Therefore the answer is:"
    assert set(INSTRUCTIONS_V1) == {
        "none", "iwl_population_T2", "iwl_mass_T2", "iwl_area_T2",
        "iwl_population_T2_ctxonly", "iwl_mass_T2_ctxonly", "iwl_area_T2_ctxonly",
        "iwl_height_T2",
    }


def test_instruction_wording():
    assert INSTRUCTIONS_V1["iwl_population_T2"] == (
        'In the following question, > means "has a larger population" and < means '
        '"has a smaller population". Answer whether the statement is true or false.'
    )
    for base in ("iwl_population_T2", "iwl_mass_T2", "iwl_area_T2"):
        assert INSTRUCTIONS_V1[f"{base}_ctxonly"] == INSTRUCTIONS_V1[base] + " " + CTXONLY_SENTENCE
    assert "\u2014" in CTXONLY_SENTENCE  # the em dash of the published runs
    assert INSTRUCTIONS_V1["iwl_height_T2"].startswith('In the following question, > means "is taller"')


def test_template_id_consistency():
    for tid, tmpl in TEMPLATES.items():
        assert tmpl.template_id == tid


# ---------- spec validation ----------


def test_spec_unknown_ids_raise():
    with pytest.raises(ValueError, match="template_id"):
        GridPromptSpec(template_id="T1", instructions_v1_id="none", query_wrapper_id="is_it_true_or_false")
    with pytest.raises(ValueError, match="instructions_v1_id"):
        GridPromptSpec(template_id="T2", instructions_v1_id="brief_T2", query_wrapper_id="is_it_true_or_false")
    with pytest.raises(ValueError, match="query_wrapper_id"):
        GridPromptSpec(template_id="T2", instructions_v1_id="none", query_wrapper_id="true_or_false_colon")


def test_spec_resolved_answer_family(spec_population):
    assert spec_population.resolved_answer_family == "true_false"
    forced = GridPromptSpec(template_id="T2", instructions_v1_id="none",
                            query_wrapper_id="is_it_true_or_false", answer_family="item_names")
    assert forced.resolved_answer_family == "item_names"


# ---------- rendering ----------


def test_render_premise_t2_english_symbol(trial_apple_book):
    pairs = trial_apple_book.context_pairs
    assert render_grid_premise(pairs[0], TEMPLATES["T2"]) == "ball > book"
    assert render_grid_premise(pairs[1], TEMPLATES["T2"]) == "ball < apple"


def test_render_context_joins_with_separator(trial_apple_book):
    assert render_grid_context(trial_apple_book, TEMPLATES["T2"], "\n") == "ball > book\nball < apple"


def test_render_query_t2_is_it_true_or_false(trial_apple_book):
    out = render_grid_query(trial_apple_book.query, TEMPLATES["T2"], "is_it_true_or_false")
    assert out == "Is it true or false that apple > book?"


def test_build_grid_user_text_with_instruction(trial_apple_book, spec_population):
    out = build_grid_user_text(trial_apple_book, spec_population)
    assert out == (
        INSTRUCTIONS_V1["iwl_population_T2"] + "\n"
        "\n"
        "ball > book\n"
        "ball < apple\n"
        "\n"
        "Is it true or false that apple > book?"
    )


def test_build_grid_user_text_no_instr_skips_preamble(trial_apple_book):
    spec = GridPromptSpec(template_id="T2", instructions_v1_id="none",
                          query_wrapper_id="is_it_true_or_false", separator="\n")
    assert build_grid_user_text(trial_apple_book, spec) == (
        "ball > book\nball < apple\n\nIs it true or false that apple > book?"
    )


def test_build_grid_messages_with_system_role(trial_apple_book, spec_population):
    msgs = build_grid_messages(trial_apple_book, spec_population)
    assert msgs == [
        {"role": "system", "content": INSTRUCTIONS_V1["iwl_population_T2"]},
        {"role": "user", "content": "ball > book\nball < apple\n\nIs it true or false that apple > book?"},
    ]


def test_build_grid_messages_none_instr_skips_system_message(trial_apple_book):
    spec = GridPromptSpec(template_id="T2", instructions_v1_id="none",
                          query_wrapper_id="is_it_true_or_false")
    msgs = build_grid_messages(trial_apple_book, spec)
    assert len(msgs) == 1 and msgs[0]["role"] == "user"


def _zero_context(trial: Trial) -> Trial:
    return Trial(**{**{f: getattr(trial, f) for f in trial.__slots__},
                    "extra": {"surface_form_mode": "zero_context"}})


def test_zero_context_renders_no_premises(trial_apple_book, spec_population):
    """The IWL test shows instruction + query only, with no dangling blank lines."""
    t = _zero_context(trial_apple_book)
    assert build_grid_user_text(t, spec_population) == (
        INSTRUCTIONS_V1["iwl_population_T2"] + "\n\nIs it true or false that apple > book?"
    )
    msgs = build_grid_messages(t, spec_population)
    assert msgs[1] == {"role": "user", "content": "Is it true or false that apple > book?"}


# ---------- reasoning pass ----------


def test_cot_sweep_spec_is_cot(spec_population):
    assert not CotSweepSpec(base=spec_population, trigger_id="none").is_cot
    assert CotSweepSpec(base=spec_population, trigger_id="passive").is_cot
    with pytest.raises(ValueError, match="trigger_id"):
        CotSweepSpec(base=spec_population, trigger_id="active")  # type: ignore[arg-type]


def test_user_body_ends_at_trigger(trial_apple_book, spec_population):
    spec = CotSweepSpec(base=spec_population, trigger_id="passive")
    body = build_reasoning_user_body(trial_apple_book, spec)
    assert body == (
        INSTRUCTIONS_V1["iwl_population_T2"] + "\n\n"
        "ball > book\nball < apple\n\nIs it true or false that apple > book?\n"
        "Let's think step by step."
    )


def test_user_body_no_trigger_ends_at_query(trial_apple_book, spec_population):
    spec = CotSweepSpec(base=spec_population, trigger_id="none")
    body = build_reasoning_user_body(trial_apple_book, spec)
    assert body.endswith("Is it true or false that apple > book?")


def test_messages_instr_to_system(trial_apple_book, spec_population):
    spec = CotSweepSpec(base=spec_population, trigger_id="passive")
    msgs = build_reasoning_messages(trial_apple_book, spec)
    assert msgs == [
        {"role": "system", "content": INSTRUCTIONS_V1["iwl_population_T2"]},
        {"role": "user", "content": "ball > book\nball < apple\n\nIs it true or false that apple > book?\n"
                                    "Let's think step by step."},
    ]


def test_messages_none_instr_no_system(trial_apple_book):
    base = GridPromptSpec(template_id="T2", instructions_v1_id="none", query_wrapper_id="is_it_true_or_false")
    msgs = build_reasoning_messages(trial_apple_book, CotSweepSpec(base=base, trigger_id="none"))
    assert len(msgs) == 1 and msgs[0]["role"] == "user"


def test_untriggered_messages_equal_the_direct_messages(trial_apple_book, spec_population):
    """With trigger 'none' (native thinking models) the reasoning-pass user turn is the
    direct user turn; the only difference is the open assistant turn."""
    direct = build_grid_messages(trial_apple_book, spec_population)
    reasoning_prompt = build_reasoning_messages(trial_apple_book, CotSweepSpec(base=spec_population, trigger_id="none"))
    assert direct == reasoning_prompt


def test_zero_context_reasoning_messages_are_the_direct_messages_plus_the_trigger(
        trial_apple_book, spec_population):
    """The IWL test's reasoning pass shows no premises: its user turn is the direct user
    turn, followed by the trigger when there is one."""
    t = _zero_context(trial_apple_book)
    direct = build_grid_messages(t, spec_population)
    assert build_reasoning_messages(t, CotSweepSpec(base=spec_population, trigger_id="none")) == direct
    triggered = build_reasoning_messages(t, CotSweepSpec(base=spec_population, trigger_id="passive"))
    assert triggered[0] == direct[0]
    assert triggered[1] == {"role": "user",
                            "content": direct[1]["content"] + "\n" + GRID_COT_TRIGGERS["passive"]}


def test_zero_context_reasoning_user_body_is_the_direct_text_plus_the_trigger(
        trial_apple_book, spec_population):
    t = _zero_context(trial_apple_book)
    direct = build_grid_user_text(t, spec_population)
    assert build_reasoning_user_body(t, CotSweepSpec(base=spec_population, trigger_id="none")) == direct
    body = build_reasoning_user_body(t, CotSweepSpec(base=spec_population, trigger_id="passive"))
    assert body == direct + "\n" + GRID_COT_TRIGGERS["passive"]
    assert "ball" not in body


def test_format_stage2_appends_trace_and_cue():
    reasoning_prompt = "...premises...\nLet's think step by step."
    trace = "  dog > car\ndog > ball  "
    out = format_answer_pass(reasoning_prompt, trace)
    assert out == reasoning_prompt + "\n  dog > car\ndog > ball\nTherefore the answer is:"
    # A reasoning-pass that already ends in a newline gets no second one.
    assert format_answer_pass("x\n", "t") == "x\nt\nTherefore the answer is:"
