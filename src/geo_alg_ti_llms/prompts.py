"""The one prompt format of the paper, and the reasoning-pass variants of it.

Every prompt is a system instruction (one per dimension, naming what
``>`` and ``<`` mean; the context conditions use the context-only wording
that adds "Base your answer only on the comparisons given — not on outside
knowledge."), then a user turn of ``A > B`` premise lines (one per line; the
uninformative context writes each with ``>`` or ``<`` at random; omitted in the
no-context IWL test) and the question
``Is it true or false that A > B?``. The direct pass scores True/False as the
next token of the assistant turn (:mod:`geo_alg_ti_llms.scoring`).

The reasoning pass builds the same user turn and either leaves the assistant
turn open for a native thinking block (trigger ``none``: Qwen3, the DeepSeek
distills, Phi-4, gpt-oss) or appends the trigger ``Let's think step by step.``
(trigger ``passive``: every other model). After the generated trace, the
answer is read after the cue ``Therefore the answer is:``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from geo_alg_ti_llms.trials import ContextPair, Query, Trial


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

AnswerCueId = Literal["direct", "extract"]

ANSWER_CUES: dict[AnswerCueId, str] = {
    # Direct (non-CoT) regime: model produces the answer immediately.
    "direct": "Answer:",
    # CoT regime, answer pass: after the generated trace, the model commits to a
    # final True / False, scored by logprob.
    "extract": "Therefore the answer is:",
}


TemplateId = Literal["T2"]

QueryWrapperId = Literal["is_it_true_or_false"]

InstructionsV1Id = Literal[
    "none",
    # One instruction per dimension, worded so it is verbatim valid
    # whether or not premises are present. This is what lets the IWL test (no
    # context) and the content conditions (congruent / reversed / shuffled
    # context) share an identical instruction, so the ONLY thing that differs
    # between them is the context block itself.
    "iwl_population_T2",
    "iwl_mass_T2",
    "iwl_area_T2",
    # Context-only variants: the same sentence plus an explicit instruction to
    # ignore outside knowledge. Used by every context condition of the paper
    # (content, uninformative, and their reasoning ladders); the no-context
    # (IWL) test keeps the original wording.
    "iwl_population_T2_ctxonly",
    "iwl_mass_T2_ctxonly",
    "iwl_area_T2_ctxonly",
    # Novel-item (ICL) test. Same wording template as the familiar-item
    # instructions above, so the IWL and ICL panels of the distance figure
    # differ ONLY in item familiarity and context presence -- never in
    # prompt format. Height of an arbitrary person is unknowable a priori,
    # which is exactly the point: the ordering must come from the context.
    "iwl_height_T2",
]

AnswerCandidateFamily = Literal["true_false", "item_names"]
"""Scoring family used by the trial set:

- `true_false`: score `True` vs `False` family candidates.
- `item_names`: score the two specific item-name tokens for the trial; the
  candidate-id resolution is per-trial, not registry-level.
"""


@dataclass(frozen=True, slots=True)
class Template:
    """A premise/query rendering scheme.

    1. **Premise rendering.** Two Python format strings, one for each
       surface form. ``premise_gt`` is called with ``(higher, lower)`` when
       ``surface_form == 'high_first'``; ``premise_lt`` is called with
       ``(lower, higher)`` when ``surface_form == 'low_first'``.

    2. **Query predicate.** ``query_predicate_gt`` is the canonical
       GT-form predicate (e.g. ``"{0} > {1}"``) used inside True/False
       wrappers; called with ``(query.a, query.b)``. The query always
       renders in GT form — query polarity (True vs False) is determined
       by which item is passed first, not by the symbol used.
    """

    template_id: TemplateId
    description: str
    premise_gt: str
    premise_lt: str
    query_predicate_gt: str
    default_answer_family: AnswerCandidateFamily
    query_predicate_lt: str = ""
    """LT-form (opposite-direction) query predicate; not rendered by the
    paper's trial sets (the query is always in GT form)."""


TEMPLATES: dict[TemplateId, Template] = {
    "T2": Template(
        template_id="T2",
        description="English symbol: 'A > B' premises with True/False "
                    "query wrapper.",
        premise_gt="{0} > {1}",
        premise_lt="{0} < {1}",
        query_predicate_gt="{0} > {1}",
        query_predicate_lt="{0} < {1}",
        default_answer_family="true_false",
    ),
}


INSTRUCTIONS_V1: dict[InstructionsV1Id, str] = {
    "none": "",
    "iwl_population_T2": (
        "In the following question, > means \"has a larger population\" and < means "
        "\"has a smaller population\". Answer whether the statement is true or false."
    ),
    "iwl_mass_T2": (
        "In the following question, > means \"is heavier\" and < means "
        "\"is lighter\". Answer whether the statement is true or false."
    ),
    "iwl_area_T2": (
        "In the following question, > means \"has a larger area\" and < means "
        "\"has a smaller area\". Answer whether the statement is true or false."
    ),
    # The context-only sentence carries an em dash.
    "iwl_population_T2_ctxonly": (
        "In the following question, > means \"has a larger population\" and < means "
        "\"has a smaller population\". Answer whether the statement is true or false. "
        "Base your answer only on the comparisons given — not on outside knowledge."
    ),
    "iwl_mass_T2_ctxonly": (
        "In the following question, > means \"is heavier\" and < means "
        "\"is lighter\". Answer whether the statement is true or false. "
        "Base your answer only on the comparisons given — not on outside knowledge."
    ),
    "iwl_area_T2_ctxonly": (
        "In the following question, > means \"has a larger area\" and < means "
        "\"has a smaller area\". Answer whether the statement is true or false. "
        "Base your answer only on the comparisons given — not on outside knowledge."
    ),
    "iwl_height_T2": (
        "In the following question, > means \"is taller\" and < means "
        "\"is shorter\". Answer whether the statement is true or false."
    ),
}


GridCotTriggerId = Literal["none", "passive"]

GRID_COT_TRIGGERS: dict[GridCotTriggerId, str] = {
    # No trigger: the model is not nudged to produce a chain. On a native
    # thinking model the open assistant turn is where it generates its block.
    "none": "",
    # Kojima et al. 2022 (arXiv:2205.11916) zero-shot CoT trigger.
    "passive": "Let's think step by step.",
}


# Query wrapper format strings; ``render_grid_query`` fills ``{predicate}``
# from the template's query_predicate_gt.
QUERY_WRAPPERS: dict[QueryWrapperId, str] = {
    "is_it_true_or_false": "Is it true or false that {predicate}?",
}


# Default answer-family per wrapper. Trial sets can override but this is
# the natural pairing.
WRAPPER_ANSWER_FAMILY: dict[QueryWrapperId, AnswerCandidateFamily] = {
    "is_it_true_or_false": "true_false",
}


@dataclass(frozen=True, slots=True)
class GridPromptSpec:
    """Fully-specified prompt format of one trial set.

    Validated against the registries on construction so unknown ids fail
    loudly at spec creation, not at scoring time.

    Attributes
    ----------
    template_id : TemplateId
    instructions_v1_id : InstructionsV1Id
    query_wrapper_id : QueryWrapperId
    separator : str
        Inter-premise separator (``"\\n"`` in every published run).
    answer_family : AnswerCandidateFamily | None
        Override of the wrapper's default scoring family. ``None`` = use
        the wrapper default.
    """

    template_id: TemplateId
    instructions_v1_id: InstructionsV1Id
    query_wrapper_id: QueryWrapperId
    separator: str = "\n"
    answer_family: AnswerCandidateFamily | None = None

    def __post_init__(self) -> None:
        if self.template_id not in TEMPLATES:
            raise ValueError(f"unknown template_id: {self.template_id!r}")
        if self.instructions_v1_id not in INSTRUCTIONS_V1:
            raise ValueError(
                f"unknown instructions_v1_id: {self.instructions_v1_id!r}"
            )
        if self.query_wrapper_id not in QUERY_WRAPPERS:
            raise ValueError(
                f"unknown query_wrapper_id: {self.query_wrapper_id!r}"
            )

    @property
    def resolved_answer_family(self) -> AnswerCandidateFamily:
        """Final scoring family for this trial set (override or wrapper default)."""
        if self.answer_family is not None:
            return self.answer_family
        return WRAPPER_ANSWER_FAMILY[self.query_wrapper_id]


# ---------- Rendering primitives ----------


def render_grid_premise(pair: ContextPair, template: Template) -> str:
    """Render one ContextPair according to ``template``.

    ``pair.surface_form`` selects ``premise_gt`` / ``premise_lt`` and the
    argument order.
    """
    if pair.surface_form == "high_first":
        return template.premise_gt.format(pair.higher, pair.lower)
    return template.premise_lt.format(pair.lower, pair.higher)


def render_grid_context(
    trial: Trial, template: Template, separator: str
) -> str:
    """Render all of a trial's context_pairs, joined by ``separator``."""
    rendered = [render_grid_premise(p, template) for p in trial.context_pairs]
    return separator.join(rendered)


def render_grid_shown_context(
    trial: Trial, template: Template, separator: str
) -> str:
    """The context text the model is shown for ``trial``.

    The one source of the context in the direct builders
    (:func:`build_grid_user_text`, :func:`build_grid_messages`), on which the
    reasoning pass is built (:func:`build_reasoning_user_body`,
    :func:`build_reasoning_messages`). In
    ``zero_context`` mode (the IWL test) the trial carries hidden premises so it
    stays a valid Trial, but the model is shown none, so the result is ``""``.
    """
    if trial.extra and trial.extra.get("surface_form_mode") == "zero_context":
        return ""
    return render_grid_context(trial, template, separator)


def render_grid_query(
    query: Query,
    template: Template,
    wrapper_id: QueryWrapperId,
) -> str:
    """Render the final question seen by the model.

    Matched-template rule: the predicate inside the wrapper uses the
    template's ``query_predicate_gt`` exactly; the query is always rendered
    in GT form.
    """
    wrapper = QUERY_WRAPPERS[wrapper_id]
    predicate = template.query_predicate_gt.format(query.a, query.b)
    return wrapper.format(predicate=predicate)


# ---------- Top-level builders ----------


def build_grid_user_text(trial: Trial, spec: GridPromptSpec) -> str:
    """Assemble the full raw-text prompt for one trial set.

    Layout::

        {instructions}        # omitted if 'none'
        {blank line}          # only if instructions is non-empty
        {premise1}{sep}{premise2}{sep}...{premiseN-1}
        {blank line}
        {query}
    """
    template = TEMPLATES[spec.template_id]
    instr = INSTRUCTIONS_V1[spec.instructions_v1_id]
    ctx = render_grid_shown_context(trial, template, spec.separator)
    qry = render_grid_query(trial.query, template, spec.query_wrapper_id)

    # English wrappers: standard layout with blank-line separators.
    parts: list[str] = []
    if instr:
        parts.append(instr)
        parts.append("")
    # Zero-context mode yields no premises, so ctx is "". Omit the empty
    # context block (and its trailing blank line) so the prompt is
    # instruction + query with no dangling blank lines.
    if ctx:
        parts.append(ctx)
        parts.append("")
    parts.append(qry)
    return "\n".join(parts)


def build_grid_messages(
    trial: Trial, spec: GridPromptSpec
) -> list[dict[str, str]]:
    """Assemble the prompt as chat-template messages.

    The instructions preamble goes into the system message; the context
    + query go into the user message. The user message carries no
    "Answer:" cue: the chat template's assistant-turn opening serves as the
    cue and the next generated token is what gets scored.
    """
    template = TEMPLATES[spec.template_id]
    instr = INSTRUCTIONS_V1[spec.instructions_v1_id]
    ctx = render_grid_shown_context(trial, template, spec.separator)
    qry = render_grid_query(trial.query, template, spec.query_wrapper_id)

    # Zero-context mode renders no premises (ctx == ""); send just the query as
    # the user turn rather than a leading pair of blank lines.
    user_body = "\n".join([ctx, "", qry]) if ctx else qry
    msgs: list[dict[str, str]] = []
    if instr:
        msgs.append({"role": "system", "content": instr})
    msgs.append({"role": "user", "content": user_body})
    return msgs


# ---------------------------------------------------------------------------
# Reasoning pass: reasoning-pass prompt and answer-pass text
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CotSweepSpec:
    """One reasoning trial set: the base prompt format + the trigger.

    ``base`` carries the held-fixed rendering (template, instructions
    preamble, query wrapper, separator, answer family). ``trigger_id`` is the
    phrase appended after the query to elicit the chain (``none`` on native
    thinking models, ``passive`` otherwise).
    """

    base: GridPromptSpec
    trigger_id: GridCotTriggerId = "none"

    def __post_init__(self) -> None:
        if self.trigger_id not in GRID_COT_TRIGGERS:
            raise ValueError(f"unknown trigger_id: {self.trigger_id!r}")

    @property
    def is_cot(self) -> bool:
        """True iff this trial set actually prompts for a chain (a trigger is set).

        ``trigger=none`` is the untriggered condition: on a native thinking
        model the chain is the model's own thinking block.
        """
        return self.trigger_id != "none"


def build_reasoning_user_body(
    test_trial: Trial,
    spec: CotSweepSpec,
) -> str:
    """Assemble the user-turn text for the reasoning-pass prompt.

    The string ends exactly where the model should begin generating its
    own trace — i.e. after the test query and (if any) the trigger. Use
    this for raw-text mode; for chat-template models use
    :func:`build_reasoning_messages` (the instruction preamble moves to
    the system message).

    Layout (raw text)::

        {instructions preamble}        # omitted, with the blank line below, if 'none'

        {test premises}                # omitted, with the blank line below, for the IWL test

        {test query}
        {trigger}                      # omitted if trigger == 'none'

    It is the direct pass's text (:func:`build_grid_user_text`) plus the
    trigger on its own line, so both passes show the same context.
    """
    text = build_grid_user_text(test_trial, spec.base)
    trigger = GRID_COT_TRIGGERS[spec.trigger_id]
    return f"{text}\n{trigger}" if trigger else text


def build_reasoning_messages(
    test_trial: Trial,
    spec: CotSweepSpec,
) -> list[dict[str, str]]:
    """Chat-template variant: instruction preamble → system, rest → user.

    The direct pass's messages (:func:`build_grid_messages`) with the trigger,
    if any, appended to the user message on its own line. The assistant turn
    — opened by ``add_generation_prompt=True`` at tokenization time — is where
    the model generates its trace.
    """
    msgs = build_grid_messages(test_trial, spec.base)
    trigger = GRID_COT_TRIGGERS[spec.trigger_id]
    if trigger:
        msgs[-1] = {"role": "user", "content": f"{msgs[-1]['content']}\n{trigger}"}
    return msgs


def format_answer_pass(reasoning_prompt: str, cot_trace: str) -> str:
    """Answer-pass prompt TEXT = reasoning-pass + model trace + extraction cue.

    The ladder scores this text when its re-tokenization reproduces the
    generated ids (``scoring.answer_pass_input``), else the ids themselves; the
    two agree wherever the trace text round-trips through the tokenizer.
    """
    sep = "" if reasoning_prompt.endswith("\n") else "\n"
    return (
        f"{reasoning_prompt}{sep}{cot_trace.rstrip()}\n{ANSWER_CUES['extract']}"
    )
