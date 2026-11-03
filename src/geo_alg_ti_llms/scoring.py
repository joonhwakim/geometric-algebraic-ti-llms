"""Reading the answer: True/False log-probabilities, and the reasoning-budget ladder.

Two regimes:

- **Direct** (no reasoning). One forward pass through the model on a prompt that
  ends at the position where the next token is the answer (a chat-templated
  assistant header, plus the model's ``thinking_suppress_prefix`` /
  ``chat_generation_suffix`` where the config sets one -- see
  :func:`build_direct_prompt`). The next-token logprobs are log-sum-exp'd
  over the two candidate families of the answer wrapper -- for ``true_false``:
  {True, " True", true, " true"} vs the False forms
  (:func:`resolve_family_candidates`) -- and ``lp_yes > lp_no`` is the
  prediction (a tie predicts the "no" side). ``logprob_yes - logprob_no``
  is the continuous signal.

- **Reasoning.** Two passes:
    1. Greedy-generate the trace from a reasoning-pass prompt that ends with the
       trigger or opens a native thinking block
       (:func:`generate_new_token_ids`), once per trial at the largest budget
       of the ladder; every smaller budget is the token-level prefix of that
       trajectory (:func:`truncate_new_ids`), which is exact under greedy
       decoding.
    2. Score the answer after the trace. The answer-pass input is the rendered
       text ``reasoning-pass + trace + "\\n" + cue`` when its re-tokenization
       reproduces the generated ids, else the ids themselves --
       ``reasoning-pass ids + generated ids + encode("\\n" + cue)``
       (:func:`answer_pass_input`) -- so the scored context is exactly the
       model's own continuation.

The chat-template helpers pin everything a template would otherwise take from
the run day (:data:`PINNED_DATE_STRING`, :data:`PINNED_STRFTIME_DATE`) and
fold the system instruction into the user turn for templates that reject or
drop the system role (Gemma-2, Phi-4-reasoning), so every stored prompt
re-renders byte for byte.

torch is imported inside the functions that need it, so the package imports
without it; the candidate resolution and chat-template helpers need only a
tokenizer.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

from geo_alg_ti_llms.prompts import build_grid_messages, build_grid_user_text


_STUB_OF = {"score_yes_no": "score", "score_yes_no_ids": "score_ids",
            "generate_new_token_ids": "generate"}


def _no_grad(fn):
    """``torch.no_grad()`` applied at call time (torch is imported lazily); the stub model of
    :mod:`geo_alg_ti_llms.stub` is answered by the stub's own function instead."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        from geo_alg_ti_llms import stub
        if isinstance(args[0], stub.StubModel):
            return getattr(stub, _STUB_OF[fn.__name__])(*args, **kwargs)
        import torch
        with torch.no_grad():
            return fn(*args, **kwargs)
    return wrapper


# ---------------------------------------------------------------------------
# Candidate token-id resolution
# ---------------------------------------------------------------------------


# Both leading-space and no-leading-space variants; the resolution helper
# picks whichever the tokenizer emits as a single id and skips the rest.
DEFAULT_TRUE_CANDIDATES: tuple[str, ...] = ("True", " True", "true", " true")
"""Tokens summed for a "True" answer."""

DEFAULT_FALSE_CANDIDATES: tuple[str, ...] = ("False", " False", "false", " false")
"""Mirror of ``DEFAULT_TRUE_CANDIDATES`` for "False"."""


def resolve_forced_choice_candidates(
    tokenizer,
    a: str,
    b: str,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Resolve per-trial candidate token ids for forced-choice naming trial sets.

    The answer space is the two specific item names (``a`` and ``b``); the
    model's logprob over each is scored. Both leading-space and
    no-leading-space variants are tried, so whichever surface form
    the tokenizer emits at the answer position is scored.

    Returns
    -------
    (a_ids, b_ids) : two tuples of single-token ids
        Empty tuples if no single-token form resolves for that item under
        ``tokenizer`` — caller should treat this as a hard error
        (forced-choice scoring requires single-token item names).
    """
    a_ids: list[int] = []
    b_ids: list[int] = []
    for item, bucket in [(a, a_ids), (b, b_ids)]:
        for variant in (item, " " + item):
            ids = tokenizer.encode(variant, add_special_tokens=False)
            if len(ids) == 1:
                bucket.append(ids[0])
    return tuple(dict.fromkeys(a_ids)), tuple(dict.fromkeys(b_ids))


@dataclass(frozen=True, slots=True)
class CandidateIds:
    """Token-id sets for the Yes and No candidate families.

    Only single-token candidates are kept — multi-token entries can't be
    scored at a single next-token position.
    """

    yes: tuple[int, ...]
    no: tuple[int, ...]
    yes_strs: tuple[str, ...]   # for diagnostics
    no_strs: tuple[str, ...]


def resolve_candidate_ids(
    tokenizer,
    *,
    yes_candidates: tuple[str, ...],
    no_candidates: tuple[str, ...],
) -> CandidateIds:
    """Encode the two candidate families, keep only single-token forms.

    The families are REQUIRED keyword arguments: a Yes/No default would let a
    caller score a True/False experiment with the wrong family. Use
    :func:`resolve_family_candidates` to go from an answer-family name to ids.
    Run once at startup; pass the result into every scoring call.
    """

    def _resolve(strs: tuple[str, ...]) -> tuple[tuple[int, ...], tuple[str, ...]]:
        """Encode, keep single-token candidates, and DE-DUPLICATE by token id.

        A tokenizer can fold two surface forms onto one id (a SentencePiece-style
        pre-tokenizer folds ' True' onto 'True' -- the mis-loaded DeepSeek case,
        now caught by ``models.load_tokenizer``). Without dedup that id would be
        counted twice by ``torch.logsumexp`` (+log 2 on the absolute logprob; the
        difference ``lp_yes - lp_no`` is unaffected when both families collide).
        """
        ids: list[int] = []
        kept: list[str] = []
        for s in strs:
            encoded = tokenizer.encode(s, add_special_tokens=False)
            if len(encoded) == 1 and encoded[0] not in ids:
                ids.append(encoded[0])
                kept.append(s)
        return tuple(ids), tuple(kept)

    yes_ids, yes_strs = _resolve(yes_candidates)
    no_ids, no_strs = _resolve(no_candidates)

    if not yes_ids:
        raise ValueError(
            f"No single-token Yes candidates found in {yes_candidates}."
        )
    if not no_ids:
        raise ValueError(
            f"No single-token No candidates found in {no_candidates}."
        )

    return CandidateIds(yes=yes_ids, no=no_ids, yes_strs=yes_strs, no_strs=no_strs)


def resolve_family_candidates(family: str, tokenizer) -> CandidateIds | None:
    """Candidate ids for an answer family name (one source for every runner).

    ``"true_false"`` resolves the default True/False surface forms;
    ``"item_names"`` (forced choice) returns ``None`` because its candidates
    are per trial.
    """
    if family == "true_false":
        return resolve_candidate_ids(tokenizer, yes_candidates=DEFAULT_TRUE_CANDIDATES,
                                     no_candidates=DEFAULT_FALSE_CANDIDATES)
    if family == "item_names":
        return None
    raise ValueError(f"unknown answer family: {family!r}")


def eos_token_ids(model, tokenizer) -> tuple[int, ...]:
    """Every id the model treats as end-of-sequence.

    ``generation_config.eos_token_id`` is a LIST for most chat models (Qwen
    ``[<|im_end|>, <|endoftext|>]``, Llama-3.1 three ids, GLM-4 three, gpt-oss
    three); passing only ``tokenizer.eos_token_id`` to ``generate`` would let a
    row that ended on a secondary EOS keep decoding. The tokenizer's id is
    always included.
    """
    ids: list[int] = []
    gc = getattr(model, "generation_config", None)
    raw = getattr(gc, "eos_token_id", None) if gc is not None else None
    if isinstance(raw, int):
        ids.append(raw)
    elif raw:
        ids.extend(int(i) for i in raw)
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in ids:
        ids.append(int(tokenizer.eos_token_id))
    return tuple(ids)


# ---------------------------------------------------------------------------
# Prompt formatting (chat template)
# ---------------------------------------------------------------------------


def build_direct_prompt(trial, spec, tokenizer, model_cfg, use_chat_template: bool) -> str:
    """The direct (no-reasoning) prompt string -- ONE source for both runners.

    The direct tests and the ladder's budget-0 row score byte-identical prompts
    (the order is fixed here: suffix, then prefix).

    - Hybrid reasoning models (``thinking_mode``, e.g. Qwen3) get
      ``enable_thinking=False`` so the template closes an empty think block and
      the next token is the answer.
    - ``thinking_suppress_prefix`` (DeepSeek-R1 distills, Phi-4, gpt-oss: reasoning
      models with no toggle) prefills a closed think block / final channel.
    - ``chat_generation_suffix`` (GLM-4: ``"\\n"``) completes the assistant header.
    """
    if not use_chat_template:
        return build_grid_user_text(trial, spec)
    thinking = bool(getattr(model_cfg, "thinking_mode", False))
    prompt = apply_chat_template_cot(
        tokenizer, build_grid_messages(trial, spec), add_generation_prompt=True,
        enable_thinking=(False if thinking else None),
    )
    prefix = getattr(model_cfg, "thinking_suppress_prefix", None)
    suffix = getattr(model_cfg, "chat_generation_suffix", None)
    # order: template header -> chat_generation_suffix (completes the header) ->
    # thinking_suppress_prefix (assistant content). No roster config sets both.
    if suffix:
        prompt += suffix
    if prefix:
        prompt += prefix
    return prompt


_SYSTEM_MERGE_WARNED: set[str] = set()
_ENABLE_THINKING_FALLBACK_WARNED: set[str] = set()
PINNED_DATE_STRING = "26 Jul 2024"   # Llama-3 template default; see apply_chat_template_safe
# Templates that call strftime_now() directly (gpt-oss: "Current date: YYYY-MM-DD") would render
# a different prompt every day. Pinned to the date every stored gpt-oss prompt carries, so a
# re-run reproduces them byte for byte.
PINNED_STRFTIME_DATE = "2026-09-13"


def merge_system_into_user(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Fold a leading system message into the first user turn.

    For chat templates that reject the system role (Gemma-2). The instruction text
    is unchanged and still precedes the context; only its role tag differs -- a
    model-forced, documented deviation from the one-instruction-per-role control.
    The rendered prompt is stored in every result row, so it stays auditable.
    """
    if not messages or messages[0]["role"] != "system":
        return messages
    sys_msg, rest = messages[0], list(messages[1:])
    if not rest or rest[0]["role"] != "user":
        raise ValueError("system merge needs a user turn right after the system turn")
    rest[0] = {"role": "user", "content": f"{sys_msg['content']}\n\n{rest[0]['content']}"}
    return rest


def apply_chat_template_safe(tokenizer, messages: list[dict[str, str]], **kwargs) -> str:
    """``apply_chat_template`` that survives templates without a system role.

    Tries the messages as given; if the template raises because of the system
    role, merges it into the user turn (see :func:`merge_system_into_user`) and
    retries, warning once per tokenizer.
    """
    # Llama-3.x templates stamp strftime_now() into the system header ("Today
    # Date: ..."), which would make the prompt differ by RUN DAY. Pin it to the
    # template's own fallback constant so every run renders the same bytes.
    tmpl = getattr(tokenizer, "chat_template", None)
    tmpl = tmpl if isinstance(tmpl, str) else ""
    if "date_string" in tmpl and "date_string" not in kwargs:
        kwargs = {**kwargs, "date_string": PINNED_DATE_STRING}
    if "strftime_now" in tmpl and "date_string" not in tmpl and "strftime_now" not in kwargs:
        # render kwargs shadow the template global, so the date is pinned
        kwargs = {**kwargs, "strftime_now": lambda fmt: PINNED_STRFTIME_DATE}
    try:
        out = tokenizer.apply_chat_template(messages, **kwargs)
    except Exception as e:   # jinja2.TemplateError subclasses vary by version
        if "system" not in str(e).lower() or not messages or messages[0]["role"] != "system":
            raise
        key = getattr(tokenizer, "name_or_path", "?")
        if key not in _SYSTEM_MERGE_WARNED:
            _SYSTEM_MERGE_WARNED.add(key)
            print(f"[prompt] {key}: template rejects the system role ({e!s:.60}); "
                  "merging the instruction into the user turn", flush=True)
        return tokenizer.apply_chat_template(merge_system_into_user(messages), **kwargs)
    # Some templates accept a system message and then silently DROP it in favor of a fixed
    # system prompt of their own (microsoft/Phi-4-reasoning replaces every instruction with
    # "You are Phi ..."; no other roster model does). A render that does not contain the
    # instruction text is not the experiment, so merge it into the user turn and render
    # again, warning once per tokenizer.
    if messages and messages[0]["role"] == "system" and isinstance(out, str):
        sys_text = messages[0]["content"].strip()
        if sys_text and sys_text not in out:
            key = getattr(tokenizer, "name_or_path", "?")
            if key not in _SYSTEM_MERGE_WARNED:
                _SYSTEM_MERGE_WARNED.add(key)
                print(f"[prompt] {key}: template drops the system message; "
                      "merging the instruction into the user turn", flush=True)
            out = tokenizer.apply_chat_template(merge_system_into_user(messages), **kwargs)
            assert sys_text in out, f"instruction still missing after the merge ({key})"
    return out


def apply_chat_template_cot(
    tokenizer,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool = True,
    enable_thinking: bool | None = None,
) -> str:
    """``apply_chat_template`` with native reasoning toggled per the model.

    Reasoning-trained chat templates (Qwen3, etc.) accept an
    ``enable_thinking`` flag that decides whether the model opens a
    ``<think>`` block. The flag is **tri-state**:

    - ``None`` (default) — never pass the kwarg. Byte-identical to a plain
      ``apply_chat_template`` call (Qwen2.5 / Llama / GLM, which don't have
      the flag).
    - ``True`` — request native thinking (the reasoning condition on a
      reasoning model). Qwen3 then opens a ``<think>`` block.
    - ``False`` — explicitly SUPPRESS thinking. This matters because Qwen3
      defaults thinking ON, so the direct scoring path must pass ``False`` or
      the scored next-token position lands on the opening ``<think>`` instead
      of the answer.

    Only the Qwen3 templates consume the flag; every roster template accepts
    and ignores unknown kwargs, so the ``TypeError`` fallback below is
    defensive only (a tokenizer that rejects the kwarg is rendered without it,
    and the fallback is printed once so it is never silent).
    """
    if enable_thinking is not None:
        try:
            return apply_chat_template_safe(
                tokenizer, messages, tokenize=False,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            key = getattr(tokenizer, "name_or_path", "?")
            if key not in _ENABLE_THINKING_FALLBACK_WARNED:
                _ENABLE_THINKING_FALLBACK_WARNED.add(key)
                print(f"[prompt] {key}: template rejects enable_thinking; rendered without it", flush=True)
    return apply_chat_template_safe(
        tokenizer, messages, tokenize=False, add_generation_prompt=add_generation_prompt,
    )


# ---------------------------------------------------------------------------
# Answer-pass input of the reasoning pass
# ---------------------------------------------------------------------------


def strip_trace_ids(tokenizer, trace_ids: list[int], *, eos_ids: tuple[int, ...]) -> list[int]:
    """The generated trace without its trailing end-of-sequence / special ids and trailing
    whitespace-only tokens (what ``rstrip()`` removes from the decoded text)."""
    special = set(getattr(tokenizer, "all_special_ids", []) or [])
    ids = list(trace_ids)
    while ids and (ids[-1] in eos_ids or ids[-1] in special
                   or tokenizer.decode([ids[-1]], skip_special_tokens=False).strip() == ""):
        ids.pop()
    return ids


def build_answer_pass_ids(tokenizer, reasoning_prompt_ids: list[int], trace_ids: list[int], *,
                     eos_ids: tuple[int, ...], cue: str) -> list[int]:
    """Answer-pass input as token ids: reasoning-pass + the model's own trace ids +
    ``encode("\\n"+cue)``."""
    return list(reasoning_prompt_ids) + strip_trace_ids(tokenizer, trace_ids, eos_ids=eos_ids) \
        + tokenizer.encode("\n" + cue, add_special_tokens=False)


def answer_pass_input(tokenizer, reasoning_prompt_text: str, reasoning_prompt_ids: list[int], trace_ids: list[int],
                 answer_text: str, *, eos_ids: tuple[int, ...], cue: str,
                 add_special_tokens: bool = False) -> tuple[list[int], str, bool]:
    """The answer-pass token sequence to score, and whether the TEXT path was faithful.

    Rule: the answer-pass prompt is the rendered text ``reasoning-pass + trace + "\\n" + cue``
    (:func:`geo_alg_ti_llms.prompts.format_answer_pass`) PROVIDED its
    re-tokenization reproduces the ids the model actually generated up to the cue
    junction (the last trace token may merge with the newline before the cue; that is
    the only tolerated difference). The text is used when its re-tokenization is
    faithful, else the ids: for the Qwen models that is nearly always the text (4 of the
    15,408 stored Qwen3-32B b1024 context-ladder rows took the ids path); where the
    re-tokenized text is NOT the generated sequence (Phi-4-reasoning: up to +262 tokens;
    OLMo-2-7B; GLM-4 / gpt-oss: a newline the model never produced), the ids are used
    instead: ``reasoning-pass ids + trace ids + encode("\\n" + cue)``. Returns
    ``(ids, source, faithful)`` with ``source`` = "text" or "ids".
    """
    tr = strip_trace_ids(tokenizer, trace_ids, eos_ids=eos_ids)
    text_ids = tokenizer.encode(answer_text, add_special_tokens=add_special_tokens)
    want = list(reasoning_prompt_ids) + tr[:-1] if tr else list(reasoning_prompt_ids)
    faithful = text_ids[:len(want)] == want
    if faithful:
        return text_ids, "text", True
    return build_answer_pass_ids(tokenizer, reasoning_prompt_ids, trace_ids, eos_ids=eos_ids, cue=cue), "ids", False


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------


def _yes_no_from_logits(
    last_logits, candidates: CandidateIds,
) -> tuple[float, float]:
    """Log-sum candidate logprobs from a final-position logit vector.

    ``last_logits`` must already be a 1-D float32 CPU tensor over the full
    vocabulary.
    """
    import torch
    logprobs = torch.log_softmax(last_logits, dim=-1)
    yes_idx = torch.tensor(candidates.yes, dtype=torch.long)
    no_idx = torch.tensor(candidates.no, dtype=torch.long)
    lp_yes = torch.logsumexp(logprobs[yes_idx], dim=0).item()
    lp_no = torch.logsumexp(logprobs[no_idx], dim=0).item()
    return lp_yes, lp_no


@_no_grad
def score_yes_no(
    model,
    tokenizer,
    prompt: str,
    candidates: CandidateIds,
    *,
    add_special_tokens: bool = False,
) -> tuple[float, float, int]:
    """Return ``(logprob_yes, logprob_no, n_tokens)`` for the next-token position.

    The first two are log-summed over the candidate sets via
    :func:`torch.logsumexp`; the third is the prompt's token count
    (returned so callers can record it in TrialResult without
    re-tokenizing).

    Parameters
    ----------
    add_special_tokens : bool, default False
        Whether to let the tokenizer prepend its BOS token. **Set False
        when the prompt was produced by a chat template** —
        ``apply_chat_template`` already includes ``<|begin_of_text|>``
        in the rendered string, and the default
        ``add_special_tokens=True`` would prepend a duplicate BOS,
        biasing the next-token distribution. Set True only for raw-text
        prompts where the prompt has no BOS.
    """
    import torch
    inputs = tokenizer(
        prompt, return_tensors="pt",
        add_special_tokens=add_special_tokens,
    ).to(model.device)
    n_tokens = int(inputs.input_ids.shape[1])
    out = model(**inputs)
    last_logits = out.logits[0, -1, :].to(dtype=torch.float32, device="cpu")
    lp_yes, lp_no = _yes_no_from_logits(last_logits, candidates)
    return lp_yes, lp_no, n_tokens


@_no_grad
def score_yes_no_ids(model, ids: list[int], candidates: CandidateIds) -> tuple[float, float, int]:
    """Like :func:`score_yes_no`, on an explicit token-id sequence."""
    import torch
    input_ids = torch.tensor([ids], dtype=torch.long, device=model.device)
    out = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
    last_logits = out.logits[0, -1, :].to(dtype=torch.float32, device="cpu")
    lp_yes, lp_no = _yes_no_from_logits(last_logits, candidates)
    return lp_yes, lp_no, len(ids)


# ---------------------------------------------------------------------------
# Token-level generation (ladder-once + fixed-batch path)
# ---------------------------------------------------------------------------
#
# The budgets of a ladder differ ONLY in ``max_new_tokens``. Under greedy
# decoding on one hardware/software stack the trace at budget b is exactly the
# first b tokens of the trace at any larger budget (verified on the country population
# runs: 6480/6480 same-hardware trial set pairs, every model; the only divergences
# were job pairs that ran on different GPU types). So one max-budget
# generation per trial, truncated at the token level, reproduces every budget —
# that is what ``generate_new_token_ids`` + ``truncate_new_ids`` provide.
# ``batch_size > 1`` decodes several trials at once (left-padded, the batch
# always padded to exactly ``batch_size`` rows so kernel shapes are the same
# for every trial of a run); it changes bf16 kernel shapes, so it is a
# documented numerical perturbation of the same kind as a GPU-type change.


def _true_new_length(
    new_ids: list[int], *, eos_ids: tuple[int, ...], stops: tuple[str, ...], tokenizer,
) -> int:
    """Where did HF's generate stop THIS row of a batch?

    After a row finishes (EOS or a stop string) generate pads it with
    ``pad_token_id`` until the batch ends; the pad may equal EOS. The row's
    true length is the first step at which either criterion fired.
    """
    stop_tails = tuple(s[-1] for s in stops)
    for i, tid in enumerate(new_ids):
        if tid in eos_ids:
            return i + 1
        if stops:
            piece = tokenizer.decode([tid], skip_special_tokens=False)
            if any(t in piece for t in stop_tails):
                text = tokenizer.decode(new_ids[: i + 1], skip_special_tokens=False)
                if any(text.endswith(s) for s in stops):
                    return i + 1
    return len(new_ids)


@_no_grad
def generate_new_token_ids(
    model,
    tokenizer,
    prompts: list[str],
    *,
    max_new_tokens: int,
    thinking_mode: bool = False,
    add_special_tokens: bool = False,
    batch_size: int = 1,
    thinking_stop_string: str = "</think>",
    eos_ids: tuple[int, ...] | None = None,
) -> list[tuple[list[int], int]]:
    """Greedy-generate ``max_new_tokens`` for each prompt; return token ids.

    ``eos_ids``: every id that ends a row (:func:`eos_token_ids`); ``None`` falls
    back to the tokenizer's single id.

    Returns one ``(new_token_ids, n_input_tokens)`` per prompt, in order.
    ``new_token_ids`` is trimmed to where generation actually stopped for that
    row (EOS, native ``</think>`` close in thinking mode, or the budget).

    ``batch_size == 1`` issues one unpadded ``generate`` call per prompt.
    ``batch_size > 1`` left-pads; every batch is padded to exactly
    ``batch_size`` rows by repeating the last prompt (discarded) so that the
    decode-step GEMM shapes are constant across the whole run.
    """
    stops: tuple[str, ...] = (thinking_stop_string,) if thinking_mode else ()
    eos = tuple(eos_ids) if eos_ids else (tokenizer.eos_token_id,)
    gen_kwargs: dict = dict(
        max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=list(eos),
    )
    if stops:
        gen_kwargs["stop_strings"] = list(stops)
        gen_kwargs["tokenizer"] = tokenizer

    out: list[tuple[list[int], int]] = []
    if batch_size <= 1:
        for prompt in prompts:
            inputs = tokenizer(
                prompt, return_tensors="pt", add_special_tokens=add_special_tokens,
            ).to(model.device)
            n_input = int(inputs.input_ids.shape[1])
            gen = model.generate(**inputs, **gen_kwargs)
            ids = gen[0, n_input:].tolist()
            out.append((ids[:_true_new_length(ids, eos_ids=eos, stops=stops, tokenizer=tokenizer)], n_input))
        return out

    orig_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        for start in range(0, len(prompts), batch_size):
            chunk = prompts[start:start + batch_size]
            n_real = len(chunk)
            chunk = chunk + [chunk[-1]] * (batch_size - n_real)   # fixed shape
            enc = tokenizer(
                chunk, return_tensors="pt", padding=True,
                add_special_tokens=add_special_tokens,
            ).to(model.device)
            padded_len = int(enc.input_ids.shape[1])
            n_inputs = enc.attention_mask.sum(dim=1).tolist()
            gen = model.generate(**enc, **gen_kwargs)
            for i in range(n_real):
                ids = gen[i, padded_len:].tolist()
                k = _true_new_length(ids, eos_ids=eos, stops=stops, tokenizer=tokenizer)
                out.append((ids[:k], int(n_inputs[i])))
    finally:
        tokenizer.padding_side = orig_side
    return out


def truncate_new_ids(
    new_ids: list[int], *, budget: int, thinking_mode: bool, tokenizer,
    thinking_stop_string: str = "</think>", eos_ids: tuple[int, ...] | None = None,
) -> tuple[str, int, str]:
    """The (trace, n_new, finish_reason) a run at ``max_new_tokens=budget``
    would have produced, derived from a longer greedy trajectory.

    ``finish_reason`` is ``"length"`` if the chain hit the budget (``length``
    outranks the others when a chain ends exactly at the budget), ``"eos"`` if
    it ended on an end-of-sequence id, ``"stop"`` if it ended on the native
    thinking close, ``"unknown"`` otherwise.
    """
    ids = new_ids[:budget]
    n_new = len(ids)
    trace = tokenizer.decode(ids, skip_special_tokens=True)
    stops: tuple[str, ...] = (thinking_stop_string,) if thinking_mode else ()
    eos = tuple(eos_ids) if eos_ids else (tokenizer.eos_token_id,)
    if n_new >= budget:
        finish = "length"
    elif ids and ids[-1] in eos:
        finish = "eos"
    # Stop detection on the un-stripped decode so a SPECIAL stop token (gpt-oss
    # ``<|end|>``) is seen.
    elif stops and any(tokenizer.decode(ids, skip_special_tokens=False).rstrip().endswith(s)
                       for s in stops):
        finish = "stop"
    else:
        finish = "unknown"
    return trace, n_new, finish
