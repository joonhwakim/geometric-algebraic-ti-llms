"""One (experiment test, item set, model) job: generate the trial sets, score, write JSONL.

Two runners, one per kind of test:

- :func:`run_direct` -- the direct tests (IWL and ICL tests, content and
  uninformative contexts, the item-spacing sets). For every feasible
  ``(N, hop)`` trial set, context mode and assignment it generates the trials,
  renders the direct prompt, scores True/False at the answer position and
  writes ``N=<N>__hop=<hop>__surface=<mode>[__assign=<assignment>].jsonl``.
- :func:`run_ladder` -- the reasoning-budget ladders. Every trial is scored
  twice: a ``none`` row (the direct prompt; byte-identical to the direct test's
  row for that trial) and a ``cot`` row per budget. The trace is generated
  ONCE per trial at the largest budget and every smaller budget is the
  token-level prefix of that trajectory (exact under greedy decoding); the
  output files are ``b<budget>__N=<N>__hop=<hop>__surface=<mode>[__assign=
  <assignment>].jsonl``.

Both take the plain config dicts the published runs were declared with
(``geometry``: ``master_seed``, ``prompt``, ``grid``, ``all_pairs``,
``n_premise_orders``, ``assignment_strategies``, ``surface_form_modes``;
``ladder``: ``defaults`` + ``budgets``), which :mod:`geo_alg_ti_llms.experiments`
builds from an experiment declaration.
"""

from __future__ import annotations

import datetime as dt
import time
from pathlib import Path

from geo_alg_ti_llms.items import DEFAULT_ITEMS_DIR, ItemSet, load_item_set
from geo_alg_ti_llms.models import ModelConfig, load_model_and_tokenizer
from geo_alg_ti_llms.prompts import (
    ANSWER_CUES,
    CotSweepSpec,
    GridPromptSpec,
    build_grid_user_text,
    build_reasoning_messages,
    build_reasoning_user_body,
    format_answer_pass,
)
from geo_alg_ti_llms.results import TrialResult, write_results
from geo_alg_ti_llms.scoring import (
    CandidateIds,
    apply_chat_template_cot,
    build_direct_prompt,
    eos_token_ids,
    generate_new_token_ids,
    resolve_family_candidates,
    resolve_forced_choice_candidates,
    score_yes_no,
    score_yes_no_ids,
    answer_pass_input,
    truncate_new_ids,
)
from geo_alg_ti_llms.trials import SurfaceFormMode, Trial, generate_dataset_constrained_hop


# ---------------------------------------------------------------------------
# Config dicts
# ---------------------------------------------------------------------------


def load_geometry(geo: dict) -> dict:
    """Validate a geometry config dict and return its normalized form.

    Returns a dict with: master_seed, prompt (dict), N_values, hop_values,
    all_pairs, n_premise_orders, surface_form_modes, assignment_strategies.
    """
    required = {"master_seed", "prompt", "grid", "surface_form_modes",
                "n_premise_orders"}
    missing = required - set(geo)
    if missing:
        raise ValueError(f"geometry config: missing {sorted(missing)}")
    grid = geo["grid"]
    if "N_values" not in grid or "hop_values" not in grid:
        raise ValueError("geometry config: grid must declare N_values and hop_values")
    assignment_strategies = geo.get("assignment_strategies", ["random"])
    if not isinstance(assignment_strategies, list) or not assignment_strategies:
        raise ValueError(
            "geometry config: 'assignment_strategies' must be a non-empty list "
            "(e.g. [random, natural, reversed]); omit it for the random default."
        )
    return {
        "master_seed": int(geo["master_seed"]),
        "prompt": geo["prompt"],
        "N_values": [int(x) for x in grid["N_values"]],
        "hop_values": [int(x) for x in grid["hop_values"]],
        "all_pairs": bool(geo.get("all_pairs", True)),
        "n_premise_orders": int(geo["n_premise_orders"]),
        "surface_form_modes": list(geo["surface_form_modes"]),
        "assignment_strategies": [str(a) for a in assignment_strategies],
    }


def load_ladder_budgets(ladder: dict) -> dict[str, dict]:
    """The ladder's budgets with its ``defaults`` merged into each."""
    defaults = ladder.get("defaults", {})
    required = {"trigger_id", "max_new_tokens"}
    trial_sets: dict[str, dict] = {}
    for cid, cspec in ladder["budgets"].items():
        merged = {**defaults, **(cspec or {})}
        merged.setdefault("answer_family", None)
        merged.setdefault("instructions_v1_id", None)
        missing = required - set(merged)
        if missing:
            raise ValueError(f"budget {cid!r}: missing required keys {sorted(missing)}")
        trial_sets[cid] = merged
    return trial_sets


def _valid_grid(
    N_values: list[int], hop_values: list[int], set_size: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Cross product of (N, hop), partitioned into feasible vs skipped.

    Feasible iff ``hop + 3 <= N <= set_size`` (the non-terminal constraint
    and the item set capacity). Returns (kept, skipped), each sorted by (N, hop).
    """
    kept, skipped = [], []
    for N in N_values:
        for hop in hop_values:
            if hop + 3 <= N <= set_size:
                kept.append((N, hop))
            else:
                skipped.append((N, hop))
    kept.sort()
    skipped.sort()
    return kept, skipped


def _trials_per_set(N: int, hop: int, n_orders: int, all_pairs: bool) -> int:
    """Number of test trials at (N, hop): all_pairs counting."""
    return (2 * (N - 2 - hop) * n_orders) if all_pairs else (2 * n_orders)


def _make_base_spec(prompt_cfg: dict, instructions_override: str | None) -> GridPromptSpec:
    return GridPromptSpec(
        template_id=prompt_cfg["template_id"],
        instructions_v1_id=(instructions_override or prompt_cfg["instructions_v1_id"]),
        query_wrapper_id=prompt_cfg["query_wrapper_id"],
        separator=prompt_cfg["separator"],
        answer_family=prompt_cfg.get("answer_family"),
    )


def _candidates_for_trial(
    family: str, static: CandidateIds | None, trial: Trial, tokenizer
) -> CandidateIds:
    """Per-trial answer candidates: forced-choice item names vary; true_false
    reuses the run-level static set."""
    if family == "item_names":
        a_ids, b_ids = resolve_forced_choice_candidates(
            tokenizer, trial.query.a, trial.query.b,
        )
        if not a_ids or not b_ids:
            raise ValueError(
                f"forced-choice resolution failed for {trial.query.a!r}/{trial.query.b!r}"
            )
        return CandidateIds(
            yes=a_ids, no=b_ids,
            yes_strs=(trial.query.a,), no_strs=(trial.query.b,),
        )
    assert static is not None
    return static


def _report_grid(
    kept: list[tuple[int, int]],
    skipped: list[tuple[int, int]],
    set_size: int,
    n_orders: int,
    all_pairs: bool,
    n_surfaces: int,
    n_assignments: int = 1,
) -> None:
    """Print the feasible trial set grid + projected trial budget (every surface form and rank
    assignment of each (N, hop) is a trial set of its own)."""
    print(f"\n=== (N, hop) grid (set_size={set_size}) ===")
    per_cell = (lambda N, hop: 2 * (N - 2 - hop) * n_orders) if all_pairs \
        else (lambda N, hop: 2 * n_orders)
    total = 0
    by_N: dict[int, list[int]] = {}
    for (N, hop) in kept:
        by_N.setdefault(N, []).append(hop)
        total += per_cell(N, hop) * n_surfaces * n_assignments
    for N in sorted(by_N):
        hops = by_N[N]
        per = [per_cell(N, h) for h in hops]
        print(f"  N={N:>2d}: hops={hops}  "
              f"trials/trial set/surface={per}")
    if skipped:
        print(f"  skipped (infeasible hop+3>N or N>set_size): {skipped}")
    print(f"feasible trial sets: {len(kept)}  × {n_surfaces} surface(s)  "
          f"× {n_assignments} assignment(s)  → projected total trials: {total}")


# ---------------------------------------------------------------------------
# Direct tests
# ---------------------------------------------------------------------------


def run_direct(
    geometry: dict,
    model_cfg: ModelConfig,
    output_dir: Path,
    *,
    dry_run: bool = False,
    items_dir: Path = DEFAULT_ITEMS_DIR,
) -> int:
    """Score every trial set of a direct test; one JSONL per trial set under ``output_dir``."""
    cfg = geometry
    prompt_cfg = cfg["prompt"]
    master_seed = int(cfg["master_seed"])
    n_orders = int(cfg["n_premise_orders"])
    all_pairs = bool(cfg.get("all_pairs", True))
    surface_modes: list[SurfaceFormMode] = list(cfg["surface_form_modes"])
    N_values = [int(x) for x in cfg["grid"]["N_values"]]
    hop_values = [int(x) for x in cfg["grid"]["hop_values"]]

    assignment_strategies = cfg.get("assignment_strategies", ["random"])
    if not isinstance(assignment_strategies, list) or not assignment_strategies:
        raise ValueError("assignment_strategies must be a non-empty list")

    spec = GridPromptSpec(
        template_id=prompt_cfg["template_id"],
        instructions_v1_id=prompt_cfg["instructions_v1_id"],
        query_wrapper_id=prompt_cfg["query_wrapper_id"],
        separator=prompt_cfg["separator"],
        answer_family=prompt_cfg.get("answer_family"),
    )
    family = spec.resolved_answer_family
    set_name = prompt_cfg["item_pool"]
    print(f"prompt spec: {spec}")
    print(f"answer family: {family}")
    print(f"item set: {set_name}")
    print(f"grid N_values={N_values}  hop_values={hop_values}")
    print(f"all_pairs={all_pairs}  n_premise_orders={n_orders}  "
          f"surface_form_modes={surface_modes}")
    print(f"assignment_strategies={assignment_strategies}")

    item_set = load_item_set(set_name, items_dir)
    set_size = len(item_set.items)
    print(f"item set {set_name}: {set_size} items")

    # Dry-run path: feasibility report without loading a model.
    if dry_run:
        kept, skipped = _valid_grid(N_values, hop_values, set_size)
        _report_grid(kept, skipped, set_size, n_orders, all_pairs,
                     len(surface_modes), len(assignment_strategies))
        return 0

    use_chat_template = model_cfg.chat_template
    print(f"loading model: {model_cfg.name}")
    model, tokenizer = load_model_and_tokenizer(model_cfg)

    if any(a in ("natural", "reversed") for a in assignment_strategies) \
            and not item_set.natural_order:
        raise ValueError(
            f"assignment_strategies={assignment_strategies} include "
            f"natural/reversed but item set {set_name!r} is order-meaningless "
            "(natural_order=False)."
        )

    kept, skipped = _valid_grid(N_values, hop_values, set_size)
    _report_grid(kept, skipped, set_size, n_orders, all_pairs,
                 len(surface_modes), len(assignment_strategies))
    if not kept:
        raise ValueError("no feasible (N, hop) trial sets for this grid + item set size")

    # Resolve answer candidates once per run (static families) or per-trial
    # (item_names).
    static_candidates: CandidateIds | None = resolve_family_candidates(family, tokenizer)
    if static_candidates is not None:
        print(f"candidates ({family}): "
              f"yes={list(static_candidates.yes)} ({list(static_candidates.yes_strs)}); "
              f"no={list(static_candidates.no)} ({list(static_candidates.no_strs)})")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grand_t0 = time.perf_counter()
    total_trials = 0
    set_lines: list[str] = []

    for (N, hop) in kept:
        for surface_mode in surface_modes:
            for assignment in assignment_strategies:
                trials = generate_dataset_constrained_hop(
                    N=N, item_set=item_set, hop=hop,
                    n_premise_orders=n_orders,
                    master_seed=master_seed,
                    surface_form_mode=surface_mode,
                    all_pairs=all_pairs,
                    assignment_strategy=assignment,
                )
                cands_list = [
                    _candidates_for_trial(family, static_candidates, t, tokenizer)
                    for t in trials
                ]
                n_pairs = N - 2 - hop
                set_tag = f"N={N}__hop={hop}__surface={surface_mode}"
                if assignment != "random":
                    set_tag += f"__assign={assignment}"
                print(f"\n--- {set_tag} ---")
                print(f"  N={N} hop={hop}  valid non-terminal pairs={n_pairs}  "
                      f"trials={len(trials)}")

                results: list[TrialResult] = []
                t0 = time.perf_counter()

                # Pass 1: render every prompt.
                prompts = [
                    build_direct_prompt(t, spec, tokenizer, model_cfg, use_chat_template)
                    for t in trials
                ]
                # Pass 2: score (one full forward per trial).
                scores = [
                    score_yes_no(model, tokenizer, p, c,
                                 add_special_tokens=not use_chat_template)
                    for p, c in zip(prompts, cands_list)
                ]

                # Pass 3: assemble results.
                for trial, prompt, (lp_yes, lp_no, n_tokens) in zip(
                    trials, prompts, scores
                ):
                    predicted = lp_yes > lp_no
                    is_correct = predicted == trial.query.correct

                    results.append(TrialResult(
                        trial_id=trial.trial_id,
                        cell_id=trial.cell_id,
                        master_seed=trial.master_seed,
                        N=trial.N,
                        item_pool=trial.item_pool,
                        assignment_strategy=trial.assignment_strategy,
                        signed_distance=trial.query.signed_distance,
                        symbolic_distance=trial.query.symbolic_distance,
                        correct=trial.query.correct,
                        model=model_cfg.name,
                        prompt_spec={
                            "template_id": spec.template_id,
                            "instructions_v1_id": spec.instructions_v1_id,
                            "query_wrapper_id": spec.query_wrapper_id,
                            "separator_repr": repr(spec.separator),
                            "answer_family": family,
                            "phase": "nxhop",
                            "N": N,
                            "hop": hop,
                            "surface_form_mode": surface_mode,
                            "assignment_strategy": assignment,
                        },
                        mode="direct",
                        predicted=predicted,
                        is_correct=is_correct,
                        logprob_yes=lp_yes,
                        logprob_no=lp_no,
                        logit_gap=lp_yes - lp_no,
                        prompt=prompt,
                        prompt_n_tokens=n_tokens,
                        timestamp_iso=dt.datetime.utcnow().isoformat(),
                        duration_seconds=0.0,
                        extra={
                            "phase": "nxhop",
                            "N": N,
                            "hop": hop,
                            "permutation_idx": trial.extra.get("permutation_idx"),
                            "rank_high": trial.extra.get("rank_high"),
                            "rank_low": trial.extra.get("rank_low"),
                            "surface_form_mode": surface_mode,
                            "assignment_strategy": assignment,
                            "answer_cue": None,
                            "actual_items": list(item_set.items),
                            "rank_to_item": list(trial.rank_to_item),
                        },
                    ))

                elapsed = time.perf_counter() - t0
                accuracy = sum(r.is_correct for r in results) / max(1, len(results))
                acc_yes = (
                    sum(r.is_correct for r in results if r.correct) /
                    max(1, sum(r.correct for r in results))
                )
                acc_no = (
                    sum(r.is_correct for r in results if not r.correct) /
                    max(1, sum(not r.correct for r in results))
                )
                out_path = output_dir / f"{set_tag}.jsonl"
                n_written = write_results(out_path, results)
                line = (
                    f"{set_tag}  acc={accuracy:.3f}  "
                    f"yes={acc_yes:.3f}  no={acc_no:.3f}  "
                    f"bias={abs(acc_yes - acc_no):.3f}  "
                    f"n={n_written}  elapsed={elapsed:.1f}s"
                )
                print(f"  -> {line}")
                set_lines.append(line)
                total_trials += n_written

    print(f"\ndone. total trials: {total_trials}  "
          f"wall: {time.perf_counter() - grand_t0:.1f}s")
    print("\n=== per trial set summary ===")
    for ln in set_lines:
        print(f"  {ln}")
    return 0


# ---------------------------------------------------------------------------
# Reasoning-budget ladders
# ---------------------------------------------------------------------------


def _result_row(
    *, trial: Trial, model_name: str, base_prompt_spec_dict: dict, mode: str,
    lp_yes: float, lp_no: float, scored_prompt: str, scored_ntok: int,
    cot_trace, cot_n_tokens, cot_finish, reasoning_prompt, reasoning_prompt_ntok,
    extra: dict,
) -> TrialResult:
    predicted = lp_yes > lp_no
    return TrialResult(
        trial_id=trial.trial_id, cell_id=trial.cell_id,
        master_seed=trial.master_seed, N=trial.N, item_pool=trial.item_pool,
        assignment_strategy=trial.assignment_strategy,
        signed_distance=trial.query.signed_distance,
        symbolic_distance=trial.query.symbolic_distance,
        correct=trial.query.correct, model=model_name,
        prompt_spec=base_prompt_spec_dict, mode=mode,
        predicted=predicted, is_correct=(predicted == trial.query.correct),
        logprob_yes=lp_yes, logprob_no=lp_no, logit_gap=lp_yes - lp_no,
        cot_trace=cot_trace, cot_n_tokens=cot_n_tokens, cot_finish_reason=cot_finish,
        prompt=scored_prompt, prompt_n_tokens=scored_ntok,
        cot_stage1_prompt=reasoning_prompt, cot_stage1_n_tokens=reasoning_prompt_ntok,
        timestamp_iso=dt.datetime.utcnow().isoformat(), duration_seconds=0.0,
        extra=extra,
    )


def _ladder_budgets_compatible(trial_sets: dict[str, dict]) -> list[str]:
    """Trial sets of a ladder must differ ONLY in ``max_new_tokens``.

    Returns the budget ids sorted by budget; raises otherwise.
    """
    ref = None
    for cid, trial_set in trial_sets.items():
        rest = {k: v for k, v in trial_set.items() if k != "max_new_tokens"}
        if ref is None:
            ref = rest
        elif rest != ref:
            diff = sorted(k for k in set(rest) | set(ref) if rest.get(k) != ref.get(k))
            raise ValueError(
                f"ladder: trial set {cid!r} differs from the first trial set in {diff}; "
                "a ladder only applies to trial sets that differ in max_new_tokens alone"
            )
    return sorted(trial_sets, key=lambda c: int(trial_sets[c]["max_new_tokens"]))


def _score_ladder_set(
    *, trial_sets: dict[str, dict], base_spec: GridPromptSpec, cot_spec: CotSweepSpec,
    family: str, N: int, hop: int, surface_mode: SurfaceFormMode,
    assignment_strategy: str,
    all_pairs: bool, n_orders: int, test_set: ItemSet,
    master_seed: int, model, tokenizer, model_cfg, static_candidates,
    use_chat_template: bool, output_dir: Path, gen_batch_size: int,
) -> list[str]:
    """Ladder scoring of one (N, hop, surface, assignment) sub-trial set.

    The direct ``none`` row and every answer-pass score are single-sequence
    forward passes; the trace is generated ONCE per trial at the largest
    budget and every smaller budget is the token-level prefix of that
    trajectory (see ``generate_new_token_ids`` in scoring.py for why this is
    exact under greedy decoding).

    Returns one log line per budget.
    """
    add_special = not use_chat_template
    budget_ids = _ladder_budgets_compatible(trial_sets)
    budgets = {cid: int(trial_sets[cid]["max_new_tokens"]) for cid in budget_ids}
    max_cid = budget_ids[-1]
    max_budget = budgets[max_cid]

    test_trials = generate_dataset_constrained_hop(
        N=N, item_set=test_set, hop=hop, n_premise_orders=n_orders,
        master_seed=master_seed, surface_form_mode=surface_mode, all_pairs=all_pairs,
        assignment_strategy=assignment_strategy,
    )
    common_spec = {
        "template_id": base_spec.template_id,
        "instructions_v1_id": base_spec.instructions_v1_id,
        "query_wrapper_id": base_spec.query_wrapper_id,
        "separator_repr": repr(base_spec.separator),
        "answer_family": family,
        "phase": "CoT", "context_N": N, "hop": hop,
        "surface_form_mode": surface_mode,
        "assignment_strategy": assignment_strategy,
    }
    # The exemplar axes of the published rows; every run used zero exemplars
    # (the exemplar item set recorded was the test item set).
    cot_axes_base = {
        "n_exemplars": 0, "trigger_id": cot_spec.trigger_id,
        "reproduce": "copy", "direction": "descending",
        "query_position": "after",
        "novel_item_exemplars": False,
        "exemplar_pool": test_set.name, "exemplar_N": N,
        "exemplar_hop": hop,
    }
    thinking = bool(getattr(model_cfg, "thinking_mode", False))
    provenance = {"generated_at_budget": max_budget, "gen_batch_size": gen_batch_size,
                  "stage2": "text-if-faithful-else-ids", "eos_ids": None}

    t0 = time.perf_counter()
    # --- pass 1: direct 'none' score per trial (single forward) ---
    none_rows: list[tuple[Trial, CandidateIds, float, float, str, int]] = []
    reasoning_prompts: list[str] = []
    eos_ids = eos_token_ids(model, tokenizer)
    cue = ANSWER_CUES["extract"]
    for trial in test_trials:
        cands = _candidates_for_trial(family, static_candidates, trial, tokenizer)
        none_prompt = build_direct_prompt(trial, base_spec, tokenizer, model_cfg, use_chat_template)
        if use_chat_template:
            reasoning_prompt = apply_chat_template_cot(
                tokenizer, build_reasoning_messages(trial, cot_spec),
                add_generation_prompt=True,
                enable_thinking=(True if thinking else None),
            )
        else:
            none_prompt = build_grid_user_text(trial, base_spec)
            reasoning_prompt = build_reasoning_user_body(trial, cot_spec)
        lp_y, lp_n, ntok = score_yes_no(
            model, tokenizer, none_prompt, cands, add_special_tokens=add_special,
        )
        none_rows.append((trial, cands, lp_y, lp_n, none_prompt, ntok))
        reasoning_prompts.append(reasoning_prompt)
    t_none = time.perf_counter() - t0

    # --- pass 2: ONE greedy trajectory per trial at the largest budget ---
    _t = time.perf_counter()
    provenance["eos_ids"] = list(eos_ids)
    trajectories = generate_new_token_ids(
        model, tokenizer, reasoning_prompts, max_new_tokens=max_budget,
        thinking_mode=thinking, add_special_tokens=add_special,
        batch_size=gen_batch_size, thinking_stop_string=model_cfg.thinking_stop_string,
        eos_ids=eos_ids,
    )
    reasoning_prompt_ids = [tokenizer.encode(s1, add_special_tokens=add_special) for s1 in reasoning_prompts]
    for (_t_, _c_, _y_, _n_, _p_, _k_), s1_ids, (_ids_, n_in) in zip(none_rows, reasoning_prompt_ids, trajectories, strict=True):
        if len(s1_ids) != n_in:
            raise RuntimeError(f"reasoning-pass re-encode ({len(s1_ids)} ids) != generation input ({n_in})")
    t_gen = time.perf_counter() - _t

    # --- pass 3: every budget = prefix of the trajectory, answer-pass scored ---
    lines: list[str] = []
    t_score = 0.0
    for cid in budget_ids:
        b = budgets[cid]
        results: list[TrialResult] = []
        for (trial, cands, lp_y0, lp_n0, none_prompt, ntok0), reasoning_prompt, s1_ids, (new_ids, n_in) in zip(
            none_rows, reasoning_prompts, reasoning_prompt_ids, trajectories, strict=True,
        ):
            results.append(_result_row(
                trial=trial, model_name=model_cfg.name,
                base_prompt_spec_dict={"cot_cell_id": cid, **common_spec, "condition": "none"},
                mode="direct", lp_yes=lp_y0, lp_no=lp_n0,
                scored_prompt=none_prompt, scored_ntok=ntok0,
                cot_trace=None, cot_n_tokens=None, cot_finish=None,
                reasoning_prompt=None, reasoning_prompt_ntok=None,
                extra={"condition": "none", "context_N": N, "hop": hop,
                       "surface_form_mode": surface_mode,
                       "assignment_strategy": assignment_strategy,
                       "rank_to_item": list(trial.rank_to_item)},
            ))
            trace, n_cot, finish = truncate_new_ids(
                new_ids, budget=b, thinking_mode=thinking, tokenizer=tokenizer,
                thinking_stop_string=model_cfg.thinking_stop_string, eos_ids=eos_ids,
            )
            # answer-pass: the rendered text when its re-tokenization is faithful to the generated
            # ids, else the ids themselves (scoring.answer_pass_input; 4 of the 15,408 stored
            # Qwen3-32B b1024 context-ladder rows took the ids path)
            answer_pass = format_answer_pass(reasoning_prompt, trace)
            answer_ids, stage2_source, answer_faithful = answer_pass_input(
                tokenizer, reasoning_prompt, s1_ids, new_ids[:b], answer_pass, eos_ids=eos_ids, cue=cue,
                add_special_tokens=add_special,
            )
            _t = time.perf_counter()
            if stage2_source == "text":
                lp_y, lp_n, answer_ntok = score_yes_no(model, tokenizer, answer_pass, cands, add_special_tokens=add_special)
            else:
                lp_y, lp_n, answer_ntok = score_yes_no_ids(model, answer_ids, cands)
                answer_pass = tokenizer.decode(answer_ids, skip_special_tokens=False)   # the record = what was scored
            t_score += time.perf_counter() - _t
            results.append(_result_row(
                trial=trial, model_name=model_cfg.name,
                base_prompt_spec_dict={"cot_cell_id": cid, **common_spec, **cot_axes_base,
                                       "max_new_tokens": b, "condition": "cot"},
                mode="cot", lp_yes=lp_y, lp_no=lp_n,
                scored_prompt=answer_pass, scored_ntok=answer_ntok,
                cot_trace=trace, cot_n_tokens=n_cot, cot_finish=finish,
                reasoning_prompt=reasoning_prompt, reasoning_prompt_ntok=n_in,
                extra={"condition": "cot", "context_N": N, "hop": hop,
                       "surface_form_mode": surface_mode,
                       "assignment_strategy": assignment_strategy,
                       "exemplar_items": list(test_set.items),
                       "rank_to_item": list(trial.rank_to_item),
                       "ladder_once": provenance,
                       # "text" = rendered text when its re-tokenization is faithful to the
                       # generated ids; "ids" = generated ids + cue otherwise (4 of the 15,408
                       # stored Qwen3-32B b1024 context-ladder rows took the ids path)
                       "stage2_source": stage2_source, "stage2_text_faithful": bool(answer_faithful)},
            ))
        none_r = [r for r in results if r.mode == "direct"]
        cot_r = [r for r in results if r.mode == "cot"]
        acc_none = sum(r.is_correct for r in none_r) / max(1, len(none_r))
        acc_cot = sum(r.is_correct for r in cot_r) / max(1, len(cot_r))
        n_trunc = sum(1 for r in cot_r if r.cot_finish_reason == "length")
        set_tag = f"{cid}__N={N}__hop={hop}__surface={surface_mode}"
        if assignment_strategy != "random":
            set_tag += f"__assign={assignment_strategy}"
        n_written = write_results(output_dir / f"{set_tag}.jsonl", results)
        line = (
            f"{set_tag}  acc_none={acc_none:.3f}  acc_cot={acc_cot:.3f}  "
            f"delta={acc_cot - acc_none:+.3f}  n_trials={len(none_r)}  "
            f"rows={n_written}  trunc={n_trunc}  "
            f"[ladder-once from b{max_budget}, batch={gen_batch_size}]"
        )
        print(f"  -> {line}")
        lines.append(line)
    elapsed = time.perf_counter() - t0
    print(f"  sub-trial set wall {elapsed:.1f}s  [none_score={t_none:.1f}s gen={t_gen:.1f}s "
          f"cot_score={t_score:.1f}s]  n_trials={len(test_trials)}  trial_sets={len(budget_ids)}")
    return lines


def run_ladder(
    geometry: dict,
    ladder: dict,
    model_cfg: ModelConfig,
    output_dir: Path,
    *,
    gen_batch_size: int = 1,
    budget_ids: list[str] | None = None,
    dry_run: bool = False,
    items_dir: Path = DEFAULT_ITEMS_DIR,
) -> int:
    """Score a reasoning-budget ladder; one JSONL per (budget, trial set) under ``output_dir``.

    ``ladder`` carries ``defaults`` and ``budgets`` (``b32`` .. ``b1024``, each a
    ``max_new_tokens``); ``budget_ids`` restricts the budgets (default: all).
    """
    geo = load_geometry(geometry)
    master_seed = geo["master_seed"]
    all_budgets = load_ladder_budgets(ladder)
    if budget_ids is None:
        budget_ids = list(all_budgets.keys())
    else:
        missing = [c for c in budget_ids if c not in all_budgets]
        if missing:
            raise ValueError(f"budget id(s) not in config: {missing}")

    set_name = geo["prompt"]["item_pool"]
    test_set = load_item_set(set_name, items_dir)
    pairs = [
        (N, hop)
        for N in sorted(set(geo["N_values"]))
        for hop in sorted(set(geo["hop_values"]))
        if hop + 3 <= N <= test_set.size
    ]

    # --- dry-run: project the grid without a model ---
    if dry_run:
        n_surf = len(geo["surface_form_modes"])
        n_assign = len(geo["assignment_strategies"])
        per_set_trials = sum(
            _trials_per_set(N, hop, geo["n_premise_orders"], geo["all_pairs"])
            for (N, hop) in pairs
        ) * n_surf * n_assign
        print(f"  prompt={geo['prompt']}")
        print(f"  item_set={set_name} (size {test_set.size})  all_pairs={geo['all_pairs']}  "
              f"n_orders={geo['n_premise_orders']}  surfaces={geo['surface_form_modes']}  "
              f"assignments={geo['assignment_strategies']}")
        print(f"  valid (N,hop) [{len(pairs)}]: {pairs}")
        print(f"\nper-trial set TEST trials (× each budget): {per_set_trials}")
        print(f"budgets: {budget_ids} ({len(budget_ids)})")
        print(f"TOTAL test trials: {per_set_trials * len(budget_ids)}  "
              f"(×2 scored rows each: none + cot)")
        return 0

    use_chat_template = model_cfg.chat_template
    print(f"loading model: {model_cfg.name}  chat_template={use_chat_template}")
    model, tokenizer = load_model_and_tokenizer(model_cfg)

    if not pairs:
        raise ValueError("no feasible (N, hop) trial sets for this grid + item set size")
    # Congruency axis: natural/reversed need an order-meaningful item set.
    needs_natural = any(a in ("natural", "reversed")
                        for a in geo["assignment_strategies"])
    if needs_natural and not test_set.natural_order:
        raise ValueError(
            f"assignment_strategies={geo['assignment_strategies']} include "
            f"natural/reversed but item set {set_name!r} is order-meaningless "
            "(natural_order=False)."
        )
    print(f"item set {set_name}: {test_set.size} items "
          f"(natural_order={test_set.natural_order})")
    print(f"valid (N,hop) [{len(pairs)}]: {pairs}")
    print(f"surfaces={geo['surface_form_modes']}  "
          f"assignments={geo['assignment_strategies']}  "
          f"all_pairs={geo['all_pairs']}  n_orders={geo['n_premise_orders']}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grand_t0 = time.perf_counter()
    all_lines: list[str] = []

    trial_sets = {cid: all_budgets[cid] for cid in budget_ids}
    _ladder_budgets_compatible(trial_sets)
    ref = trial_sets[budget_ids[0]]
    base_spec = _make_base_spec(geo["prompt"], ref.get("instructions_v1_id"))
    cot_spec = CotSweepSpec(base=base_spec, trigger_id=ref["trigger_id"])
    family = base_spec.resolved_answer_family
    candidates = resolve_family_candidates(family, tokenizer)
    if not cot_spec.is_cot and not getattr(model_cfg, "thinking_mode", False):
        print("WARNING: trigger_id=none on a NON-thinking model: the 'cot' rows "
              "are UNTRIGGERED free generation.", flush=True)
    print(f"\n=== ladder over trial sets {budget_ids}  gen_batch_size={gen_batch_size} ===")
    print(f"  base={base_spec}")
    print(f"  cot={cot_spec}  family={family}")
    for (N, hop) in pairs:
        for surface_mode in geo["surface_form_modes"]:
            for assignment in geo["assignment_strategies"]:
                print(f"--- ladder :: N={N} hop={hop} surface={surface_mode} "
                      f"assign={assignment} ---")
                all_lines.extend(_score_ladder_set(
                    trial_sets=trial_sets, base_spec=base_spec, cot_spec=cot_spec,
                    family=family, N=N, hop=hop, surface_mode=surface_mode,
                    assignment_strategy=assignment,
                    all_pairs=geo["all_pairs"], n_orders=geo["n_premise_orders"],
                    test_set=test_set,
                    master_seed=master_seed, model=model, tokenizer=tokenizer,
                    model_cfg=model_cfg, static_candidates=candidates,
                    use_chat_template=use_chat_template, output_dir=output_dir,
                    gen_batch_size=gen_batch_size,
                ))

    print(f"\ndone. wall: {time.perf_counter() - grand_t0:.1f}s")
    print("\n=== per-set summary (acc_none / acc_cot / delta) ===")
    for ln in all_lines:
        print(f"  {ln}")
    return 0


# ---------------------------------------------------------------------------
# Prompt-only replay (no model): what the runners would render
# ---------------------------------------------------------------------------


def direct_trial_sets(geometry: dict, items_dir: Path = DEFAULT_ITEMS_DIR):
    """Yield ``(set_tag, spec, item_set, trials)`` for every trial set a direct test scores,
    in the runner's order, without loading a model."""
    geo = load_geometry(geometry)
    item_set = load_item_set(geo["prompt"]["item_pool"], items_dir)
    spec = _make_base_spec(geo["prompt"], None)
    kept, _ = _valid_grid(geo["N_values"], geo["hop_values"], item_set.size)
    for (N, hop) in kept:
        for surface_mode in geo["surface_form_modes"]:
            for assignment in geo["assignment_strategies"]:
                trials = generate_dataset_constrained_hop(
                    N=N, item_set=item_set, hop=hop,
                    n_premise_orders=geo["n_premise_orders"],
                    master_seed=geo["master_seed"],
                    surface_form_mode=surface_mode,
                    all_pairs=geo["all_pairs"],
                    assignment_strategy=assignment,
                )
                set_tag = f"N={N}__hop={hop}__surface={surface_mode}"
                if assignment != "random":
                    set_tag += f"__assign={assignment}"
                yield set_tag, spec, item_set, trials


def ladder_trial_sets(geometry: dict, ladder: dict, items_dir: Path = DEFAULT_ITEMS_DIR):
    """Yield ``(set_tag_suffix, base_spec, cot_spec, item_set, trials)`` for every
    sub-trial set a ladder scores (the file name is ``b<budget>__<suffix>.jsonl``)."""
    geo = load_geometry(geometry)
    trial_sets = load_ladder_budgets(ladder)
    budget_ids = _ladder_budgets_compatible(trial_sets)
    ref = trial_sets[budget_ids[0]]
    item_set = load_item_set(geo["prompt"]["item_pool"], items_dir)
    base_spec = _make_base_spec(geo["prompt"], ref.get("instructions_v1_id"))
    cot_spec = CotSweepSpec(base=base_spec, trigger_id=ref["trigger_id"])
    pairs = [
        (N, hop)
        for N in sorted(set(geo["N_values"]))
        for hop in sorted(set(geo["hop_values"]))
        if hop + 3 <= N <= item_set.size
    ]
    for (N, hop) in pairs:
        for surface_mode in geo["surface_form_modes"]:
            for assignment in geo["assignment_strategies"]:
                trials = generate_dataset_constrained_hop(
                    N=N, item_set=item_set, hop=hop,
                    n_premise_orders=geo["n_premise_orders"],
                    master_seed=geo["master_seed"],
                    surface_form_mode=surface_mode,
                    all_pairs=geo["all_pairs"],
                    assignment_strategy=assignment,
                )
                suffix = f"N={N}__hop={hop}__surface={surface_mode}"
                if assignment != "random":
                    suffix += f"__assign={assignment}"
                yield suffix, base_spec, cot_spec, item_set, trials


__all__ = [
    "run_direct", "run_ladder", "direct_trial_sets", "ladder_trial_sets",
    "load_geometry", "load_ladder_budgets",
]
