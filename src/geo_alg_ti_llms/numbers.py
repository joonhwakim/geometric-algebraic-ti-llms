"""Every number the paper prints about the language models, recomputed from the tidy tables.

``numbers()`` returns one row per printed number: ``id``, ``site`` (where the paper prints it),
``printed``, ``computed``, ``source`` (the value of ``values()`` it comes from) and ``match``
(whether ``computed``, written at the printed precision, equals ``printed``). A claim the paper
states as a bound is printed as ``>= x`` or ``< x`` and holds when the value is on that side.
"""

from __future__ import annotations

import json
import math
import operator
import re
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction
from pathlib import Path

import pandas as pd

from geo_alg_ti_llms import analysis, items, roster, tables

RESULTS = "Results"
METHODS = "Methods, large language models"
EXTENDED_METHODS = "Extended methods, large language models"
EXTENDED_RESULTS = "Extended results, large language models"
SDE_FIGURE = "Symbolic distance effects (SDE) figure"
CONTENT_FIGURE = "Content effects in transitive inference figure"
MECHANISMS_FIGURE = "Mechanisms underlying the geometric and algebraic strategies figure"
SDE_CAPTION = SDE_FIGURE + " caption"
ITEM_TABLE = EXTENDED_METHODS + ", main LLM item sets table, spacing row"

# (id, site, printed, source)
PAPER_NUMBERS: tuple[tuple[str, str, str, str], ...] = (
    ("llm-1", METHODS, "22", "models"),
    ("llm-2", METHODS, "10", "native_models"),
    ("llm-3", METHODS, "12", "prompted_models"),
    ("llm-4", METHODS, "11", "items"),
    ("llm-5", METHODS, "10", "relations"),
    ("llm-6", METHODS, "10", "premise_orders"),
    ("llm-7", EXTENDED_METHODS, "42", "iwl_trials_per_set"),
    ("llm-8", EXTENDED_METHODS, "420", "icl_trials"),
    ("llm-9", EXTENDED_METHODS, "19", "spacing_sets"),
    ("llm-10", EXTENDED_METHODS, "2", "spacing_ratio_min"),
    ("llm-11", EXTENDED_METHODS, "56.2", "spacing_ratio_max"),
    ("llm-12", EXTENDED_METHODS, "378", "competence_trials"),
    ("llm-13", EXTENDED_METHODS, "Llama-3.2-1B, Llama-3.2-3B, Phi-4-reasoning, and Qwen3-1.7B",
     "not_competent"),
    ("llm-14", RESULTS + ", symbolic distance effects", "15", "slopes_lower_right"),
    ("llm-15", RESULTS + ", symbolic distance effects", "18", "slopes_drawn"),
    ("llm-16", RESULTS + ", symbolic distance effects", "18", "budget_upper_right"),
    ("llm-17", RESULTS + ", symbolic distance effects", "19", "budget_drawn"),
    ("llm-18", SDE_CAPTION, "15", "slopes_lower_right"),
    ("llm-19", SDE_CAPTION, "18", "slopes_drawn"),
    ("llm-20", SDE_CAPTION, "18", "budget_upper_right"),
    ("llm-21", SDE_CAPTION, "19", "budget_drawn"),
    ("llm-22", RESULTS + ", content effects", "81", "congruent_above_reversed"),
    ("llm-23", RESULTS + ", content effects", "110", "model_dimensions"),
    ("llm-24", RESULTS + ", content effects", "98", "recovery_positive"),
    ("llm-25", RESULTS + ", content effects", "110", "recovery_points"),
    ("llm-26", CONTENT_FIGURE + ", prior-strength panel", "-0.30", "prior_strength_rho"),
    ("llm-27", EXTENDED_RESULTS, "82", "iwl_d6_above_d1"),
    ("llm-28", EXTENDED_RESULTS, "110", "iwl_panels"),
    ("llm-29", EXTENDED_RESULTS, "15", "icl_d6_below_d1"),
    ("llm-30", EXTENDED_RESULTS, "22", "icl_panels"),
    ("llm-31", EXTENDED_RESULTS, "15", "icl_budget_rises"),
    ("llm-32", EXTENDED_RESULTS, "17", "icl_budget_compared"),
    ("llm-33", EXTENDED_RESULTS, "61", "iwl_budget_falls"),
    ("llm-34", EXTENDED_RESULTS, "75", "iwl_budget_compared"),
    ("llm-35", EXTENDED_RESULTS + ", reasoning requirements caption", "two", "city_iwl_trials_d6"),
    ("llm-36", EXTENDED_RESULTS, "81", "congruent_above_reversed"),
    ("llm-37", EXTENDED_RESULTS, "110", "model_dimensions"),
    ("llm-38", EXTENDED_RESULTS, "89", "recovered_10_points"),
    ("llm-39", EXTENDED_RESULTS, "110", "model_dimensions"),
    ("llm-40", EXTENDED_RESULTS, "98", "near_above_far_256"),
    ("llm-41", EXTENDED_RESULTS, "110", "model_dimensions"),
    ("llm-42", EXTENDED_RESULTS + ", recovery with reasoning caption", "120",
     "context_ladder_trials_d1"),
    ("llm-43", EXTENDED_RESULTS + ", recovery with reasoning caption", "20",
     "context_ladder_trials_d6"),
    ("llm-44", EXTENDED_RESULTS, "57", "reversed_budget_rises"),
    ("llm-45", EXTENDED_RESULTS, "66", "reversed_budget_compared"),
    ("llm-46", EXTENDED_RESULTS, "60", "shuffled_budget_rises"),
    ("llm-47", EXTENDED_RESULTS, "66", "shuffled_budget_compared"),
    ("llm-48", EXTENDED_RESULTS, "52", "congruent_budget_rises"),
    ("llm-49", EXTENDED_RESULTS, "89", "congruent_budget_compared"),
    ("llm-50", EXTENDED_RESULTS + ", reasoning traces", ">= 0.90", "composed_icl_qwen3_512"),
    ("llm-51", EXTENDED_RESULTS + ", reasoning traces", ">= 0.90", "composed_icl_qwen25_512"),
    ("llm-52", EXTENDED_RESULTS + ", reasoning traces", ">= 0.90", "composed_reversed_qwen3_512"),
    ("llm-53", EXTENDED_RESULTS + ", reasoning traces", "< 0.90", "composed_reversed_qwen25_512"),
    ("llm-54", EXTENDED_RESULTS + ", example traces", "seven", "qwen3_incorrect_trials"),
    ("llm-55", EXTENDED_RESULTS + ", example traces", "one", "qwen3_incorrect_finished"),
    ("llm-56", EXTENDED_RESULTS + ", example traces", "693", "trace_tokens_qwen3_correct"),
    ("llm-57", EXTENDED_RESULTS + ", example traces", "1,022", "trace_tokens_qwen3_incorrect"),
    ("llm-58", EXTENDED_RESULTS + ", example traces", "267", "trace_tokens_qwen25_correct"),
    ("llm-59", EXTENDED_RESULTS + ", example traces", "311", "trace_tokens_qwen25_incorrect"),
    ("llm-60", EXTENDED_METHODS + ", interventions", "64", "blocks"),
    ("llm-61", EXTENDED_METHODS + ", interventions", "252", "head_scoring_trials"),
    ("llm-62", EXTENDED_METHODS + ", interventions", "32", "heads_ablated"),
    ("llm-63", EXTENDED_METHODS + ", interventions", "64", "random_heads_outside_top"),
    ("llm-64", METHODS, "1", "parameters_min_b"),
    ("llm-65", METHODS, "70", "parameters_max_b"),
    ("llm-66", METHODS, "seven", "innermost_ranks"),
    ("llm-67", METHODS, "five", "dimensions"),
    ("llm-68", METHODS + ", mechanistic interventions", "4,096", "heads_total"),
    ("llm-69", EXTENDED_METHODS, "17.8", "ratio_d6_countries"),
    ("llm-70", EXTENDED_METHODS, "17.8", "ratio_d6_animals"),
    ("llm-71", EXTENDED_METHODS, "17.8", "ratio_d6_countryarea"),
    ("llm-72", EXTENDED_METHODS, "5.6", "ratio_d6_usstates"),
    ("llm-73", ITEM_TABLE, "1.62", "step_countries"),
    ("llm-74", ITEM_TABLE, "1.33", "step_usstates"),
    ("llm-75", ITEM_TABLE, "even", "step_citieslinear"),
    ("llm-76", ITEM_TABLE, "1.62", "step_animals"),
    ("llm-77", ITEM_TABLE, "1.62", "step_countryarea"),
    ("llm-78", EXTENDED_RESULTS + ", example traces", "four", "example_trace_distance"),
    ("llm-79", EXTENDED_METHODS + ", interventions", "16", "smaller_head_set"),
    ("llm-80", EXTENDED_METHODS + ", interventions", "US states", "impaired_by_smaller_head_set"),
    ("llm-81", EXTENDED_RESULTS + ", reasoning requirements under content conditions caption",
     "120", "context_ladder_trials_d1"),
    ("llm-82", EXTENDED_RESULTS + ", reasoning requirements under content conditions caption",
     "20", "context_ladder_trials_d6"),
    ("llm-83", SDE_FIGURE + ", reasoning-budget panel", "0.98", "icl_test_accuracy_largest_budget"),
    ("llm-84", SDE_FIGURE + ", reasoning-budget panel", "1.00", "iwl_test_accuracy_largest_budget"),
    ("llm-85", CONTENT_FIGURE + ", reasoning-budget panel", "1.00",
     "congruent_accuracy_largest_budget"),
    ("llm-86", CONTENT_FIGURE + ", reasoning-budget panel", "0.92",
     "reversed_accuracy_largest_budget"),
    ("llm-87", CONTENT_FIGURE + ", reasoning-budget panel", "0.95",
     "shuffled_accuracy_largest_budget"),
    ("llm-88", METHODS + ", mechanistic interventions", "32", "heads_ablated"),
    ("llm-89", RESULTS + ", independent mechanisms underlie the geometric and algebraic strategies",
     "32", "heads_ablated"),
    ("llm-90", MECHANISMS_FIGURE + " caption", "32", "heads_ablated"),
    ("llm-91", EXTENDED_METHODS, "22", "models"),
    ("llm-92", EXTENDED_METHODS, "1", "parameters_min_b"),
    ("llm-93", EXTENDED_METHODS, "70", "parameters_max_b"),
    ("llm-94", EXTENDED_METHODS, "Ten", "native_models"),
    ("llm-95", EXTENDED_METHODS, "12", "prompted_models"),
    ("llm-96", EXTENDED_METHODS + ", interventions", "4,096", "heads_total"),
    ("llm-97", EXTENDED_METHODS + ", IWL and ICL tests", "10", "premise_orders"),
)


def values(results_root: Path = analysis.RESULTS_ROOT) -> dict[str, object]:
    """Every source value, keyed as ``PAPER_NUMBERS`` names them."""
    direct = analysis.direct_trials(results_root)
    ladder = analysis.ladder_trials(results_root)
    out: dict[str, object] = {}
    kinds = direct.drop_duplicates("model").set_index("model")["reasoning_kind"].astype(str)
    out.update(models=len(kinds), native_models=int((kinds == "native").sum()),
               prompted_models=int((kinds == "prompted").sum()),
               parameters_min_b=min(roster.PARAMETERS_B[m] for m in kinds.index),
               parameters_max_b=max(roster.PARAMETERS_B[m] for m in kinds.index))
    icl = direct[(direct["model"] == "qwen3-32b") & (direct["condition"] == roster.ICL_CONDITION)]
    # The geometry every exported row was checked against (the two ends are never queried).
    out["items"] = tables.N_ITEMS
    out["relations"] = out["items"] - 1
    out["premise_orders"] = _premise_orders(results_root)
    out["innermost_ranks"] = tables.INNERMOST_RANKS
    out["dimensions"] = len(roster.DIMENSIONS)
    out.update(_spacing())
    inner = icl[icl["innermost"]]
    out["icl_trials"] = len(inner)
    iwl = direct[(direct["model"] == "qwen3-32b") & (direct["condition"] == roster.IWL_CONDITION)
                 & direct["innermost"]]
    per_set = iwl.groupby("dim", observed=True).size()
    out["iwl_trials_per_set"] = int(per_set.iloc[0]) if per_set.nunique() == 1 else math.nan
    item_sets = pd.read_parquet(Path(results_root) / "prior_strength" / "tables" / "pools.parquet")
    out.update(spacing_sets=len(item_sets), spacing_ratio_min=float(item_sets["value_ratio"].min()),
               spacing_ratio_max=float(item_sets["value_ratio"].max()))

    competence = analysis.competence(direct)
    out["competence_trials"] = int(competence["n"].iloc[0]) if competence["n"].nunique() == 1 else math.nan
    not_competent = competence.loc[~competence["competent"], "model"]
    out["not_competent"] = _name_list(sorted(roster.MODEL_LABELS[m] for m in not_competent))
    competent = competence.loc[competence["competent"], "model"]
    slopes = analysis.distance_slope_points(direct)
    slopes = slopes[slopes["model"].isin(competent)]
    out.update(slopes_drawn=len(slopes),
               slopes_lower_right=analysis.quadrant_counts(slopes["x"], slopes["y"])["lower_right"])
    budgets = analysis.budget_points(ladder)
    budgets = budgets[~budgets["omitted"]]
    out.update(budget_drawn=len(budgets),
               budget_upper_right=analysis.quadrant_counts(budgets["x"], budgets["y"])["upper_right"])
    prior = analysis.prior_strength_points(direct, analysis.spacing_trials(results_root))
    prior = prior[prior["model"].isin(competent)]
    out["prior_strength_rho"] = analysis.spearman_rho(prior["x"], prior["y"])
    recovery = analysis.recovery_points(ladder)
    out.update(recovery_points=len(recovery), recovery_positive=int((recovery["y"] > 0).sum()))

    gap = analysis.context_accuracy(direct, "congruent") - analysis.context_accuracy(direct, "reversed")
    out.update(model_dimensions=len(gap), congruent_above_reversed=int((gap > 0).sum()))
    change = analysis.accuracy_change(direct, roster.IWL_CONDITION, dimensions=roster.DIMENSIONS)
    out.update(iwl_panels=len(change), iwl_d6_above_d1=int((change > 0).sum()))
    change = analysis.accuracy_change(direct, roster.ICL_CONDITION, dimensions=(roster.ICL_DIMENSION,))
    out.update(icl_panels=len(change), icl_d6_below_d1=int((change < 0).sum()))
    change = analysis.budget_change(ladder, roster.ICL_CONDITION, dimensions=(roster.ICL_DIMENSION,))
    out.update(icl_budget_compared=len(change), icl_budget_rises=int((change > 0).sum()))
    change = analysis.budget_change(ladder, roster.IWL_CONDITION, dimensions=roster.DIMENSIONS)
    out.update(iwl_budget_compared=len(change), iwl_budget_falls=int((change < 0).sum()))
    for context in ("reversed", "shuffled", "congruent"):
        change = analysis.budget_change(ladder, context, dimensions=roster.DIMENSIONS, pooled=())
        out[f"{context}_budget_compared"] = len(change)
        out[f"{context}_budget_rises"] = int((change > 0).sum())
    city = ladder[(ladder["model"] == "qwen3-32b") & (ladder["condition"] == roster.IWL_CONDITION)
                  & (ladder["dimension"] == "citieslinear") & (ladder["d"] == 6)
                  & (ladder["budget"] == 0)]
    out["city_iwl_trials_d6"] = len(city)
    out["recovered_10_points"] = _recovered_10_points(ladder)
    by_distance = analysis.ladder_accuracy(ladder, "reversed", by_distance=True)
    near_far = by_distance[by_distance["budget"] == 256].pivot_table(
        index=["model", "dimension"], columns="d", values="is_correct")
    out["near_above_far_256"] = int((near_far[1] > near_far[6]).sum())
    one = ladder[(ladder["model"] == "qwen3-32b") & (ladder["dimension"] == "usstates")
                 & (ladder["condition"] == "reversed") & (ladder["budget"] == 0)]
    per_d = one.groupby("d").size()
    out.update(context_ladder_trials_d1=int(per_d[1]), context_ladder_trials_d6=int(per_d[6]))
    out.update(_accuracy_largest_budget(ladder))

    chains = analysis.chain_steps(results_root)
    for key, model in (("qwen3", "qwen3-32b"), ("qwen25", "qwen2.5-32b-instruct")):
        # Exact fractions of whole counts, so that a share on the 90% bound compares as equal.
        icl_share = [Fraction(int(k), int(n)) for k, n in analysis.composed_counts(
            chains, model, roster.ICL_CONDITION, 512).itertuples(index=False)]
        rev_share = [Fraction(int(k), int(n)) for k, n in analysis.composed_counts(
            chains, model, "reversed", 512).itertuples(index=False)]
        out[f"composed_icl_{key}_512"] = min(icl_share)
        # Short of 90% at every distance: the largest share stays below it.
        out[f"composed_reversed_{key}_512"] = max(rev_share) if key == "qwen25" else min(rev_share)
    out.update(_traces(results_root))
    out.update(_interventions(results_root))
    return out


# The single-model reasoning-budget panels: the model and the dimension of each test or context,
# as the figure scripts set them (MODEL and DIMENSION in figures/fig_symbolic_distance.py and
# figures/fig_content_effects.py).
_PANEL_MODEL = "qwen3-32b"
_SDE_DIMENSION = "countries"
_CONTENT_DIMENSION = "usstates"


def _accuracy_largest_budget(ladder: pd.DataFrame) -> dict[str, float]:
    """The accuracy at the largest budget, distances two to six, of each test and context, as
    the single-model reasoning-budget panels print it beside its line
    (``plots.make_accuracy_numbers``)."""
    own = ladder[ladder["model"] == _PANEL_MODEL]
    series = {
        "icl_test": own[own["condition"] == roster.ICL_CONDITION],
        "iwl_test": own[(own["condition"] == roster.IWL_CONDITION)
                        & (own["dimension"] == _SDE_DIMENSION)],
        **{context: own[(own["condition"] == context) & (own["dimension"] == _CONTENT_DIMENSION)]
           for context in ("congruent", "reversed", "shuffled")},
    }
    out = {}
    for name, rows in series.items():
        if rows.empty:
            raise ValueError(f"numbers: no {name} ladder rows of {_PANEL_MODEL}")
        out[f"{name}_accuracy_largest_budget"] = analysis.accuracy_at_budget(
            rows, budget=int(rows["budget"].max()), min_distance=2,
            fname=f"numbers {name}")["accuracy"]
    return out


def _recovered_10_points(ladder: pd.DataFrame) -> int:
    """Model-dimension pairs whose reversed-context accuracy at 1,024 tokens is at least 10
    points above the direct answer's, compared in whole trials so the boundary is exact."""
    rows = ladder[(ladder["condition"] == "reversed") & ladder["dimension"].isin(roster.DIMENSIONS)
                  & ladder["innermost"] & ladder["budget"].isin((0, 1024))]
    k = rows.groupby(["model", "dimension", "budget"], observed=True)["is_correct"].agg(["sum", "size"])
    k = k.unstack("budget")
    if not (k[("size", 0)] == k[("size", 1024)]).all():
        raise ValueError("recovery: the direct and 1,024-token answers cover different trials")
    return int(((k[("sum", 1024)] - k[("sum", 0)]) * 10 >= k[("size", 0)]).sum())


def _spacing() -> dict[str, object]:
    """Each main item set's value ratio between items six ranks apart, and its spacing between
    neighbors (the ratio, or "even" for equal steps), from the set's provenance record."""
    out: dict[str, object] = {}
    for dim, item_set in roster.PRIMARY_SETS.items():
        path = items.DEFAULT_ITEMS_DIR / "sources" / f"{item_set}_provenance.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        if int(record["sens_d"]) != 6:
            raise ValueError(f"{item_set}: the provenance ratio is not over six ranks")
        out[f"ratio_d6_{dim}"] = float(record["sens_ratio"])
        out[f"step_{dim}"] = ("even" if record["spacing"] == "linear"
                              else float(record["step_ratio"]))
    return out


def _premise_orders(results_root: Path) -> int:
    frame = pd.read_parquet(Path(results_root) / "symbolic_distance" / "tables" / "direct.parquet",
                            columns=["model", "condition", "premise_order"])
    rows = frame[(frame["model"] == "qwen3-32b") & (frame["condition"] == roster.ICL_CONDITION)]
    return int(rows["premise_order"].nunique())


def _traces(results_root: Path) -> dict[str, object]:
    path = Path(results_root) / "reasoning_traces" / "tables" / "trace_examples.json"
    table = json.loads(path.read_text(encoding="utf-8"))
    groups = table["selection"]["groups"]
    examples = {e["id"]: e for e in table["examples"]}
    return {
        "qwen3_incorrect_trials": int(groups["qwen3-32b | incorrect"]["trials"]),
        "qwen3_incorrect_finished": int(groups["qwen3-32b | incorrect"]["finished_within_budget"]),
        "trace_tokens_qwen3_correct": int(examples["qwen3-32b-correct"]["cot_n_tokens"]),
        "trace_tokens_qwen3_incorrect": int(examples["qwen3-32b-incorrect"]["cot_n_tokens"]),
        "trace_tokens_qwen25_correct": int(examples["qwen2.5-32b-instruct-correct"]["cot_n_tokens"]),
        "trace_tokens_qwen25_incorrect": int(examples["qwen2.5-32b-instruct-incorrect"]["cot_n_tokens"]),
        "example_trace_distance": (int(table["examples"][0]["d"])
                                   if len({e["d"] for e in table["examples"]}) == 1 else math.nan),
    }


def _interventions(results_root: Path) -> dict[str, object]:
    trials = analysis.intervention_trials(results_root)["heads"]
    heads = pd.read_parquet(Path(results_root) / "components" / "tables" / "heads.parquet")
    one = trials[(trials["dim"] == "usstates") & (trials["condition"] == "intact")]
    scored = one[one["test"].isin(("cot_reversed", "cot_icl"))]
    sets = {name: set(zip(g["layer"], g["head"]))
            for name, g in heads[heads["dim"] == "usstates"].groupby("set")}
    random_outside = not (sets["rand32_s0"] & sets["top64"])
    probe = analysis.probe_table(results_root)
    blocks = int(probe["layer"].max())          # hidden state 0 is the embeddings
    # Heads are numbered 0..63 within a block; the selected sets reach both ends.
    per_block = int(heads["head"].max()) + 1
    return {"blocks": blocks, "head_scoring_trials": len(scored),
            "heads_ablated": len(sets["top32"]), "heads_total": blocks * per_block,
            "random_heads_outside_top": len(sets["top64"]) if random_outside else math.nan,
            "smaller_head_set": len(sets["top16"]),
            "impaired_by_smaller_head_set": _impaired_by_top16(trials)}


# How the paper names a dimension of the interventions in running text.
_SHORT_NAMES = {"countries": "countries", "cities": "cities", "usstates": "US states"}


def _impaired_by_top16(trials: pd.DataFrame) -> str:
    """The dimensions where ablating the 16 top heads lowers accuracy after reasoning below the
    intact model's on both the reversed context and the ICL test (distances two to six)."""
    impaired = []
    for dim in _SHORT_NAMES:
        rows = trials[(trials["dim"] == dim) & (trials["budget"] == 1024)]
        acc = {(test, condition): analysis.pooled_accuracy(
                   rows[(rows["test"] == test) & (rows["condition"] == condition)],
                   d_min=2, d_max=6)["acc"]
               for test in ("cot_reversed", "cot_icl") for condition in ("intact", "top16")}
        if all(acc[(t, "top16")] < acc[(t, "intact")] for t in ("cot_reversed", "cot_icl")):
            impaired.append(_SHORT_NAMES[dim])
    return _name_list(impaired)


def _name_list(names) -> str:
    """Names as running text lists them: "A", "A and B", "A, B, and C"."""
    names = list(names)
    if len(names) < 3:
        return " and ".join(names)
    return ", ".join(names[:-1]) + ", and " + names[-1]


# ── Writing a value as the paper writes it ──────────────────────────────────────────────────
_WORDS = ("none", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")
_NUMBER = re.compile(r"^([+-]?)(\d[\d,]*)?(?:\.(\d+))?(%?)$")


def format_like(value, printed: str) -> str:
    """``value`` written as ``printed`` writes its number.

    The same decimals (rounded half away from zero), a thousands separator, a leading zero and
    a sign where ``printed`` has them; a number word for a word; ``>= x`` or ``< x`` while the
    value is on that side of x (a ``Fraction`` of whole counts is compared with x exactly); a text
    value as it is. A NaN is written "nan".
    """
    if isinstance(value, str):
        return value
    for op, holds in ((">= ", operator.ge), ("< ", operator.lt)):
        if printed.startswith(op):
            bound = printed[len(op):]
            exact = isinstance(value, Fraction)
            ok = holds(value, Fraction(bound)) if exact else holds(float(value), float(bound))
            return printed if ok else format_like(value, bound)
    if math.isnan(float(value)):
        return "nan"
    if printed.lower() in _WORDS:
        whole = float(value).is_integer() and 0 <= value < len(_WORDS)
        if not whole:
            return str(value)
        word = _WORDS[int(value)]
        return word.capitalize() if printed[0].isupper() else word
    match = _NUMBER.match(printed)
    if match is None:
        raise ValueError(f"no number format in {printed!r}")
    sign, whole, fraction, percent = match.groups()
    exact = Decimal(f"{float(value):.12g}")
    rounded = exact.quantize(Decimal(1).scaleb(-len(fraction or "")), rounding=ROUND_HALF_UP)
    text = format(abs(rounded), ",f" if whole and "," in whole else "f")
    if whole is None and text.startswith("0."):
        text = text[1:]
    if rounded < 0:
        text = "-" + text
    elif sign == "+" and rounded > 0:
        text = "+" + text
    return text + percent


def numbers(results_root: Path = analysis.RESULTS_ROOT) -> pd.DataFrame:
    """Every number of ``PAPER_NUMBERS``, recomputed and compared at the printed precision."""
    got = values(results_root)
    rows = []
    for id_, site, printed, source in PAPER_NUMBERS:
        computed = format_like(got[source], printed)
        rows.append({"id": id_, "site": site, "printed": printed, "computed": computed,
                     "source": source, "match": computed == printed})
    return pd.DataFrame(rows)
