"""The estimators behind the LLM panels and numbers, on small synthetic frames."""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from geo_alg_ti_llms import analysis, numbers


def ladder_frame(acc_by_budget: dict[int, float], n: int = 40, d: int = 3, seed: int = 0) -> pd.DataFrame:
    """One model's ladder at one distance: ``n`` trials, each budget correct at the given rate."""
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n):
        for b, p in acc_by_budget.items():
            rows.append({"model": "m", "condition": "reversed", "dimension": "usstates",
                         "item_pool": "p", "innermost": True, "d": d, "trial_key": f"k{t:03d}",
                         "budget": b, "is_correct": bool(rng.random() < p)})
    return pd.DataFrame(rows)


def test_wilson_interval_matches_the_closed_form():
    lo, hi = analysis.wilson_ci(8, 10, z=1.96)
    assert lo == pytest.approx(0.4901568, abs=1e-6) and hi == pytest.approx(0.9433191, abs=1e-6)
    lo, hi = analysis.wilson_ci(0, 0)
    assert np.isnan(lo) and np.isnan(hi)


def test_reasoning_budget_is_the_first_budget_reaching_the_criterion():
    assert analysis.reasoning_budget([0, 32, 64, 128], [0.5, 0.7, 0.95, 1.0]) == 64
    assert analysis.reasoning_budget([128, 0, 64], [1.0, 0.95, 0.5]) == 0
    assert math.isnan(analysis.reasoning_budget([0, 32], [np.nan, np.nan]))


def test_running_best_keeps_the_row_of_the_first_budget_that_reached_the_best():
    table = pd.DataFrame({"budget": [64, 0, 128, 256], "n": [10, 10, 10, 10],
                          "accuracy": [0.6, 0.7, 0.7, 0.9], "ci_lo": [0.5, 0.65, 0.6, 0.8],
                          "ci_hi": [0.7, 0.75, 0.8, 0.95]})
    best = analysis.running_best(table)
    assert list(best["budget"]) == [0, 64, 128, 256]
    assert list(best["accuracy"]) == [0.7, 0.7, 0.7, 0.9]
    assert list(best["ci_lo"]) == [0.65, 0.65, 0.65, 0.8]      # the tie keeps budget 0's band


def test_accuracy_at_budget_pools_distances_from_the_minimum():
    df = pd.concat([ladder_frame({0: 0.5, 1024: 0.9}, d=d, seed=d) for d in (1, 2, 3)])
    got = analysis.accuracy_at_budget(df, budget=1024, min_distance=2)
    rows = df[(df["budget"] == 1024) & (df["d"] >= 2)]
    assert got["n"] == len(rows) == 80 and got["n_correct"] == int(rows["is_correct"].sum())
    assert got["ci_lo"] <= got["accuracy"] <= got["ci_hi"] and got["n_item_sets"] == 1
    with pytest.raises(ValueError, match="accuracy_at_budget: no innermost rows"):
        analysis.accuracy_at_budget(df[df["d"] == 1], budget=1024, min_distance=2)


def test_budget_matrix_matches_a_pivot_table():
    df = ladder_frame({0: 0.5, 64: 0.8, 1024: 0.9}, n=25)
    df = df.drop(index=[3, 10]).reset_index(drop=True)      # a trial missing at a budget
    matrix, budgets = analysis.budget_matrix(df)
    pivot = df.assign(y=df["is_correct"].astype(float)).pivot_table(
        index="trial_key", columns="budget", values="y", aggfunc="mean")
    assert np.array_equal(matrix, pivot.to_numpy(float), equal_nan=True)
    assert list(budgets) == [0, 64, 1024]


def test_budget_to_criterion_flags_a_ladder_that_never_beats_chance():
    good = analysis.budget_to_criterion(ladder_frame({0: 0.5, 64: 0.95, 128: 1.0}), n_boot=50)
    assert good["above_chance"] and good["budget"] == 64 and good["boot_sd"] >= 0
    flat = analysis.budget_to_criterion(ladder_frame({0: 0.5, 64: 0.5}), n_boot=0)
    assert not flat["above_chance"] and math.isnan(flat["boot_sd"])


def test_the_bootstrap_is_reproducible_and_reads_the_default_at_call_time(monkeypatch):
    df = ladder_frame({0: 0.4, 64: 0.7, 128: 0.9})
    a = analysis.budget_to_criterion(df, n_boot=200)
    b = analysis.budget_to_criterion(df, n_boot=200)
    assert a["boot_sd"] == b["boot_sd"]
    monkeypatch.setattr(analysis, "N_BOOT", 200)
    assert analysis.budget_to_criterion(df)["boot_sd"] == a["boot_sd"]


def test_budget_summary_omits_a_ladder_with_too_few_usable_distances():
    usable = pd.concat([ladder_frame({0: 0.5, 64: 1.0}, d=d, seed=d) for d in (1, 2, 3)])
    summary = analysis.budget_summary(usable)
    assert not summary["omitted"] and summary["usable"] == [1, 2, 3]
    assert summary["slope"] == pytest.approx(0.0)
    short = pd.concat([ladder_frame({0: 0.5, 64: 1.0}, d=1), ladder_frame({0: 0.5}, d=2)])
    assert analysis.budget_summary(short)["omitted"]


def test_guard_refuses_an_unpinned_axis():
    df = ladder_frame({0: 0.6})
    df.loc[df.index[:5], "dimension"] = "countries"
    with pytest.raises(ValueError, match="dimension"):
        analysis.trial_accuracy_by(df, ["d"])
    assert len(analysis.trial_accuracy_by(df, ["d"], pooled=("dimension",))) == 1


def test_binomial_tail_and_rank_correlation():
    assert analysis.binomial_sf(7, 10) == pytest.approx(0.171875)
    assert analysis.binomial_sf(0, 5) == 1.0 and analysis.binomial_sf(6, 5) == 0.0
    assert analysis.spearman_rho([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert analysis.spearman_rho([1, 2, 3], [5, 5, 5]) != analysis.spearman_rho([1, 2, 3], [5, 5, 5])


def test_a_point_on_a_reference_line_counts_in_no_corner():
    counts = analysis.quadrant_counts([0.1, -0.2, analysis.snapped(1e-17), 0.3], [-0.1, 0.2, 0.5, 0.4])
    assert counts == {"upper_right": 1, "upper_left": 1, "lower_right": 1, "lower_left": 0,
                      "on_ref": 1}


def test_competence_pools_the_grid_sets_of_the_five_dimensions():
    rows = []
    for model, rate in (("qwen3-32b", 1.0), ("llama-3.2-1b", 0.5)):
        for dim in ("countries", "countries_rep2", "countries_rep3", "usstates"):
            for t in range(20):
                rows.append({"model": model, "condition": "no_context", "innermost": True, "d": 1,
                             "dim": dim, "is_correct": t < 20 * rate})
    frame = pd.DataFrame(rows)
    frame["dimension"] = frame["dim"].str.replace(r"_rep\d+$", "", regex=True)
    frame["grid_set"] = frame["dim"] != "countries_rep3"
    table = analysis.competence(frame).set_index("model")
    assert table.loc["qwen3-32b", "n"] == 60 and table.loc["qwen3-32b", "competent"]
    assert not table.loc["llama-3.2-1b", "competent"]


def test_composed_fraction_needs_one_dimension():
    rows = pd.DataFrame([{"model": "m", "condition": "reversed", "dim": dim, "innermost": True,
                          "d": d, "budget": 512, "inferences_needed": d - 1,
                          "inferences_composed": d - 1 if dim == "usstates" else 0}
                         for dim in ("usstates", "countries") for d in range(1, 7)])
    shares = analysis.composed_fraction(rows[rows["dim"] == "usstates"], "m", "reversed", 512)
    assert list(shares.index) == [2, 3, 4, 5, 6] and (shares == 1.0).all()
    with pytest.raises(ValueError, match="dimensions"):
        analysis.composed_fraction(rows, "m", "reversed", 512)
    counts = analysis.composed_counts(rows[rows["dim"] == "usstates"], "m", "reversed", 512)
    assert counts.to_dict("index") == {d: {"inferences_composed": d - 1, "inferences_needed": d - 1}
                                       for d in range(2, 7)}
    with pytest.raises(ValueError, match="dimensions"):
        analysis.composed_counts(rows, "m", "reversed", 512)


@pytest.mark.parametrize("value, printed, expected", [
    (15, "15", "15"), (-0.3001, "-0.30", "-0.30"), (1022, "1,022", "1,022"), (56.2, "56.2", "56.2"),
    (1, "one", "one"), (0.93, ">= 0.90", ">= 0.90"), (0.85, ">= 0.90", "0.85"),
    (0.84, "< 0.90", "< 0.90"), ("a, b", "a, b", "a, b"), (float("nan"), "3", "nan"),
    (0.125, "0.13", "0.13"), (10, "Ten", "Ten"), (7, "seven", "seven"),
    (Fraction(162, 180), ">= 0.90", ">= 0.90"),
    (Fraction(161, 180), ">= 0.90", "0.89"), (Fraction(162, 180), "< 0.90", "0.90"),
])
def test_values_are_written_at_the_printed_precision(value, printed, expected):
    assert numbers.format_like(value, printed) == expected


def test_names_are_listed_as_running_text():
    assert numbers._name_list(["A"]) == "A"
    assert numbers._name_list(["A", "B"]) == "A and B"
    assert numbers._name_list(["A", "B", "C"]) == "A, B, and C"
