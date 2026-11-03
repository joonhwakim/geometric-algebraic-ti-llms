"""The panel functions on small synthetic frames: each draws onto its axes and returns stats."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from geo_alg_ti_llms import plots  # noqa: E402
from geo_alg_ti_llms.plots import grid, representation, style  # noqa: E402

BUDGETS = (0, 64, 512, 1024)


def trials(conditions, *, model="m", budgets=(0,), n=12, rate=0.8, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for condition in conditions:
        for d in range(1, 7):
            for t in range(n):
                for b in budgets:
                    rows.append({"model": model, "condition": condition, "dimension": "usstates",
                                 "item_pool": "p", "innermost": True, "d": d,
                                 "trial_key": f"{condition}|{d}|{t}", "budget": b,
                                 "is_correct": bool(rng.random() < rate * (b + 64) / (b + 128) + 0.2)})
    return pd.DataFrame(rows)


@pytest.fixture
def ax():
    fig, ax = plt.subplots()
    yield ax
    plt.close(fig)


def test_accuracy_by_distance_draws_one_line_per_condition(ax):
    stats = plots.make_accuracy_by_distance(ax, trials(("congruent", "no_context")),
                                            conditions=("congruent", "no_context"),
                                            open_marker_conditions=("no_context",),
                                            linestyles={"no_context": "--"}, show_legend=True)
    assert set(stats["per_condition"]) == {"congruent", "no_context"}
    assert stats["per_condition"]["congruent"]["n_by_distance"][1] == 12
    lines = [l for l in ax.get_lines() if l.get_label() in ("Congruent", "No context")]
    assert [l.get_linestyle() for l in lines] == ["-", "--"]
    assert lines[1].get_markerfacecolor() == style.OPEN_MARKER_FACE


def test_a_panel_refuses_more_than_one_model(ax):
    two = pd.concat([trials(("congruent",)), trials(("congruent",), model="other")])
    with pytest.raises(ValueError, match="pin the model"):
        plots.make_accuracy_by_distance(ax, two, conditions=("congruent",))


def test_budget_by_distance_marks_a_distance_never_above_chance(ax):
    df = trials(("reversed",), budgets=BUDGETS, rate=0.9)
    df.loc[df["d"] == 6, "is_correct"] = df.loc[df["d"] == 6, "trial_key"].str[-1].isin(list("02468"))
    stats = plots.make_budget_by_distance(ax, df, conditions=("reversed",), n_boot=20,
                                          break_at_unreached=True)
    assert stats["unreached"]["reversed"] == [6]
    assert style.UNREACHED_LABEL in [h.get_label() for h in ax.collections]


def test_accuracy_by_budget_takes_one_shade_per_distance(ax):
    stats = plots.make_accuracy_by_budget(ax, trials(("congruent", "reversed"), budgets=BUDGETS),
                                          conditions=("congruent", "reversed"))
    assert set(stats["per_condition"]["reversed"]["accuracy_by_distance_and_budget"]) == set(range(1, 7))
    assert len({l.get_color() for l in ax.get_lines()[:6]}) == 6


def test_accuracy_by_budget_can_draw_the_best_accuracy_so_far(ax):
    df = trials(("reversed",), budgets=BUDGETS)
    raw = plots.make_accuracy_by_budget(ax, df, conditions=("reversed",))
    best = plots.make_accuracy_by_budget(ax, df, conditions=("reversed",),
                                         budget_stat="running_best",
                                         ylabel=style.ACCURACY_BEST_SO_FAR_LABEL)
    for d in range(1, 7):
        raw_d = raw["per_condition"]["reversed"]["accuracy_by_distance_and_budget"][d]
        best_d = best["per_condition"]["reversed"]["accuracy_by_distance_and_budget"][d]
        so_far = np.maximum.accumulate([raw_d[b] for b in BUDGETS])
        assert [best_d[b] for b in BUDGETS] == list(so_far)
    assert ax.get_ylabel() == style.ACCURACY_BEST_SO_FAR_LABEL
    with pytest.raises(ValueError, match="budget_stat"):
        plots.make_accuracy_by_budget(ax, df, conditions=("reversed",), budget_stat="max")


def test_accuracy_panels_take_their_ticks_and_budget_range():
    fig, (left, right) = plt.subplots(1, 2)
    plots.make_accuracy_by_distance(left, trials(("congruent",)), conditions=("congruent",),
                                    yticks=(0.0, 0.5, 1.0))
    plots.make_accuracy_by_budget(right, trials(("congruent",), budgets=BUDGETS),
                                  conditions=("congruent",), xlim=(-40.0, 1064.0),
                                  yticks=(0.0, 0.5, 1.0))
    assert list(left.get_yticks()) == list(right.get_yticks()) == [0.0, 0.5, 1.0]
    assert right.get_xlim() == (-40.0, 1064.0)
    assert left.get_ylim() == right.get_ylim() == (-0.02, 1.02)
    plt.close(fig)


def test_accuracy_bars_pool_distances_two_to_six_at_the_largest_budget(ax):
    df = trials(("congruent", "reversed"), budgets=BUDGETS)
    stats = plots.make_accuracy_bars(ax, df, conditions=("congruent", "reversed"))
    assert stats["budget"] == 1024
    rows = df[(df["budget"] == 1024) & (df["d"] >= 2) & (df["condition"] == "reversed")]
    got = stats["per_condition"]["reversed"]
    assert got["n"] == len(rows) == 60 and got["n_correct"] == int(rows["is_correct"].sum())
    assert got["ci_lo"] <= got["accuracy"] <= got["ci_hi"]
    assert [p.get_facecolor()[:3] for p in ax.patches] == [
        matplotlib.colors.to_rgb(style.CONDITION_COLORS[c]) for c in ("congruent", "reversed")]
    assert ax.get_ylabel() == style.ACCURACY_REASONING_LABEL


def test_accuracy_numbers_write_the_bars_values_beside_each_line(ax):
    df = trials(("congruent", "reversed"), budgets=BUDGETS, rate=0.9)
    conditions = ("congruent", "reversed")
    budget = plots.make_budget_by_distance(ax, df, conditions=conditions, n_boot=0)
    fig, other = plt.subplots()
    bars = plots.make_accuracy_bars(other, df, conditions=conditions)
    plt.close(fig)
    got = plots.make_accuracy_numbers(ax, df, conditions=conditions,
                                      anchors={"reversed": (5.0, 100.0)})
    for c in conditions:
        mine = got["per_condition"][c]
        assert {k: mine[k] for k in bars["per_condition"][c]} == bars["per_condition"][c]
        assert mine["text"] == f"{bars['per_condition'][c]['accuracy']:.2f}"
        assert got["artists"][c].get_color() == style.CONDITION_COLORS[c]
    end = (6.0, float(budget["per_condition"]["congruent"]["budget_by_distance"][6]))
    assert got["artists"]["congruent"].xy == end
    assert got["artists"]["reversed"].get_position() == (5.0, 100.0)
    with pytest.raises(ValueError, match="no line"):
        plots.make_accuracy_numbers(ax, trials(("shuffled",), budgets=BUDGETS),
                                    conditions=("shuffled",))


def test_a_number_placed_by_its_line_stays_inside_the_axes():
    fig, ax = plt.subplots(figsize=(1.5, 1.3))
    ends = {"congruent": 1000.0, "reversed": 0.0}      # one line ends at the top, one at the bottom
    for condition, top in ends.items():
        ax.plot([1, 6], [500.0, top], color=style.CONDITION_COLORS[condition], marker="o")
    ax.set_ylim(-20.0, 1020.0)
    df = trials(tuple(ends), budgets=BUDGETS)
    got = plots.make_accuracy_numbers(ax, df, conditions=tuple(ends))
    fig.canvas.draw()
    box = ax.get_window_extent()
    for condition in ends:
        text = got["artists"][condition].get_window_extent()
        assert box.x0 <= text.x0 and text.x1 <= box.x1 + 1e-6, condition
        assert box.y0 <= text.y0 and text.y1 <= box.y1 + 1e-6, condition
    plt.close(fig)


def test_congruent_distance_colormap_runs_through_the_congruent_color():
    cmap = matplotlib.colormaps[style.DISTANCE_CMAPS["congruent"]]
    assert style.DISTANCE_CMAPS["congruent"] == "congruent_greens"
    assert [matplotlib.colors.to_hex(cmap(x)) for x in (0.0, 0.5, 1.0)] == ["#f2fbf6", "#3fb780",
                                                                            "#0b3d2c"]


def test_composed_inferences_draws_each_distance_and_its_need(ax):
    rows = [{"model": "m", "condition": "icl_people", "dim": "countries", "innermost": True,
             "d": d, "trial_key": f"{d}|{t}", "budget": b, "inferences_needed": d - 1,
             "inferences_composed": min(d - 1, b // 256)}
            for d in range(2, 7) for t in range(4) for b in BUDGETS]
    stats = plots.make_composed_inferences_by_budget(ax, pd.DataFrame(rows), cmap="Blues")
    assert stats["per_distance"][6]["need"] == 5
    assert stats["per_distance"][6]["mean_by_budget"][1024] == 4
    two = pd.DataFrame(rows).assign(dim=lambda f: f["trial_key"].str[-1].map(
        {"0": "countries", "1": "usstates", "2": "countries", "3": "usstates"}))
    with pytest.raises(ValueError, match="dim"):
        plots.make_composed_inferences_by_budget(ax, two, cmap="Blues")


def test_model_scatter_counts_the_corners_and_tints_one(ax):
    points = pd.DataFrame({"x": [0.1, 0.2, -0.1, 0.0], "y": [-0.1, -0.2, 0.1, 0.3],
                           "x_se": 0.01, "y_lo": -0.3, "y_hi": 0.4})
    stats = plots.make_model_scatter(ax, points, xlabel="x", ylabel="y", x_err="x_se",
                                     y_err=("y_lo", "y_hi"), quadrant="lower_right", symmetric=True)
    assert stats["quadrant_counts"]["lower_right"] == 2 and stats["quadrant_counts"]["on_ref"] == 1
    assert len(ax.patches) == 1
    lo, hi = ax.get_xlim()
    assert lo == pytest.approx(-hi)


def test_model_scatter_sizes_every_point_so_only_edged_points_are_stroked(ax):
    """With one size for all points the PDF writer reuses one stroked marker, and every point
    without an edge gets a hairline; one size per point keeps them edgeless."""
    points = pd.DataFrame({"x": [0.1, 0.2, 0.3], "y": [0.1, 0.2, 0.3],
                           "main": [True, False, False]})
    plots.make_model_scatter(ax, points, xlabel="x", ylabel="y", emphasis_col="main")
    [dots] = [c for c in ax.collections if len(c.get_offsets()) == len(points)]
    assert len(dots.get_sizes()) == len(points)
    assert list(dots.get_linewidths()) == [0.8, 0.0, 0.0]


def intervention_trials():
    runs = {}
    for run, conditions in (("rank", ("intact", "erase", "rand_erase", "invert", "write_0.25",
                                      "write_0.5", "write_0.75", "write_1.0", "write_rand")),
                            ("heads", ("intact", "top32", "rand32_s0"))):
        rows = []
        for _, test, budget in representation.GROUPS:
            for condition in conditions:
                for d in range(1, 7):
                    for t in range(6):
                        rows.append({"dim": "usstates", "test": test, "condition": condition,
                                     "budget": budget, "d": d, "is_correct": (t + d) % 3 != 0,
                                     "agree_written": t % 2 == 0})
        runs[run] = pd.DataFrame(rows)
    return runs


def test_intervention_panels_draw_from_both_runs():
    fig, axes = plt.subplots(1, 5)
    runs = intervention_trials()
    bars = plots.make_intervention_bars(axes[0], runs)
    assert set(bars["groups"]["iwl|0"]) == {key for key, _, _ in representation.BARS}
    plots.make_edit_by_distance(axes[1], runs)
    write = plots.make_write_by_strength(axes[2], runs, control_ax=axes[3])
    assert set(write["series"]) == {0, 1024}
    probe = pd.DataFrame([{"dim": dim, "layer": layer, "acc_axis_heldout": 0.5 + layer / 200}
                          for dim in ("countries", "cities", "usstates") for layer in range(65)])
    assert plots.make_probe_by_layer(axes[4], probe)["peak"]["usstates"] == pytest.approx(0.82)
    plt.close(fig)


def test_intervention_bars_refuse_runs_that_disagree_on_the_intact_model():
    runs = intervention_trials()
    flip = (runs["heads"]["condition"] == "intact") & (runs["heads"]["d"] == 3)
    runs["heads"].loc[flip, "is_correct"] = ~runs["heads"].loc[flip, "is_correct"]
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="disagree"):
        plots.make_intervention_bars(ax, runs)
    plt.close(fig)


def test_a_page_has_one_row_per_model_and_one_column_per_dimension():
    fig, axes = grid.page(["A", "B-\nC"], ["one", "two", "three"])
    assert axes.shape == (2, 3)
    assert fig.get_size_inches()[1] == pytest.approx(grid.CHROME_HEIGHT_IN + 2 * grid.ROW_HEIGHT_IN)
    plt.close(fig)


def test_a_page_takes_the_width_of_its_row_header_gutter():
    default, _ = grid.page(["A"], ["one"])
    narrow, _ = grid.page(["A"], ["one"], gutter_in=grid.GUTTER_IN - 0.1)
    for fig in (default, narrow):
        fig.canvas.draw()
    shift = default.axes[0].get_position().x0 - narrow.axes[0].get_position().x0
    assert shift * style.FULL_WIDTH_IN == pytest.approx(0.1, abs=0.01)
    header = narrow.axes[0].texts[0].xy[0] * style.FULL_WIDTH_IN
    assert header == pytest.approx(grid.GUTTER_IN - 0.1 - grid.ROW_HEADER_PAD_PT / 72.0)
    plt.close(default)
    plt.close(narrow)


def test_distance_shades_run_from_light_to_dark():
    shades = style.distance_colors((1, 2, 3, 4, 5, 6), "Oranges")
    assert len(set(shades)) == 6
    lightness = [sum(int(s[i:i + 2], 16) for i in (1, 3, 5)) for s in shades]
    assert lightness == sorted(lightness, reverse=True)


def test_every_model_row_label_fits_in_the_page_gutter():
    from geo_alg_ti_llms import roster
    style.apply_style()
    fig = plt.figure()
    renderer = fig.canvas.get_renderer()
    room = grid.GUTTER_IN - grid.ROW_HEADER_PAD_PT / 72.0
    for label in roster.MODEL_ROW_LABELS.values():
        text = fig.text(0, 0, label, fontweight="bold", fontsize=style.FONT_TICK_PT)
        assert text.get_window_extent(renderer).width / fig.dpi <= room, label
    plt.close(fig)
