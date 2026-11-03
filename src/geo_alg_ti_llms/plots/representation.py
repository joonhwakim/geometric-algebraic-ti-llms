"""Panels of the interventions on Qwen3-32B: the rank direction and the context-reading heads.

The trial tables come from two runs: the rank-direction run (``rank``: the direction erased,
inverted, erased along a random direction, or overwritten toward another ordering) and the head
run (``heads``: the 32 heads that attend most to the relevant context relations while the model
reasons, or 32 random heads, ablated). The caller pins one dimension. Accuracy is k / n over
trials with its binomial standard error, the analog of the transformer panels' one SE.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.axes import Axes
from matplotlib.patches import Patch
from matplotlib.transforms import offset_copy

from geo_alg_ti_llms import analysis, roster
from geo_alg_ti_llms.plots import style
from geo_alg_ti_llms.plots.distance import BAND_ZORDER, LINE_ZORDER, draw_series

# A group is one test at its reasoning budget: the tick label, the test, the budget.
GROUPS: tuple[tuple[str, str, int], ...] = (
    ("IWL test", "iwl", 0), ("Reversed", "content_reversed", 0),
    ("Reversed", "cot_reversed", 1024), ("ICL test", "cot_icl", 1024),
)
# A bar is one model state: its key, the run and the condition.
BARS: tuple[tuple[str, str, str], ...] = (
    ("intact", "rank", "intact"), ("rank_erased", "rank", "erase"),
    ("random_direction", "rank", "rand_erase"), ("heads", "heads", "top32"),
    ("random_heads", "heads", "rand32_s0"),
)
BAR_LABELS: dict[str, str] = {
    "intact": "Intact", "rank_erased": "Rank direction erased",
    "random_direction": "Random direction erased", "heads": "Context-reading heads ablated",
    "random_heads": "Random heads ablated",
}
BAR_COLORS: dict[str, str] = {
    "intact": style.EDIT_INTACT, "rank_erased": style.EDIT_ERASED,
    "random_direction": style.EDIT_ERASED, "heads": style.CONTEXT_HEADS,
    "random_heads": style.CONTEXT_HEADS,
}
OPEN_BARS: tuple[str, ...] = ("random_direction", "random_heads")
BAR_WIDTH: float = 0.16
CONTROL_BAR_WIDTH: float = 0.7
BUDGET_FORMAT: str = "{:,} tokens"
# The budget row: its top below the axis, and the bracket over the groups it names.
BUDGET_ROW_OFFSET_PT: float = 16.0
BRACKET_INSET: float = 0.42
BRACKET_DROP_PT: float = 1.5
EDIT_LABELS: dict[str, str] = {"intact": "Intact", "invert": "Rank direction inverted",
                               "erase": "Rank direction erased",
                               "rand_erase": "Random direction erased"}
EDIT_COLORS: dict[str, str] = {"intact": style.EDIT_INTACT, "invert": style.EDIT_INVERTED,
                               "erase": style.EDIT_ERASED, "rand_erase": style.EDIT_ERASED}
# A write series is one test at one budget and the write strengths drawn on it.
WRITE_SERIES: tuple[tuple[str, int, tuple[str, ...]], ...] = (
    ("content_reversed", 0, ("intact", "write_0.25", "write_0.5", "write_0.75", "write_1.0")),
    ("cot_reversed", 1024, ("intact", "write_1.0")),
)
WRITE_LABELS: dict[int, str] = {0: "0 reasoning tokens", 1024: "1,024 reasoning tokens"}
WRITE_COLORS: dict[int, str] = {0: style.NO_STEPS, 1024: style.MAX_STEPS}
WRITE_CONTROL: str = "write_rand"
PROBE_COLORS: dict[str, str] = {"countries": style.GRAY_SHADES[2], "cities": style.GRAY_SHADES[1],
                                "usstates": style.EDIT_INTACT}
PROBE_MARKERS: dict[str, str] = {"countries": "o", "cities": "s", "usstates": "D"}
PROBE_LAYER_TICKS: tuple[int, ...] = (0, 16, 32, 48, 64)
WRITE_X_PAD: float = 0.07


def _rows(trials: Mapping[str, pd.DataFrame], *, run: str, test: str, condition: str,
          budget: int) -> pd.DataFrame:
    df = trials[run]
    rows = df[(df["test"] == test) & (df["condition"] == condition) & (df["budget"] == budget)]
    if rows.empty:
        raise ValueError(f"no {run} trials for {test}, {condition}, budget {budget}")
    if rows["dim"].nunique() != 1:
        raise ValueError("pin one dimension first")
    return rows


def budget_spans(budgets: Sequence[int], centers: Sequence[float]) -> list[tuple[str, float, float]]:
    """(text, first x, last x) for each run of neighboring groups that share a budget."""
    spans: list[list] = []
    for budget, x in zip(budgets, centers):
        if spans and spans[-1][3] == budget:
            spans[-1][2] = x
        else:
            spans.append([BUDGET_FORMAT.format(int(budget)), x, x, budget])
    return [(text, first, last) for text, first, last, _ in spans]


def add_budget_row(ax: Axes, spans: Sequence[tuple[str, float, float]]) -> None:
    """A second row of x labels under the tick labels, with a bracket over each span's groups."""
    blend = ax.get_xaxis_transform()
    line_at = offset_copy(blend, fig=ax.figure, y=-(BUDGET_ROW_OFFSET_PT - BRACKET_DROP_PT),
                          units="points")
    for text, first, last in spans:
        ax.annotate(text, xy=((first + last) / 2.0, 0.0), xycoords=blend,
                    xytext=(0.0, -BUDGET_ROW_OFFSET_PT), textcoords="offset points", ha="center",
                    va="top", annotation_clip=False, fontsize=rcParams["xtick.labelsize"])
        if last > first:
            ax.plot([first - BRACKET_INSET, last + BRACKET_INSET], [0.0, 0.0], transform=line_at,
                    color=style.REFERENCE_LINE, lw=style.AXES_LINEWIDTH_PT, clip_on=False,
                    solid_capstyle="butt")


def intervention_legend_handles() -> list[Patch]:
    """The bars' key in three columns: intact over a blank, each manipulation over its control."""
    handles = [Patch(facecolor=style.OPEN_MARKER_FACE if key in OPEN_BARS else BAR_COLORS[key],
                     edgecolor=BAR_COLORS[key], linewidth=style.AXES_LINEWIDTH_PT,
                     label=BAR_LABELS[key]) for key, _, _ in BARS]
    return [handles[0], Patch(visible=False, label=" "), *handles[1:]]


def make_intervention_bars(ax: Axes, trials: Mapping[str, pd.DataFrame], *, d_min: int = 2,
                           d_max: int = 6) -> dict:
    """Grouped bars per test: intact, the rank direction erased, the heads ablated, the controls.

    Accuracy pools distances ``d_min``..``d_max``. The intact bar reads the rank-direction run;
    the head run's intact model must give the same counts in every group.
    """
    centers = np.arange(len(GROUPS), dtype=float)
    stats: dict = {"groups": {}}
    for label, test, budget in GROUPS:
        both = {run: analysis.pooled_accuracy(_rows(trials, run=run, test=test, condition="intact",
                                                    budget=budget), d_min=d_min, d_max=d_max)
                for run in ("rank", "heads")}
        if len({(v["k"], v["n"]) for v in both.values()}) != 1:
            raise ValueError(f"make_intervention_bars: the runs' intact models disagree on {test}")
        stats["groups"][f"{test}|{budget}"] = {}
    for index, (key, run, condition) in enumerate(BARS):
        offset = (index - (len(BARS) - 1) / 2.0) * BAR_WIDTH
        got = [analysis.pooled_accuracy(_rows(trials, run=run, test=test, condition=condition,
                                              budget=budget), d_min=d_min, d_max=d_max)
               for _, test, budget in GROUPS]
        for (_, test, budget), g in zip(GROUPS, got):
            stats["groups"][f"{test}|{budget}"][key] = {"k": g["k"], "n": g["n"],
                                                        "acc": round(g["acc"], 4)}
        color = BAR_COLORS[key]
        ax.bar(centers + offset, [g["acc"] for g in got], width=BAR_WIDTH,
               yerr=[g["se"] for g in got],
               color=style.OPEN_MARKER_FACE if key in OPEN_BARS else color, edgecolor=color,
               linewidth=style.AXES_LINEWIDTH_PT, label=BAR_LABELS[key], zorder=LINE_ZORDER,
               error_kw={"ecolor": color, "lw": style.AXES_LINEWIDTH_PT,
                         "capsize": style.ERRORBAR_CAPSIZE, "capthick": style.AXES_LINEWIDTH_PT})
    ax.set_xticks(centers)
    ax.set_xticklabels([label for label, _, _ in GROUPS])
    ax.set_xlim(centers[0] - 0.5, centers[-1] + 0.5)
    add_budget_row(ax, budget_spans([b for _, _, b in GROUPS], centers))
    ax.set_ylabel(style.ACCURACY_LABEL)
    ax.set_ylim(-0.02, 1.02)
    ax.set_yticks((0.0, 0.2, 0.4, 0.6, 0.8, 1.0))
    style.chance_line(ax)
    return stats


def make_edit_by_distance(ax: Axes, trials: Mapping[str, pd.DataFrame], *,
                          marker_size: float = 3.5, legend_kwargs: Mapping | None = None) -> dict:
    """Direct IWL-test accuracy against distance under each edit of the rank direction, with the
    random-direction control dashed."""
    stats: dict = {"series": {}}
    d_values = analysis.DISTANCES
    series = [(c, False) for c in ("intact", "invert", "erase")] + [("rand_erase", True)]
    for condition, is_control in series:
        rows = _rows(trials, run="rank", test="iwl", condition=condition, budget=0)
        rows = rows[rows["d"].isin(list(d_values))]
        table = rows.groupby("d")["is_correct"].agg(["sum", "size"])
        acc = (table["sum"] / table["size"]).to_numpy()
        se = analysis.binomial_se(table["sum"], table["size"])
        draw_series(ax, table.index, acc, acc - se, acc + se, color=EDIT_COLORS[condition],
                    label=EDIT_LABELS[condition], marker_size=marker_size, open_marker=is_control,
                    ls="--" if is_control else "-")
        stats["series"][condition] = dict(zip(table.index.astype(int), np.round(acc, 4)))
    style.chance_line(ax)
    ax.set_xticks(list(d_values))
    ax.set_xlabel(style.DISTANCE_LABEL)
    ax.set_ylabel(style.ACCURACY_LABEL)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(**{"loc": "lower right", **dict(legend_kwargs or {})})
    return stats


def make_write_by_strength(ax: Axes, trials: Mapping[str, pd.DataFrame], *, control_ax: Axes,
                           d_min: int = 2, d_max: int = 6, marker_size: float = 3.5,
                           legend_kwargs: Mapping | None = None) -> dict:
    """Agreement with an ordering written along the rank direction, by write strength.

    One series per reasoning budget on the reversed context; the control (the same write at
    full strength along a random direction) is drawn as open bars on ``control_ax``.
    """
    stats: dict = {"series": {}, "control": {}}
    strengths = set()
    for index, (test, budget, conditions) in enumerate(WRITE_SERIES):
        xs, got = [], []
        for condition in conditions:
            xs.append(0.0 if condition == "intact" else float(condition.removeprefix("write_")))
            got.append(analysis.pooled_accuracy(
                _rows(trials, run="rank", test=test, condition=condition, budget=budget),
                "agree_written", d_min=d_min, d_max=d_max))
        strengths.update(xs)
        acc = np.array([g["acc"] for g in got])
        se = np.array([g["se"] for g in got])
        draw_series(ax, xs, acc, acc - se, acc + se, color=WRITE_COLORS[budget],
                    label=WRITE_LABELS[budget], marker_size=marker_size)
        stats["series"][budget] = dict(zip(xs, np.round(acc, 4)))
        ctrl = analysis.pooled_accuracy(_rows(trials, run="rank", test=test, condition=WRITE_CONTROL,
                                              budget=budget), "agree_written", d_min=d_min, d_max=d_max)
        stats["control"][budget] = round(ctrl["acc"], 4)
        control_ax.bar([index], [ctrl["acc"]], width=CONTROL_BAR_WIDTH, yerr=[ctrl["se"]],
                       color=style.OPEN_MARKER_FACE, edgecolor=WRITE_COLORS[budget],
                       linewidth=style.AXES_LINEWIDTH_PT, zorder=LINE_ZORDER,
                       error_kw={"ecolor": WRITE_COLORS[budget], "lw": style.AXES_LINEWIDTH_PT,
                                 "capsize": style.ERRORBAR_CAPSIZE,
                                 "capthick": style.AXES_LINEWIDTH_PT})
    ticks = sorted(strengths)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{s:g}" for s in ticks])
    ax.set_xlim(ticks[0] - WRITE_X_PAD, ticks[-1] + WRITE_X_PAD)
    ax.set_xlabel("Write strength")
    ax.set_ylabel("Agreement with written ordering")
    ax.set_ylim(-0.02, 1.02)
    style.chance_line(ax)
    control_ax.set_xticks([])
    margin = (1.0 - CONTROL_BAR_WIDTH) / 2.0
    control_ax.set_xlim(-0.5 - margin, len(WRITE_SERIES) - 0.5 + margin)
    control_ax.set_xlabel("Random\ndirection")
    control_ax.tick_params(axis="y", left=False, labelleft=False)
    control_ax.spines["left"].set_visible(False)
    control_ax.set_ylim(*ax.get_ylim())
    style.chance_line(control_ax)
    ax.legend(**{"loc": "upper left", **dict(legend_kwargs or {})})
    return stats


def make_probe_by_layer(ax: Axes, probe: pd.DataFrame, *, marker_size: float = 2.5,
                        legend_kwargs: Mapping | None = None) -> dict:
    """The rank direction's held-out ordering accuracy at each hidden state, one line per dimension.

    Layer 0 is the token embeddings, 1-64 the outputs of the model's blocks.
    """
    stats: dict = {"peak": {}}
    for dim in PROBE_COLORS:
        rows = probe[probe["dim"] == dim].sort_values("layer")
        ax.plot(rows["layer"], rows["acc_axis_heldout"], color=PROBE_COLORS[dim],
                marker=PROBE_MARKERS[dim], markersize=marker_size, markevery=8,
                label=roster.DIMENSION_LABELS[dim], zorder=LINE_ZORDER)
        stats["peak"][dim] = round(float(rows["acc_axis_heldout"].max()), 4)
    ax.set_xticks(list(PROBE_LAYER_TICKS))
    ax.set_xticklabels(["Embed." if t == 0 else str(t) for t in PROBE_LAYER_TICKS])
    ax.set_xlabel("Layer")
    ax.set_ylabel("Probe accuracy")
    ax.set_ylim(-0.02, 1.02)
    ax.axhline(style.CHANCE, color=style.REFERENCE_LINE, ls="--", lw=style.AXES_LINEWIDTH_PT,
               zorder=BAND_ZORDER)
    ax.legend(**{"loc": "lower right", **dict(legend_kwargs or {})})
    return stats
