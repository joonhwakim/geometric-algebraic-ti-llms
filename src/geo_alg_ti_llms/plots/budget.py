"""Panels over the reasoning budget: accuracy in a context, and the inferences a chain composed.

One model per panel (the caller pins it), one line per symbolic distance in shades of the
condition's colormap (light near, dark far). Accuracy is raw at each budget, or with
``budget_stat="running_best"`` the best accuracy at that budget or below (with the Wilson
interval of the budget that reached it); composed inferences carry plus and minus one standard
error over trials.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from geo_alg_ti_llms import analysis
from geo_alg_ti_llms.plots import style
from geo_alg_ti_llms.plots.distance import BAND_ZORDER, draw_series

BAND_ALPHA = 0.18   # lighter than the distance panels' bands: six distances overlap here
_BUDGET_STATS: tuple[str, ...] = ("raw", "running_best")


def make_accuracy_by_budget(
    ax: Axes, ladder: pd.DataFrame, *, conditions: Sequence[str],
    xticks: Sequence[int] = style.BUDGET_TICKS, marker_size: float = 3.5,
    ylim: tuple[float, float] | None = (-0.02, 1.02), budget_stat: str = "raw",
    ylabel: str = style.ACCURACY_LABEL, xlim: tuple[float, float] | None = None,
    yticks: Sequence[float] | None = None,
) -> dict:
    """Accuracy against the reasoning budget in each context, one line per distance.

    ``budget_stat`` ``"raw"`` plots the accuracy at each budget; ``"running_best"`` plots, at
    each budget, the best accuracy at that budget or below, with the interval of the first
    budget that reached it (``analysis.running_best``). Conditions are drawn in the order
    given, so the last sits on top. ``xlim`` and ``yticks`` set the budget axis's range and the
    accuracy ticks (matplotlib's choices when None).
    """
    fname = "make_accuracy_by_budget"
    if budget_stat not in _BUDGET_STATS:
        raise ValueError(f"{fname}: budget_stat is one of {_BUDGET_STATS}, not {budget_stat!r}")
    d_values = analysis.DISTANCES
    rows = ladder[ladder["innermost"]]
    if rows["model"].nunique() > 1:
        raise ValueError(f"{fname}: pin the model first")
    stats = {"conditions": list(conditions), "budget_stat": budget_stat, "per_condition": {}}
    for condition in conditions:
        sub = analysis.pick(rows, condition=condition, d=[int(d) for d in d_values])
        per, lows, highs = {}, {}, {}
        for d, shade in zip(d_values, style.distance_colors(d_values, style.DISTANCE_CMAPS[condition])):
            table = analysis.trial_accuracy_by(analysis.pick(sub, d=int(d)), ["budget"], fname=fname)
            if budget_stat == "running_best":
                table = analysis.running_best(table)
            draw_series(ax, table["budget"], table["accuracy"], table["ci_lo"], table["ci_hi"],
                        color=shade, label=None, marker_size=marker_size, band_alpha=BAND_ALPHA)
            budgets = table["budget"].astype(int)
            per[int(d)] = dict(zip(budgets, table["accuracy"].round(4)))
            lows[int(d)] = dict(zip(budgets, table["ci_lo"].round(4)))
            highs[int(d)] = dict(zip(budgets, table["ci_hi"].round(4)))
        stats["per_condition"][condition] = {"accuracy_by_distance_and_budget": per,
                                             "ci_lo_by_distance_and_budget": lows,
                                             "ci_hi_by_distance_and_budget": highs}
    style.chance_line(ax)
    ax.set_xticks(list(xticks))
    ax.set_xlabel(style.REASONING_BUDGET_LABEL)
    ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if yticks is not None:
        ax.set_yticks(list(yticks))
    return stats


def make_composed_inferences_by_budget(
    ax: Axes, chains: pd.DataFrame, *, cmap: str, marker_size: float = 3.0,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None, xticks: Sequence[int] = style.BUDGET_TICKS,
    yticks: Sequence[int] | None = None, legend_kwargs: Mapping | None = None,
) -> dict:
    """The mean number of inferences a chain had composed by each budget, one line per distance.

    Distances two to six (an adjacent pair needs none). A query at distance d needs d - 1
    compositions, drawn as a dotted line in the distance's shade; a distance takes the shade it
    has on the panels that draw distances one to six.
    """
    fname = "make_composed_inferences_by_budget"
    d_values = analysis.DISTANCES[1:]
    rows = chains[chains["innermost"] & chains["d"].isin(d_values)]
    for col in ("model", "condition", "dim"):
        if rows[col].nunique() != 1:
            raise ValueError(f"{fname}: pin one {col} first")
    shade_of = dict(zip(analysis.DISTANCES, style.distance_colors(analysis.DISTANCES, cmap)))
    stats = {"model": str(rows["model"].iloc[0]), "condition": str(rows["condition"].iloc[0]),
             "per_distance": {}}
    for d in d_values:
        sub = rows[rows["d"] == d]
        need = int(sub["inferences_needed"].iloc[0])
        table = sub.groupby("budget")["inferences_composed"].agg(["mean", "std", "size"])
        se = (table["std"].fillna(0.0) / np.sqrt(table["size"])).to_numpy()
        mean = table["mean"].to_numpy()
        ax.axhline(need, color=shade_of[d], ls=":", lw=style.AXES_LINEWIDTH_PT, zorder=BAND_ZORDER)
        draw_series(ax, table.index.to_numpy(float), mean, mean - se, mean + se,
                    color=shade_of[d], label=str(d), marker_size=marker_size, band_alpha=BAND_ALPHA)
        stats["per_distance"][int(d)] = {
            "need": need, "trials": int(table["size"].iloc[0]),
            "mean_by_budget": dict(zip(table.index.astype(int), np.round(mean, 4)))}
    ax.set_xticks(list(xticks))
    if yticks is not None:
        ax.set_yticks(list(yticks))
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel(style.REASONING_BUDGET_LABEL)
    ax.set_ylabel(style.COMPOSED_INFERENCES_LABEL)
    ax.legend(title=style.DISTANCE_LEGEND_TITLE, **dict(legend_kwargs or {}))
    return stats
