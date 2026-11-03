"""Panels over symbolic distance: direct accuracy, and the reasoning budget needed; and each
condition's accuracy at one reasoning budget, as bars or as numbers beside its line.

Each panel draws one model (the caller pins it), one line per condition, onto the axes it is
given and returns the statistics it drew. Accuracy carries a Wilson interval over trials; the
reasoning budget a band of plus and minus its bootstrap standard deviation over trials.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import to_hex

from geo_alg_ti_llms import analysis
from geo_alg_ti_llms.plots import style

BAND_ZORDER = 1
LINE_ZORDER = 3
UNREACHED_ZORDER = 4
_NUMBER_ZORDER = 5
_NUMBER_OFFSET_PT = 4.0             # a number's gap above or below the end of its line
_BAR_WIDTH = 0.7                    # a bar's share of the space between bar centers


def draw_series(ax: Axes, x, y, lo, hi, *, color: str, label: str | None, marker_size: float,
                open_marker: bool = False, ls: str = "-", band_alpha: float = 0.25) -> None:
    """A line with markers over a translucent band from ``lo`` to ``hi``."""
    ax.fill_between(np.asarray(x, float), np.asarray(lo, float), np.asarray(hi, float),
                    color=color, alpha=band_alpha, lw=0, zorder=BAND_ZORDER)
    ax.plot(np.asarray(x, float), np.asarray(y, float), color=color, marker="o",
            markersize=marker_size, ls=ls, markeredgecolor=color,
            markerfacecolor=style.OPEN_MARKER_FACE if open_marker else color,
            label=label, zorder=LINE_ZORDER)


def _one_model(df: pd.DataFrame, fname: str) -> pd.DataFrame:
    rows = df[df["innermost"]]
    if rows["model"].nunique() > 1:
        raise ValueError(f"{fname}: pin the model first (rows carry "
                         f"{sorted(rows['model'].unique())})")
    return rows


def make_accuracy_by_distance(
    ax: Axes, trials: pd.DataFrame, *, conditions: Sequence[str],
    labels: Mapping[str, str] | None = None, open_marker_conditions: Sequence[str] = (),
    linestyles: Mapping[str, str] | None = None, marker_size: float = 3.5,
    ylim: tuple[float, float] | None = (-0.02, 1.02), ylabel: str = style.ACCURACY_LABEL,
    show_legend: bool = False, legend_kwargs: Mapping | None = None,
    yticks: Sequence[float] | None = None,
) -> dict:
    """Direct accuracy against symbolic distance, one line per condition, at the innermost ranks.

    A condition's item sets are pooled (the IWL test's replicates); every other swept axis must
    be pinned. Conditions in ``open_marker_conditions`` are drawn with open markers, and
    ``linestyles`` gives a condition its own line style: the paper's control idiom. ``yticks``
    places the accuracy ticks (matplotlib's choice when None).
    """
    fname = "make_accuracy_by_distance"
    labels = {**style.CONDITION_LABELS, **(labels or {})}
    linestyles = dict(linestyles or {})
    rows = _one_model(trials, fname)
    stats = {"conditions": list(conditions), "per_condition": {}}
    drawn = set()
    for condition in conditions:
        table = analysis.trial_accuracy_by(analysis.pick(rows, condition=condition), ["d"],
                                           pooled=("item_pool",), fname=fname)
        draw_series(ax, table["d"], table["accuracy"], table["ci_lo"], table["ci_hi"],
                    color=style.CONDITION_COLORS[condition], label=labels[condition],
                    marker_size=marker_size, open_marker=condition in open_marker_conditions,
                    ls=linestyles.get(condition, "-"))
        drawn.update(int(d) for d in table["d"])
        stats["per_condition"][condition] = {
            "accuracy_by_distance": dict(zip(table["d"].astype(int), table["accuracy"].round(4))),
            "n_by_distance": dict(zip(table["d"].astype(int), table["n"].astype(int))),
        }
    style.chance_line(ax)
    ax.set_xticks(sorted(drawn))
    ax.set_xlabel(style.DISTANCE_LABEL)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if yticks is not None:
        ax.set_yticks(list(yticks))
    if show_legend:
        ax.legend(**dict(legend_kwargs or {}))
    return stats


def _accuracy_at_budget(ladder: pd.DataFrame, *, conditions: Sequence[str], budget: int | None,
                        min_distance: int, fname: str) -> tuple[int, dict[str, dict]]:
    """Each condition's accuracy at one budget, pooled over trials, item sets and distances.

    The budget is the largest the rows hold when None; the record of each condition is
    ``analysis.accuracy_at_budget``'s, unrounded.
    """
    rows = _one_model(ladder, fname)
    budget = int(rows["budget"].max()) if budget is None else int(budget)
    return budget, {c: analysis.accuracy_at_budget(analysis.pick(rows, condition=c), budget=budget,
                                                   min_distance=min_distance, fname=fname)
                    for c in conditions}


def _rounded(record: dict) -> dict:
    """A pooled-accuracy record as the stats report it: counts whole, rates to four places."""
    return {k: (round(v, 4) if k in ("accuracy", "ci_lo", "ci_hi") else v)
            for k, v in record.items()}


def make_accuracy_bars(
    ax: Axes, ladder: pd.DataFrame, *, conditions: Sequence[str], budget: int | None = None,
    min_distance: int = 2, bar_width: float = _BAR_WIDTH,
    ylabel: str = style.ACCURACY_REASONING_LABEL, ylim: tuple[float, float] = (0.0, 1.02),
) -> dict:
    """One bar per condition: accuracy at one reasoning budget, pooled over distances.

    Each condition's rows at ``budget`` (the largest budget when None), the innermost ranks and
    distances from ``min_distance`` are pooled over trials and item sets; the error bar is the
    68% Wilson interval. Bars are in ``conditions`` order and their conditions' colors, with no
    tick labels (a legend elsewhere keys the colors).
    """
    fname = "make_accuracy_bars"
    budget, pooled = _accuracy_at_budget(ladder, conditions=conditions, budget=budget,
                                         min_distance=min_distance, fname=fname)
    stats = {"conditions": list(conditions), "budget": budget, "min_distance": min_distance,
             "per_condition": {c: _rounded(pooled[c]) for c in conditions}}
    for i, condition in enumerate(conditions):
        acc, lo, hi = (pooled[condition][k] for k in ("accuracy", "ci_lo", "ci_hi"))
        color = style.CONDITION_COLORS[condition]
        ax.bar([i], [acc], width=bar_width, yerr=[[max(acc - lo, 0.0)], [max(hi - acc, 0.0)]],
               color=color, edgecolor=color, linewidth=style.AXES_LINEWIDTH_PT,
               error_kw={"ecolor": style.ERROR_COLOR, "lw": style.AXES_LINEWIDTH_PT,
                         "capsize": style.ERRORBAR_CAPSIZE, "capthick": style.AXES_LINEWIDTH_PT},
               zorder=LINE_ZORDER)
    ax.set_xticks([])
    ax.set_xlim(-0.5, len(conditions) - 0.5)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    return stats


def _line_end(ax: Axes, color: str) -> tuple[float, float] | None:
    """The last point (largest x) of the lines with markers drawn on ``ax`` in ``color``, or None
    when there is none. Lines are matched by color only."""
    ends = []
    for line in ax.get_lines():
        unmarked = line.get_marker() in (None, "None", "", " ")
        if unmarked or to_hex(line.get_color()) != to_hex(color):
            continue
        xy = np.column_stack([np.asarray(line.get_xdata(), float),
                              np.asarray(line.get_ydata(), float)])
        xy = xy[np.isfinite(xy).all(axis=1)]
        if len(xy):
            ends.append(xy[np.argmax(xy[:, 0])])
    return tuple(float(v) for v in max(ends, key=lambda p: p[0])) if ends else None


def make_accuracy_numbers(
    ax: Axes, ladder: pd.DataFrame, *, conditions: Sequence[str], budget: int | None = None,
    min_distance: int = 2, number_format: str = "{:.2f}", fontsize: float = style.FONT_LEGEND_PT,
    anchors: Mapping[str, tuple[float, float]] | None = None, ha: str | None = None,
    va: str | None = None,
) -> dict:
    """Each condition's accuracy at one reasoning budget, pooled over distances, as a number in
    the condition's color beside its line.

    The estimand is ``make_accuracy_bars``'s: each condition's rows at ``budget`` (the largest
    budget when None), the innermost ranks and distances from ``min_distance``, pooled over
    trials and item sets, with the 68% Wilson interval in the stats. Each number is formatted
    with ``number_format`` and placed at ``anchors[condition]`` (data units; aligned left and
    bottom by default), or else on the last point of the condition's line already drawn on
    ``ax`` (``make_budget_by_distance``): aligned right on that point and a few points above it
    in the lower half of the panel, below it in the upper half, so it stays inside the axes.
    ``ha`` and ``va`` override the alignment. Lines are matched by color only, so a condition
    that shares its color with another on the same axes (the IWL test and the uninformative
    context) needs an anchor; a condition with neither a line nor an anchor raises. The stats
    hold each number's text, and ``artists`` the Text objects, so a caller can move them once
    the figure's layout is final.
    """
    fname = "make_accuracy_numbers"
    budget, pooled = _accuracy_at_budget(ladder, conditions=conditions, budget=budget,
                                         min_distance=min_distance, fname=fname)
    stats = {"conditions": list(conditions), "budget": budget, "min_distance": min_distance,
             "per_condition": {}, "artists": {}}
    for condition in conditions:
        color = style.CONDITION_COLORS[condition]
        label = number_format.format(pooled[condition]["accuracy"])
        anchor = (anchors or {}).get(condition)
        if anchor is not None:
            offset = (0.0, 0.0)
            text = ax.text(*anchor, label, color=color, fontsize=fontsize, ha=ha or "left",
                           va=va or "bottom", zorder=_NUMBER_ZORDER)
        else:
            anchor = _line_end(ax, color)
            if anchor is None:
                raise ValueError(f"{fname}: no line in {color} for {condition!r} on these axes; "
                                 "draw it first or pass anchors")
            low, high = ax.get_ylim()
            upper = (anchor[1] - low) / (high - low) > 0.5
            offset = (0.0, -_NUMBER_OFFSET_PT if upper else _NUMBER_OFFSET_PT)
            text = ax.annotate(label, anchor, xytext=offset, textcoords="offset points",
                               color=color, fontsize=fontsize, ha=ha or "right",
                               va=va or ("top" if upper else "bottom"), zorder=_NUMBER_ZORDER)
        stats["per_condition"][condition] = {**_rounded(pooled[condition]),
                                             "text": text.get_text(),
                                             "position": [float(v) for v in anchor],
                                             "offset_pt": list(offset)}
        stats["artists"][condition] = text
    return stats


def criterion_by_distance(ladder: pd.DataFrame, condition: str, *, pool_item_sets: bool = True,
                          n_boot: int | None = None,
                          fname: str = "criterion_by_distance") -> dict[int, dict]:
    """Per distance, ``analysis.budget_to_criterion`` on one condition of one model's ladder: the
    reasoning budget and its bootstrap that :func:`make_budget_by_distance` draws."""
    sub = analysis.pick(_one_model(ladder, fname), condition=condition)
    pooled = ("item_pool",) if pool_item_sets else ()
    return {d: analysis.budget_to_criterion(analysis.pick(sub, d=d), n_boot=n_boot, pooled=pooled,
                                            fname=fname)
            for d in sorted(int(d) for d in sub["d"].unique())}


def make_budget_by_distance(
    ax: Axes, ladder: pd.DataFrame, *, conditions: Sequence[str],
    labels: Mapping[str, str] | None = None, pool_item_sets: bool = True,
    n_boot: int | None = None, clip_band: bool = False, break_at_unreached: bool = False,
    unreached_step: float | None = None, unreached_by_condition: bool = False,
    unreached_offset: float = 0.3, unreached_size: float = 12.0, marker_size: float = 3.5,
    ylim: tuple[float, float] | None = None,
    yticks: Sequence[float] = style.BUDGET_TICKS, ylabel: str = style.REASONING_BUDGET_LABEL,
    criteria: Mapping[str, Mapping[int, dict]] | None = None,
) -> dict:
    """The reasoning budget each distance needs, one line per condition.

    Per condition and distance, the smallest budget whose accuracy reaches 90% of its maximum
    over budgets, with a band of plus and minus the bootstrap standard deviation (clipped at
    zero, and at the largest budget with ``clip_band``). A distance whose best accuracy is not
    above chance has no budget: it is marked with a cross ``unreached_step`` tokens above the
    largest budget (default: the spacing of the top two budgets), in its condition's color and
    raised ``unreached_offset`` of a step per condition with ``unreached_by_condition``, else
    in gray. ``break_at_unreached`` breaks a line there instead of joining across it.
    ``criteria`` gives a condition's :func:`criterion_by_distance` on these rows, with the same
    ``pool_item_sets`` and ``n_boot``, when it is already computed (a condition drawn in several
    panels is then bootstrapped once).
    """
    fname = "make_budget_by_distance"
    labels = {**style.CONDITION_LABELS, **(labels or {})}
    rows = _one_model(ladder, fname)
    budgets = sorted(int(b) for b in rows["budget"].unique())
    top = budgets[-1]
    step = float(budgets[-1] - budgets[-2]) if unreached_step is None else float(unreached_step)
    stats = {"conditions": list(conditions), "budgets": budgets, "per_condition": {},
             "unreached": {}}
    drawn = set()
    for condition in conditions:
        got = dict((criteria or {}).get(condition) or criterion_by_distance(
            rows, condition, pool_item_sets=pool_item_sets, n_boot=n_boot, fname=fname))
        distances = sorted(got)
        cap = float(top) if clip_band else np.inf
        band = {d: (max(0.0, g["budget"] - g["boot_sd"]), min(cap, g["budget"] + g["boot_sd"]))
                for d, g in got.items()}
        runs: list[list[int]] = [[d for d in distances if got[d]["above_chance"]]]
        if break_at_unreached:
            runs = [[]]
            for d in distances:
                if not got[d]["above_chance"]:
                    runs.append([])
                else:
                    runs[-1].append(d)
        for i, run in enumerate(r for r in runs if r):
            draw_series(ax, run, [got[d]["budget"] for d in run], [band[d][0] for d in run],
                        [band[d][1] for d in run], color=style.CONDITION_COLORS[condition],
                        label=labels[condition] if i == 0 else None, marker_size=marker_size)
        drawn.update(distances)
        stats["unreached"][condition] = [d for d in distances if not got[d]["above_chance"]]
        stats["per_condition"][condition] = {
            "budget_by_distance": {d: got[d]["budget"] for d in distances},
            "boot_sd_by_distance": {d: round(got[d]["boot_sd"], 2) for d in distances},
            "n_by_distance": {d: got[d]["n_trials"] for d in distances},
        }
    unreached = [(i, c, d) for i, c in enumerate(conditions) for d in stats["unreached"][c]]
    for i, condition, d in unreached:
        lift = 1.0 + (i * unreached_offset if unreached_by_condition else 0.0)
        ax.scatter([d], [top + step * lift], marker="x", s=unreached_size, zorder=UNREACHED_ZORDER,
                   color=style.CONDITION_COLORS[condition] if unreached_by_condition
                   else style.REFERENCE_LINE)
    if unreached:
        ax.scatter([], [], marker="x", s=unreached_size, color=style.REFERENCE_LINE,
                   label=style.UNREACHED_LABEL)
    ax.set_xticks(sorted(drawn))
    ax.set_xlabel(style.DISTANCE_LABEL)
    ax.set_ylabel(ylabel)
    ax.set_yticks(list(yticks))
    if ylim is not None:
        ax.set_ylim(*ylim)
    return stats
