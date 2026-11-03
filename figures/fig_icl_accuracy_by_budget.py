"""The appendix figure "LLM ICL accuracy by reasoning budget": ICL test accuracy by reasoning
budget, one panel per model: the given names, one line per symbolic distance (light near, dark
far), raw at each budget. The four groups of the roster are the rows (models trained to reason
in the first two, models prompted to reason in the last two).
Tables: symbolic_distance/ladders.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from _shared import ladder, main
from geo_alg_ti_llms import analysis, plots, roster
from geo_alg_ti_llms.plots import grid, style   # grid.finish: shared ticks and edge labels

NAME = "icl_accuracy_by_budget"
CONDITION = roster.ICL_CONDITION
N_COLS = 6
ROW_HEIGHT_IN = 1.2          # one row of panels with their two-line titles
CHROME_HEIGHT_IN = 0.62      # the x label, tick labels and the legend under the grid
PADS = {"w_pad": 0.03, "h_pad": 0.03, "wspace": 0.05, "hspace": 0.06}
TITLE_PAD_PT = 2.5
MARKER_SIZE = 2.5
XTICKS = (0, 512, 1024)
YTICKS = (0.0, 0.5, 1.0)
XLIM = (-40, 1064)
XLABEL = style.REASONING_BUDGET_LABEL.replace(" (", "\n(")   # on one line it is wider than a panel


def build(args: argparse.Namespace, *, title_size: float = style.FONT_TICK_PT) -> Figure:
    """The page; one legend under the grid keys the distance shades. Each panel is titled with
    its model's name at ``title_size`` points."""
    rows = analysis.pick(ladder(args.results), condition=CONDITION)
    groups = list(roster.PAGES.values())
    style.apply_style()
    fig = plt.figure(figsize=(style.FULL_WIDTH_IN, CHROME_HEIGHT_IN + ROW_HEIGHT_IN * len(groups)))
    fig.get_layout_engine().set(**PADS)
    axes = fig.subplots(len(groups), N_COLS, sharex=True, sharey=True, squeeze=False)
    for r, models in enumerate(groups):
        for c in range(N_COLS):
            ax = axes[r, c]
            if c >= len(models):
                ax.set_axis_off()
                continue
            plots.make_accuracy_by_budget(ax, rows[rows["model"] == models[c]], conditions=(CONDITION,),
                                          xticks=XTICKS, marker_size=MARKER_SIZE)
            ax.set_xlim(*XLIM)
            ax.set_title(roster.MODEL_ROW_LABELS[models[c]], pad=TITLE_PAD_PT,
                         fontsize=title_size)
    grid.finish(axes, xlabel=XLABEL, ylabel=style.ACCURACY_LABEL, yticks=YTICKS, xticks=XTICKS)
    shades = style.distance_colors(analysis.DISTANCES, style.DISTANCE_CMAPS[CONDITION])
    handles = [Line2D([], [], color=s, marker="o", markersize=MARKER_SIZE) for s in shades]
    fig.legend(handles, [str(d) for d in analysis.DISTANCES], loc="outside lower center",
               ncol=len(handles), title=style.DISTANCE_LEGEND_TITLE, handlelength=1.5,
               frameon=False, fontsize=style.FONT_LEGEND_PT, title_fontsize=style.FONT_LEGEND_PT)
    return fig


if __name__ == "__main__":
    main(build, name=NAME)
