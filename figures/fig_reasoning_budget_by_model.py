"""The appendix figures "LLM reasoning requirements: <family> models" (one per page of the
roster): the reasoning budget each symbolic distance needs on the IWL and ICL tests, one panel
per model (rows) and dimension (columns). The IWL test pools every item set of the dimension;
the ICL test (given names) is drawn in every column of a row. A distance whose best accuracy is
not above chance at any budget is marked unreached, by a cross in its test's color.
Tables: symbolic_distance/ladders.
"""

from __future__ import annotations

import argparse

import numpy as np
from matplotlib.figure import Figure
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D

from _shared import PAGES, ladder, main, model_page
from geo_alg_ti_llms import plots, roster
from geo_alg_ti_llms.plots import distance, grid, style

NAME = "reasoning_budget_by_model"
CONDITIONS = (roster.ICL_CONDITION, roster.IWL_CONDITION)
TOP_BUDGET = 1024
UNREACHED_STEP = 80.0
UNREACHED_SIZE = 12.0
# Room for two crosses stacked one step apart at a distance neither test reaches.
PANEL = dict(clip_band=True, break_at_unreached=True, unreached_step=UNREACHED_STEP,
             unreached_size=UNREACHED_SIZE, unreached_by_condition=True, unreached_offset=1.0,
             ylim=(-32.0, TOP_BUDGET + UNREACHED_STEP * 2.0 + 48.0), yticks=(0, 512, 1024))
# On one line the budget label runs taller than a panel.
YLABEL = style.REASONING_BUDGET_LABEL.replace(" (", "\n(")


def panel_rows(ladders, model: str, dimension: str):
    """One model's ICL ladder and its IWL ladder on every item set of one dimension."""
    rows = ladders[ladders["model"] == model]
    return rows[(rows["condition"] == roster.ICL_CONDITION)
                | ((rows["condition"] == roster.IWL_CONDITION) & (rows["dimension"] == dimension))]


def legend_handles(conditions, labels):
    """Each condition's line and, under one label, its unreached cross, as a panel draws them;
    pass ``LEGEND_HANDLERS`` with them so the crosses sit side by side."""
    handles = [Line2D([], [], color=style.CONDITION_COLORS[c], marker="o", markersize=3.5)
               for c in conditions]
    handles.append(tuple(Line2D([], [], marker="x", ls="none", color=style.CONDITION_COLORS[c],
                                markersize=float(np.sqrt(UNREACHED_SIZE))) for c in conditions))
    return handles, [labels[c] for c in conditions] + [style.UNREACHED_LABEL]


LEGEND_HANDLERS = {tuple: HandlerTuple(ndivide=None)}


def build(args: argparse.Namespace, page: str, *,
          gutter_in: float = grid.GUTTER_IN) -> Figure:
    """One page of the figure, the row headers in a gutter ``gutter_in`` inches wide."""
    ladders = ladder(args.results)
    icl = {}        # the ICL test is the same in every column of a row: bootstrapped once per model

    def draw(ax, model, dimension):
        rows = panel_rows(ladders, model, dimension)
        if model not in icl:
            icl[model] = distance.criterion_by_distance(rows, roster.ICL_CONDITION)
        plots.make_budget_by_distance(ax, rows, conditions=CONDITIONS, labels=style.TEST_LABELS,
                                      ylabel=YLABEL, criteria={roster.ICL_CONDITION: icl[model]},
                                      **PANEL)

    fig, _ = model_page(page, draw, xlabel=style.DISTANCE_LABEL, ylabel=YLABEL,
                        yticks=PANEL["yticks"], gutter_in=gutter_in)
    grid.legend(fig, *legend_handles(CONDITIONS, style.TEST_LABELS), handler_map=LEGEND_HANDLERS)
    return fig


if __name__ == "__main__":
    main(build, name=NAME, pages=PAGES)
