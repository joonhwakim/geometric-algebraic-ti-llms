"""The appendix figures "LLM symbolic-distance results: <family> models" (one per page of the
roster): direct accuracy by symbolic distance on the IWL and ICL tests, one panel per model
(rows) and dimension (columns). The IWL test pools the dimension's first two item sets; the ICL
test (given names) is one measurement, drawn in every column of a row.
Tables: symbolic_distance/direct.
"""

from __future__ import annotations

import argparse

from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from _shared import PAGES, direct, main, model_page
from geo_alg_ti_llms import plots, roster
from geo_alg_ti_llms.plots import grid, style

NAME = "symbolic_distance_by_model"
CONDITIONS = (roster.ICL_CONDITION, roster.IWL_CONDITION)


def panel_rows(trials, model: str, dimension: str):
    """One model's ICL test rows and its IWL test rows on the grid sets of one dimension."""
    rows = trials[trials["model"] == model]
    return rows[(rows["condition"] == roster.ICL_CONDITION)
                | ((rows["condition"] == roster.IWL_CONDITION) & (rows["dimension"] == dimension)
                   & rows["grid_set"])]


def build(args: argparse.Namespace, page: str, *,
          gutter_in: float = grid.GUTTER_IN) -> Figure:
    """One page of the figure, the row headers in a gutter ``gutter_in`` inches wide."""
    trials = direct(args.results)

    def draw(ax, model, dimension):
        plots.make_accuracy_by_distance(ax, panel_rows(trials, model, dimension),
                                        conditions=CONDITIONS)

    fig, _ = model_page(page, draw, xlabel=style.DISTANCE_LABEL, ylabel=style.ACCURACY_LABEL,
                        yticks=(0.0, 0.5, 1.0), gutter_in=gutter_in)
    grid.legend(fig, [Line2D([], [], color=style.CONDITION_COLORS[c], marker="o", markersize=3.5)
                      for c in CONDITIONS], [style.TEST_LABELS[c] for c in CONDITIONS])
    return fig


if __name__ == "__main__":
    main(build, name=NAME, pages=PAGES)
