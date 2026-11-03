"""The appendix figures "LLM content effects: <family> models" (one per page of the roster):
direct accuracy by symbolic distance with no context and under the uninformative, congruent,
reversed and shuffled contexts, one panel per model (rows) and dimension (columns). The contexts
were run with the instruction to answer only from the comparisons given; the IWL test pools the
dimension's first two item sets.
Tables: content_effects/direct, symbolic_distance/direct.
"""

from __future__ import annotations

import argparse

from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from _shared import PAGES, direct, main, model_page
from geo_alg_ti_llms import plots, roster
from geo_alg_ti_llms.plots import grid, style

NAME = "content_effects_by_model"
# The query alone is drawn last, as the control (open markers on a dashed line).
CONDITIONS = ("unrelated_context", "congruent", "reversed", "shuffled", roster.IWL_CONDITION)
CONTROL = (roster.IWL_CONDITION,)


def build(args: argparse.Namespace, page: str, *,
          gutter_in: float = grid.GUTTER_IN) -> Figure:
    """One page of the figure, the row headers in a gutter ``gutter_in`` inches wide."""
    trials = direct(args.results)
    trials = trials[trials["grid_set"]]

    def draw(ax, model, dimension):
        rows = trials[(trials["model"] == model) & (trials["dimension"] == dimension)]
        plots.make_accuracy_by_distance(ax, rows, conditions=CONDITIONS,
                                        open_marker_conditions=CONTROL,
                                        linestyles={c: "--" for c in CONTROL})

    fig, _ = model_page(page, draw, xlabel=style.DISTANCE_LABEL, ylabel=style.ACCURACY_LABEL,
                        yticks=(0.0, 0.5, 1.0), gutter_in=gutter_in)
    handles = [Line2D([], [], color=style.CONDITION_COLORS[c], marker="o", markersize=3.5,
                      ls="--" if c in CONTROL else "-", markeredgecolor=style.CONDITION_COLORS[c],
                      markerfacecolor=style.OPEN_MARKER_FACE if c in CONTROL
                      else style.CONDITION_COLORS[c]) for c in CONDITIONS]
    grid.legend(fig, handles, [style.CONDITION_LABELS[c] for c in CONDITIONS])
    return fig


if __name__ == "__main__":
    main(build, name=NAME, pages=PAGES)
