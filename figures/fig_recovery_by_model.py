"""The appendix figures "Recovery with reasoning: <family> models" (one per page of the roster):
accuracy by reasoning budget under the congruent and reversed contexts, one line per symbolic
distance (light near, dark far), one panel per model (rows) and dimension (columns). Budget zero
is the direct answer; each larger budget scores the first B tokens of the same reasoning chain.
Tables: content_effects/ladders.
"""

from __future__ import annotations

import argparse

from matplotlib.figure import Figure
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.lines import Line2D

from _shared import PAGES, ladder, main, model_page
from geo_alg_ti_llms import analysis, plots
from geo_alg_ti_llms.plots import grid, style

NAME = "recovery_by_model"
CONDITIONS = ("congruent", "reversed")       # congruent first: the reversed lines sit on top
XTICKS = (0, 512, 1024)
# A narrow panel bunches the eleven budgets at the left of a token axis: smaller markers.
MARKER_SIZE = 2.0
XLABEL = style.REASONING_BUDGET_LABEL.replace(" (", "\n(")


class _ShortStroke(HandlerLine2D):
    """A legend handle drawn as a stroke over the right ``fraction`` of its box."""

    def __init__(self, fraction: float):
        super().__init__()
        self.fraction = float(fraction)

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize,
                       trans):
        short = width * self.fraction
        return super().create_artists(legend, orig_handle, xdescent - (width - short), ydescent,
                                      short, height, fontsize, trans)


def build(args: argparse.Namespace, page: str, *, gutter_in: float = grid.GUTTER_IN,
          key_stroke: float = 1.0) -> Figure:
    """One page of the figure; the legend keys the contexts and the distance shades.

    The row headers sit in a gutter ``gutter_in`` inches wide; each distance shade of the key is
    a stroke ``key_stroke`` of the legend's handle length, at its label's side.
    """
    ladders = ladder(args.results)

    def draw(ax, model, dimension):
        rows = ladders[(ladders["model"] == model) & (ladders["dimension"] == dimension)]
        plots.make_accuracy_by_budget(ax, rows, conditions=CONDITIONS, xticks=XTICKS,
                                      marker_size=MARKER_SIZE)

    fig, _ = model_page(page, draw, xlabel=XLABEL, ylabel=style.ACCURACY_LABEL,
                        yticks=(0.0, 0.5, 1.0), xticks=XTICKS, gutter_in=gutter_in)
    handles = [Line2D([], [], color=style.CONDITION_COLORS[c], marker="o", markersize=MARKER_SIZE)
               for c in CONDITIONS]
    labels = [style.CONDITION_LABELS[c] for c in CONDITIONS]
    handles.append(Line2D([], [], ls="none"))
    labels.append(style.DISTANCE_LEGEND_TITLE)
    key = [Line2D([], [], color=shade)
           for shade in style.distance_colors(analysis.DISTANCES, style.DISTANCE_CMAPS["key"])]
    handles += key
    labels += [str(d) for d in analysis.DISTANCES]
    short = None if key_stroke == 1.0 else {h: _ShortStroke(key_stroke) for h in key}
    grid.legend(fig, handles, labels, handler_map=short)
    return fig


if __name__ == "__main__":
    main(build, name=NAME, pages=PAGES)
