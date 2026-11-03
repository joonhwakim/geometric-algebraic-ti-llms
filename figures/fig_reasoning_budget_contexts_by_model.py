"""The appendix figures "LLM reasoning requirements under content conditions: <family> models"
(one per page of the roster): the reasoning budget each symbolic distance needs under the
congruent, reversed and shuffled contexts, one panel per model (rows) and dimension (columns). A
distance whose best accuracy is not above chance at any budget is marked unreached, in its
context's color.
Tables: content_effects/ladders.
"""

from __future__ import annotations

import argparse

from matplotlib.figure import Figure

from _shared import PAGES, ladder, main, model_page
from fig_reasoning_budget_by_model import TOP_BUDGET, UNREACHED_SIZE, UNREACHED_STEP, YLABEL
from fig_reasoning_budget_by_model import LEGEND_HANDLERS, legend_handles
from geo_alg_ti_llms import plots
from geo_alg_ti_llms.plots import grid, style

NAME = "reasoning_budget_contexts_by_model"
CONDITIONS = ("congruent", "reversed", "shuffled")
# Room for three crosses stacked one step apart at a distance no context reaches.
PANEL = dict(clip_band=True, break_at_unreached=True, unreached_step=UNREACHED_STEP,
             unreached_size=UNREACHED_SIZE, unreached_by_condition=True, unreached_offset=1.0,
             ylim=(-32.0, TOP_BUDGET + UNREACHED_STEP * 3.0 + 48.0), yticks=(0, 512, 1024))


def build(args: argparse.Namespace, page: str, *,
          gutter_in: float = grid.GUTTER_IN) -> Figure:
    """One page of the figure, the row headers in a gutter ``gutter_in`` inches wide."""
    ladders = ladder(args.results)

    def draw(ax, model, dimension):
        rows = ladders[(ladders["model"] == model) & (ladders["dimension"] == dimension)]
        plots.make_budget_by_distance(ax, rows, conditions=CONDITIONS, pool_item_sets=False,
                                      ylabel=YLABEL, **PANEL)

    fig, _ = model_page(page, draw, xlabel=style.DISTANCE_LABEL, ylabel=YLABEL,
                        yticks=PANEL["yticks"], gutter_in=gutter_in)
    grid.legend(fig, *legend_handles(CONDITIONS, style.CONDITION_LABELS),
                handler_map=LEGEND_HANDLERS)
    return fig


if __name__ == "__main__":
    main(build, name=NAME, pages=PAGES)
