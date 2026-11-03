"""The appendix figure "LLM mechanism results: Qwen3-32B": the rank direction's held-out
ordering accuracy by layer on three dimensions; IWL test accuracy by distance on US state
population with the rank direction intact, erased or inverted, or a random direction erased;
agreement with another ordering written along the rank direction, with no reasoning and after
1,024 reasoning tokens; and the intervention bars on country and city population.
Tables: representation/probe, trials; components/trials.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from _shared import H_PAD_IN, W_PAD_IN, main
from fig_mechanisms import D_MAX, D_MIN, dimension_trials
from geo_alg_ti_llms import analysis, roster
from geo_alg_ti_llms.plots import representation, style

NAME = "representation"
HEIGHT_IN = 4.75
TOP = ("probe", "erase", "write", "strip")
TOP_WIDTHS = (1.0, 1.0, 0.92, 0.2)
BOTTOM = ("bars_countries", "bars_cities")
ROW_HEIGHTS = (1.0, 0.04, 1.0)              # the top row, a gap, the bottom row
KEY_BAND_IN = 0.30                          # the bars' key, in a band under the bottom row
LETTERS = {"probe": "A", "erase": "B", "write": "C", "bars_countries": "D", "bars_cities": "E"}
EDIT_DIMENSION = "usstates"
KEY = {"ncol": 3, "frameon": False, "handlelength": 1.5, "columnspacing": 1.2,
       "handletextpad": 0.5, "labelspacing": 0.3, "borderaxespad": 0.0}


def build(args: argparse.Namespace) -> Figure:
    """Two rows with their own column splits; one key under the bars."""
    style.apply_style()
    fig = plt.figure(figsize=(style.FULL_WIDTH_IN, HEIGHT_IN))
    band = KEY_BAND_IN / HEIGHT_IN
    fig.get_layout_engine().set(w_pad=W_PAD_IN, h_pad=H_PAD_IN, wspace=0.05, hspace=0.04,
                                rect=(0.0, band, 1.0, 1.0 - band))
    rows = fig.add_gridspec(3, 1, height_ratios=ROW_HEIGHTS)
    top = rows[0].subgridspec(1, len(TOP), width_ratios=TOP_WIDTHS)
    bottom = rows[2].subgridspec(1, len(BOTTOM))
    axs = {name: fig.add_subplot(top[0, i]) for i, name in enumerate(TOP)}
    axs.update({name: fig.add_subplot(bottom[0, i]) for i, name in enumerate(BOTTOM)})
    edits = dimension_trials(args.results, EDIT_DIMENSION)
    representation.make_probe_by_layer(axs["probe"], analysis.probe_table(args.results),
                                       legend_kwargs={"handlelength": 1.5})
    representation.make_edit_by_distance(axs["erase"], edits,
                                         legend_kwargs={"handlelength": 1.5, "labelspacing": 0.3})
    representation.make_write_by_strength(axs["write"], edits, control_ax=axs["strip"],
                                          d_min=D_MIN, d_max=D_MAX,
                                          legend_kwargs={"handlelength": 1.5})
    axs["strip"].xaxis.labelpad += 5.0      # level with the write panel's x label
    for name, dim in (("bars_countries", "countries"), ("bars_cities", "cities")):
        representation.make_intervention_bars(axs[name], dimension_trials(args.results, dim),
                                              d_min=D_MIN, d_max=D_MAX)
        axs[name].set_title(roster.DIMENSION_LABELS[dim], pad=4.0)
    for name in ("erase", "write"):
        axs[name].set_title(roster.DIMENSION_LABELS[EDIT_DIMENSION], pad=4.0)
    for name, letter in LETTERS.items():
        style.panel_letter(axs[name], letter)
    fig.canvas.draw()
    left, right = axs["bars_countries"].get_position(), axs["bars_cities"].get_position()
    fig.legend(handles=representation.intervention_legend_handles(), loc="lower center",
               bbox_to_anchor=((left.x0 + right.x1) / 2, 0.03 / HEIGHT_IN), **KEY)
    return fig


if __name__ == "__main__":
    main(build, name=NAME)
