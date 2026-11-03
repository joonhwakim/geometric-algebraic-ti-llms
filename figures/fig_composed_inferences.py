"""The appendix figure "Composed inferences in LLM reasoning chains": the inferences Qwen3-32B
and Qwen2.5-32B-Instruct had composed in their reasoning chains by each reasoning budget, one
line per symbolic distance (two to six), on the ICL test and under a reversed context (US state
population); dotted lines mark the number each distance needs.
Tables: reasoning_traces/chain_steps.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from _shared import letter_panels, main, mosaic_figure
from geo_alg_ti_llms import analysis, plots, roster
from geo_alg_ti_llms.plots import style

NAME = "composed_inferences"
HEIGHT_IN = 4.3
MOSAIC = (("icl_qwen3", "icl_qwen25"), ("rev_qwen3", "rev_qwen25"))
LETTERS = {"icl_qwen3": "A", "icl_qwen25": "B", "rev_qwen3": "C", "rev_qwen25": "D"}
PANELS = {
    "icl_qwen3": ("qwen3-32b", "icl_people"), "icl_qwen25": ("qwen2.5-32b-instruct", "icl_people"),
    "rev_qwen3": ("qwen3-32b", "reversed"), "rev_qwen25": ("qwen2.5-32b-instruct", "reversed"),
}
TITLES = {"icl_people": "ICL test", "reversed": "Reversed context"}
CMAPS = {"icl_people": style.DISTANCE_CMAPS["icl_people"], "reversed": style.DISTANCE_CMAPS["reversed"]}
PANEL = dict(xlim=(-24, 1048), ylim=(-0.2, 5.4), yticks=(0, 1, 2, 3, 4, 5),
             legend_kwargs={"loc": "upper left", "frameon": True, "framealpha": 1.0})
# The distance keys: edgeless opaque boxes flush with each panel's corner, so the dotted lines
# stop behind them; the y axis is drawn above the box.
LEGEND_RC = {"legend.handlelength": 1.4, "legend.labelspacing": 0.25, "legend.edgecolor": "none",
             "legend.borderaxespad": 0.0, "legend.fancybox": False}


def build(args: argparse.Namespace) -> Figure:
    """The four panels."""
    chains = analysis.chain_steps(args.results)
    # The figure and the shared style first: inside the block the style would override the legend
    # settings, and leaving the block would undo the style (its fonts) before the figure is saved.
    fig, axs = mosaic_figure(MOSAIC, width_in=style.FULL_WIDTH_IN, height_in=HEIGHT_IN,
                             wspace=0.06, hspace=0.08)
    with plt.rc_context(LEGEND_RC):
        for name, (model, condition) in PANELS.items():
            rows = chains[(chains["model"] == model) & (chains["condition"] == condition)]
            plots.make_composed_inferences_by_budget(axs[name], rows, cmap=CMAPS[condition], **PANEL)
            axs[name].set_title(f"{roster.MODEL_LABELS[model]}, {TITLES[condition]}", pad=4.0)
            axs[name].spines["left"].set_zorder(6)
        letter_panels(axs, LETTERS)
        fig.canvas.draw()
    return fig


if __name__ == "__main__":
    main(build, name=NAME)
