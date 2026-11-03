"""The LLM panel of the figure "Mechanisms underlying the geometric and algebraic strategies":
Qwen3-32B's accuracy on US state population with the rank direction or a random direction
erased at every hidden state, and with the 32 context-reading heads or 32 random heads ablated,
on the IWL test and the reversed context with no reasoning and on the reversed context and the
ICL test after 1,024 reasoning tokens.
Tables: representation/trials, components/trials.
"""

from __future__ import annotations

import argparse

from matplotlib.figure import Figure

from _shared import column_titles, main, mosaic_figure
from geo_alg_ti_llms import analysis, roster
from geo_alg_ti_llms.plots import representation, style

NAME = "mechanisms"
DIMENSION = "usstates"
HEIGHT_IN = 2.6
# Adjacent pairs need no inference, so each bar pools distances two and above.
D_MIN, D_MAX = 2, 6
# The key runs above the panel's title, intact alone, then each manipulation over its control.
KEY = {"loc": "outside upper center", "ncol": 3, "frameon": False, "handlelength": 1.2,
       "columnspacing": 0.8}


def dimension_trials(results, dimension: str) -> dict:
    """Both runs' trials on one dimension."""
    return {run: frame[frame["dim"] == dimension]
            for run, frame in analysis.intervention_trials(results).items()}


def build(args: argparse.Namespace) -> Figure:
    """The bars, with their key above the panel."""
    fig, axs = mosaic_figure((("bars",),), width_in=style.COLUMN_WIDTH_IN, height_in=HEIGHT_IN)
    representation.make_intervention_bars(axs["bars"], dimension_trials(args.results, DIMENSION),
                                          d_min=D_MIN, d_max=D_MAX)
    fig.legend(handles=representation.intervention_legend_handles(), **KEY)
    column_titles(axs, {"bars": f"LLM ({roster.MODEL_LABELS['qwen3-32b']})"})
    return fig


if __name__ == "__main__":
    main(build, name=NAME)
