"""The LLM panels of the figure "Content effects in transitive inference": Qwen3-32B's direct
accuracy by distance on US state population with no context and under the uninformative,
congruent, reversed and shuffled contexts, the reasoning budget the three informative contexts
need, with each context's accuracy at 1,024 reasoning tokens (distances 2 to 6) shown beside
its line, and the congruent and reversed contexts' best accuracy so far by reasoning budget;
over the roster, IWL accuracy against reversed-context accuracy for every model and item set,
and each model's content effect against its recovery with reasoning on every dimension.
Tables: content_effects/direct, ladders; symbolic_distance/direct; prior_strength/direct.
"""

from __future__ import annotations

import argparse

from matplotlib.figure import Figure

from _shared import column_titles, direct, ladder, letter_panels, main, mosaic_figure
from fig_symbolic_distance import BAND, BAND_XLIM, distance_key
from geo_alg_ti_llms import analysis, plots, roster
from geo_alg_ti_llms.plots import style

NAME = "content_effects"
MODEL = "qwen3-32b"
DIMENSION = "usstates"
# The query alone is drawn last, as the control (open markers on a dashed line).
CONDITIONS = ("unrelated_context", "congruent", "reversed", "shuffled", roster.IWL_CONDITION)
BUDGET_CONDITIONS = ("congruent", "reversed", "shuffled")
BAND_CONDITIONS = ("congruent", "reversed")      # congruent first: the reversed lines sit on top
MOSAIC = (("acc", "prior"), ("budget", "recovery"), ("band", "band"))
LETTERS = {"acc": "A", "prior": "B", "budget": "C", "recovery": "D", "band": "E"}
TITLES = {"acc": f"LLM ({roster.MODEL_LABELS[MODEL]})", "budget": f"LLM ({roster.MODEL_LABELS[MODEL]})",
          "prior": "LLMs", "recovery": "LLMs", "band": f"LLM ({roster.MODEL_LABELS[MODEL]})"}
HEIGHT_IN = 4.9
LEGEND = {"loc": "lower left", "handlelength": 1.5, "labelspacing": 0.2, "fontsize": 5.0,
          "frameon": True, "framealpha": 0.5}
PRIOR = dict(xlim=(0.24, 1.00), ylim=(0.25, 0.74), xticks=(0.4, 0.6, 0.8, 1.0), x_ref=0.5,
             y_ref=0.5, alpha=0.6, emphasis_col="is_main", fit_line=True, show_rho=True)
# Ticks every 256 tokens up to the drawn data.
BUDGET_YTICKS = (0, 256, 512, 768)
# Each context's accuracy at the largest budget sits clear of the lines and bands (data units).
NUMBER_ANCHORS = {"congruent": (5.15, 130.0), "reversed": (5.15, 705.0), "shuffled": (5.15, 260.0)}
# The two contexts of the band panel, named in their colors under the rising lines (data units).
BAND_NAMES = {"congruent": (560.0, 0.33), "reversed": (560.0, 0.21)}


def build(args: argparse.Namespace) -> Figure:
    """Assemble the five panels; the conditions' legend sits on the accuracy panel."""
    trials, ladders = direct(args.results), ladder(args.results)
    fig, axs = mosaic_figure(MOSAIC, width_in=style.COLUMN_WIDTH_IN, height_in=HEIGHT_IN)
    own = trials[(trials["model"] == MODEL) & (trials["dimension"] == DIMENSION)]
    plots.make_accuracy_by_distance(axs["acc"], own, conditions=CONDITIONS,
                                    open_marker_conditions=(roster.IWL_CONDITION,),
                                    linestyles={roster.IWL_CONDITION: "--"},
                                    ylabel=style.ACCURACY_NO_REASONING_LABEL, show_legend=True,
                                    legend_kwargs=LEGEND)
    own_ladder = ladders[(ladders["model"] == MODEL) & (ladders["dimension"] == DIMENSION)]
    plots.make_budget_by_distance(axs["budget"], own_ladder, conditions=BUDGET_CONDITIONS,
                                  unreached_by_condition=True, unreached_offset=0.8,
                                  yticks=BUDGET_YTICKS)
    plots.make_accuracy_numbers(axs["budget"], own_ladder, conditions=BUDGET_CONDITIONS,
                                anchors=NUMBER_ANCHORS)
    plots.make_accuracy_by_budget(axs["band"], own_ladder, conditions=BAND_CONDITIONS, **BAND)
    axs["band"].set_xlim(*BAND_XLIM)
    for condition, (x, y) in BAND_NAMES.items():
        axs["band"].text(x, y, style.CONDITION_LABELS[condition].lower(),
                         color=style.CONDITION_COLORS[condition], fontsize=style.FONT_LEGEND_PT)
    distance_key(axs["band"], style.DISTANCE_CMAPS["key"])
    competent = analysis.competence(trials).query("competent")["model"]
    prior = analysis.prior_strength_points(trials, analysis.spacing_trials(args.results))
    prior = prior[prior["model"].isin(competent)]
    outside = ~prior["x"].between(*PRIOR["xlim"]) | ~prior["y"].between(*PRIOR["ylim"])
    if outside.any():
        raise ValueError(f"{int(outside.sum())} prior-strength points fall outside the panel")
    plots.make_model_scatter(axs["prior"], prior, xlabel="No-context accuracy",
                             ylabel="Reversed accuracy", **PRIOR)
    plots.make_model_scatter(axs["recovery"], analysis.recovery_points(ladders),
                             xlabel="Congruent − reversed", ylabel="Recovery slope",
                             quadrant="upper_right", symmetric=True)
    column_titles(axs, TITLES)
    letter_panels(axs, LETTERS)
    return fig


if __name__ == "__main__":
    main(build, name=NAME)
