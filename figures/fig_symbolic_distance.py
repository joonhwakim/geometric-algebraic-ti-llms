"""The LLM panels of the figure "Symbolic distance effects (SDE)": Qwen3-32B's direct accuracy
and reasoning budget by distance on the IWL test (country population) and the ICL test (given
names), with each test's accuracy at 1,024 reasoning tokens (distances 2 to 6) shown beside
its line, and the ICL test's best accuracy so far by reasoning budget; over the roster, each
model's two accuracy slopes and its two reasoning-budget measures.
Tables: symbolic_distance/direct, ladders.
"""

from __future__ import annotations

import argparse

from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from _shared import column_titles, direct, ladder, letter_panels, main, mosaic_figure
from geo_alg_ti_llms import analysis, plots, roster
from geo_alg_ti_llms.plots import style

NAME = "symbolic_distance"
MODEL = "qwen3-32b"
DIMENSION = "countries"
CONDITIONS = (roster.ICL_CONDITION, roster.IWL_CONDITION)
MOSAIC = (("acc", "slopes"), ("budget", "budget_points"), ("band", "band"))
LETTERS = {"acc": "A", "slopes": "B", "budget": "C", "budget_points": "D", "band": "E"}
TITLES = {"acc": f"LLM ({roster.MODEL_LABELS[MODEL]})", "budget": f"LLM ({roster.MODEL_LABELS[MODEL]})",
          "slopes": "LLMs", "budget_points": "LLMs", "band": f"LLM ({roster.MODEL_LABELS[MODEL]})"}
HEIGHT_IN = 4.9
# Ticks every 256 tokens up to the drawn data.
BUDGET_YTICKS = (0, 256, 512)
# Each test's accuracy at the largest budget, placed (data units) above the last segment of its
# line, clear of the lines and bands.
NUMBER_ANCHORS = {roster.ICL_CONDITION: (5.15, 430.0), roster.IWL_CONDITION: (5.15, 40.0)}
BAND = dict(budget_stat="running_best", ylabel=style.ACCURACY_BEST_SO_FAR_LABEL, marker_size=3.0)
BAND_XLIM = (-40.0, 1064.0)
DISTANCE_KEY = {"loc": "lower right", "ncol": 3, "title": style.DISTANCE_LEGEND_TITLE,
                "title_fontsize": style.FONT_LEGEND_PT, "handlelength": 1.2,
                "handletextpad": 0.4, "columnspacing": 0.8, "labelspacing": 0.2}


def model_rows(frame, model: str = MODEL, dimension: str = DIMENSION):
    """One model's ICL test rows and its IWL test rows on one dimension (every item set)."""
    rows = frame[frame["model"] == model]
    return rows[(rows["condition"] == roster.ICL_CONDITION)
                | ((rows["condition"] == roster.IWL_CONDITION) & (rows["dimension"] == dimension))]


def distance_key(ax, cmap: str) -> None:
    """A key to the distance shades of ``cmap``, light near and dark far."""
    shades = style.distance_colors(analysis.DISTANCES, cmap)
    handles = [Line2D([], [], color=s, marker="o", markersize=3.0) for s in shades]
    ax.legend(handles, [str(d) for d in analysis.DISTANCES], **DISTANCE_KEY)


def build(args: argparse.Namespace) -> Figure:
    """Assemble the five panels; the tests' legend sits on the accuracy panel."""
    trials, ladders = direct(args.results), ladder(args.results)
    fig, axs = mosaic_figure(MOSAIC, width_in=style.COLUMN_WIDTH_IN, height_in=HEIGHT_IN)
    plots.make_accuracy_by_distance(axs["acc"], model_rows(trials), conditions=CONDITIONS,
                                    labels=style.TEST_LABELS, ylabel=style.ACCURACY_NO_REASONING_LABEL,
                                    show_legend=True, legend_kwargs={"loc": "lower left"})
    own = model_rows(ladders)
    plots.make_budget_by_distance(axs["budget"], own, conditions=CONDITIONS,
                                  labels=style.TEST_LABELS, yticks=BUDGET_YTICKS)
    plots.make_accuracy_numbers(axs["budget"], own, conditions=CONDITIONS, anchors=NUMBER_ANCHORS)
    plots.make_accuracy_by_budget(axs["band"], own, conditions=(roster.ICL_CONDITION,), **BAND)
    axs["band"].set_xlim(*BAND_XLIM)
    distance_key(axs["band"], style.DISTANCE_CMAPS[roster.ICL_CONDITION])
    competent = analysis.competence(trials).query("competent")["model"]
    slopes = analysis.distance_slope_points(trials)
    plots.make_model_scatter(axs["slopes"], slopes[slopes["model"].isin(competent)],
                             xlabel="IWL test slope", ylabel="ICL test slope", x_err="x_se",
                             y_err=("y_lo", "y_hi"), quadrant="lower_right", symmetric=True)
    budgets = analysis.budget_points(ladders)
    plots.make_model_scatter(axs["budget_points"], budgets[~budgets["omitted"]],
                             xlabel="ICL − IWL budget (tokens)", ylabel="ICL budget slope",
                             x_err=("x_lo", "x_hi"), y_err=("y_lo", "y_hi"),
                             quadrant="upper_right", symmetric=True)
    column_titles(axs, TITLES)
    letter_panels(axs, LETTERS)
    return fig


if __name__ == "__main__":
    main(build, name=NAME)
