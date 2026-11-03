"""Panel functions for the LLM panels of the paper's figures.

Every function is ``make_<name>(ax, df, **settings) -> dict``: it draws onto the given axes
only, never creates a figure or saves a file, takes its settings as keywords, and returns the
statistics it drew. Trials are the only axis a panel averages over; every other swept axis
must be pinned first, and the panel raises naming it otherwise.
"""

from geo_alg_ti_llms.plots import budget, distance, grid, representation, scatter, style
from geo_alg_ti_llms.plots.budget import make_accuracy_by_budget, make_composed_inferences_by_budget
from geo_alg_ti_llms.plots.distance import (
    make_accuracy_bars,
    make_accuracy_by_distance,
    make_accuracy_numbers,
    make_budget_by_distance,
)
from geo_alg_ti_llms.plots.representation import (
    make_edit_by_distance,
    make_intervention_bars,
    make_probe_by_layer,
    make_write_by_strength,
)
from geo_alg_ti_llms.plots.scatter import make_model_scatter

__all__ = [
    "budget", "distance", "grid", "representation", "scatter", "style",
    "make_accuracy_bars", "make_accuracy_by_budget", "make_accuracy_by_distance",
    "make_accuracy_numbers", "make_budget_by_distance",
    "make_composed_inferences_by_budget", "make_edit_by_distance", "make_intervention_bars",
    "make_model_scatter", "make_probe_by_layer", "make_write_by_strength",
]
