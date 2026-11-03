"""Colors, fonts, sizes and labels shared by every LLM panel.

The values are those of the paper's transformer and human panels, so an LLM panel sits beside
them without a seam. The IWL test is red and the ICL test blue; the uninformative context
takes the IWL red (true relations among other items only), and the congruent, reversed and
shuffled contexts keep one color each. A distance family takes shades of its context's
colormap, light for near pairs and dark for far ones; the congruent family runs through the
congruent color itself (``congruent_greens``, registered with matplotlib). Fonts are real
points at print size; PDFs keep editable text (TrueType) and are saved without a tight bounding
box, so a figure keeps its physical size.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, to_hex

IWL: str = "#d62728"
ICL: str = "#1f77b4"
CONGRUENT: str = "#3fb780"
REVERSED: str = "#ff7f0e"
SHUFFLED: str = "#875a5a"
REFERENCE_LINE: str = "#7f7f7f"
CHANCE: float = 0.5
NO_STEPS: str = "#7f7f7f"           # no reasoning
MAX_STEPS: str = "#ff7f0e"          # the largest reasoning budget
OPEN_MARKER_FACE: str = "#ffffff"
NEUTRAL_POINT: str = "#333333"      # a point of a cross-model scatter
QUADRANT_TINT: str = "#1f77b4"      # the corner a cross-model scatter's claim counts
QUADRANT_ALPHA: float = 0.08
# The interventions: intact near black, the rank direction erased teal, inverted magenta, the
# context-reading heads ablated purple; a control is drawn open in the color it matches.
EDIT_INTACT: str = "#222222"
EDIT_ERASED: str = "#1b9e77"
EDIT_INVERTED: str = "#e7298a"
CONTEXT_HEADS: str = "#7570b3"
GRAY_SHADES: tuple[str, ...] = ("#c6c6c6", "#9a9a9a", "#6e6e6e", "#424242")

FONT_FAMILY: tuple[str, ...] = ("Helvetica", "Arial", "DejaVu Sans")
FONT_BODY_PT: float = 7.0
FONT_TICK_PT: float = 6.0
FONT_LEGEND_PT: float = 6.0
FONT_TITLE_PT: float = 7.0
FONT_PANEL_LETTER_PT: float = 9.0
AXES_LINEWIDTH_PT: float = 0.6
LINE_WIDTH_PT: float = 1.0
TICK_WIDTH_PT: float = 0.6
TICK_LENGTH_PT: float = 2.5
ERRORBAR_CAPSIZE: float = 1.5
ERROR_COLOR: str = "#333333"        # error bars on a bar, dark enough to show on any color
PANEL_LETTER_OFFSET_PT: tuple[float, float] = (-14.0, 2.0)
RASTER_DPI: int = 600
PREVIEW_DPI: int = 200
COLUMN_WIDTH_IN: float = 3.5
FULL_WIDTH_IN: float = 7.0
MAX_HEIGHT_IN: float = 8.75

# The conditions of the tidy tables; the content contexts were run under the instruction to
# answer only from the comparisons given.
CONDITION_COLORS: dict[str, str] = {
    "no_context": IWL, "icl_people": ICL, "icl_cities": ICL, "congruent": CONGRUENT,
    "reversed": REVERSED, "shuffled": SHUFFLED, "unrelated_context": IWL,
}
CONDITION_LABELS: dict[str, str] = {
    "no_context": "No context", "icl_people": "ICL test", "icl_cities": "ICL test",
    "congruent": "Congruent", "reversed": "Reversed", "shuffled": "Shuffled",
    "unrelated_context": "Uninformative",
}
TEST_LABELS: dict[str, str] = {"no_context": "IWL test", "icl_people": "ICL test",
                               "icl_cities": "ICL test"}
# The congruent context's distance family runs from near white through the congruent color to
# a dark green of the same hue, so its shades match the context's line.
_CONGRUENT_DISTANCE_CMAP: str = "congruent_greens"
_CONGRUENT_DISTANCE_ANCHORS: tuple[str, str, str] = ("#f2fbf6", CONGRUENT, "#0b3d2c")
colormaps.register(LinearSegmentedColormap.from_list(_CONGRUENT_DISTANCE_CMAP,
                                                     _CONGRUENT_DISTANCE_ANCHORS), force=True)
DISTANCE_CMAPS: dict[str, str] = {"icl_people": "Blues", "congruent": _CONGRUENT_DISTANCE_CMAP,
                                  "reversed": "Oranges", "key": "Greys"}

DISTANCE_LABEL: str = "Symbolic distance"
ACCURACY_LABEL: str = "Accuracy"
ACCURACY_NO_REASONING_LABEL: str = "Accuracy (no reasoning)"   # a direct-answer panel
ACCURACY_REASONING_LABEL: str = "Accuracy (reasoning)"         # a bar panel at one budget
ACCURACY_BEST_SO_FAR_LABEL: str = "Accuracy (best so far)"     # the running best over budgets
REASONING_BUDGET_LABEL: str = "Reasoning budget (tokens)"
COMPOSED_INFERENCES_LABEL: str = "Composed inferences"
DISTANCE_LEGEND_TITLE: str = "Distance"
UNREACHED_LABEL: str = "unreached"
BUDGET_TICKS: tuple[int, ...] = (0, 256, 512, 768, 1024)

# Single-hue colormaps are pale at zero, so a gradient samples them from 0.35 up.
_SPAN: tuple[float, float] = (0.35, 1.0)


def apply_style() -> None:
    """Set the shared matplotlib parameters (fonts, TrueType text, constrained layout)."""
    matplotlib.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "font.family": "sans-serif", "font.sans-serif": list(FONT_FAMILY),
        "font.size": FONT_BODY_PT, "axes.titlesize": FONT_TITLE_PT,
        "axes.labelsize": FONT_BODY_PT, "xtick.labelsize": FONT_TICK_PT,
        "ytick.labelsize": FONT_TICK_PT, "legend.fontsize": FONT_LEGEND_PT,
        "mathtext.fontset": "dejavusans",
        "axes.linewidth": AXES_LINEWIDTH_PT, "lines.linewidth": LINE_WIDTH_PT,
        "xtick.major.width": TICK_WIDTH_PT, "ytick.major.width": TICK_WIDTH_PT,
        "xtick.major.size": TICK_LENGTH_PT, "ytick.major.size": TICK_LENGTH_PT,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "legend.handlelength": 2.0, "legend.labelspacing": 0.5,
        "figure.constrained_layout.use": True, "figure.dpi": PREVIEW_DPI,
        "savefig.dpi": PREVIEW_DPI, "savefig.bbox": "standard",
    })


def distance_colors(d_values: Sequence[int], cmap: str) -> list[str]:
    """One shade of ``cmap`` per distance, evenly spaced from light (near) to dark (far)."""
    ordered = [int(d) for d in d_values]
    low, high = _SPAN
    ranks = {d: i for i, d in enumerate(sorted(set(ordered)))}
    n = len(ranks)
    positions = [high if n == 1 else low + (high - low) * ranks[d] / (n - 1) for d in ordered]
    return [to_hex(colormaps[cmap](p)) for p in positions]


def panel_letter(ax: Axes, letter: str) -> None:
    """A bold panel letter, ``PANEL_LETTER_OFFSET_PT`` from the top left corner of the axes."""
    ax.annotate(letter, xy=(0.0, 1.0), xycoords="axes fraction", xytext=PANEL_LETTER_OFFSET_PT,
                textcoords="offset points", fontsize=FONT_PANEL_LETTER_PT, fontweight="bold",
                ha="left", va="bottom", annotation_clip=False)


def chance_line(ax: Axes) -> None:
    """The dashed chance reference, under the data."""
    ax.axhline(CHANCE, color=REFERENCE_LINE, ls="--", lw=AXES_LINEWIDTH_PT, zorder=1)
