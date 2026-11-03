"""The page every per-model appendix figure shares: one row per model, one column per dimension.

Layout only: no data, no estimand. Row headers are written horizontally in a gutter at the left
edge that constrained layout keeps free; column headers are the first row's bold titles; axis
labels sit on the bottom row and the first column only; one legend runs under the grid. Every
page has the same row height, so a panel has nearly the same size on every page.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from geo_alg_ti_llms.plots import style

ROW_HEIGHT_IN: float = 1.05
CHROME_HEIGHT_IN: float = 1.05      # the column headers, the bottom axis label and the legend
GUTTER_IN: float = 0.90           # holds the widest model label (two lines, bold)
ROW_HEADER_PAD_PT: float = 4.0
COLUMN_HEADER_PAD_PT: float = 3.0
PADS: dict[str, float] = {"w_pad": 0.03, "h_pad": 0.03, "wspace": 0.05, "hspace": 0.07}


def page(row_labels: Sequence[str], col_labels: Sequence[str], *,
         gutter_in: float = GUTTER_IN) -> tuple[Figure, np.ndarray]:
    """A new full-width page with one equal axes per row and column, headed by both; the row
    headers sit in a gutter ``gutter_in`` inches wide."""
    style.apply_style()
    fig = plt.figure(figsize=(style.FULL_WIDTH_IN, CHROME_HEIGHT_IN + ROW_HEIGHT_IN * len(row_labels)))
    gutter = gutter_in / style.FULL_WIDTH_IN
    fig.get_layout_engine().set(rect=(gutter, 0.0, 1.0 - gutter, 1.0), **PADS)
    axes = fig.subplots(len(row_labels), len(col_labels), sharex=True, sharey=True, squeeze=False)
    header_x = (gutter_in - ROW_HEADER_PAD_PT / 72.0) / style.FULL_WIDTH_IN
    for r, label in enumerate(row_labels):
        header = axes[r, 0].annotate(label, xy=(header_x, 0.5),
                                     xycoords=("figure fraction", "axes fraction"), ha="right",
                                     va="center", fontweight="bold", annotation_clip=False,
                                     fontsize=style.FONT_TICK_PT)
        header.set_in_layout(False)
    for c, label in enumerate(col_labels):
        axes[0, c].set_title(label, fontweight="bold", pad=COLUMN_HEADER_PAD_PT,
                             fontsize=style.FONT_BODY_PT)
    return fig, axes


def finish(axes: np.ndarray, *, xlabel: str, ylabel: str, yticks: Sequence[float] | None,
           xticks: Sequence[float] | None = None) -> None:
    """The same ticks on every panel; the x label on the bottom row, the y label on column one."""
    last = axes.shape[0] - 1
    for (r, c), ax in np.ndenumerate(axes):
        if yticks is not None:
            ax.set_yticks(list(yticks))
        if xticks is not None:
            ax.set_xticks(list(xticks))
        ax.set_xlabel(xlabel if r == last else "")
        ax.set_ylabel(ylabel if c == 0 else "")


def legend(fig: Figure, handles: Sequence, labels: Sequence[str], *,
           handler_map: Mapping | None = None) -> None:
    """One frameless single-row legend under the grid (``handler_map`` as matplotlib's)."""
    fig.legend(list(handles), list(labels), loc="outside lower center", ncol=len(labels),
               handlelength=1.5, frameon=False, fontsize=style.FONT_LEGEND_PT,
               handler_map=handler_map)
