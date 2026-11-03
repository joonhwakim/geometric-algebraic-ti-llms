"""The cross-model scatter: one point per model (or per model and item set or dimension).

The panel draws a finished table of points (``analysis.distance_slope_points`` and its
siblings, already filtered to the models the figure draws). Reference lines divide it into
corners and the corner the claim counts is tinted; ``fit_line`` adds the least-squares line and
``show_rho`` writes the rank correlation. The statistics carry the corner counts.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator

from geo_alg_ti_llms import analysis
from geo_alg_ti_llms.plots import style

POINT_SIZE: float = 8.0
POINT_ALPHA: float = 0.85
EMPHASIS_EDGE_WIDTH: float = 0.8    # the edge of a point ``emphasis_col`` marks
PAD: float = 0.08                   # a symmetric axis reaches this much past the widest point
QUADRANTS: tuple[str, ...] = ("upper_right", "upper_left", "lower_right", "lower_left")


def _err(points: pd.DataFrame, spec, values: np.ndarray):
    """Error-bar half-widths: None, one column of half-widths, or a (lower, upper) bound pair."""
    if spec is None:
        return None
    if isinstance(spec, str):
        half = points[spec].to_numpy(float)
        return np.vstack([half, half])
    lower, upper = (points[c].to_numpy(float) for c in spec)
    return np.vstack([np.clip(values - lower, 0.0, None), np.clip(upper - values, 0.0, None)])


def _reach(values: np.ndarray, err, pad: float) -> float:
    ends = [np.abs(values)] if err is None else [np.abs(values - err[0]), np.abs(values + err[1])]
    return float(np.nanmax(np.concatenate(ends))) * (1.0 + pad)


def make_model_scatter(
    ax: Axes, points: pd.DataFrame, *, xlabel: str, ylabel: str, x_err=None, y_err=None,
    x_ref: float = 0.0, y_ref: float = 0.0, quadrant: str | None = None,
    symmetric: bool = False, xlim=None, ylim=None,
    xticks: Sequence[float] | None = None, alpha: float = POINT_ALPHA,
    emphasis_col: str | None = None, fit_line: bool = False, show_rho: bool = False,
) -> dict:
    """Column ``x`` against column ``y``, one point per row of ``points``.

    ``x_err`` / ``y_err`` name a column of half-widths or a (lower, upper) pair of bound
    columns. With ``symmetric`` both axes reach equally either side of zero, out to the widest
    point or bar end times ``1 + PAD``; ``xlim`` / ``ylim`` override. ``quadrant`` tints one
    corner about the reference values; ``emphasis_col`` edges the rows it marks.
    """
    if points.empty:
        raise ValueError("make_model_scatter: no points to draw")
    if quadrant is not None and quadrant not in QUADRANTS:
        raise ValueError(f"make_model_scatter: quadrant is one of {QUADRANTS}")
    xs, ys = points["x"].to_numpy(float), points["y"].to_numpy(float)
    xe, ye = _err(points, x_err, xs), _err(points, y_err, ys)
    ax.axvline(x_ref, color=style.REFERENCE_LINE, ls="--", lw=style.AXES_LINEWIDTH_PT, zorder=1)
    ax.axhline(y_ref, color=style.REFERENCE_LINE, ls="--", lw=style.AXES_LINEWIDTH_PT, zorder=1)
    if xe is not None or ye is not None:
        ax.errorbar(xs, ys, xerr=None if xe is None else np.nan_to_num(xe),
                    yerr=None if ye is None else np.nan_to_num(ye), fmt="none",
                    elinewidth=style.AXES_LINEWIDTH_PT, alpha=0.6, capsize=0.0,
                    ecolor=style.NEUTRAL_POINT, zorder=2)
    edges = (points[emphasis_col].astype(bool).to_numpy() if emphasis_col
             else np.zeros(len(points), bool))
    # One size per point: with a single size the PDF writer reuses one marker for every point,
    # stroked whenever any point has an edge, which draws a hairline around every other point.
    ax.scatter(xs, ys, s=np.full(len(points), POINT_SIZE), c=style.NEUTRAL_POINT, alpha=alpha,
               linewidths=np.where(edges, EMPHASIS_EDGE_WIDTH, 0.0),
               edgecolors=np.where(edges, "black", "none"), zorder=3)
    rho = analysis.spearman_rho(xs, ys)
    if fit_line:
        slope, intercept = np.polyfit(xs, ys, 1)
        span = np.array([xs.min(), xs.max()])
        ax.plot(span, slope * span + intercept, color="#000000", lw=0.8, zorder=2.5)
    if xlim is None and symmetric:
        xlim = (-_reach(xs, xe, PAD), _reach(xs, xe, PAD))
    if ylim is None and symmetric:
        ylim = (-_reach(ys, ye, PAD), _reach(ys, ye, PAD))
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if xticks is not None:
        ax.set_xticks(list(xticks))
    elif symmetric:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, symmetric=True, steps=[1, 2, 2.5, 5, 10]))
    if symmetric:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, symmetric=True, steps=[1, 2, 2.5, 5, 10]))
    final_x, final_y = ax.get_xlim(), ax.get_ylim()
    ax.set_xlim(*final_x)
    ax.set_ylim(*final_y)
    if quadrant is not None:
        left = x_ref if quadrant.endswith("right") else final_x[0]
        right = final_x[1] if quadrant.endswith("right") else x_ref
        bottom = y_ref if quadrant.startswith("upper") else final_y[0]
        top = final_y[1] if quadrant.startswith("upper") else y_ref
        ax.add_patch(Rectangle((left, bottom), right - left, top - bottom, lw=0, zorder=0.5,
                               facecolor=style.QUADRANT_TINT, alpha=style.QUADRANT_ALPHA))
    if show_rho:
        ax.text(0.97, 0.97, f"ρ = {rho:+.2f}".replace("-", "−"), transform=ax.transAxes,
                ha="right", va="top", fontsize=rcParams["legend.fontsize"])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return {"n_points": int(len(points)), "spearman": rho,
            "quadrant_counts": analysis.quadrant_counts(xs, ys, x_ref=x_ref, y_ref=y_ref)}
