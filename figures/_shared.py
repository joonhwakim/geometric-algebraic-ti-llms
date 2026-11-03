"""Shared plumbing for the figure scripts: the tables, sizing, the per-model pages, saving, and
the command line.

Each ``fig_<topic>.py`` redraws the LLM panels of one paper figure from the committed tidy
tables under ``results/`` and writes it to ``figures/out/``: a script's ``build(args)`` returns
the figure, or, for a per-model appendix figure, ``build(args, page)`` returns one of its four
pages (``PAGES``). Figures are built at their final physical size in inches with constrained
layout and saved without a tight bounding box, so a PDF is placed at full scale.
"""

from __future__ import annotations

import argparse
import functools
import logging
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from geo_alg_ti_llms import analysis, roster
from geo_alg_ti_llms.plots import grid, style

RESULTS_ROOT: Path = analysis.RESULTS_ROOT
OUT_DIR: Path = Path(__file__).resolve().parent / "out"
PREVIEW_FORMAT: str = "png"
PAGES: tuple[str, ...] = tuple(roster.PAGES)
W_PAD_IN: float = 0.042
H_PAD_IN: float = 0.042
WSPACE: float = 0.085
HSPACE: float = 0.05
TITLE_PAD: float = 4.0
SAVE_RC: dict[str, object] = {"pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "standard"}


@functools.lru_cache(maxsize=4)
def direct(results_root: Path = RESULTS_ROOT):
    """Every direct answer (``analysis.direct_trials``), read once per process."""
    return analysis.direct_trials(results_root)


@functools.lru_cache(maxsize=4)
def ladder(results_root: Path = RESULTS_ROOT):
    """Every reasoning-budget ladder at the innermost ranks, read once per process."""
    return analysis.ladder_trials(results_root)


def mosaic_figure(mosaic: Sequence[Sequence[str]], *, width_in: float, height_in: float,
                  wspace: float = WSPACE, hspace: float = HSPACE, **gridspec) -> tuple[Figure, dict]:
    """A figure at exact physical size with constrained layout and one axes per mosaic name."""
    style.apply_style()
    fig = plt.figure(figsize=(width_in, height_in))
    axes = fig.subplot_mosaic([list(row) for row in mosaic], empty_sentinel=".", **gridspec)
    fig.get_layout_engine().set(w_pad=W_PAD_IN, h_pad=H_PAD_IN, wspace=wspace, hspace=hspace)
    return fig, axes


def letter_panels(axes: dict, letters: dict[str, str]) -> None:
    """Add a bold letter to each named axes."""
    for name, letter in letters.items():
        style.panel_letter(axes[name], letter)


def column_titles(axes: dict, titles: dict[str, str]) -> None:
    """A bold title over each named axes, naming the system its column shows."""
    for name, text in titles.items():
        axes[name].set_title(text, fontweight="bold", pad=TITLE_PAD)


def model_page(page: str, draw: Callable[[object, str, str], None], *, xlabel: str, ylabel: str,
               yticks: Sequence[float] | None, xticks: Sequence[float] | None = None,
               gutter_in: float = grid.GUTTER_IN) -> tuple[Figure, np.ndarray]:
    """One page of a per-model figure: ``draw(ax, model, dimension)`` fills each panel."""
    models = roster.PAGES[page]
    fig, axes = grid.page([roster.MODEL_ROW_LABELS[m] for m in models],
                          [roster.DIMENSION_HEADERS[d] for d in roster.DIMENSIONS],
                          gutter_in=gutter_in)
    for r, model in enumerate(models):
        for c, dim in enumerate(roster.DIMENSIONS):
            draw(axes[r, c], model, dim)
    grid.finish(axes, xlabel=xlabel, ylabel=ylabel, yticks=yticks, xticks=xticks)
    return fig, axes


@contextmanager
def _quiet_font_subsetting() -> Iterator[None]:
    """Hold back the font subsetting reports while a figure is written."""
    logger = logging.getLogger("fontTools")
    level = logger.level
    logger.setLevel(max(level, logging.ERROR))
    try:
        yield
    finally:
        logger.setLevel(level)


def save(fig: Figure, name: str, *, out_dir: Path = OUT_DIR,
         formats: Sequence[str] = ("pdf",)) -> list[Path]:
    """Write the figure in each format plus a PNG preview, close it, and return the paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    with matplotlib.rc_context(SAVE_RC), _quiet_font_subsetting():
        for fmt in dict.fromkeys([*formats, PREVIEW_FORMAT]):
            path = out_dir / f"{name}.{fmt}"
            fig.savefig(path, dpi=style.PREVIEW_DPI if fmt == PREVIEW_FORMAT else style.RASTER_DPI)
            paths.append(path)
    plt.close(fig)
    return paths


def main(build: Callable, *, name: str, pages: Sequence[str] | None = None,
         argv: Sequence[str] | None = None) -> list[Path]:
    """Parse ``--results``, ``--out`` and ``--format``, build the figure (or each page), save it."""
    matplotlib.use("Agg")
    parser = argparse.ArgumentParser(description=f"redraw the {name.replace('_', ' ')} figure")
    parser.add_argument("--results", type=Path, default=RESULTS_ROOT,
                        help="directory holding <experiment>/tables/")
    parser.add_argument("--out", type=Path, default=OUT_DIR, help="directory written to")
    parser.add_argument("--format", action="append", default=None,
                        help="an output format; repeat for several (pdf)")
    if pages:
        parser.add_argument("--page", choices=tuple(pages), action="append", default=None,
                            help="one page of the figure; repeat for several (every page)")
    args = parser.parse_args(argv)
    formats = tuple(args.format or ("pdf",))
    written = []
    if pages:
        for page in args.page or pages:
            written += save(build(args, page), f"{name}_{page}", out_dir=args.out, formats=formats)
    else:
        written += save(build(args), name, out_dir=args.out, formats=formats)
    for path in written:
        print(path)
    return written
