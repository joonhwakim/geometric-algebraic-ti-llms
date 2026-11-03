"""Every figure script draws from the committed tables, and the committed numbers table is
what the tables give."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from geo_alg_ti_llms import analysis, cli, numbers, roster  # noqa: E402
from geo_alg_ti_llms.plots import style  # noqa: E402

ROOT = analysis.RESULTS_ROOT.parent
TOPICS = ("symbolic_distance", "content_effects", "mechanisms", "representation",
          "composed_inferences", "symbolic_distance_by_model", "reasoning_budget_by_model",
          "content_effects_by_model", "recovery_by_model", "reasoning_budget_contexts_by_model",
          "icl_accuracy_by_budget", "example_traces")


def test_every_figure_script_is_listed():
    assert set(cli.FIGURE_TOPICS) == set(TOPICS)


# Every figure, and every page of a per-model figure (five bootstrap draws keep it quick).
BUILDS = [(t, None) for t in TOPICS if t != "example_traces" and not t.endswith("_by_model")]
BUILDS += [(t, page) for t in TOPICS if t.endswith("_by_model") for page in roster.PAGES]


@pytest.mark.parametrize("topic, page", BUILDS)
def test_a_figure_builds_from_the_committed_tables(topic, page, monkeypatch):
    monkeypatch.setattr(analysis, "N_BOOT", 5)
    module = cli._figure_module(f"fig_{topic}")
    args = cli.argparse.Namespace(results=analysis.RESULTS_ROOT)
    fig = module.build(args) if page is None else module.build(args, page)
    drawn = [ax for ax in fig.axes if ax.has_data()]
    assert drawn, topic
    if page is not None:
        assert len(drawn) == len(roster.PAGES[page]) * len(roster.DIMENSIONS)
    width, height = fig.get_size_inches()
    assert width <= style.FULL_WIDTH_IN + 1e-9 and height <= style.MAX_HEIGHT_IN
    plt.close(fig)


def test_the_recovery_key_can_shorten_its_distance_strokes():
    module = cli._figure_module("fig_recovery_by_model")
    args = cli.argparse.Namespace(results=analysis.RESULTS_ROOT)
    lengths = {}
    for stroke in (1.0, 0.5):
        fig = module.build(args, "qwen3", key_stroke=stroke)
        fig.canvas.draw()
        key = fig.legends[0].legend_handles[-1]    # the farthest distance's stroke
        lengths[stroke] = float(max(key.get_xdata()) - min(key.get_xdata()))
        plt.close(fig)
    assert lengths[0.5] == pytest.approx(lengths[1.0] / 2)


def test_the_icl_accuracy_figure_can_size_its_panel_titles():
    module = cli._figure_module("fig_icl_accuracy_by_budget")
    fig = module.build(cli.argparse.Namespace(results=analysis.RESULTS_ROOT), title_size=7.0)
    sizes = {ax.title.get_fontsize() for ax in fig.axes if ax.get_title()}
    assert sizes == {7.0}
    plt.close(fig)


def test_the_example_traces_are_written_verbatim(tmp_path):
    [path] = cli.draw_figure("example_traces", ["--out", str(tmp_path)])
    text = path.read_text(encoding="utf-8")
    assert text.count("Reasoning chain (") == 4 and "Reasoning chain (1,022 tokens)" in text


def test_the_figure_command_writes_a_pdf_and_a_preview(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "N_BOOT", 5)
    assert cli.main(["figure", "--name", "mechanisms", "--out", str(tmp_path)]) == 0
    assert {p.name for p in tmp_path.iterdir()} == {"mechanisms.pdf", "mechanisms.png"}


def test_a_page_of_a_single_page_figure_is_refused(tmp_path, capsys):
    assert cli.main(["figure", "--name", "mechanisms", "--page", "qwen3", "--out", str(tmp_path)]) == 1
    assert "per-model figures only" in capsys.readouterr().err


def test_every_committed_number_is_as_printed():
    table = pd.read_csv(ROOT / "results" / "numbers.csv", dtype=str, keep_default_na=False)
    assert list(table.columns) == ["id", "site", "printed", "computed", "source", "match"]
    assert len(table) == len(numbers.PAPER_NUMBERS)
    assert (table["match"] == "True").all() and (table["printed"] == table["computed"]).all()


def test_the_numbers_command_reproduces_the_committed_table(tmp_path):
    out = tmp_path / "numbers.csv"
    assert cli.main(["numbers", "--out", str(out)]) == 0
    assert out.read_text() == (ROOT / "results" / "numbers.csv").read_text()
