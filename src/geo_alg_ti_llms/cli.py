"""One executable with one subcommand per step of the reproduction path.

``run-direct`` scores a direct test of an experiment on one item set and one
model; ``run-ladder`` scores a reasoning-budget ladder test the same way;
``jobs`` lists the (item set, test) jobs an experiment declares; ``smoke`` runs the
whole pipeline on a stub model (:mod:`geo_alg_ti_llms.smoke`); ``export`` turns
the stored runs of an experiment into its committed tidy tables and manifest
(``--experiment reasoning_traces`` derives the composed-inference table and the
example traces from two ladders of the other experiments; ``representation`` and
``components`` export the stored intervention runs). ``represent`` fits the rank
direction (``fit``: extract + fit; ``probe``: refit from the stored representations)
and runs the interventions along it (``intervene``); ``components`` scores the
context-reading heads (``score``) and mean-ablates the declared head sets (``ablate``). ``numbers``
recomputes every number the paper prints about the models into ``results/numbers.csv``;
``figure`` redraws one figure from the committed tables (``figures/fig_<topic>.py``).
"""

from __future__ import annotations

import argparse
import importlib
import sys
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from geo_alg_ti_llms import components, experiments, interp, represent, run, smoke, tables, traces
from geo_alg_ti_llms.models import load_model_config, model_config_path

# The intervention experiments run as stages of their own subcommand, not as (item set, test) jobs.
STAGES = {"representation": ("represent", ("fit", "probe", "intervene")),
          "components": ("components", ("score", "ablate"))}
# The modules of the inference extra (pyproject.toml), which the analysis install does not have.
INFERENCE_MODULES = frozenset({"torch", "transformers", "accelerate", "tokenizers", "safetensors",
                               "huggingface_hub", "scipy"})


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and dispatch to one subcommand."""
    parser = build_parser()
    args = parser.parse_args(argv)
    _, handler = _COMMANDS[args.command]
    try:
        return handler(args)
    except (ValueError, FileNotFoundError) as error:
        print(f"{args.command}: {error}", file=sys.stderr)
        return 1
    except ModuleNotFoundError as error:
        if (error.name or "").split(".")[0] not in INFERENCE_MODULES:
            raise
        print(f"{args.command}: {error.name} is not installed; running a model needs the inference "
              "extra (uv sync --extra inference)", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    """Build the parser for the executable and its subcommands."""
    parser = argparse.ArgumentParser(
        prog="geo-alg-ti-llms",
        description="Large language model experiments for the paper 'Geometric and "
                    "algebraic inference in humans and neural networks'.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    for verb, (summary, _) in _COMMANDS.items():
        sub = subcommands.add_parser(verb, help=summary, description=summary)
        _ARGUMENTS[verb](sub)
    return parser


def _add_experiment_arguments(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--experiment", required=True,
                     help="experiment name (a directory under experiments/; export also takes "
                          "reasoning_traces, representation and components)")
    sub.add_argument("--experiments-dir", type=Path, default=experiments.DEFAULT_EXPERIMENTS_DIR)


def _add_run_arguments(sub: argparse.ArgumentParser) -> None:
    _add_experiment_arguments(sub)
    sub.add_argument("--item-set", required=True,
                     help="set key: a dimension (its main item set), <dimension>_rep<k>, or "
                          "<dimension>_scale<ratio> (see `jobs`)")
    sub.add_argument("--test", required=True, help="test name in the experiment declaration")
    sub.add_argument("--model", required=True,
                     help="model short name (configs/models/<short_name>.yaml) or a YAML path")
    sub.add_argument("--out", required=True, type=Path,
                     help="directory the trial set JSONL files are written to")
    sub.add_argument("--dry-run", action="store_true",
                     help="print the trial set grid + projected trial counts and exit (no model load)")


def _add_run_ladder_arguments(sub: argparse.ArgumentParser) -> None:
    _add_run_arguments(sub)
    sub.add_argument("--gen-batch-size", type=int, default=1,
                     help="trials decoded per generate() call; every batch is padded to "
                          "exactly this many rows (16 in the published runs)")
    sub.add_argument("--budgets", default="all",
                     help="budgets to write: 'all' or a comma list such as 'b32,b1024'")


def _add_smoke_arguments(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--out", type=Path, default=None,
                     help="where the stub runs and their tables are written (default: a new "
                          "temporary directory)")


def _add_jobs_arguments(sub: argparse.ArgumentParser) -> None:
    _add_experiment_arguments(sub)
    sub.add_argument("--test", default=None, help="restrict to one test")


def _add_export_arguments(sub: argparse.ArgumentParser) -> None:
    _add_experiment_arguments(sub)
    sub.add_argument("--runs", type=Path, default=None,
                     help=f"root of the published runs (default: ${tables.RUNS_ENV})")
    sub.add_argument("--what", choices=("trials", "labels", "prompt-examples", "steps", "examples"),
                     default="trials",
                     help="trials: direct.parquet + ladders.parquet (+ pools.parquet) and the "
                          "manifest (default); labels: query_labels.parquet (content_effects, "
                          "after the trials of content_effects and prior_strength); "
                          "prompt-examples: prompt_examples.json (symbolic_distance); "
                          "steps: chain_steps.parquet (reasoning_traces, after the trials of "
                          "symbolic_distance and content_effects); examples: trace_examples.json "
                          "(reasoning_traces, after steps)")
    sub.add_argument("--out", type=Path, default=None,
                     help="tables directory (default results/<experiment>/tables)")
    sub.add_argument("--results-dir", type=Path, default=tables.DEFAULT_RESULTS_DIR,
                     help="where the other experiments' tables are read for the label coverage")
    sub.add_argument("--models", default=None,
                     help="comma list of model short names (default: every configs/models/*.yaml)")
    sub.add_argument("--tests", default=None,
                     help="comma list of tests to export (default: all; a ladder's direct twin is "
                          "read for the twin check either way)")
    sub.add_argument("--work-dir", type=Path, default=None,
                     help="cache one frame per model here so an interrupted export resumes "
                          "(with the same tests and runs directory only)")
    sub.add_argument("--check-ladders", action="store_true",
                     help="labels: also compare every shuffled ladder row with its direct twin")


def _add_intervention_arguments(sub: argparse.ArgumentParser, stages: tuple[str, ...]) -> None:
    sub.add_argument("stage", choices=stages)
    sub.add_argument("--dim", required=True, choices=list(represent.DIMS),
                     help="dimension: countries | cities | usstates")
    sub.add_argument("--model", default="qwen3-32b",
                     help="model short name (configs/models/<short_name>.yaml) or a YAML path")
    sub.add_argument("--runs", type=Path, default=None,
                     help=f"root of the published runs the trials are read from (default: ${tables.RUNS_ENV})")
    sub.add_argument("--experiments-dir", type=Path, default=experiments.DEFAULT_EXPERIMENTS_DIR,
                     help="where the intervention.yaml declarations are read from")
    sub.add_argument("--out", required=True, type=Path,
                     help="the run directory: the fit / head scores are written here, the intervention "
                          "into --tag below it")
    sub.add_argument("--tag", default=None,
                     help="subdirectory of --out for the intervention files (default: the declaration's)")
    sub.add_argument("--tests", default=None, help="comma list of tests (default: the four scored conditions)")
    sub.add_argument("--conditions", default=None, help="comma list of conditions (default: the declaration's)")
    sub.add_argument("--n-orders", type=int, default=3, help="premise orders re-scored (0..n-1)")
    sub.add_argument("--budget", type=int, default=1024, help="tokens generated per reasoning chain")
    sub.add_argument("--score-budgets", default="256,512,1024",
                     help="comma list of budgets at which each chain is read out")
    sub.add_argument("--batch", type=int, default=16, help="chains generated per batch")


def _add_represent_arguments(sub: argparse.ArgumentParser) -> None:
    _add_intervention_arguments(sub, STAGES["representation"][1])
    sub.add_argument("--layers", default="all",
                     help="hidden-state indices edited: 'all' (0..n_layers) or a comma list")
    sub.add_argument("--layer-stride", type=int, default=1, help="fit: extract every k-th hidden state")
    sub.add_argument("--k-partner", type=int, default=3, help="fit: partners per item and occurrence kind")
    sub.add_argument("--axis", type=Path, default=None,
                     help="intervene: the fitted rank direction (default: rank_axis.npz in the run "
                          "directory)")


def _add_components_arguments(sub: argparse.ArgumentParser) -> None:
    _add_intervention_arguments(sub, STAGES["components"][1])
    sub.add_argument("--max-seq", type=int, default=1200,
                     help="score: the most prompt + chain tokens a chain may have (longer is an error)")
    sub.add_argument("--heads", type=Path, default=None,
                     help="ablate: selected_heads.json (default: --out/selected_heads.json)")


def _csv(value: str | None) -> list[str] | None:
    return [v.strip() for v in value.split(",") if v.strip()] if value else None


@contextmanager
def _declared_in(path: Path) -> Iterator[None]:
    """A key missing from the declaration or model config at ``path``, as a one-line error."""
    try:
        yield
    except KeyError as error:
        raise ValueError(f"{path}: missing {error.args[0]!r}") from None


def _load_experiment(name: str, experiments_dir: Path) -> dict:
    with _declared_in(experiments.experiment_path(name, experiments_dir)):
        return experiments.load_experiment(name, experiments_dir)


def _load_declaration(name: str, experiments_dir: Path) -> dict:
    with _declared_in(interp.declaration_path(name, experiments_dir)):
        return interp.load_declaration(name, experiments_dir)


def _model_config(model: str):
    path = Path(model)
    if not path.suffix == ".yaml":
        path = model_config_path(model)
    with _declared_in(path):
        return load_model_config(path)


def _resolve(args):
    exp = _load_experiment(args.experiment, args.experiments_dir)
    model_cfg = _model_config(args.model)
    with _declared_in(experiments.experiment_path(args.experiment, args.experiments_dir)):
        return experiments.resolve(exp, args.item_set, args.test, model_cfg), model_cfg


def cmd_run_direct(args) -> int:
    spec, model_cfg = _resolve(args)
    if spec.runner != "direct":
        raise ValueError(f"test {args.test!r} is a {spec.runner} test; use run-{spec.runner}")
    print(f"experiment={spec.experiment} test={spec.test} item_set={spec.item_set} model={model_cfg.short_name}")
    return run.run_direct(spec.geometry, model_cfg, args.out, dry_run=args.dry_run)


def cmd_run_ladder(args) -> int:
    spec, model_cfg = _resolve(args)
    if spec.runner != "ladder":
        raise ValueError(f"test {args.test!r} is a {spec.runner} test; use run-{spec.runner}")
    budget_ids = None if args.budgets == "all" else [c.strip() for c in args.budgets.split(",") if c.strip()]
    print(f"experiment={spec.experiment} test={spec.test} item_set={spec.item_set} model={model_cfg.short_name} "
          f"trigger={spec.ladder['defaults']['trigger_id']}")
    return run.run_ladder(spec.geometry, spec.ladder, model_cfg, args.out,
                          gen_batch_size=args.gen_batch_size, budget_ids=budget_ids,
                          dry_run=args.dry_run)


def cmd_jobs(args) -> int:
    if args.experiment in STAGES:
        verb, stages = STAGES[args.experiment]
        raise ValueError(f"{args.experiment} declares no (item set, test) jobs; it runs in stages: "
                         f"{verb} {{{'|'.join(stages)}}} --dim <dimension> "
                         "(scripts/slurm/interventions.sh)")
    exp = _load_experiment(args.experiment, args.experiments_dir)
    with _declared_in(experiments.experiment_path(args.experiment, args.experiments_dir)):
        jobs = [(set_key, test, exp["tests"][test]) for set_key, test in experiments.jobs(exp)]
    for set_key, test, spec in jobs:
        if args.test and test != args.test:
            continue
        print(f"{spec['runner']}\t{set_key}\t{test}\t{spec.get('published_as', test)}")
    return 0


def cmd_represent(args) -> int:
    model_cfg = _model_config(args.model)
    decl = _load_declaration("representation", args.experiments_dir)
    if args.stage == "fit":
        represent.extract_representations(args.dim, model_cfg, args.out, layer_stride=args.layer_stride,
                                          k_partner=args.k_partner)
        represent.fit_rank_axis(args.dim, args.out)
        return 0
    if args.stage == "probe":
        represent.fit_rank_axis(args.dim, args.out)
        return 0
    runs = tables.runs_root(args.runs)
    axis = args.axis or (args.out / "rank_axis.npz")
    if not axis.exists():
        raise FileNotFoundError(f"rank direction {axis} not found (run `represent fit` first)")
    if args.layers == "all":
        import numpy as np
        layers = [int(l) for l in np.load(axis)["layers"]]
    else:
        layers = [int(l) for l in args.layers.split(",") if l.strip()]
    out = args.out / (args.tag or decl["published_as"]["trials"])
    return represent.run_intervention(
        runs, args.dim, model_cfg, axis, out, layers=layers,
        tests=_csv(args.tests) or represent.TESTS, conditions=_csv(args.conditions) or represent.CONDITIONS,
        n_orders=args.n_orders, budget=args.budget, score_budgets=[int(b) for b in args.score_budgets.split(",")],
        batch=args.batch)


def cmd_components(args) -> int:
    model_cfg = _model_config(args.model)
    decl = _load_declaration("components", args.experiments_dir)
    runs = tables.runs_root(args.runs)
    if args.stage == "score":
        components.score_heads(runs, args.dim, model_cfg, args.out, tests=_csv(args.tests) or components.SCORE_TESTS,
                               n_orders=args.n_orders, max_seq=args.max_seq)
        return 0
    heads = args.heads or (args.out / "selected_heads.json")
    if not heads.exists():
        raise FileNotFoundError(f"head sets {heads} not found (run `components score` first)")
    out = args.out / (args.tag or decl["published_as"]["trials"])
    return components.run_ablation(
        runs, args.dim, model_cfg, heads, out, tests=_csv(args.tests) or represent.TESTS,
        conditions=_csv(args.conditions), n_orders=args.n_orders, budget=args.budget,
        score_budgets=[int(b) for b in args.score_budgets.split(",")], batch=args.batch)


FIGURES_DIR = tables._REPO / "figures"
FIGURE_TOPICS = tuple(sorted(p.stem.removeprefix("fig_") for p in FIGURES_DIR.glob("fig_*.py")))


def _figure_module(stem: str):
    """``figures/<stem>.py``, imported with ``figures/`` on the path (its helpers: ``_shared``)."""
    if str(FIGURES_DIR) not in sys.path:
        sys.path.insert(0, str(FIGURES_DIR))
    return importlib.import_module(stem)


def draw_figure(name: str, argv: Sequence[str] = ()) -> list[Path]:
    """Run ``figures/fig_<name>.py`` as its command line does, with ``argv``."""
    module = _figure_module(f"fig_{name}")
    if name == "example_traces":    # text, not a figure: the script writes it itself
        return module.main(list(argv))
    pages = _figure_module("_shared").PAGES if name.endswith("_by_model") else None
    return _figure_module("_shared").main(module.build, name=module.NAME, pages=pages,
                                          argv=list(argv))


def cmd_figure(args) -> int:
    if args.page and not args.name.endswith("_by_model"):
        raise ValueError(f"--page applies to the per-model figures only, not {args.name}")
    argv = ["--results", str(args.results)] + (["--out", str(args.out)] if args.out else [])
    for page in args.page or ():
        argv += ["--page", page]
    draw_figure(args.name, argv)
    return 0


def cmd_numbers(args) -> int:
    from geo_alg_ti_llms import numbers

    table = numbers.numbers(args.results)
    path = args.out or (args.results / "numbers.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, lineterminator="\n")
    differ = table[~table["match"]]
    print(f"{len(table)} numbers, {len(table) - len(differ)} as printed; wrote {path}")
    for row in differ.itertuples():
        print(f"  {row.id} ({row.site}): printed {row.printed}, computed {row.computed}")
    return 1 if len(differ) else 0


def _add_figure_arguments(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--name", required=True, choices=FIGURE_TOPICS, help="the figure's topic")
    sub.add_argument("--page", action="append", default=None,
                     help="one page of a per-model figure; repeat for several (every page)")
    sub.add_argument("--results", type=Path, default=tables._REPO / "results",
                     help="directory holding <experiment>/tables/")
    sub.add_argument("--out", type=Path, default=None, help="directory written to (figures/out)")


def _add_numbers_arguments(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--results", type=Path, default=tables._REPO / "results",
                     help="directory holding <experiment>/tables/")
    sub.add_argument("--out", type=Path, default=None,
                     help="the table written (results/numbers.csv)")


def cmd_smoke(args) -> int:
    return smoke.run_smoke(args.out or Path(tempfile.mkdtemp(prefix="geo-alg-ti-llms-smoke-")))


def cmd_export(args) -> int:
    runs = tables.runs_root(args.runs)
    if args.experiment in interp.EXPERIMENTS:
        if args.what != "trials":
            raise ValueError(f"--what {args.what} is not an {args.experiment} table")
        out = args.out or (args.results_dir / args.experiment / "tables")
        interp.export_interp(runs, out, args.experiment, experiments_dir=args.experiments_dir)
        return 0
    if args.experiment == traces.EXPERIMENT:
        out = args.out or (args.results_dir / traces.EXPERIMENT / "tables")
        if args.what == "steps":
            traces.export_chain_steps(runs, out, results_dir=args.results_dir,
                                      experiments_dir=args.experiments_dir)
        elif args.what == "examples":
            traces.export_trace_examples(runs, out, results_dir=args.results_dir,
                                         experiments_dir=args.experiments_dir)
        else:
            raise ValueError(f"--what {args.what} is not a {traces.EXPERIMENT} table (steps, examples)")
        return 0
    if args.what in ("steps", "examples"):
        raise ValueError(f"--what {args.what} belongs to --experiment {traces.EXPERIMENT}")
    exp = _load_experiment(args.experiment, args.experiments_dir)
    out = args.out or (args.results_dir / args.experiment / "tables")
    models = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else None
    tests = [a.strip() for a in args.tests.split(",") if a.strip()] if args.tests else None
    if args.what == "trials":
        tables.export_trials(exp, runs, out, models=models, tests=tests, work_dir=args.work_dir)
    elif args.what == "labels":
        if args.experiment != "content_effects":
            raise ValueError("--what labels belongs to the content_effects experiment")
        spacing = _load_experiment("prior_strength", args.experiments_dir)
        tables.export_query_labels(exp, spacing, runs, out, results_dir=args.results_dir,
                                   models=models, check_ladders=args.check_ladders)
    else:
        if args.experiment != "symbolic_distance":
            raise ValueError("--what prompt-examples belongs to the symbolic_distance experiment")
        tables.export_prompt_examples(runs, out, experiments_dir=args.experiments_dir)
    return 0


_COMMANDS = {
    "run-direct": ("score a direct test on one item set and one model", cmd_run_direct),
    "run-ladder": ("score a reasoning-budget ladder on one item set and one model", cmd_run_ladder),
    "jobs": ("list the (item set, test) jobs an experiment declares", cmd_jobs),
    "smoke": ("run the whole pipeline on a stub model in seconds", cmd_smoke),
    "export": ("write an experiment's tidy tables and manifest from the stored runs", cmd_export),
    "represent": ("fit the rank direction of a dimension, probe it by layer, or intervene along it", cmd_represent),
    "components": ("score the context-reading heads of a dimension, or mean-ablate the declared head sets", cmd_components),
    "numbers": ("recompute every number the paper prints about the models", cmd_numbers),
    "figure": ("redraw one figure from the committed tables", cmd_figure),
}
_ARGUMENTS = {
    "run-direct": _add_run_arguments,
    "run-ladder": _add_run_ladder_arguments,
    "jobs": _add_jobs_arguments,
    "smoke": _add_smoke_arguments,
    "export": _add_export_arguments,
    "represent": _add_represent_arguments,
    "components": _add_components_arguments,
    "numbers": _add_numbers_arguments,
    "figure": _add_figure_arguments,
}


if __name__ == "__main__":
    raise SystemExit(main())
