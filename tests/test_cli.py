"""The executable: every documented invocation parses, the model-free ones run, and the
intervention subcommands reach their handlers with the declared defaults (heavy functions
stubbed)."""

from __future__ import annotations

import re
import shlex
import tomllib
from pathlib import Path

import numpy as np
import pytest
import yaml

from geo_alg_ti_llms import cli, components, experiments, interp, represent, tables
from geo_alg_ti_llms.models import model_config_path

ROOT = Path(__file__).resolve().parent.parent
REPRODUCE = ROOT / "REPRODUCE.md"
SLURM = sorted((ROOT / "scripts" / "slurm").glob("*.sh"))

# Values substituted for the shell variables of scripts/slurm/*.sh and the <dir> of REPRODUCE.md.
SHELL_VALUES = {
    "EXPERIMENT": "symbolic_distance", "ITEM_SET": "countries", "TEST": "iwl", "MODEL": "qwen3-32b",
    "OUT": "/tmp/out", "GEN_BATCH": "16", "DIM": "countries", "RUNS_ROOT": "/tmp/runs",
    "VENV": "/tmp/venv",
}
_VAR = re.compile(r'"?\$\{?([A-Za-z_][A-Za-z0-9_]*)(?::-[^}]*)?\}?"?')


def _substitute(line: str) -> str:
    return _VAR.sub(lambda m: SHELL_VALUES[m.group(1)], line)


def _invocations(path: Path) -> list[list[str]]:
    """Every ``geo-alg-ti-llms …`` command line of a file, as argv (without the program name)."""
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"\\\n\s*", " ", text)                       # join continuation lines
    found = []
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        m = re.search(r'geo-alg-ti-llms"?\s+(.*)$', line)
        if not m or not m.group(1).strip():
            continue
        rest = _substitute(m.group(1)).replace("<dir>", "/tmp/out").replace("<topic>", "mechanisms")
        rest = rest.split(";;")[0].split("|")[0]
        argv = shlex.split(rest)
        if argv and argv[0] in cli._COMMANDS:
            if argv[0] == "run-ladder" and "--test" in argv and argv[argv.index("--test") + 1] == "iwl":
                argv[argv.index("--test") + 1] = "budget_iwl"      # job.sh's $TEST is generic: a ladder test here
            found.append(argv)
    return found


def _documented() -> list[tuple[str, list[str]]]:
    out = [(REPRODUCE.name, argv) for argv in _invocations(REPRODUCE)]
    for script in SLURM:
        out += [(script.name, argv) for argv in _invocations(script)]
    return out


DOCUMENTED = _documented()


def test_the_documented_invocations_were_found():
    verbs = sorted({argv[0] for _, argv in DOCUMENTED})
    assert verbs == ["components", "export", "figure", "jobs", "numbers", "represent", "run-direct",
                     "run-ladder", "smoke"]
    assert sum(1 for _, a in DOCUMENTED if a[0] == "export") == 9          # REPRODUCE.md lists nine
    assert sum(1 for _, a in DOCUMENTED if a[0] in ("represent", "components")) == 4   # interventions.sh


@pytest.mark.parametrize("source,argv", DOCUMENTED, ids=[f"{s}:{' '.join(a[:3])}" for s, a in DOCUMENTED])
def test_every_documented_invocation_parses(source, argv):
    args = cli.build_parser().parse_args(argv)
    assert args.command == argv[0]
    if args.command not in ("smoke", "numbers", "figure"):
        assert args.experiments_dir == experiments.DEFAULT_EXPERIMENTS_DIR
    if args.command == "figure":
        assert args.name in cli.FIGURE_TOPICS
    if args.command in ("run-direct", "run-ladder", "jobs", "export"):
        assert args.experiment in experiments.list_experiments() + list(interp.EXPERIMENTS) + ["reasoning_traces"]
    if args.command in ("represent", "components"):
        assert args.dim in represent.DIMS


@pytest.mark.parametrize("argv", [a for _, a in DOCUMENTED if a[0] in ("run-direct", "run-ladder")],
                         ids=lambda a: a[0])
def test_run_lines_dry_run_without_a_model(argv, tmp_path, capsys, monkeypatch):
    import geo_alg_ti_llms.run as runmod

    def boom(cfg):
        raise AssertionError("dry run must not load a model")
    monkeypatch.setattr(runmod, "load_model_and_tokenizer", boom)
    argv = [tmp_path.as_posix() if a == "/tmp/out" else a for a in argv] + ["--dry-run"]
    assert cli.main(argv) == 0
    out = capsys.readouterr().out
    assert "experiment=" in out and "test=" in out


def test_the_wrong_runner_is_refused(tmp_path, capsys):
    rc = cli.main(["run-ladder", "--experiment", "symbolic_distance", "--item-set", "countries",
                   "--test", "iwl", "--model", "qwen3-32b", "--out", tmp_path.as_posix(), "--dry-run"])
    assert rc == 1
    assert "is a direct test" in capsys.readouterr().err


@pytest.mark.parametrize("experiment", experiments.list_experiments())
def test_jobs_lists_every_declared_job(experiment, capsys):
    assert cli.main(["jobs", "--experiment", experiment]) == 0
    lines = capsys.readouterr().out.rstrip("\n").split("\n")
    exp = experiments.load_experiment(experiment)
    assert len(lines) == len(list(experiments.jobs(exp)))
    for line in lines:
        runner, item_set, test, stored = line.split("\t")
        assert runner in ("direct", "ladder")
        assert test in exp["tests"]


@pytest.mark.parametrize("experiment", interp.EXPERIMENTS)
def test_jobs_names_the_stages_of_an_intervention_experiment(experiment, capsys):
    assert cli.main(["jobs", "--experiment", experiment]) == 1
    verb, stages = cli.STAGES[experiment]
    err = capsys.readouterr().err
    assert err.startswith(f"jobs: {experiment} declares no (item set, test) jobs")
    assert f"{verb} {{{'|'.join(stages)}}} --dim <dimension>" in err


def _missing(monkeypatch, name):
    import geo_alg_ti_llms.run as runmod

    def fail(cfg):
        raise ModuleNotFoundError(f"No module named '{name}'", name=name)
    monkeypatch.setattr(runmod, "load_model_and_tokenizer", fail)


def test_a_missing_inference_module_is_a_one_line_error(monkeypatch, tmp_path, capsys):
    argv = ["run-direct", "--experiment", "symbolic_distance", "--item-set", "countries",
            "--test", "iwl", "--model", "qwen3-32b", "--out", tmp_path.as_posix()]
    _missing(monkeypatch, "torch")
    assert cli.main(argv) == 1
    assert capsys.readouterr().err == ("run-direct: torch is not installed; running a model needs the "
                                       "inference extra (uv sync --extra inference)\n")
    _missing(monkeypatch, "some_helper")            # not an inference module: raised as it is
    with pytest.raises(ModuleNotFoundError, match="some_helper"):
        cli.main(argv)


def test_the_inference_modules_are_the_inference_extra():
    extra = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    names = {re.split(r"[=<>!~ ]", r)[0] for r in extra["project"]["optional-dependencies"]["inference"]}
    assert names == cli.INFERENCE_MODULES


def test_a_key_missing_from_a_declaration_is_named(tmp_path, capsys):
    decl = yaml.safe_load(experiments.experiment_path("symbolic_distance").read_text(encoding="utf-8"))
    del decl["tests"]["iwl"]["assignment"]
    path = tmp_path / "symbolic_distance" / "experiment.yaml"
    path.parent.mkdir()
    path.write_text(yaml.safe_dump(decl), encoding="utf-8")
    assert cli.main(["jobs", "--experiment", "symbolic_distance", "--experiments-dir", tmp_path.as_posix()]) == 1
    assert capsys.readouterr().err == f"jobs: {path}: missing 'assignment'\n"
    config = yaml.safe_load(model_config_path("qwen3-32b").read_text(encoding="utf-8"))
    del config["short_name"]
    model = tmp_path / "model.yaml"
    model.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert cli.main(["run-direct", "--experiment", "symbolic_distance", "--item-set", "countries",
                     "--test", "iwl", "--model", model.as_posix(), "--out", tmp_path.as_posix(),
                     "--dry-run"]) == 1
    assert capsys.readouterr().err == f"run-direct: {model}: missing 'short_name'\n"


@pytest.fixture
def intervention_stubs(monkeypatch, tmp_path):
    """Record the arguments of the heavy intervention functions instead of running them."""
    calls: dict[str, list] = {}

    def record(name):
        def f(*args, **kwargs):
            calls.setdefault(name, []).append((args, kwargs))
            return 0
        return f
    monkeypatch.setattr(represent, "extract_representations", record("extract"))
    monkeypatch.setattr(represent, "fit_rank_axis", record("fit"))
    monkeypatch.setattr(represent, "run_intervention", record("intervene"))
    monkeypatch.setattr(components, "score_heads", record("score"))
    monkeypatch.setattr(components, "run_ablation", record("ablate"))
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setenv(tables.RUNS_ENV, runs.as_posix())
    return calls, runs


def _out(tmp_path, dim="countries"):
    out = tmp_path / "rank_intervention" / dim / "qwen3-32b"
    out.mkdir(parents=True, exist_ok=True)
    return out


def test_represent_fit_extracts_then_fits(intervention_stubs, tmp_path):
    calls, _ = intervention_stubs
    out = _out(tmp_path)
    assert cli.main(["represent", "fit", "--dim", "countries", "--model", "qwen3-32b", "--out", out.as_posix()]) == 0
    (args, kwargs), = calls["extract"]
    assert args[0] == "countries" and args[1].short_name == "qwen3-32b" and args[2] == out
    assert kwargs == {"layer_stride": 1, "k_partner": 3}
    (args, kwargs), = calls["fit"]
    assert args == ("countries", out) and kwargs == {}


def test_represent_probe_only_refits(intervention_stubs, tmp_path):
    calls, _ = intervention_stubs
    out = _out(tmp_path)
    assert cli.main(["represent", "probe", "--dim", "countries", "--out", out.as_posix()]) == 0
    assert set(calls) == {"fit"}
    assert calls["fit"][0][0] == ("countries", out)


def test_represent_intervene_defaults_are_the_declarations(intervention_stubs, tmp_path):
    calls, runs = intervention_stubs
    out = _out(tmp_path)
    np.savez(out / "rank_axis.npz", layers=np.arange(5))
    assert cli.main(["represent", "intervene", "--dim", "countries", "--model", "qwen3-32b",
                     "--out", out.as_posix()]) == 0
    (args, kwargs), = calls["intervene"]
    decl = interp.load_declaration("representation")
    assert args[0] == runs and args[1] == "countries" and args[2].short_name == "qwen3-32b"
    assert args[3] == out / "rank_axis.npz"
    assert args[4] == out / decl["published_as"]["trials"] == out / "intervene_scrub_ctxonly"
    assert kwargs == {"layers": [0, 1, 2, 3, 4], "tests": represent.TESTS, "conditions": represent.CONDITIONS,
                      "n_orders": 3, "budget": 1024, "score_budgets": [256, 512, 1024], "batch": 16}


def test_represent_intervene_options_are_forwarded(intervention_stubs, tmp_path):
    calls, runs = intervention_stubs
    out = _out(tmp_path, "usstates")
    axis = tmp_path / "axis.npz"
    np.savez(axis, layers=np.arange(3))
    assert cli.main(["represent", "intervene", "--dim", "usstates", "--out", out.as_posix(), "--axis", axis.as_posix(),
                     "--tag", "mytag", "--layers", "2,4", "--conditions", "intact,erase", "--tests", "iwl",
                     "--n-orders", "1", "--budget", "48", "--score-budgets", "16,48", "--batch", "4",
                     "--runs", runs.as_posix()]) == 0
    (args, kwargs), = calls["intervene"]
    assert args[3] == axis and args[4] == out / "mytag"
    assert kwargs == {"layers": [2, 4], "tests": ["iwl"], "conditions": ["intact", "erase"],
                      "n_orders": 1, "budget": 48, "score_budgets": [16, 48], "batch": 4}


def test_represent_intervene_needs_the_axis(intervention_stubs, tmp_path, capsys):
    calls, _ = intervention_stubs
    out = _out(tmp_path)
    assert cli.main(["represent", "intervene", "--dim", "countries", "--out", out.as_posix()]) == 1
    assert "run `represent fit` first" in capsys.readouterr().err
    assert calls == {}


def test_components_score_defaults(intervention_stubs, tmp_path):
    calls, runs = intervention_stubs
    out = tmp_path / "head_ablation" / "countries" / "qwen3-32b"
    assert cli.main(["components", "score", "--dim", "countries", "--model", "qwen3-32b", "--out", out.as_posix()]) == 0
    (args, kwargs), = calls["score"]
    assert args[0] == runs and args[1] == "countries" and args[2].short_name == "qwen3-32b" and args[3] == out
    assert kwargs == {"tests": components.SCORE_TESTS, "n_orders": 3, "max_seq": 1200}


def test_components_ablate_defaults_are_the_declarations(intervention_stubs, tmp_path):
    calls, runs = intervention_stubs
    out = tmp_path / "head_ablation" / "countries" / "qwen3-32b"
    out.mkdir(parents=True)
    (out / "selected_heads.json").write_text("{}", encoding="utf-8")
    assert cli.main(["components", "ablate", "--dim", "countries", "--out", out.as_posix()]) == 0
    (args, kwargs), = calls["ablate"]
    decl = interp.load_declaration("components")
    assert args[0] == runs and args[1] == "countries" and args[3] == out / "selected_heads.json"
    assert args[4] == out / decl["published_as"]["trials"] == out / "ablate"
    assert kwargs == {"tests": represent.TESTS, "conditions": None, "n_orders": 3, "budget": 1024,
                      "score_budgets": [256, 512, 1024], "batch": 16}


def test_components_ablate_needs_the_head_sets(intervention_stubs, tmp_path, capsys):
    calls, _ = intervention_stubs
    out = tmp_path / "head_ablation" / "countries" / "qwen3-32b"
    assert cli.main(["components", "ablate", "--dim", "countries", "--out", out.as_posix()]) == 1
    assert "run `components score` first" in capsys.readouterr().err
    assert calls == {}


def test_intervention_without_runs_root_is_a_clean_error(intervention_stubs, monkeypatch, tmp_path, capsys):
    monkeypatch.delenv(tables.RUNS_ENV)
    out = tmp_path / "x"
    assert cli.main(["components", "score", "--dim", "cities", "--out", out.as_posix()]) == 1
    assert tables.RUNS_ENV in capsys.readouterr().err


def test_interventions_sh_lines_dispatch(intervention_stubs, tmp_path):
    """The four STAGE lines of scripts/slurm/interventions.sh reach their handlers."""
    calls, runs = intervention_stubs
    SHELL_VALUES["RUNS_ROOT"] = runs.as_posix()
    try:
        lines = [a for a in _invocations(ROOT / "scripts" / "slurm" / "interventions.sh")
                 if a[0] in ("represent", "components")]
    finally:
        SHELL_VALUES["RUNS_ROOT"] = "/tmp/runs"
    assert [a[:2] for a in lines] == [["represent", "fit"], ["represent", "intervene"],
                                      ["components", "score"], ["components", "ablate"]]
    rep = runs / "rank_intervention" / "countries" / "qwen3-32b"
    rep.mkdir(parents=True)
    np.savez(rep / "rank_axis.npz", layers=np.arange(2))
    comp = runs / "head_ablation" / "countries" / "qwen3-32b"
    comp.mkdir(parents=True)
    (comp / "selected_heads.json").write_text("{}", encoding="utf-8")
    for argv in lines:
        assert cli.main(argv) == 0
    assert sorted(calls) == ["ablate", "extract", "fit", "intervene", "score"]
    assert calls["intervene"][0][0][4] == rep / "intervene_scrub_ctxonly"
    assert calls["ablate"][0][0][4] == comp / "ablate"


def test_smoke_defaults_to_a_new_temporary_directory(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(cli.smoke, "run_smoke", lambda out: seen.append(out) or 0)
    monkeypatch.setattr(cli.tempfile, "mkdtemp", lambda prefix: str(tmp_path / f"{prefix}x"))
    assert cli.main(["smoke"]) == 0 and seen == [tmp_path / "geo-alg-ti-llms-smoke-x"]
    assert cli.main(["smoke", "--out", str(tmp_path / "mine")]) == 0 and seen[-1] == tmp_path / "mine"
