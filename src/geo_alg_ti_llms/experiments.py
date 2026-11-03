"""Experiment declarations: ``experiments/<name>/experiment.yaml`` → runner configs.

One declaration per experiment names the dimensions (an item set, the
instruction naming the dimension, the context-only wording of it, replicate or
item-spacing sets), the tests (which runner, context condition, rank
assignment, context mode, item set and instruction), the shared trial geometry,
the reasoning budgets and the trigger per model kind. :func:`resolve` turns
one ``(experiment, set key, test, model)`` into the plain config dicts that
:mod:`geo_alg_ti_llms.run` consumes, so the runners' bodies are those of the
published runs.

Schema of ``experiment.yaml``::

    name: symbolic_distance
    question: ...
    geometry:                      # shared by every test
      master_seed: 42
      template_id: T2
      query_wrapper_id: is_it_true_or_false
      separator: "\\n"
      N_values: [11]
      hop_values: [1, 2, ..., 10]  # trial sets with hop + 3 > N are skipped
      all_pairs: true
      n_premise_orders: 10
    budgets: [32, 64, ..., 1024]   # the reasoning ladder (ladder tests only)
    triggers:                      # reasoning-pass trigger by model kind
      native_thinking: none        # model config thinking_mode: true
      prompted: passive
    dimensions:
      countries:
        name: country population
        instruction: iwl_population_T2
        instruction_context_only: iwl_population_T2_ctxonly
        item_set: countries_by_population_11 # the main item set ...
        replicates: 3                        # ... and its _rep2, _rep3 twins
        # or, for the item-spacing experiment, explicit keyed item sets:
        # item_sets: {scale2: countries_by_population_scale2, ...}
    tests:
      iwl:
        runner: direct                       # direct | ladder
        condition: no_context                # a name, or {assignment: name}
        assignment: natural                  # one strategy or a list
        surface: zero_context                # fixed_gt | zero_context | disjoint_context
        item_set: dimension                  # dimension | dimension_disjoint | <item set name>
        instruction: dimension               # dimension | dimension_context_only | <id>
        item_sets: all                       # main | all (default main)
        dimensions: [countries]              # optional restriction
        n_premise_orders: 1                  # optional override
        published_as: iwl                    # results directory of the published run

A *set key* is how the CLI names an item set: the dimension (its main item set), or
``<dimension>_rep2`` / ``<dimension>_scale2`` for a replicate or keyed item set.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from geo_alg_ti_llms.models import ModelConfig


DEFAULT_EXPERIMENTS_DIR: Path = Path(__file__).resolve().parents[2] / "experiments"
"""The repository's ``experiments`` directory."""

TEST_RUNNERS = ("direct", "ladder")
SURFACE_MODES = ("fixed_gt", "zero_context", "disjoint_context")
ASSIGNMENTS = ("natural", "reversed", "random")


@dataclass(frozen=True, slots=True)
class RunSpec:
    """One resolved job: the dicts the runners take, plus its labels."""

    experiment: str
    test: str
    runner: str                 # "direct" | "ladder"
    dimension: str
    set_key: str
    item_set: str
    conditions: dict[str, str]  # assignment -> condition name
    geometry: dict[str, Any]
    ladder: dict[str, Any] | None   # defaults + trial sets (ladder tests only)
    published_as: str


def experiment_path(name: str, experiments_dir: str | Path = DEFAULT_EXPERIMENTS_DIR) -> Path:
    return Path(experiments_dir) / name / "experiment.yaml"


def load_experiment(name: str, experiments_dir: str | Path = DEFAULT_EXPERIMENTS_DIR) -> dict:
    """Load and validate ``experiments/<name>/experiment.yaml``."""
    path = experiment_path(name, experiments_dir)
    with path.open("r", encoding="utf-8") as f:
        exp = yaml.safe_load(f)
    if exp.get("name") != name:
        raise ValueError(f"{path}: name field {exp.get('name')!r} != directory {name!r}")
    for key in ("geometry", "dimensions", "tests"):
        if key not in exp:
            raise ValueError(f"{path}: missing {key!r}")
    for test, spec in exp["tests"].items():
        if spec.get("runner") not in TEST_RUNNERS:
            raise ValueError(f"{path}: test {test!r}: runner must be one of {TEST_RUNNERS}")
        if spec.get("surface") not in SURFACE_MODES:
            raise ValueError(f"{path}: test {test!r}: surface must be one of {SURFACE_MODES}")
        for a in _assignments(spec):
            if a not in ASSIGNMENTS:
                raise ValueError(f"{path}: test {test!r}: assignment {a!r} not in {ASSIGNMENTS}")
        if spec["runner"] == "ladder" and "budgets" not in exp:
            raise ValueError(f"{path}: ladder test {test!r} needs top-level 'budgets'")
    return exp


def list_experiments(experiments_dir: str | Path = DEFAULT_EXPERIMENTS_DIR) -> list[str]:
    return sorted(p.parent.name for p in Path(experiments_dir).glob("*/experiment.yaml"))


def _assignments(test_spec: dict) -> list[str]:
    a = test_spec["assignment"]
    return list(a) if isinstance(a, list) else [a]


def set_keys(exp: dict, dimension: str) -> dict[str, str]:
    """Set key → item set name for one dimension (the main item set first)."""
    q = exp["dimensions"][dimension]
    if "item_sets" in q:
        return {f"{dimension}_{tag}": item_set for tag, item_set in q["item_sets"].items()}
    keys = {dimension: q["item_set"]}
    for i in range(2, int(q.get("replicates", 1)) + 1):
        keys[f"{dimension}_rep{i}"] = f"{q['item_set']}_rep{i}"
    return keys


def test_dimensions(exp: dict, test: str) -> list[str]:
    """The dimensions a test runs on (all, unless the test restricts them)."""
    spec = exp["tests"][test]
    return list(spec.get("dimensions") or exp["dimensions"].keys())


def test_set_keys(exp: dict, test: str) -> dict[str, tuple[str, str]]:
    """Set key → (dimension, item set name) for every item set a test runs on."""
    spec = exp["tests"][test]
    out: dict[str, tuple[str, str]] = {}
    for dimension in test_dimensions(exp, test):
        keys = set_keys(exp, dimension)
        if spec.get("item_sets", "main") != "all":
            keys = dict([next(iter(keys.items()))])
        for key, item_set in keys.items():
            out[key] = (dimension, _test_item_set(spec, item_set))
    return out


def _test_item_set(spec: dict, dimension_set: str) -> str:
    item_set = spec["item_set"]
    if item_set == "dimension":
        return dimension_set
    if item_set == "dimension_disjoint":
        return f"{dimension_set}_disjoint"
    return item_set


def _test_instruction(spec: dict, q: dict) -> str:
    instr = spec["instruction"]
    if instr == "dimension":
        return q["instruction"]
    if instr == "dimension_context_only":
        return q["instruction_context_only"]
    return instr


def _conditions(spec: dict) -> dict[str, str]:
    cond = spec["condition"]
    assignments = _assignments(spec)
    if isinstance(cond, dict):
        missing = [a for a in assignments if a not in cond]
        if missing:
            raise ValueError(f"condition mapping lacks assignments {missing}")
        return {a: cond[a] for a in assignments}
    return {a: cond for a in assignments}


def trigger_for(exp: dict, model_cfg: ModelConfig) -> str:
    """The reasoning-pass trigger for a model: ``none`` on native thinking
    models (``thinking_mode: true``), the prompted trigger otherwise."""
    triggers = exp.get("triggers", {"native_thinking": "none", "prompted": "passive"})
    return triggers["native_thinking"] if model_cfg.thinking_mode else triggers["prompted"]


def resolve(
    exp: dict, set_key: str, test: str, model_cfg: ModelConfig | None = None,
) -> RunSpec:
    """Resolve one ``(experiment, set key, test[, model])`` into runner configs.

    ``model_cfg`` is needed for ladder tests (the trigger depends on the model).
    """
    if test not in exp["tests"]:
        raise ValueError(f"experiment {exp['name']!r} has no test {test!r}; "
                         f"tests: {sorted(exp['tests'])}")
    spec = exp["tests"][test]
    keys = test_set_keys(exp, test)
    if set_key not in keys:
        raise ValueError(f"test {test!r} does not run on item set {set_key!r}; "
                         f"item sets: {sorted(keys)}")
    dimension, item_set = keys[set_key]
    q = exp["dimensions"][dimension]
    g = exp["geometry"]
    geometry = {
        "master_seed": int(g["master_seed"]),
        "prompt": {
            "template_id": g["template_id"],
            "item_pool": item_set,
            "instructions_v1_id": _test_instruction(spec, q),
            "query_wrapper_id": g["query_wrapper_id"],
            "separator": g["separator"],
        },
        "grid": {"N_values": list(g["N_values"]), "hop_values": list(g["hop_values"])},
        "all_pairs": bool(g.get("all_pairs", True)),
        "n_premise_orders": int(spec.get("n_premise_orders", g["n_premise_orders"])),
        "assignment_strategies": _assignments(spec),
        "surface_form_modes": [spec["surface"]],
    }
    ladder = None
    if spec["runner"] == "ladder":
        if model_cfg is None:
            raise ValueError("a ladder test needs the model config (its trigger depends on the model)")
        budgets = [int(b) for b in exp["budgets"]]
        ladder = {
            "defaults": {
                "trigger_id": trigger_for(exp, model_cfg),
                "max_new_tokens": max(budgets),
                "instructions_v1_id": None,
            },
            "budgets": {f"b{b}": {"max_new_tokens": b} for b in budgets},
        }
    return RunSpec(
        experiment=exp["name"], test=test, runner=spec["runner"], dimension=dimension,
        set_key=set_key, item_set=item_set, conditions=_conditions(spec),
        geometry=geometry, ladder=ladder, published_as=spec.get("published_as", test),
    )


def jobs(exp: dict) -> list[tuple[str, str]]:
    """Every ``(set key, test)`` the experiment declares, tests in declaration order."""
    out: list[tuple[str, str]] = []
    for test in exp["tests"]:
        for key in test_set_keys(exp, test):
            out.append((key, test))
    return out
