"""The whole pipeline on tiny inputs: trials, prompts, scoring, stored rows, export, manifest.

``run_smoke`` takes the symbolic-distance declaration cut down to one dimension (country
population, main item set only), two reasoning budgets and three tests (the IWL and ICL tests
direct, the IWL ladder), runs them on the stub model into the published layout, exports
the tables as the real export would and verifies the manifest. Seconds on a laptop.
"""

from __future__ import annotations

import copy
from pathlib import Path

from geo_alg_ti_llms import experiments, stub, tables
from geo_alg_ti_llms.run import run_direct, run_ladder

DIMENSION = "countries"
BUDGETS = [32, 64]
TESTS = ("iwl", "icl", "budget_iwl")


def small_experiment() -> dict:
    """``symbolic_distance`` reduced to one item set, two budgets and three tests."""
    exp = copy.deepcopy(experiments.load_experiment("symbolic_distance"))
    exp["dimensions"] = {DIMENSION: {**exp["dimensions"][DIMENSION], "replicates": 1}}
    exp["budgets"] = BUDGETS
    exp["tests"] = {a: exp["tests"][a] for a in TESTS}
    return exp


def run_smoke(out: Path, log=print) -> int:
    out = Path(out)
    exp = small_experiment()
    cfg = stub.model_config()
    models_dir = out / "models"
    stub.write_config(models_dir)
    runs = out / "runs"
    specs = tables.test_specs(exp)
    for test in TESTS:
        for dim in specs[test].item_sets:
            spec = experiments.resolve(exp, dim, test, cfg)
            target = tables.test_dir(runs, specs[test], dim, cfg.short_name)
            log(f"== {test} on {dim} -> {target}")
            if spec.runner == "direct":
                run_direct(spec.geometry, cfg, target)
            else:
                run_ladder(spec.geometry, spec.ladder, cfg, target, gen_batch_size=4)
    tables_dir = out / "tables"
    manifest = tables.export_trials(exp, runs, tables_dir, models=[cfg.short_name],
                                    models_dir=models_dir, log=log)
    problems = tables.verify_manifest(tables_dir)
    if problems:
        raise AssertionError(f"smoke manifest: {problems}")
    rows = {t["table"]: t["rows"] for t in manifest["tables"]}
    log(f"smoke ok: {rows} under {tables_dir}")
    return 0
