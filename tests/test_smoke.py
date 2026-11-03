"""The smoke command: the whole pipeline on the stub model, ending in a verified manifest."""

from __future__ import annotations

import pandas as pd

from geo_alg_ti_llms import smoke, tables


def test_smoke_runs_end_to_end(tmp_path):
    assert smoke.run_smoke(tmp_path, log=lambda *_: None) == 0
    assert tables.verify_manifest(tmp_path / "tables") == []
    direct = pd.read_parquet(tmp_path / "tables" / "direct.parquet")
    ladders = pd.read_parquet(tmp_path / "tables" / "ladders.parquet")
    assert sorted(direct["test"].unique()) == ["icl", "iwl"] and len(direct) == 72 + 720
    assert sorted(ladders["budget"].unique()) == [0, *smoke.BUDGETS] and len(ladders) == 3 * 72
    assert set(direct["model"]) == {"stub"} and set(direct["reasoning_kind"]) == {"native"}
