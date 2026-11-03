"""Context-reading heads: the statement reader, the head-score bookkeeping on a toy attention
tensor, the mean-ablation hook on a toy model (torch), the head-set selection, and the committed
components tables."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from geo_alg_ti_llms import components, interp, tables
from geo_alg_ti_llms.components import (
    HeadAblator,
    K_LIST,
    N_OUTSIDE,
    accumulate_masses,
    chain_record,
    premise_lines,
    select_head_sets,
    statements,
    token_positions,
)

_ROOT = Path(__file__).resolve().parent.parent
TABLES = _ROOT / "results" / "components" / "tables"
DIMS = ("countries", "cities", "usstates")
ITEMS = ["Peru", "Chile", "Spain", "Japan", "Brazil"]
REASONING_PROMPT = ("<|im_start|>system\nx<|im_end|>\n<|im_start|>user\n"
          "Chile > Peru\nSpain > Chile\nJapan > Spain\nBrazil > Japan\n\n"
          "Is it true or false that Spain > Peru?<|im_end|>\n<|im_start|>assistant\n<think>\n")


# --- the statement reader ----------------------------------------------------------------

def test_statements_read_symbols_and_words():
    trace = ("Spain > Chile and Chile is larger than Peru, so Spain is above Peru. "
             "Peru < Spain. Japan has a larger population than Peru. The United States ranks higher.")
    found = statements(trace, ITEMS + ["United States"])
    pairs = [(s["x"], s["y"]) for s in found]
    assert pairs == [("Spain", "Chile"), ("Chile", "Peru"), ("Spain", "Peru"), ("Peru", "Spain"), ("Japan", "Peru")]
    for s in found:
        assert trace[s["start"]:s["end"]].startswith(s["x"]) and trace[s["start"]:s["end"]].endswith(s["y"])
    assert statements("Spain > Spain", ITEMS) == []
    assert [(s["x"], s["y"]) for s in statements("USA > Peru", ["United States", "Peru"])] == [("United States", "Peru")]


def test_premise_lines_and_token_positions():
    (s, e), lines = premise_lines(REASONING_PROMPT, ITEMS)
    assert REASONING_PROMPT[s:e] == "Chile > Peru\nSpain > Chile\nJapan > Spain\nBrazil > Japan\n\n"
    assert [L["items"] for L in lines] == [["Peru", "Chile"], ["Chile", "Spain"], ["Spain", "Japan"], ["Japan", "Brazil"]]
    assert all(REASONING_PROMPT[L["start"]:L["end"]].count(">") == 1 for L in lines)
    offsets = [(0, 3), (3, 6), (6, 10), (10, 11)]
    assert token_positions(offsets, 4, 8) == [1, 2]
    assert token_positions(offsets, 10, 11) == [3]


class _CharTokenizer:
    """Every character one token, so offsets are trivial."""
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=True):
        return {"input_ids": [ord(c) for c in text], "offset_mapping": [(i, i + 1) for i in range(len(text))]}


def test_chain_record_splits_composed_inferences_from_restatements():
    trace = "Spain > Chile. So Spain > Peru. Done"
    t = dict(stage1=REASONING_PROMPT, trace=trace, items=ITEMS, arm="cot_reversed", trial_id=1)   # stored keys
    r = chain_record(_CharTokenizer(), t, max_seq=10_000)
    n_s1 = len(REASONING_PROMPT)
    assert r["n_s1"] == n_s1 and len(r["ids"]) == n_s1 + len(trace)
    (s, e), lines = premise_lines(REASONING_PROMPT, ITEMS)
    assert r["prem_all"] == list(range(s, e))
    # "Spain > Chile" are adjacent ranks (a restatement); "Spain > Peru" is two ranks apart
    # (composed)
    copy_pos = {p for p, _ in r["copy"]}; deriv_pos = {p for p, _ in r["deriv"]}
    assert copy_pos == set(range(n_s1, n_s1 + len("Spain > Chile")))
    start = n_s1 + trace.index("Spain > Peru")
    assert deriv_pos == set(range(start, start + len("Spain > Peru")))
    rel = r["deriv"][0][1]
    relevant = [L for L in lines if "Spain" in L["items"] or "Peru" in L["items"]]
    assert rel == sorted(p for L in relevant for p in range(L["start"], L["end"]))
    assert set(r["ctrl"]) == set(range(n_s1, len(r["ids"]))) - copy_pos - deriv_pos


def test_chain_record_refuses_a_chain_longer_than_max_seq():
    t = dict(stage1=REASONING_PROMPT, trace="Spain > Peru.", items=ITEMS, arm="cot_icl", trial_id=7)
    n = len(REASONING_PROMPT) + len("Spain > Peru.")
    assert len(chain_record(_CharTokenizer(), t, max_seq=n)["ids"]) == n
    with pytest.raises(ValueError, match=f"cot_icl trial 7: prompt and chain are {n} tokens"):
        chain_record(_CharTokenizer(), t, max_seq=n - 1)


# --- the bookkeeping on a toy attention tensor (torch) ---------------------------------------

def test_accumulate_masses_on_a_toy_attention_tensor():
    torch = pytest.importorskip("torch")
    H, T = 2, 6
    w = torch.zeros(H, T, T)
    w[0, 4, 0] = 0.5; w[0, 4, 1] = 0.3; w[0, 4, 5] = 0.2          # head 0 at the inference token
    w[1, 4, 1] = 1.0
    w[0, 5, 0] = 0.4; w[0, 5, 2] = 0.6                             # head 0 at the control token
    w[0, 3, 1] = 0.7; w[1, 3, 2] = 0.9                             # the restatement token
    rec = dict(prem_all=[0, 1, 2], deriv=[(4, [1])], copy=[(3, [1, 2])], ctrl=[5])
    L = 1
    rel = np.zeros((L, H)); all_ = np.zeros((L, H)); cp = np.zeros((L, H)); ctrl = np.zeros((L, H))
    accumulate_masses(w, rec, 0, rel, all_, cp, ctrl)
    assert np.allclose(rel[0], [0.3, 1.0])         # mass onto the relevant premise line
    assert np.allclose(all_[0], [0.8, 1.0])        # mass onto the whole premise block
    assert np.allclose(cp[0], [0.7, 0.9])          # restatement token onto its lines
    assert np.allclose(ctrl[0], [1.0, 0.0])        # control token onto the block
    accumulate_masses(w, rec, 0, rel, all_, cp, ctrl)
    assert np.allclose(rel[0], [0.6, 2.0])         # sums accumulate over chains
    empty = dict(prem_all=[0], deriv=[], copy=[], ctrl=[])
    accumulate_masses(w, empty, 0, rel, all_, cp, ctrl)
    assert np.allclose(rel[0], [0.6, 2.0])


# --- the mean-ablation hook on a toy model (torch) ----------------------------------------------

def _toy_attention_model(torch, n_layers, H, D):
    nn = torch.nn

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.o_proj = nn.Linear(H * D, H * D, bias=False)

        def forward(self, x):
            return self.o_proj(x)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = Attn()

        def forward(self, x):
            return self.self_attn(x)

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Block() for _ in range(n_layers)])

        def forward(self, x):
            for blk in self.layers:
                x = blk(x)
            return x

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()

        def forward(self, x):
            return self.model(x)

    torch.manual_seed(0)
    return Toy()


def test_mean_ablation_replaces_the_selected_head_slices():
    torch = pytest.importorskip("torch")
    L, H, D, B, T = 2, 4, 3, 2, 5
    model = _toy_attention_model(torch, L, H, D)
    means = np.random.default_rng(0).standard_normal((L, H, D)).astype(np.float32)
    abl = HeadAblator(model, means, H, D)
    seen = {}
    for l, blk in enumerate(model.model.layers):
        blk.self_attn.o_proj.register_forward_pre_hook(lambda m, a, l=l: seen.__setitem__(l, a[0].clone()))
    x = torch.randn(B, T, H * D)
    with torch.no_grad():
        intact = model(x).clone()
        inputs_intact = {l: v.clone() for l, v in seen.items()}
        abl.set_heads([(0, 1), (0, 3), (1, 0)])
        ablated = model(x).clone()
    assert not torch.allclose(ablated, intact)
    assert abl.n_edits == B * T * 3
    for l, heads in ((0, [1, 3]), (1, [0])):
        got = seen[l].view(B, T, H, D)
        for h in range(H):
            if h in heads:
                assert torch.allclose(got[:, :, h], torch.as_tensor(means[l, h]).expand(B, T, D))
            elif l == 0:
                assert torch.allclose(got[:, :, h], inputs_intact[l].view(B, T, H, D)[:, :, h])
    abl.set_heads([])
    with torch.no_grad():
        assert torch.allclose(model(x), intact)
    abl.remove()


# --- the head-set selection ----------------------------------------------------------------

def test_head_set_selection():
    ranked = [(l, h) for l in range(4) for h in range(64)]      # 256 heads in score order
    sel = select_head_sets(ranked)
    assert set(sel) == {f"top{k}" for k in K_LIST} | {f"rand{k}_s0" for k in K_LIST}
    top = set(ranked[:N_OUTSIDE])
    for k in K_LIST:
        assert sel[f"top{k}"] == ranked[:k]
        r = sel[f"rand{k}_s0"]
        assert len(r) == k == len(set(r)) and not (set(r) & top)
        assert r == sorted(r, key=ranked.index)
        rng = np.random.default_rng(100 * k)
        rest = ranked[N_OUTSIDE:]
        assert r == [rest[i] for i in sorted(rng.choice(len(rest), size=k, replace=False))]
    assert sel["top8"] == sel["top16"][:8] == sel["top32"][:8]
    two = select_head_sets(ranked, k_list=(8,), seeds=(0, 1))
    assert two["rand8_s0"] == sel["rand8_s0"] and two["rand8_s1"] != two["rand8_s0"]
    assert components.conditions_for("cot_icl") == components.CHAIN_CONDITIONS
    assert components.conditions_for("iwl") == components.DIRECT_CONDITIONS


# --- the committed tables ---------------------------------------------------------------

# Accuracy under the headline head sets, from the stored figure data: (correct, trials).
PANEL_E = {
    "countries": {("iwl", 0, "top32"): (100, 126), ("content_reversed", 0, "top32"): (67, 126),
                  ("cot_reversed", 1024, "top32"): (101, 126), ("cot_icl", 1024, "top32"): (109, 126),
                  ("iwl", 0, "rand32_s0"): (102, 126), ("content_reversed", 0, "rand32_s0"): (73, 126),
                  ("cot_reversed", 1024, "rand32_s0"): (124, 126), ("cot_icl", 1024, "rand32_s0"): (121, 126)},
    "cities": {("iwl", 0, "top32"): (55, 84), ("content_reversed", 0, "top32"): (64, 126),
               ("cot_reversed", 1024, "top32"): (92, 126), ("cot_icl", 1024, "top32"): (93, 126),
               ("iwl", 0, "rand32_s0"): (55, 84), ("content_reversed", 0, "rand32_s0"): (70, 126),
               ("cot_reversed", 1024, "rand32_s0"): (119, 126), ("cot_icl", 1024, "rand32_s0"): (126, 126)},
    "usstates": {("iwl", 0, "top32"): (92, 126), ("content_reversed", 0, "top32"): (64, 126),
                 ("cot_reversed", 1024, "top32"): (100, 126), ("cot_icl", 1024, "top32"): (108, 126),
                 ("iwl", 0, "rand32_s0"): (97, 126), ("content_reversed", 0, "rand32_s0"): (57, 126),
                 ("cot_reversed", 1024, "rand32_s0"): (119, 126), ("cot_icl", 1024, "rand32_s0"): (121, 126)},
}
# Change in IWL-direct and ICL-chain accuracy per head set, as counts:
# (iwl - intact, icl - intact).
PANEL_D = {
    "countries": {"top16": (0, 0), "top32": (-5, -14), "rand16_s0": (-2, -2), "rand32_s0": (-3, -2)},
    "cities": {"top16": (0, 2), "top32": (-4, -30), "rand16_s0": (-3, 0), "rand32_s0": (-4, 3)},
    "usstates": {"top16": (0, -9), "top32": (-5, -15), "rand16_s0": (2, -1), "rand32_s0": (0, -2)},
}


@pytest.fixture(scope="module")
def committed():
    if not (TABLES / "MANIFEST.yaml").exists():
        pytest.skip("components tables not exported")
    manifest = tables.load_manifest(TABLES / "MANIFEST.yaml")
    trials = pd.read_parquet(TABLES / "trials.parquet")
    heads = pd.read_parquet(TABLES / "heads.parquet")
    return manifest, trials, heads


def test_manifest_digests(committed):
    assert tables.verify_manifest(TABLES) == []
    manifest, trials, heads = committed
    entry = {t["table"]: t for t in manifest["tables"]}
    assert set(entry) == {"trials", "heads"}
    assert entry["trials"]["rows"] == len(trials) == 8778 and entry["heads"]["rows"] == len(heads) == 1080
    assert entry["trials"]["trial_sets"] == interp.trial_set_digests(trials)
    for block in manifest["reproduces_stored_summaries"].values():
        if isinstance(block, dict) and "mismatches" in block:
            assert block["mismatches"] == []


def test_trial_table_shape(committed):
    _, trials, _ = committed
    assert list(trials.columns) == interp.TRIAL_COLUMNS[:10] + ["k"] + interp.TRIAL_COLUMNS[10:]
    assert set(trials["budget"].unique()) == {0, 1024}
    assert trials.duplicated(["dim", "test", "condition", "budget", "pool", "d", "trial_id"]).sum() == 0
    k_of = trials["condition"].map(lambda c: 0 if c == "intact" else int(c.removeprefix("top").removeprefix("rand").split("_")[0]))
    assert (trials["k"] == k_of).all()
    decl = interp.load_declaration("components")
    for (dim, test), g in trials.groupby(["dim", "test"]):
        assert set(g["condition"]) == set(decl["conditions"][test]), (dim, test)
        assert g.groupby("condition").size().nunique() == 1


def test_head_sets(committed):
    manifest, _, heads = committed
    meta = {t["table"]: t for t in manifest["tables"]}["heads"]["selected_heads"]
    for dim in DIMS:
        g = heads[heads["dim"] == dim]
        sets = {s: list(zip(x["layer"], x["head"])) for s, x in g.groupby("set")}
        for k in K_LIST:
            assert len(sets[f"top{k}"]) == k and len(sets[f"rand{k}_s0"]) == k == len(sets[f"rand{k}_s1"])
            assert sets[f"top{k}"] == sets["top64"][:k]
            assert not set(sets[f"rand{k}_s0"]) & set(sets["top64"])
            assert (g[g["set"] == f"top{k}"]["rank"].values == np.arange(1, k + 1)).all()
        assert meta[dim]["top_sets_are_prefixes_of_each_other"] and meta[dim]["n_chains"] == 252
        assert (meta[dim]["L"], meta[dim]["H"]) == (64, 64) and heads["layer"].max() < 64 and heads["head"].max() < 64


def test_headline_bars_recompute_from_the_table(committed):
    manifest, trials, _ = committed
    recorded = {t["table"]: t for t in manifest["tables"]}["trials"]["panel_numbers"]

    def count(dim, test, budget, cond):
        g = trials[(trials.dim == dim) & (trials.test == test) & (trials.budget == budget) & (trials.condition == cond)]
        return int(g["is_correct"].sum()), len(g)

    for dim in DIMS:
        for (test, budget, cond), (k, n) in PANEL_E[dim].items():
            assert count(dim, test, budget, cond) == (k, n), (dim, test, cond)
        i0, n0 = count(dim, "iwl", 0, "intact"); c0, nc = count(dim, "cot_icl", 1024, "intact")
        for cond, (di, dc) in PANEL_D[dim].items():
            ki, ni = count(dim, "iwl", 0, cond); kc, ncc = count(dim, "cot_icl", 1024, cond)
            assert (ni, ncc) == (n0, nc) and (ki - i0, kc - c0) == (di, dc), (dim, cond)
        for r in recorded[dim]:
            if r["panel"] == "E":
                k, n = PANEL_E[dim][(r["test"], r["budget"], r["condition"])]
                assert r["n"] == n and r["value"] == k / n
            if r["panel"] == "D":
                di, dc = PANEL_D[dim][r["condition"]]
                assert np.isclose(r["d_iwl_direct"], di / r["n_direct"]) and np.isclose(r["d_icl_chain"], dc / r["n_chain"])


def test_ablating_the_context_reading_heads_hurts_the_chains_more_than_random_heads(committed):
    _, trials, _ = committed
    for dim in DIMS:
        def acc(test, budget, cond):
            g = trials[(trials.dim == dim) & (trials.test == test) & (trials.budget == budget) & (trials.condition == cond)]
            return g["is_correct"].mean()
        assert acc("cot_icl", 1024, "top32") < acc("cot_icl", 1024, "rand32_s0") - 0.05, dim
        assert acc("cot_reversed", 1024, "top32") < acc("cot_reversed", 1024, "rand32_s0") - 0.05, dim
        assert abs(acc("iwl", 0, "top32") - acc("iwl", 0, "intact")) < 0.1, dim
