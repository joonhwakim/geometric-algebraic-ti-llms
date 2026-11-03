"""The rank direction: the closed-form fit and edits on a synthetic matrix, the mention scanner
and the editor's hooks on a toy model (torch), and the committed representation tables."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from geo_alg_ti_llms import interp, represent, tables
from geo_alg_ti_llms.represent import (
    RankEditor,
    ShrunkCov,
    axis_from,
    axis_from_dense,
    ledoit_wolf,
    pairwise_acc,
)

_ROOT = Path(__file__).resolve().parent.parent
TABLES = _ROOT / "results" / "representation" / "tables"
DIMS = ("countries", "cities", "usstates")


def _synthetic(n=60, d=8, seed=0):
    """Rows with a rank signal along one direction plus correlated noise."""
    rng = np.random.default_rng(seed)
    z = np.repeat(np.linspace(-1.5, 1.5, 12), n // 12)
    z = (z - z.mean()) / z.std()
    direction = rng.standard_normal(d); direction /= np.linalg.norm(direction)
    A = rng.standard_normal((d, d))
    X = 3.0 + z[:, None] * direction[None] * 2.0 + rng.standard_normal((n, d)) @ A * 0.3
    return X, z


# --- the closed form ------------------------------------------------------------------

def test_shrunk_covariance_matches_the_dense_ledoit_wolf():
    X, _ = _synthetic()
    cov = ShrunkCov(X)
    S = ledoit_wolf(X)
    v = np.random.default_rng(1).standard_normal(X.shape[1])
    assert np.allclose(cov.apply(v, 1.0), S @ v, atol=1e-8)
    assert np.allclose(cov.apply(cov.apply(v, -0.5), 0.5), v, atol=1e-8)


def test_axis_matches_the_dense_reference_fit():
    X, z = _synthetic()
    fast, dense = axis_from(X, z), axis_from_dense(X, z)
    for key in ("mu", "v_read", "v_write"):
        assert np.allclose(fast[key], dense[key], atol=1e-6), key


def test_read_and_write_directions_are_dual_and_the_coordinate_tracks_rank():
    X, z = _synthetic()
    ax = axis_from(X, z, rng=np.random.default_rng(5))
    # v_read . v_write = u^T Sigma^-1/2 Sigma^1/2 u = 1, for the fitted and the random direction
    assert np.isclose(ax["v_read"] @ ax["v_write"], 1.0, atol=1e-8)
    assert np.isclose(ax["v_read_rand"] @ ax["v_write_rand"], 1.0, atol=1e-8)
    a = (X - ax["mu"]) @ ax["v_read"]
    assert np.corrcoef(a, z)[0, 1] > 0.9


def test_erase_invert_write_and_random_control_edits():
    X, z = _synthetic()
    ax = axis_from(X, z, rng=np.random.default_rng(7))
    cov = ShrunkCov(X)
    mu, vr, vw = ax["mu"], ax["v_read"], ax["v_write"]
    x = X[3]
    a = vr @ (x - mu)
    assert abs(a) > 1e-3
    erased = x + (0.0 - a) * vw
    assert np.isclose(vr @ (erased - mu), 0.0, atol=1e-8)
    inverted = x + (-a - a) * vw
    assert np.isclose(vr @ (inverted - mu), -a, atol=1e-8)
    alpha, beta = np.polyfit(z, (X - mu) @ vr, 1)[::-1]
    for eta in (0.25, 0.5, 1.0):
        target = (1 - eta) * a + eta * (alpha + beta * 0.8)
        written = x + (target - a) * vw
        assert np.isclose(vr @ (written - mu), target, atol=1e-8)
    # the random control: a unit Gaussian direction u_r, v_read_rand = Sigma^-1/2 u_r
    u_r = cov.apply(ax["v_read_rand"], 0.5)
    assert np.isclose(np.linalg.norm(u_r), 1.0, atol=1e-8)
    # the random direction's erase removes its own coordinate and barely touches the fitted one
    a_r = ax["v_read_rand"] @ (x - mu)
    r_erased = x + (0.0 - a_r) * ax["v_write_rand"]
    assert np.isclose(ax["v_read_rand"] @ (r_erased - mu), 0.0, atol=1e-8)
    # a fresh seed gives a different random direction; the same seed the same one
    again = axis_from(X, z, rng=np.random.default_rng(7))
    other = axis_from(X, z, rng=np.random.default_rng(8))
    assert np.allclose(again["v_read_rand"], ax["v_read_rand"])
    assert not np.allclose(other["v_read_rand"], ax["v_read_rand"])


def test_pairwise_accuracy():
    z = np.array([0.0, 1.0, 2.0, 2.0])
    assert pairwise_acc(np.array([0.0, 1.0, 2.0, 5.0]), z) == 1.0
    assert pairwise_acc(np.array([2.0, 1.0, 0.0, 0.0]), z) == 0.0
    assert pairwise_acc(np.array([0.0, 2.0, 1.0, 1.0]), z) == 0.6   # two of five ordered pairs wrong


def test_written_ordering_is_near_orthogonal_to_the_real_one():
    pytest.importorskip("scipy")
    from scipy.stats import spearmanr
    items = [f"i{k}" for k in range(11)]
    z = represent.written_ordering(items, items, seed=3)
    assert set(z) == set(items)
    vals = np.array([z[i] for i in items])
    assert np.isclose(vals.mean(), 0.0, atol=1e-9) and np.isclose(vals.std(), 1.0, atol=1e-9)
    assert abs(spearmanr(vals, np.arange(11)).correlation) <= 0.2
    assert represent.written_ordering(items, items, seed=3) == z


def test_written_orderings_do_not_depend_on_the_process_hash_salt():
    pytest.importorskip("scipy")
    import os
    import subprocess
    import sys
    code = ("from geo_alg_ti_llms import represent; "
            "t = dict(cell_id='phaseD|N=11|pool=x', trial_id=7, items=['Chile', 'Peru', 'Spain'],"
            " pool='countries_by_population_11', A='Peru', B='Spain'); "
            "represent.set_order = lambda *_: ['Spain', 'Chile', 'Peru']; "
            "represent.attach_written_ordering([t]); print(t['z_new'])")
    out = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                          env={**os.environ, "PYTHONHASHSEED": salt}).stdout for salt in ("1", "2")}
    assert len(out) == 1


# --- the mention scanner --------------------------------------------------------------

def test_fit_prompts_and_spans(mock_tokenizer):
    items = ["Alpha", "Beta", "Gamma", "Delta", "Eps", "Zeta", "Eta", "Theta", "Iota", "Kappa", "Lambda", "Mu"]
    recs = represent.build_prompts("countries", items, np.random.default_rng(0), k_partner=2)
    assert len(recs) == len(items) * (1 + 2 + 2)
    for r in recs:
        s, e = r["span"]
        assert r["prompt"][s:e] == r["item"]
        assert r["kind"] in ("qA", "qB", "prem")
    prem = [r for r in recs if r["kind"] == "prem"]
    assert all(r["prompt"].count(" > ") == 11 for r in prem)   # nine premise lines, the query, the instruction


def test_scanner_flags_the_last_subtoken_of_every_mention(mock_tokenizer):
    ed = RankEditor.__new__(RankEditor)
    ed.tok = mock_tokenizer; ed.variants = {}
    items = ["Spain", "Peru", "Chile"]
    ed.set_items(items)
    ids = mock_tokenizer.encode("Is it true or false that Spain > Peru ? Spain again")
    found = ed.scan(ids, items)
    spain, peru = mock_tokenizer.encode("Spain")[0], mock_tokenizer.encode("Peru")[0]
    assert found == sorted({(i, "Spain") for i, t in enumerate(ids) if t == spain}
                           | {(i, "Peru") for i, t in enumerate(ids) if t == peru})
    assert ed._match(ids, len(ids) - 2, items) == "Spain"
    assert ed._match(ids, len(ids) - 1, items) is None
    assert ed._match(ids, 0, items) is None


# --- the editor on a toy model (torch) ---------------------------------------------------

def _toy_model(torch, n_vocab, d, n_layers):
    nn = torch.nn

    class Block(nn.Module):
        def forward(self, h):
            return (h * 1.0,)

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(n_vocab, d)
            self.layers = nn.ModuleList([Block() for _ in range(n_layers)])

        def forward(self, input_ids):
            h = self.embed_tokens(input_ids)
            for blk in self.layers:
                h = blk(h)[0]
            return h

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()

        def forward(self, input_ids):
            return self.model(input_ids)

    torch.manual_seed(0)
    return Toy()


def _axis_dict(X, z, layers):
    out = {}
    for l in layers:
        ax = axis_from(X, z, rng=np.random.default_rng(1000 + l))
        for k, v in ax.items():
            out[f"L{l}_{k}"] = v.astype(np.float32)
        alpha, beta = np.polyfit(z, (X - ax["mu"]) @ ax["v_read"], 1)[::-1]
        out[f"L{l}_alpha"] = np.float32(alpha); out[f"L{l}_beta"] = np.float32(beta)
    return out


@pytest.mark.parametrize("condition", ["erase", "invert", "write_1.0", "rand_erase", "write_rand"])
def test_editor_edits_only_the_flagged_positions(mock_tokenizer, condition):
    torch = pytest.importorskip("torch")
    X, z = _synthetic(d=8)
    items = ["Spain", "Peru", "Chile"]
    prompt = "Is it true or false that Spain > Peru ?"
    ids = mock_tokenizer.encode(prompt)
    layers = [0, 2]
    axis = _axis_dict(X, z, layers)
    model = _toy_model(torch, n_vocab=64, d=8, n_layers=3)
    # a trial with a written ordering the writes aim at
    trial = dict(items=items, z_new={"Spain": 1.2, "Peru": -0.3, "Chile": 0.1})
    ed = RankEditor(model, mock_tokenizer, axis, layers, {})
    ed.set_items(items)
    ed.trials = [trial]
    x = torch.tensor([ids])
    with torch.no_grad():
        intact = model(x).clone()
        ed.condition = condition; ed.eta = 1.0
        edited = model(x).clone()
    flagged = {p for p, _ in ed.scan(ids, items)}
    assert flagged and ed.n_edits == len(flagged) * len(layers)
    for p in range(len(ids)):
        if p not in flagged:
            assert torch.allclose(edited[0, p], intact[0, p])
    rand = condition in ("rand_erase", "write_rand")
    l = layers[-1]
    mu = torch.as_tensor(axis[f"L{l}_mu"]); vr = torch.as_tensor(axis[f"L{l}_v_read_rand" if rand else f"L{l}_v_read"])
    alpha, beta = float(axis[f"L{l}_alpha"]), float(axis[f"L{l}_beta"])
    for p, it in ed.scan(ids, items):
        a0 = torch.dot(vr, intact[0, p] - mu).item()
        a1 = torch.dot(vr, edited[0, p] - mu).item()
        if condition in ("erase", "rand_erase"):
            assert abs(a1) < 1e-4
        elif condition == "invert":
            # the toy blocks are the identity, so the inversion at the second edited layer undoes
            # the first
            assert abs(a1 - a0 * (-1) ** len(layers)) < 1e-4
        else:
            assert abs(a1 - (alpha + beta * trial["z_new"][it])) < 1e-4
    ed.remove()
    with torch.no_grad():
        assert torch.allclose(model(x), intact)


def test_editor_follows_one_decode_step(mock_tokenizer):
    torch = pytest.importorskip("torch")
    X, z = _synthetic(d=8)
    items = ["Spain", "Peru"]
    axis = _axis_dict(X, z, [0])
    model = _toy_model(torch, n_vocab=64, d=8, n_layers=1)
    ed = RankEditor(model, mock_tokenizer, axis, [0], {})
    ed.set_items(items); ed.trials = [dict(items=items, z_new={})]; ed.condition = "erase"
    prefix = mock_tokenizer.encode("that Peru >")
    with torch.no_grad():
        model(torch.tensor([prefix]))                       # prefill: the Peru mention is edited
        n_prefill = ed.n_edits
        model(torch.tensor([[mock_tokenizer.encode("Spain")[0]]]))   # one decode step completing a name
        assert ed.n_edits == n_prefill + 1
        model(torch.tensor([[mock_tokenizer.encode("?")[0]]]))       # one that does not
        assert ed.n_edits == n_prefill + 1
    assert ed.ids[0] == prefix + mock_tokenizer.encode("Spain") + mock_tokenizer.encode("?")
    ed.remove()


# --- the fit and the editor on a small Qwen3 (transformers) ------------------------------

def _qwen3(torch, n_vocab=64, n_layers=3):
    transformers = pytest.importorskip("transformers")
    cfg = transformers.Qwen3Config(vocab_size=n_vocab, hidden_size=16, intermediate_size=32,
                                   num_hidden_layers=n_layers, num_attention_heads=2, num_key_value_heads=1,
                                   head_dim=8, max_position_embeddings=64)
    torch.manual_seed(0)
    return transformers.Qwen3ForCausalLM(cfg).eval()


def test_the_fit_reads_each_block_output_and_the_last_before_the_final_norm():
    torch = pytest.importorskip("torch")
    model = _qwen3(torch)
    n = model.config.num_hidden_layers
    ids = torch.randint(0, 64, (1, 10), generator=torch.Generator().manual_seed(1))
    positions = [3, 7]
    states = represent.record_hidden_states(model, ids, positions, list(range(n + 1)))
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True)
        normed = model.model.norm(torch.as_tensor(states[n]))
    for l in range(n):
        assert np.array_equal(states[l], out.hidden_states[l][0, positions].numpy()), l
    # the library's last hidden state is the final normalization of the last block's output
    assert torch.allclose(normed, out.hidden_states[n][0, positions], atol=1e-6)
    assert not np.allclose(states[n], out.hidden_states[n][0, positions].numpy())


@pytest.mark.parametrize("condition", ["erase", "rand_erase"])
def test_an_erase_leaves_no_coordinate_where_the_fit_reads(mock_tokenizer, condition):
    torch = pytest.importorskip("torch")
    model = _qwen3(torch)
    layers = list(range(model.config.num_hidden_layers + 1))
    items = ["Spain", "Peru", "Chile", "Mexico", "Brazil", "Japan"]
    # an axis per hidden state, fit on the states the fit reads at the end of a few prompts per item
    rows, z = {l: [] for l in layers}, []
    for k, it in enumerate(items):
        for lead in ("Is it true that", "that", "Is it"):
            ids = mock_tokenizer.encode(f"{lead} {it}")
            got = represent.record_hidden_states(model, torch.tensor([ids]), [len(ids) - 1], layers)
            for l in layers:
                rows[l].append(got[l][0])
            z.append(k)
    z = (np.array(z) - np.mean(z)) / np.std(z)
    axis = {}
    for l in layers:
        ax = axis_from(np.array(rows[l], dtype=np.float64), z, rng=np.random.default_rng(1000 + l))
        axis.update({f"L{l}_{k}": v.astype(np.float32) for k, v in ax.items()})
        axis[f"L{l}_alpha"] = np.float32(0.0); axis[f"L{l}_beta"] = np.float32(1.0)
    ed = RankEditor(model, mock_tokenizer, axis, layers, {})
    ed.set_items(items); ed.trials = [dict(items=items, z_new={})]; ed.condition = condition
    ids = mock_tokenizer.encode("Is it true or false that Spain > Peru ?")
    flagged = [p for p, _ in ed.scan(ids, items)]
    edited = represent.record_hidden_states(model, torch.tensor([ids]), flagged, layers)
    ed.remove()
    intact = represent.record_hidden_states(model, torch.tensor([ids]), flagged, layers)
    rand = condition == "rand_erase"
    for l in layers:
        mu, vr = axis[f"L{l}_mu"], axis[f"L{l}_v_read_rand" if rand else f"L{l}_v_read"]
        before, after = (intact[l] - mu) @ vr, (edited[l] - mu) @ vr
        assert np.abs(before).max() > 1e-3, l
        assert np.allclose(after, 0.0, atol=1e-4 * max(1.0, np.abs(before).max())), l


# --- the committed tables ---------------------------------------------------------------

# Accuracy of the four scored conditions under the headline interventions: (correct, trials).
PANEL_E = {
    "countries": {("iwl", 0, "intact"): (105, 126), ("content_reversed", 0, "intact"): (72, 126),
                  ("cot_reversed", 1024, "intact"): (113, 126), ("cot_icl", 1024, "intact"): (123, 126),
                  ("iwl", 0, "erase"): (81, 126), ("content_reversed", 0, "erase"): (74, 126),
                  ("cot_reversed", 1024, "erase"): (122, 126), ("cot_icl", 1024, "erase"): (124, 126),
                  ("iwl", 0, "rand_erase"): (104, 126), ("content_reversed", 0, "rand_erase"): (73, 126),
                  ("cot_reversed", 1024, "rand_erase"): (119, 126), ("cot_icl", 1024, "rand_erase"): (122, 126)},
    "cities": {("iwl", 0, "intact"): (59, 84), ("content_reversed", 0, "intact"): (69, 126),
               ("cot_reversed", 1024, "intact"): (116, 126), ("cot_icl", 1024, "intact"): (123, 126),
               ("iwl", 0, "erase"): (44, 84), ("content_reversed", 0, "erase"): (74, 126),
               ("cot_reversed", 1024, "erase"): (122, 126), ("cot_icl", 1024, "erase"): (126, 126),
               ("iwl", 0, "rand_erase"): (57, 84), ("content_reversed", 0, "rand_erase"): (70, 126),
               ("cot_reversed", 1024, "rand_erase"): (121, 126), ("cot_icl", 1024, "rand_erase"): (125, 126)},
    "usstates": {("iwl", 0, "intact"): (97, 126), ("content_reversed", 0, "intact"): (61, 126),
                 ("cot_reversed", 1024, "intact"): (117, 126), ("cot_icl", 1024, "intact"): (123, 126),
                 ("iwl", 0, "erase"): (69, 126), ("content_reversed", 0, "erase"): (71, 126),
                 ("cot_reversed", 1024, "erase"): (119, 126), ("cot_icl", 1024, "erase"): (125, 126),
                 ("iwl", 0, "rand_erase"): (96, 126), ("content_reversed", 0, "rand_erase"): (67, 126),
                 ("cot_reversed", 1024, "rand_erase"): (116, 126), ("cot_icl", 1024, "rand_erase"): (126, 126)},
}
# Agreement with the written ordering on the reversed context by write strength.
PANEL_C = {
    "countries": {("content_reversed", "intact"): 66, ("content_reversed", "write_0.25"): 72,
                  ("content_reversed", "write_0.5"): 72, ("content_reversed", "write_0.75"): 73,
                  ("content_reversed", "write_1.0"): 73, ("content_reversed", "write_rand"): 68,
                  ("cot_reversed", "intact"): 65, ("cot_reversed", "write_1.0"): 69, ("cot_reversed", "write_rand"): 64},
    "cities": {("content_reversed", "intact"): 66, ("content_reversed", "write_0.25"): 78,
               ("content_reversed", "write_0.5"): 78, ("content_reversed", "write_0.75"): 78,
               ("content_reversed", "write_1.0"): 78, ("content_reversed", "write_rand"): 68,
               ("cot_reversed", "intact"): 61, ("cot_reversed", "write_1.0"): 65, ("cot_reversed", "write_rand"): 67},
    "usstates": {("content_reversed", "intact"): 59, ("content_reversed", "write_0.25"): 71,
                 ("content_reversed", "write_0.5"): 72, ("content_reversed", "write_0.75"): 73,
                 ("content_reversed", "write_1.0"): 73, ("content_reversed", "write_rand"): 63,
                 ("cot_reversed", "intact"): 63, ("cot_reversed", "write_1.0"): 67, ("cot_reversed", "write_rand"): 64},
}


@pytest.fixture(scope="module")
def committed():
    if not (TABLES / "MANIFEST.yaml").exists():
        pytest.skip("representation tables not exported")
    manifest = tables.load_manifest(TABLES / "MANIFEST.yaml")
    trials = pd.read_parquet(TABLES / "trials.parquet")
    probe = pd.read_parquet(TABLES / "probe.parquet")
    return manifest, trials, probe


def test_manifest_digests(committed):
    assert tables.verify_manifest(TABLES) == []
    manifest, trials, probe = committed
    entry = {t["table"]: t for t in manifest["tables"]}
    assert set(entry) == {"trials", "probe"}
    assert entry["trials"]["rows"] == len(trials) == 9450 and entry["probe"]["rows"] == len(probe) == 195
    assert entry["trials"]["trial_sets"] == interp.trial_set_digests(trials)
    for block in manifest["reproduces_stored_summaries"].values():
        if isinstance(block, dict) and "mismatches" in block:
            assert block["mismatches"] == []


def test_trial_table_shape(committed):
    _, trials, _ = committed
    assert list(trials.columns) == interp.TRIAL_COLUMNS[:10] + ["eta"] + interp.TRIAL_COLUMNS[10:]
    assert sorted(trials["d"].unique()) == [1, 2, 3, 4, 5, 6]
    assert set(trials["budget"].unique()) == {0, 1024}
    assert (trials["rank_low"] >= tables.DROP_LO).all() and (trials["rank_high"] <= tables.DROP_HI).all()
    assert trials.duplicated(["dim", "test", "condition", "budget", "pool", "d", "trial_id"]).sum() == 0
    writes = trials["condition"].str.startswith("write_")
    assert trials.loc[~writes, "eta"].isna().all()
    assert (trials.loc[trials["condition"] == "write_rand", "eta"] == 1.0).all()
    assert trials["agree_world"].isna().tolist() == (trials["pool"] == interp.NOVEL_SET).tolist()
    decl = interp.load_declaration("representation")
    for (dim, test), g in trials.groupby(["dim", "test"]):
        assert set(g["condition"]) == set(decl["conditions"][test]), (dim, test)
        assert g.groupby("condition").size().nunique() == 1


def test_probe_above_chance_at_every_layer(committed):
    _, _, probe = committed
    assert set(probe["dim"]) == set(DIMS)
    for dim, g in probe.groupby("dim"):
        assert list(g["layer"]) == list(range(65))
        assert (g["acc_axis_heldout"] > 0.55).all(), dim
        assert (g["acc_ridge_heldout"] > 0.55).all(), dim
        assert (g["n_occ"] == g["n_occurrences"]).all() and g["n_items"].nunique() == 1
        assert (g["beta"] > 0).all() and (g["r_a_z"] > 0.9).all()


def test_headline_bars_recompute_from_the_table(committed):
    manifest, trials, _ = committed
    recorded = {t["table"]: t for t in manifest["tables"]}["trials"]["panel_numbers"]
    for dim in DIMS:
        for (test, budget, cond), (k, n) in PANEL_E[dim].items():
            g = trials[(trials.dim == dim) & (trials.test == test) & (trials.budget == budget) & (trials.condition == cond)]
            assert (int(g["is_correct"].sum()), len(g)) == (k, n), (dim, test, cond)
        for (test, cond), k in PANEL_C[dim].items():
            budget = 1024 if test.startswith("cot") else 0
            g = trials[(trials.dim == dim) & (trials.test == test) & (trials.budget == budget) & (trials.condition == cond)]
            assert (int(g["agree_written"].sum()), len(g)) == (k, 126), (dim, test, cond)
        for r in recorded[dim]:
            if r["panel"] == "E":
                k, n = PANEL_E[dim][(r["test"], r["budget"], r["condition"])]
                assert r["n"] == n and r["value"] == k / n


def test_erasing_the_rank_direction_hurts_the_iwl_test_and_not_the_chains(committed):
    _, trials, _ = committed
    for dim in DIMS:
        def acc(test, budget, cond):
            g = trials[(trials.dim == dim) & (trials.test == test) & (trials.budget == budget) & (trials.condition == cond)]
            return g["is_correct"].mean()
        assert acc("iwl", 0, "erase") < acc("iwl", 0, "intact") - 0.1, dim
        assert abs(acc("iwl", 0, "rand_erase") - acc("iwl", 0, "intact")) < 0.05, dim
        assert acc("cot_reversed", 1024, "erase") > 0.9 and acc("cot_icl", 1024, "erase") > 0.9, dim


def test_declarations_agree_with_the_code():
    """The intervention declarations name the tests, conditions and item sets the runners use."""
    rep, comp = interp.load_declaration("representation"), interp.load_declaration("components")
    for decl in (rep, comp):
        assert interp.tests_of(decl) == represent.TESTS
        assert {d: v["item_set"] for d, v in decl["dimensions"].items()} == \
            {d: sets[0] for d, sets in represent.ITEM_SETS.items()}
    assert {t: tuple(c) for t, c in rep["conditions"].items()} == represent.TEST_CONDITIONS
    from geo_alg_ti_llms import components
    assert {t: tuple(c) for t, c in comp["conditions"].items()} == \
        {t: components.conditions_for(t) for t in represent.TESTS}
    from geo_alg_ti_llms.plots.representation import BARS
    assert rep["headline_conditions"] == [c for _, run, c in BARS if run == "rank"]
    assert comp["headline_conditions"] == [c for _, run, c in BARS if run == "heads"]


def test_stored_paths_are_written_relative_to_the_runs_root():
    stored = {"axis": "a/b/rank_intervention/countries/qwen3-32b/rank_axis.npz",
              "layers": [0, 1], "tag": "rank_intervention", "other": "a/b"}
    assert interp.relative_to_runs(stored, "rank_intervention") == {
        "axis": "rank_intervention/countries/qwen3-32b/rank_axis.npz", "layers": [0, 1],
        "tag": "rank_intervention", "other": "a/b"}
