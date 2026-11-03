"""The rank direction: its fit at every hidden state, the probe by layer, and the
interventions along it (erase, invert, write; random-direction control).

A dimension's familiar items are forwarded in the prompt format of the runs
(the query alone with the item in position A or B; a premise line among ten
random items of the fit set) and the residual vector at the LAST subtoken of
every mention is kept at every hidden state (the token embeddings and each
block's output, the last block's before the final normalization: the states
the intervention edits). Per hidden state the rank direction is the closed form of the
paper's fit: Ledoit-Wolf covariance Sigma with floored eigenvalues, b = mean of
(h - mu) z over mentions (z the standardized rank), u = Sigma^-1/2 b normalized,
coordinate a = u^T Sigma^-1/2 (h - mu), edit h' = h + (a' - a) Sigma^1/2 u, and
the OLS a ~ alpha + beta z that a write aims at. The probe by layer is the
held-out pairwise ordering accuracy over items (5-fold) of the axis coordinate
and of a ridge probe.

The intervention edits the residual stream at the last subtoken of every item
mention -- in the prompt, in every generated reasoning token and at the answer
pass -- at every hidden state, with gains intact (a' = a), erase (a' = 0),
invert (a' = -a), a write toward a random ordering of the trial's items with
|Spearman rho| <= 0.2 to the real one at strength eta (a' = (1 - eta) a +
eta (alpha + beta z_new)), and the same edits along a random unit direction
(``rand_erase``, ``write_rand``). It is scored on the stored trials of the
IWL test (no context), the reversed context (direct), the reversed context
after a reasoning chain and the ICL test (given names) after a reasoning chain,
the reversed context under the context-only instruction, the ICL test under its
own instruction.

torch, transformers and scipy are imported inside the functions that need them.
"""

from __future__ import annotations

import csv
import json
import os
import re
import time
import zlib
from pathlib import Path

import numpy as np

from geo_alg_ti_llms.items import DEFAULT_ITEMS_DIR, load_item_set
from geo_alg_ti_llms.models import ModelConfig, load_model_and_tokenizer
from geo_alg_ti_llms.prompts import ANSWER_CUES, INSTRUCTIONS_V1


# ---------------------------------------------------------------------------
# The fit sets and the prompts of the representation extraction
# ---------------------------------------------------------------------------

FAMILIARITY_FLOOR = 1_200_000        # 2024 page views (the item sets used about 1M)
DIMS = {
    "countries": dict(table="countries_full_familiarity.csv", name_col="country",
                      value_col="population", views_col="pageviews_2024", instr_id="iwl_population_T2",
                      item_sets=["countries_by_population_11", "countries_by_population_11_rep2", "countries_by_population_11_rep3"]),
    "usstates": dict(table="us_states_population_2024.csv", name_col="state",
                     value_col="population_2024", views_col=None, instr_id="iwl_population_T2",
                     item_sets=["us_states_by_population_11", "us_states_by_population_11_rep2", "us_states_by_population_11_rep3"]),
    "cities": dict(table="cities_band_familiarity.csv", name_col="display",
                   value_col="population_2025", views_col="pageviews_2024", instr_id="iwl_population_T2",
                   item_sets=["cities_by_population_11", "cities_by_population_11_rep2"]),
}
"""Per dimension: the source table of its familiar items (``data/items/sources``), the
instruction naming it, and the experimental item sets (in the fit set except four cities:
Jakarta, Beirut, Athens, Kuala Lumpur)."""

# The stored Qwen3 direct prompt of the runs (system instruction, user query, empty think block).
TEMPLATE_QUERY = ('<|im_start|>system\n{instr}<|im_end|>\n<|im_start|>user\n'
                  'Is it true or false that {A} > {B}?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')
TEMPLATE_CONTEXT = ('<|im_start|>system\n{instr}<|im_end|>\n<|im_start|>user\n{premises}\n\n'
                    'Is it true or false that {A} > {B}?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')


def load_fit_set(dim: str, items_dir: str | Path = DEFAULT_ITEMS_DIR) -> list[tuple[str, float]]:
    """The familiar items of a dimension with their values, ascending (rank 0 = smallest)."""
    d = DIMS[dim]
    rows = []
    with open(Path(items_dir) / "sources" / d["table"], newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                v = float(r[d["value_col"]])
                views = float(r[d["views_col"]] or 0) if d["views_col"] else float("inf")   # all 50 states are familiar
            except ValueError:
                continue
            if views >= FAMILIARITY_FLOOR and r[d["name_col"]].strip():
                rows.append((r[d["name_col"]].strip(), v))
    # de-duplicate by name, keep the first row
    seen, out = set(), []
    for n, v in rows:
        if n not in seen:
            seen.add(n); out.append((n, v))
    return sorted(out, key=lambda t: t[1])           # ascending value -> rank 0 = smallest


def set_items(name: str, items_dir: str | Path = DEFAULT_ITEMS_DIR) -> list[str]:
    return list(load_item_set(name, items_dir).items)


def build_prompts(dim: str, items: list[str], rng: np.random.Generator, k_partner: int = 3):
    """One record per (item, occurrence): prompt text, char span of the target mention, item,
    kind."""
    instr = INSTRUCTIONS_V1[DIMS[dim]["instr_id"]]
    recs = []
    n = len(items)
    for i, it in enumerate(items):
        others = [j for j in range(n) if j != i]
        # query position A (causally independent of B): one prompt
        b = items[rng.choice(others)]
        p = TEMPLATE_QUERY.format(instr=instr, A=it, B=b)
        recs.append(dict(prompt=p, item=it, kind="qA", span=_span(p, it, after="that ")))
        # query position B: k partners
        for _ in range(k_partner):
            a = items[rng.choice(others)]
            p = TEMPLATE_QUERY.format(instr=instr, A=a, B=it)
            recs.append(dict(prompt=p, item=it, kind="qB", span=_span(p, it, after=" > ")))
        # premise line: item first or second in one of 10 lines over random items of the fit set,
        # natural order
        for _ in range(k_partner):
            lines_items = list(rng.choice(others, size=10, replace=False))
            pos = int(rng.integers(0, 10)); lines_items[pos] = i
            ordered = sorted(lines_items)                       # ascending value
            lines = []
            for j in range(len(ordered) - 1):
                lo, hi = items[ordered[j]], items[ordered[j + 1]]
                lines.append(f"{hi} > {lo}")
            rng.shuffle(lines)
            q1, q2 = items[rng.choice(others)], items[rng.choice(others)]
            p = TEMPLATE_CONTEXT.format(instr=instr, premises="\n".join(lines), A=q1, B=q2)
            recs.append(dict(prompt=p, item=it, kind="prem", span=_span_premise(p, it)))
    return recs


def _span(prompt: str, item: str, after: str) -> tuple[int, int]:
    q = prompt.index("Is it true or false that ")
    s = prompt.index(after + item, q) + len(after)
    return s, s + len(item)


def _span_premise(prompt: str, item: str) -> tuple[int, int]:
    """The first whole-word mention of ``item`` in the premise block (a name inside a longer one,
    such as Sudan in South Sudan, also matches; in the fit sets the true mention comes first)."""
    u = prompt.index("<|im_start|>user\n") + len("<|im_start|>user\n")
    q = prompt.index("\n\nIs it true or false")
    m = re.search(r"(?<![A-Za-z])" + re.escape(item) + r"(?![A-Za-z])", prompt[u:q])
    return u + m.start(), u + m.end()


def subtoken_indices(tokenizer, prompt: str, span: tuple[int, int]) -> list[int]:
    """Token positions covering the mention's character span (a leading-space token counts)."""
    enc = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    s, e = span
    cover = [i for i, (a, b) in enumerate(enc["offset_mapping"]) if a < e and b > s]
    assert cover, (prompt[s:e], span)
    return cover


def last_subtoken_index(tokenizer, prompt: str, span: tuple[int, int]) -> int:
    return subtoken_indices(tokenizer, prompt, span)[-1]


def hidden_state_modules(model) -> list:
    """The module whose output is hidden state ``l``, for ``l`` = 0..n_layers: the token embedding,
    then each block. The fit reads these outputs and the editor changes them; the final
    normalization after the last block is neither read nor edited."""
    return [model.model.embed_tokens, *model.model.layers]


def record_hidden_states(model, input_ids, positions: list[int], layers: list[int]) -> dict[int, np.ndarray]:
    """One forward pass of ``input_ids`` (one row): the float32 vectors at ``positions`` of every
    hidden state in ``layers``, read from the outputs of :func:`hidden_state_modules`."""
    import torch
    modules = hidden_state_modules(model)
    states: dict[int, np.ndarray] = {}

    def keep(l):
        def hook(m, a, out):
            h = out[0] if isinstance(out, tuple) else out
            states[l] = h[0, positions].float().cpu().numpy()
        return hook

    handles = [modules[l].register_forward_hook(keep(l)) for l in layers]
    try:
        with torch.no_grad():
            model(input_ids=input_ids)
    finally:
        for h in handles:
            h.remove()
    return states


def extract_representations(dim: str, model_cfg: ModelConfig, out_dir: Path, *,
                            layer_stride: int = 1, k_partner: int = 3,
                            items_dir: str | Path = DEFAULT_ITEMS_DIR) -> None:
    """Forward every item of the fit set in the prompts of :func:`build_prompts` and keep the
    residual vector at the last subtoken of the mention at every hidden state
    (:func:`record_hidden_states`; ``reps.npz``: ``L<l>`` arrays of float16, one row per
    occurrence; ``reps_meta.json``)."""
    import torch
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    item_set = load_fit_set(dim, items_dir)
    items = [n for n, _ in item_set]
    exp = {p: set_items(p, items_dir) for p in DIMS[dim]["item_sets"]}
    missing = {p: [x for x in its if x not in items] for p, its in exp.items()}
    print(f"fit set: {len(items)} items (floor {FAMILIARITY_FLOOR:,} views); experimental items missing from it: {missing}")
    recs = build_prompts(dim, items, np.random.default_rng(42), k_partner=k_partner)
    print(f"{len(recs)} occurrences: " + ", ".join(f"{k}={sum(r['kind']==k for r in recs)}" for k in ("qA", "qB", "prem")))
    cfg = model_cfg
    model, tok = load_model_and_tokenizer(cfg)
    model.eval()
    L = model.config.num_hidden_layers + 1
    layers = list(range(0, L, layer_stride))
    X = {l: [] for l in layers}
    occ_idx, sub_idx, n_sub = [], [], []          # per stored vector: occurrence, position in mention, mention length
    t0 = time.time()
    with torch.no_grad():
        for k, r in enumerate(recs):
            idxs = subtoken_indices(tok, r["prompt"], r["span"])
            idxs = idxs[-1:]
            ids = tok(r["prompt"], add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
            states = record_hidden_states(model, ids, idxs, layers)
            for j in range(len(idxs)):
                for l in layers:
                    X[l].append(states[l][j])
                occ_idx.append(k); sub_idx.append(j); n_sub.append(len(idxs))
            if k % 100 == 0:
                print(f"  {k}/{len(recs)}  {time.time() - t0:.0f}s", flush=True)
    meta = dict(model=cfg.name, layers=layers, kinds=[r["kind"] for r in recs],
                item_idx=[items.index(r["item"]) for r in recs], items=items, span="last")
    reps = {f"L{l}": np.stack(X[l]).astype(np.float16) for l in layers}
    overflow = [k for k, v in reps.items() if not np.isfinite(v).all()]
    if overflow:
        raise ValueError(f"hidden states beyond the float16 range at {overflow}")
    np.savez_compressed(out_dir / "reps.npz", **reps)
    (out_dir / "reps_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    print("saved reps for", len(recs), "occurrences x", len(layers), "layers;", len(occ_idx), "vectors (span last)")


# ---------------------------------------------------------------------------
# The closed-form fit
# ---------------------------------------------------------------------------


def ledoit_wolf(X: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf (2004) shrinkage toward the scaled identity, closed form (no per-row loop):
    sum_i ||x_i x_i^T - S||_F^2 = sum_i ||x_i||^4 - n ||S||_F^2 for centered rows and
    S = X^T X / n."""
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    Xc = X - X.mean(0)
    S = Xc.T @ Xc / n
    mu = np.trace(S) / d
    delta = ((S - mu * np.eye(d)) ** 2).sum() / d
    sq = (Xc ** 2).sum(1)
    beta = ((sq ** 2).sum() - n * (S ** 2).sum()) / (n * n) / d
    rho = min(max(beta, 0.0) / delta, 1.0) if delta > 0 else 1.0
    return (1 - rho) * S + rho * mu * np.eye(d)


def _lw_params(Xc: np.ndarray):
    """Ledoit-Wolf shrinkage intensity rho and target scale mu for centered rows (closed form)."""
    n, d = Xc.shape
    sq = (Xc ** 2).sum(1)
    tr_S = sq.sum() / n                                  # trace(S), S = Xc^T Xc / n
    fro_S2 = ((Xc @ Xc.T) ** 2).sum() / (n * n)          # ||S||_F^2 via the Gram matrix
    mu = tr_S / d
    delta = (fro_S2 - 2 * mu * tr_S + d * mu * mu) / d   # ||S - mu I||_F^2 / d
    beta = ((sq ** 2).sum() - n * fro_S2) / (n * n) / d
    rho = min(max(beta, 0.0) / delta, 1.0) if delta > 0 else 1.0
    return rho, mu


class ShrunkCov:
    """Sigma = (1-rho) S + rho mu I with S = Xc^T Xc / n, held as (V, lambda) from the SVD of Xc, so
    Sigma^{p} v = V diag(g(lambda) - g(0)) V^T v + g(0) v with g(l) = ((1-rho) l + rho mu)^p.
    Exact, and O(n d) per application instead of an eigendecomposition of the d x d matrix
    (n = 861 << d = 5120).
    Eigenvalues are floored at 1e-8 * max as in the paper's fit."""

    def __init__(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        self.mu_x = X.mean(0)
        Xc = X - self.mu_x
        n = len(Xc)
        self.rho, self.mu = _lw_params(Xc)
        try:
            _, sv, Vt = np.linalg.svd(Xc, full_matrices=False)
        except np.linalg.LinAlgError:
            # LAPACK gesdd can fail to converge on an ill-conditioned n x d block: fall back to the
            # eigendecomposition of the n x n Gram matrix, which gives the same S and V exactly
            # (Xc = U S V^T  =>  Xc Xc^T = U S^2 U^T, V = Xc^T U / S)
            G = Xc @ Xc.T
            w, U = np.linalg.eigh(G)
            order = np.argsort(w)[::-1]; w, U = w[order], U[:, order]
            keep = w > 1e-12 * w.max()
            sv = np.sqrt(w[keep]); Vt = ((Xc.T @ U[:, keep]) / sv).T
        lam = (1 - self.rho) * (sv ** 2) / n + self.rho * self.mu
        floor = 1e-8 * max(lam.max(), self.rho * self.mu)
        self.lam = np.maximum(lam, floor)
        self.lam0 = max(self.rho * self.mu, floor)          # eigenvalue on the complement of the sample span
        self.V = Vt.T                                        # d x n

    def apply(self, v: np.ndarray, power: float) -> np.ndarray:
        c = self.V.T @ v
        return self.V @ ((self.lam ** power - self.lam0 ** power) * c) + (self.lam0 ** power) * v


def axis_from(X: np.ndarray, z: np.ndarray, rng: np.random.Generator | None = None):
    """The paper's fit. Returns mu, v_read (= Sigma^-1/2 u), v_write (= Sigma^1/2 u), plus the
    random-u pair."""
    cov = ShrunkCov(X)
    mu = cov.mu_x
    b = ((X - mu) * z[:, None]).mean(0)
    u = cov.apply(b, -0.5); u /= np.linalg.norm(u)
    out = dict(mu=mu, v_read=cov.apply(u, -0.5), v_write=cov.apply(u, 0.5))
    if rng is not None:
        ur = rng.standard_normal(len(mu)); ur /= np.linalg.norm(ur)
        out.update(v_read_rand=cov.apply(ur, -0.5), v_write_rand=cov.apply(ur, 0.5))
    return out


def axis_from_dense(X: np.ndarray, z: np.ndarray):
    """Reference implementation (explicit d x d eigendecomposition); used only to verify
    axis_from."""
    mu = X.mean(0); Xc = X - mu
    S = ledoit_wolf(X)
    w, V = np.linalg.eigh(S)
    w = np.maximum(w, 1e-8 * w.max())
    S_mh = (V * (w ** -0.5)) @ V.T; S_ph = (V * (w ** 0.5)) @ V.T
    b = (Xc * z[:, None]).mean(0)
    u = S_mh @ b; u /= np.linalg.norm(u)
    return dict(mu=mu, v_read=S_mh @ u, v_write=S_ph @ u)


def pairwise_acc(scores: np.ndarray, z: np.ndarray) -> float:
    i, j = np.triu_indices(len(z), 1)
    m = z[i] != z[j]
    return float(np.mean(np.sign(scores[i] - scores[j])[m] == np.sign(z[i] - z[j])[m]))


def fit_rank_axis(dim: str, out_dir: Path, *, layer_stride_fit: int = 1) -> None:
    """Per hidden state, from ``reps.npz``: the axis (``rank_axis.npz``: ``L<l>_{mu,v_read,v_write,
    v_read_rand,v_write_rand,alpha,beta}``, the random direction seeded ``1000 + l``) and the
    probe by layer (``probe_by_layer.csv``: held-out pairwise ordering accuracy of the axis
    coordinate and of a
    ridge probe, 5-fold over items, seed 42); ``fit_manifest.json`` records the fit."""
    out_dir = Path(out_dir)
    rd = out_dir
    meta = json.loads((rd / "reps_meta.json").read_text(encoding="utf-8"))
    reps = np.load(rd / "reps.npz")
    items, item_idx = meta["items"], np.array(meta["item_idx"])
    M = len(items)
    z_item = (np.arange(M) - np.arange(M).mean()) / np.arange(M).std()      # standardized rank, ascending value
    z = z_item[item_idx]
    rng = np.random.default_rng(42)
    folds = np.array_split(rng.permutation(M), 5)
    axis, rows = {}, []
    class Ridge:                                        # closed-form ridge regression (numpy; no scikit-learn needed)
        def __init__(self, alpha=10.0): self.alpha = alpha
        def fit(self, X, y):
            self.mu, self.ym = X.mean(0), y.mean(); Xc = X - self.mu
            self.w = np.linalg.solve(Xc.T @ Xc + self.alpha * np.eye(X.shape[1]), Xc.T @ (y - self.ym)); return self
        def predict(self, X): return (X - self.mu) @ self.w + self.ym
    for l in meta["layers"][::layer_stride_fit]:
        X = reps[f"L{l}"].astype(np.float32)
        ax = axis_from(X, z, rng=np.random.default_rng(1000 + l))
        a = (X - ax["mu"]) @ ax["v_read"]
        alpha, beta = np.polyfit(z, a, 1)[::-1]
        # held-out pairwise ordering accuracy over ITEMS (mean coordinate over an item's
        # occurrences)
        acc_axis, acc_ridge = [], []
        for f in folds:
            te = np.isin(item_idx, f); tr = ~te
            axf = axis_from(X[tr], z[tr])
            a_te = (X[te] - axf["mu"]) @ axf["v_read"]
            s = np.array([a_te[item_idx[te] == i].mean() for i in f])
            acc_axis.append(pairwise_acc(s, z_item[f]))
            rg = Ridge(alpha=10.0).fit(X[tr], z[tr])
            p = rg.predict(X[te]); s2 = np.array([p[item_idx[te] == i].mean() for i in f])
            acc_ridge.append(pairwise_acc(s2, z_item[f]))
        rows.append(dict(layer=l, n_occ=len(X), acc_axis_heldout=float(np.mean(acc_axis)),
                         acc_ridge_heldout=float(np.mean(acc_ridge)) if acc_ridge else np.nan,
                         alpha=float(alpha), beta=float(beta), r_a_z=float(np.corrcoef(a, z)[0, 1]),
                         a_std=float(a.std())))
        for k, v in ax.items():
            axis[f"L{l}_{k}"] = v.astype(np.float32)
        axis[f"L{l}_alpha"] = np.float32(alpha); axis[f"L{l}_beta"] = np.float32(beta)
        print(f"layer {l:3d}: axis held-out pairwise {rows[-1]['acc_axis_heldout']:.3f}  ridge {rows[-1]['acc_ridge_heldout']:.3f}  "
              f"corr(a,z) {rows[-1]['r_a_z']:.3f}  alpha {alpha:+.3f} beta {beta:+.3f}", flush=True)
    import pandas as pd
    pd.DataFrame(rows).to_csv(out_dir / "probe_by_layer.csv", index=False)
    np.savez_compressed(out_dir / "rank_axis.npz", layers=np.array(meta["layers"]), items=np.array(items),
                        z_item=z_item.astype(np.float32), **axis)
    (out_dir / "fit_manifest.json").write_text(json.dumps(dict(
        dim=dim, model=meta["model"], n_items=M, n_occurrences=int(len(item_idx)),
        familiarity_floor=FAMILIARITY_FLOOR, kinds=sorted(set(meta["kinds"])), fit_date=time.strftime("%Y-%m-%d"),
        method="Ledoit-Wolf covariance, eigenvalues floored at 1e-8*max; u = Sigma^-1/2 b normalized; "
               "a = u^T Sigma^-1/2 (x-mu); edit x + (a'-a) Sigma^1/2 u; alpha/beta by OLS of a on z"), indent=1), encoding="utf-8")
    print("wrote", out_dir / "rank_axis.npz", "and probe_by_layer.csv")


# ---------------------------------------------------------------------------
# The stored trials the interventions are scored on
# ---------------------------------------------------------------------------

CUE = ANSWER_CUES["extract"]
ITEM_SETS = {"countries": ["countries_by_population_11", "countries_by_population_11_rep2", "countries_by_population_11_rep3"],
         "cities": ["cities_by_population_11", "cities_by_population_11_rep2"],
         "usstates": ["us_states_by_population_11", "us_states_by_population_11_rep2", "us_states_by_population_11_rep3"]}
DIM_TAG = {"countries": ["countries", "countries_rep2", "countries_rep3"], "cities": ["cities", "cities_rep2"],
           "usstates": ["usstates", "usstates_rep2", "usstates_rep3"]}
CONDITIONS = ["intact", "erase", "invert", "rand_erase", "write_0.25", "write_0.5", "write_0.75", "write_1.0", "write_rand"]
TEST_CONDITIONS = {"iwl": tuple(CONDITIONS), "content_reversed": tuple(CONDITIONS),
                   "cot_reversed": ("intact", "erase", "rand_erase", "write_1.0", "write_rand"),
                   "cot_icl": ("intact", "erase", "rand_erase")}
"""The conditions each test is scored under (the write ladder on the direct tests only)."""
TESTS = ["iwl", "content_reversed", "cot_reversed", "cot_icl"]
"""The four scored conditions: IWL test direct, reversed context direct, reversed context after a
chain, ICL test (given names) after a chain."""
ICL_SET = "novel_names_11"      # the ICL test's item set
ICL_HOME = "countries"           # the given-name test (no world knowledge) was run once, with the
                                 # country population runs; it is dimension-neutral, so every
                                 # dimension reads the same trials
THINK_EMPTY = "<think>\n\n</think>\n\n"     # Qwen3 direct prompt = reasoning-pass prompt + this


def icl_tag(runs: Path, dim: str, model: str, chain: bool = False) -> str:
    """The run tag that holds the given-name test for `dim`: its own tag when its runs include
    the test, else ICL_HOME (the trials do not depend on the dimension)."""
    R = Path(runs)
    tag = DIM_TAG[dim][0]
    d = (R / "cot_sweep" / f"final_{tag}" / "budget_icl" / model) if chain else (R / "nxhop" / f"final_{tag}" / "icl" / model)
    if (any(d.glob("b1024__*.jsonl")) if chain else any(d.glob("*.jsonl"))):
        return tag
    print(f"  [icl] given-name test not run under final_{tag}; reading final_{ICL_HOME} (dimension-neutral test)", flush=True)
    return ICL_HOME


def set_order(name: str, items_dir: str | Path = DEFAULT_ITEMS_DIR) -> list[str]:
    return list(load_item_set(name, items_dir).items)   # ascending value


def _query(prompt: str) -> tuple[str, str]:
    m = re.search(r"Is it true or false that (.+?) > (.+?)\?", prompt)
    return m.group(1), m.group(2)


def _premises(prompt: str) -> str:
    """The premise block (between the user-turn start and the query): the premise order identity for
    fixed assignments, where every premise order has the same item->rank map and differs only in
    order."""
    u = prompt.index("<|im_start|>user\n") + len("<|im_start|>user\n")
    q = prompt.index("Is it true or false", u)
    return prompt[u:q].strip()


def _reasoning_prompt_from_direct(prompt: str) -> str:
    """Reasoning-pass (thinking-on) prompt derived from a stored direct prompt: drop the empty think
    block."""
    assert prompt.endswith(THINK_EMPTY), "direct prompt does not end with the empty think block (not a Qwen3 template?)"
    return prompt[: -len(THINK_EMPTY)]


def load_trials(runs: Path, dim: str, model: str, test: str, n_orders: int) -> list[dict]:
    """The stored trials of one test: the IWL test (every item set of the dimension), the reversed
    context (context-only instruction; premise orders ``0..n_orders-1``) and the two reasoning tests
    (reasoning-pass prompts of the context-only reversed ladder and of the ICL ladder, innermost 7
    ranks). When the context-only ladder is absent, the chain trials are derived from the
    context-only DIRECT rows (same trials; reasoning-pass prompt = direct prompt minus the empty
    think block)."""
    R = Path(runs)
    sfx = "_ctxonly"
    rows = []
    if test == "iwl":
        srcs = [(R / "nxhop" / f"final_{t}" / "iwl" / model, p, None) for t, p in zip(DIM_TAG[dim], ITEM_SETS[dim])]
    elif test in ("content_reversed",):
        srcs = [(R / "nxhop" / f"final_{DIM_TAG[dim][0]}" / f"content{sfx}" / model, ITEM_SETS[dim][0],
                 {"content_reversed": "reversed"}[test])]
    elif test == "cot_icl":
        ldir = R / "cot_sweep" / f"final_{icl_tag(R, dim, model, chain=True)}" / "budget_icl" / model
        assert any(ldir.glob("b1024__*.jsonl")), f"budget_icl ladder not found for {model}"
        srcs = [(ldir, ICL_SET, None)]
    elif test in ("cot_reversed",):
        ladder = {"cot_reversed": "budget_reversed"}[test] + sfx
        ldir = R / "cot_sweep" / f"final_{DIM_TAG[dim][0]}" / ladder / model
        if not any(ldir.glob("b1024__*.jsonl")):
            # fallback: derive the chain trials from the direct rows of the same (context-only)
            # content test
            print(f"  [{test}] ladder {ladder} not found for {model}: deriving reasoning-pass prompts from content{sfx} direct rows", flush=True)
            derived = load_trials(R, dim, model, "content_reversed", n_orders)
            out = []
            for t in derived:
                if 2 <= t["rank_low"] and t["rank_high"] <= 8:
                    t = dict(t, arm=test, prompt=_reasoning_prompt_from_direct(t["prompt"])); out.append(t)
            return out
        srcs = [(ldir, ITEM_SETS[dim][0], None)]
    else:
        raise ValueError(test)
    order_of: dict = {}
    if test.startswith("cot"):
        # the chain rows carry no premise order index; the direct content rows of the same geometry
        # (same seeds -> same trials) do, keyed by (hop, context order, query)
        ddir = R / "nxhop" / f"final_{icl_tag(R, dim, model) if test == 'cot_icl' else DIM_TAG[dim][0]}" / ("icl" if test == "cot_icl" else f"content{sfx}") / model
        for f in sorted(ddir.glob("*.jsonl")):
            for l in open(f, encoding="utf-8"):
                if l.strip():
                    r = json.loads(l); ex = r["extra"]
                    order_of[(int(ex["hop"]), _premises(r["prompt"])) + _query(r["prompt"])] = int(ex["permutation_idx"])
    for d, item_set, assign in srcs:
        order = set_order(item_set); val = {it: i for i, it in enumerate(order)}
        files = sorted(d.glob("b1024__*.jsonl" if test.startswith("cot") else "*.jsonl"))
        for f in files:
            for l in open(f, encoding="utf-8"):
                if not l.strip():
                    continue
                r = json.loads(l)
                ex = r["extra"]
                if assign and r.get("assignment_strategy") != assign:
                    continue
                if test.startswith("cot") and r.get("mode") != "cot":
                    continue
                A, B = _query(r["cot_stage1_prompt"] if test.startswith("cot") else r["prompt"])
                ctx = list(ex["rank_to_item"])
                if test.startswith("cot"):
                    ex = dict(ex, rank_low=min(ctx.index(A), ctx.index(B)), rank_high=max(ctx.index(A), ctx.index(B)),
                              premise_order=order_of.get((int(ex["hop"]), _premises(r["cot_stage1_prompt"]), A, B), 99))
                # a chain row's premise order is the one looked up above (99 when not found)
                perm = int(ex["premise_order"] if test.startswith("cot") else ex.get("permutation_idx", 0))
                if test != "iwl" and perm >= n_orders:
                    continue
                if test.startswith("cot") and not (2 <= int(ex["rank_low"]) and int(ex["rank_high"]) <= 8):
                    continue
                rows.append(dict(arm=test, pool=item_set, trial_id=r["trial_id"], cell_id=r["cell_id"],
                                 assignment=r.get("assignment_strategy"), d=int(r["symbolic_distance"]),
                                 rank_low=int(ex["rank_low"]), rank_high=int(ex["rank_high"]),
                                 perm=perm, items=list(ex["rank_to_item"]),
                                 A=A, B=B, correct=bool(r["correct"]),
                                 world_truth=(bool(val[A] > val[B]) if test not in ("icl", "cot_icl") else None),
                                 prompt=r["cot_stage1_prompt"] if test.startswith("cot") else r["prompt"],
                                 stored_predicted=r.get("predicted")))
    return rows


def written_ordering(items: list[str], world_order: list[str], seed: int) -> dict[str, float]:
    """Random ordering of the trial's items with |Spearman rho| <= .2 to the world order;
    z standardized."""
    from scipy.stats import spearmanr
    rng = np.random.default_rng(seed)
    w = np.array([world_order.index(it) for it in items], dtype=float)
    for _ in range(10_000):
        p = rng.permutation(len(items)).astype(float)
        if abs(spearmanr(p, w).correlation) <= 0.2:
            break
    z = (p - p.mean()) / p.std()
    return {it: float(z[i]) for i, it in enumerate(items)}


def attach_written_ordering(trials: list[dict], items_dir: str | Path = DEFAULT_ITEMS_DIR) -> None:
    """Give every trial its random written ordering (``z_new``) and the answer it implies for the
    queried pair (``written_truth``)."""
    for t in trials:
        seed = zlib.crc32(f"{t['cell_id']}|{t['trial_id']}".encode())  # the same in every process
        t["z_new"] = written_ordering(t["items"], set_order(t["pool"], items_dir), seed=seed)
        t["written_truth"] = bool(t["z_new"][t["A"]] > t["z_new"][t["B"]])


# ---------------------------------------------------------------------------
# The editor
# ---------------------------------------------------------------------------


class RankEditor:
    """Forward hooks that edit item-final positions at the chosen hidden-state layers (the outputs
    of :func:`hidden_state_modules`, where the fit reads them).

    Occurrence detection is a suffix match of the token ids against each item's tokenizations
    (bare and leading-space), so it works identically for the prompt (prefill), for each generated
    token (T = 1 decode steps) and for the answer-pass sequence. A mention is matched by the item's
    tokens ending at a position, without checking the next token, so tokens that end inside a
    longer word or another item's name would also match; no edited position of the published runs
    (the dimension's item sets and the given names) was such a match."""

    def __init__(self, model, tokenizer, axis: dict, layers: list[int], z_item_world: dict[str, float]):
        self.model, self.tok, self.axis, self.layers = model, tokenizer, axis, layers
        self.z_world = z_item_world
        self.condition, self.eta = "intact", 0.0
        self.trials: list[dict] = []           # one per batch row (row r -> trials[min(r, len-1)])
        self.variants: dict[str, list[tuple[int, ...]]] = {}
        self.ids: list[list[int]] = []
        self.flags: list[tuple[int, int, str]] = []
        self.n_edits = 0
        self.handles = []
        self._install()

    def set_items(self, items: list[str]):
        for it in items:
            if it not in self.variants:
                vs = {tuple(self.tok.encode(it, add_special_tokens=False)),
                      tuple(self.tok.encode(" " + it, add_special_tokens=False))}
                self.variants[it] = sorted(vs, key=len, reverse=True)

    def _match(self, ids: list[int], end: int, items: list[str]) -> str | None:
        for it in items:
            for v in self.variants[it]:
                n = len(v)
                if end + 1 >= n and tuple(ids[end + 1 - n:end + 1]) == v:
                    return it
        return None

    def _match_len(self, ids: list[int], end: int, items: list[str]) -> tuple[str | None, int]:
        for it in items:
            for v in self.variants[it]:
                n = len(v)
                if end + 1 >= n and tuple(ids[end + 1 - n:end + 1]) == v:
                    return it, n
        return None, 0

    def scan(self, ids: list[int], items: list[str]) -> list[tuple[int, str]]:
        """Edited positions of a full sequence: the last subtoken of every mention."""
        out = []
        for p in range(0, len(ids)):
            it, n = self._match_len(ids, p, items)
            if not it:
                continue
            out += [(p, it)]
        return sorted(set(out))

    def _pre(self, module, args, kwargs):
        input_ids = args[0] if args else kwargs.get("input_ids")
        B, T = input_ids.shape
        seqs = input_ids.tolist()
        self.flags = []
        if T > 1:                                       # prefill (or a full answer-pass sequence)
            self.ids = [list(s) for s in seqs]
            for r in range(B):
                items = self.trials[min(r, len(self.trials) - 1)]["items"]
                self.flags += [(r, p, it) for p, it in self.scan(self.ids[r], items)]
        else:                                           # one decode step
            for r in range(B):
                self.ids[r].append(seqs[r][0])
                items = self.trials[min(r, len(self.trials) - 1)]["items"]
                it = self._match(self.ids[r], len(self.ids[r]) - 1, items)
                if it:
                    self.flags.append((r, 0, it))

    def _edit(self, hidden, layer: int):
        if self.condition == "intact" or not self.flags:
            return hidden
        import torch
        ax = self.axis
        rand = self.condition in ("rand_erase", "write_rand")
        mu = ax[f"L{layer}_mu"]; vr = ax[f"L{layer}_v_read_rand" if rand else f"L{layer}_v_read"]
        vw = ax[f"L{layer}_v_write_rand" if rand else f"L{layer}_v_write"]
        alpha, beta = float(ax[f"L{layer}_alpha"]), float(ax[f"L{layer}_beta"])
        dev = hidden.device
        mu_t, vr_t, vw_t = (torch.as_tensor(v, device=dev, dtype=torch.float32) for v in (mu, vr, vw))
        for r, p, it in self.flags:
            x = hidden[r, p].float()
            a = torch.dot(vr_t, x - mu_t)
            if self.condition in ("erase", "rand_erase"):
                a_new = torch.zeros_like(a)
            elif self.condition == "invert":
                a_new = -a
            elif self.condition.startswith("write"):
                tr = self.trials[min(r, len(self.trials) - 1)]
                eta = 1.0 if self.condition == "write_rand" else self.eta
                a_new = (1 - eta) * a + eta * (alpha + beta * tr["z_new"][it])
            else:
                raise ValueError(self.condition)
            hidden[r, p] = (x + (a_new - a) * vw_t).to(hidden.dtype)
            self.n_edits += 1
        return hidden

    def _install(self):
        modules = hidden_state_modules(self.model)
        self.handles.append(modules[0].register_forward_pre_hook(self._pre, with_kwargs=True))
        for l in self.layers:
            def hook(m, a, out, l=l):
                if isinstance(out, tuple):
                    return (self._edit(out[0], l),) + tuple(out[1:])
                return self._edit(out, l)
            self.handles.append(modules[l].register_forward_hook(hook))

    def remove(self):
        for h in self.handles:
            h.remove()


# ---------------------------------------------------------------------------
# The intervention run
# ---------------------------------------------------------------------------


def run_intervention(runs: Path, dim: str, model_cfg: ModelConfig, axis_path: Path, out_dir: Path, *,
                     layers: list[int], tests: list[str] = TESTS, conditions: list[str] = CONDITIONS,
                     n_orders: int = 3, budget: int = 1024, score_budgets: list[int] = (256, 512, 1024),
                     batch: int = 16, items_dir: str | Path = DEFAULT_ITEMS_DIR) -> int:
    """Score every test under every condition with the editor installed at ``layers``; one
    ``<test>__<condition>.jsonl`` per pair (direct tests: one row per trial at budget 0; reasoning
    tests:
    greedy ``budget``-token chains in batches, read out at every ``score_budgets`` prefix).
    ``run_manifest.json`` records the setting; every run appends to
    ``run_manifest_history.jsonl``."""
    a_out = Path(out_dir); a_out.mkdir(parents=True, exist_ok=True)
    axis_path = Path(axis_path)
    cfg = model_cfg
    model_short = cfg.short_name
    axis = dict(np.load(axis_path)) if axis_path.exists() else {}
    items_all = [str(x) for x in axis["items"]] if axis else []
    z_world_item = {it: float(z) for it, z in zip(items_all, axis["z_item"])} if axis else {}

    trials = {test: load_trials(runs, dim, model_short, test, n_orders) for test in tests}
    for test, tr in trials.items():
        attach_written_ordering(tr, items_dir)
        print(f"{test}: {len(tr)} trials", flush=True)

    from geo_alg_ti_llms.scoring import (resolve_family_candidates, score_yes_no_ids, eos_token_ids,
                                         generate_new_token_ids, answer_pass_input, strip_trace_ids)
    model, tok = load_model_and_tokenizer(cfg); model.eval()
    cands = resolve_family_candidates("true_false", tok)
    eos = eos_token_ids(model, tok)
    ed = RankEditor(model, tok, axis, list(layers), z_world_item)
    manifest = dict(dim=dim, model=cfg.name, axis=str(axis_path), layers=list(layers), arms=list(tests), conditions=list(conditions), ctxonly=True,
                    n_perms=n_orders, budget=budget, batch=batch, started=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    job=os.environ.get("SLURM_JOB_ID"))
    mf = a_out / "run_manifest.json"
    first_run = not mf.exists()
    if first_run:
        mf.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    else:
        prev = json.loads(mf.read_text(encoding="utf-8"))
        for k in ("layers", "axis", "ctxonly"):
            assert prev.get(k, False) == manifest[k], f"{mf} describes a different setting ({k}: {prev.get(k)!r} vs {manifest[k]!r}); use another output directory"
    with open(a_out / "run_manifest_history.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(manifest, event="started")) + "\n")

    def score_ids(ids: list[int], trial: dict) -> dict:
        ed.trials = [trial]; ed.n_edits = 0
        lp_yes, lp_no, _ = score_yes_no_ids(model, ids, cands)
        pred = lp_yes > lp_no
        return dict(predicted=bool(pred), lp_yes=lp_yes, lp_no=lp_no, is_correct=bool(pred == trial["correct"]),
                    agree_world=(bool(pred == trial["world_truth"]) if trial["world_truth"] is not None else None),
                    agree_written=bool(pred == trial["written_truth"]),
                    n_edits=ed.n_edits)

    base = {k: v for k, v in manifest.items() if k in ("layers",)}
    for cond in conditions:
        ed.condition = cond
        ed.eta = float(cond.split("_")[1]) if cond.startswith("write_") and cond != "write_rand" else 1.0
        for test, tr in trials.items():
            if cond not in TEST_CONDITIONS[test]:
                continue
            outf = a_out / f"{test}__{cond}.jsonl"
            if outf.exists() and sum(1 for _ in open(outf, encoding="utf-8")) >= len(tr) * (len(score_budgets) if test.startswith("cot") else 1):
                print(f"skip {outf.name} (complete)", flush=True); continue
            t0 = time.time(); n = 0
            with open(outf, "w", encoding="utf-8") as fh:
                if not test.startswith("cot"):
                    for t in tr:
                        ed.set_items(t["items"])
                        ids = tok.encode(t["prompt"], add_special_tokens=False)
                        rec = {k: v for k, v in t.items() if k not in ("prompt", "z_new")}
                        rec.update(condition=cond, eta=ed.eta if cond.startswith("write") else None, budget=0, edit_span="last", **base, **score_ids(ids, t))
                        fh.write(json.dumps(rec) + "\n"); n += 1
                else:
                    for s in range(0, len(tr), batch):
                        chunk = tr[s:s + batch]
                        for t in chunk:
                            ed.set_items(t["items"])
                        ed.trials = chunk; ed.n_edits = 0
                        gen = generate_new_token_ids(model, tok, [t["prompt"] for t in chunk], max_new_tokens=budget,
                                                     thinking_mode=bool(cfg.thinking_mode), batch_size=batch, eos_ids=eos,
                                                     thinking_stop_string=cfg.thinking_stop_string)
                        n_gen_edits = ed.n_edits
                        for t, (trace_ids, n_in) in zip(chunk, gen):
                            s1_ids = tok.encode(t["prompt"], add_special_tokens=False)
                            for b in score_budgets:
                                tr_b = trace_ids[:b]
                                text = t["prompt"] + tok.decode(strip_trace_ids(tok, tr_b, eos_ids=eos)) + "\n" + CUE
                                ids2, src, faithful = answer_pass_input(tok, t["prompt"], s1_ids, tr_b, text, eos_ids=eos, cue=CUE)
                                rec = {k: v for k, v in t.items() if k not in ("prompt", "z_new")}
                                rec.update(condition=cond, eta=ed.eta if cond.startswith("write") else None, budget=b,
                                           trace_n_tokens=len(trace_ids), cut=len(trace_ids) > b, stage2_source=src,
                                           n_gen_edits_batch=n_gen_edits,
                                           edit_span="last", **base, **score_ids(ids2, t))
                                if b == score_budgets[-1]:
                                    rec["trace_text"] = tok.decode(trace_ids, skip_special_tokens=True)
                                fh.write(json.dumps(rec) + "\n"); n += 1
                        print(f"  {test}/{cond}: {min(s + batch, len(tr))}/{len(tr)} chains  {time.time() - t0:.0f}s", flush=True)
            print(f"wrote {outf.name}: {n} rows in {time.time() - t0:.0f}s", flush=True)
    ed.remove()
    manifest["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    if first_run:
        mf.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    with open(a_out / "run_manifest_history.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(manifest, event="finished")) + "\n")
    return 0


__all__ = [
    "DIMS", "FAMILIARITY_FLOOR", "ITEM_SETS", "DIM_TAG", "TESTS", "CONDITIONS", "TEST_CONDITIONS",
    "load_fit_set", "build_prompts", "subtoken_indices", "last_subtoken_index",
    "hidden_state_modules", "record_hidden_states", "extract_representations", "ledoit_wolf", "ShrunkCov", "axis_from", "axis_from_dense",
    "pairwise_acc", "fit_rank_axis", "load_trials", "written_ordering", "attach_written_ordering",
    "RankEditor", "run_intervention",
]
