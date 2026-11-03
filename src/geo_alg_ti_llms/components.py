"""Context-reading heads: their score during reasoning, and their mean ablation.

An inference composes two relations that share an item; when a reasoning chain
contains one ("X > Y", "X is larger than Y", ...), some attention heads must
read the context relations that mention X or Y. Every head of the model is
scored on the stored reasoning chains of the reversed context (under the
context-only instruction) and of the ICL test (given names, under its own
instruction): one forward pass over prompt + chain with eager attention, and per
head the attention mass from the tokens of every composed inference in the chain
(the two items two or more context ranks apart) onto the context relations
mentioning either item, averaged over tokens and chains. Restatements of a shown
relation (adjacent items) and every other chain token are scored alongside as
controls. The same pass records each head's mean output vector (over all
positions and chains).

The ablation replaces the ablated heads' outputs (their slice of the
``o_proj`` input) by those mean vectors at every position -- the prompt, every
generated token and the answer pass, the loci of the rank-direction edits -- and
is scored on the same four conditions (IWL test direct, reversed context direct,
reversed context after a chain, ICL test after a chain). Head sets: the
``k`` highest-scoring heads (``top<k>``) and ``k`` random heads drawn outside
the top 64 (``rand<k>_s<seed>``, seeded ``100 k + seed``).

torch and transformers are imported inside the functions that need them.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import numpy as np

from geo_alg_ti_llms.items import DEFAULT_ITEMS_DIR
from geo_alg_ti_llms.models import ModelConfig, load_model_and_tokenizer, load_tokenizer
from geo_alg_ti_llms.represent import (
    TESTS,
    CUE,
    DIM_TAG,
    ICL_SET,
    ITEM_SETS,
    _premises,
    _query,
    attach_written_ordering,
    icl_tag,
    load_trials,
    set_order,
)


K_LIST = (8, 16, 32, 64)
"""Sizes of the head sets selected (the paper reports top-32 against random-32)."""
N_OUTSIDE = 64
"""Random sets are drawn from the heads outside the top this many."""
SCORE_TESTS = ("cot_reversed", "cot_icl")
"""The chains the heads are scored on."""
DIRECT_CONDITIONS = ("intact", "top8", "top16", "top32", "rand8_s0", "rand16_s0", "rand32_s0")
CHAIN_CONDITIONS = ("intact", "top16", "top32", "rand16_s0", "rand32_s0")

ALIASES = {"United States": ["United States of America", "the United States", "United States", "U.S.A.", "USA", "US", "America"]}
_CMP = r"(?:is|are|has|have|was|were)?\s*(?:\w+ly\s+)?(?:a\s+|the\s+)?"
GT_WORDS = (_CMP + r"(?:larger|bigger|greater|higher|more\s+populous|more\s+populated|more)(?:\s+(?:population|populated|people|populous))?\s+than"
            r"|(?:is|are|ranks?|comes?|sits?|lies?|was|were)?\s*(?:\w+ly\s+)?(?:above|ahead\s+of|over)")
LT_WORDS = (_CMP + r"(?:smaller|lower|lesser|less|fewer|less\s+populous|less\s+populated)(?:\s+(?:population|populated|people|populous))?\s+than"
            r"|(?:is|are|ranks?|comes?|sits?|lies?|was|were)?\s*(?:\w+ly\s+)?(?:below|behind|beneath|under)")


# ---------------------------------------------------------------------------
# The stored chains and their relational statements
# ---------------------------------------------------------------------------


def test_rows(runs: Path, dim: str, model: str, test: str, n_orders: int) -> list[dict]:
    """Stored 1,024-token reasoning rows with their chains, premise orders ``0..n_orders-1``,
    innermost
    ranks 2..8 (the trials of :func:`geo_alg_ti_llms.represent.load_trials`)."""
    R = Path(runs)
    tag = DIM_TAG[dim][0]
    if test == "cot_icl":
        it = icl_tag(R, dim, model, chain=True)        # given-name test is dimension-neutral; falls back to the country runs
        ldir, ddir, item_set = R / "cot_sweep" / f"final_{it}" / "budget_icl" / model, R / "nxhop" / f"final_{it}" / "icl" / model, ICL_SET
    elif test == "cot_iwl":
        ldir, ddir, item_set = R / "cot_sweep" / f"final_{tag}" / "budget_iwl" / model, None, ITEM_SETS[dim][0]
    else:
        ladder = {"cot_reversed": "budget_reversed_ctxonly", "cot_natural": "budget_natural_ctxonly", "cot_shuffled": "budget_shuffled_ctxonly"}[test]
        ldir, ddir, item_set = R / "cot_sweep" / f"final_{tag}" / ladder / model, R / "nxhop" / f"final_{tag}" / "content_ctxonly" / model, ITEM_SETS[dim][0]
    assign = {"cot_reversed": "reversed", "cot_natural": "natural", "cot_shuffled": "random"}.get(test)
    order_of = {}
    if ddir is not None:
        for f in sorted(ddir.glob("*.jsonl")):
            for l in open(f, encoding="utf-8"):
                if l.strip():
                    r = json.loads(l); ex = r["extra"]
                    order_of[(int(ex["hop"]), _premises(r["prompt"])) + _query(r["prompt"])] = int(ex["permutation_idx"])
    order = set_order(item_set); val = {it: i for i, it in enumerate(order)}
    rows = []
    for f in sorted(ldir.glob("b1024__*.jsonl")):
        for l in open(f, encoding="utf-8"):
            if not l.strip():
                continue
            r = json.loads(l)
            if r.get("mode") != "cot" or (assign and r.get("assignment_strategy") != assign):
                continue
            ex = r["extra"]; s1 = r["cot_stage1_prompt"]; A, B = _query(s1); ctx = list(ex["rank_to_item"])
            rl, rh = min(ctx.index(A), ctx.index(B)), max(ctx.index(A), ctx.index(B))
            if test == "cot_iwl":
                premise_order = 0
            else:
                premise_order = order_of.get((int(ex["hop"]), _premises(s1), A, B), 99)
            if premise_order >= n_orders or not (2 <= rl and rh <= 8):
                continue
            rows.append(dict(arm=test, pool=item_set, trial_id=r["trial_id"], cell_id=r["cell_id"], assignment=r.get("assignment_strategy"),
                             d=int(r["symbolic_distance"]), rank_low=rl, rank_high=rh, perm=premise_order, items=ctx, A=A, B=B,
                             correct=bool(r["correct"]), world_truth=(bool(val[A] > val[B]) if test != "cot_icl" else None),
                             stage1=s1, trace=r["cot_trace"], stored_predicted=r.get("predicted"), stored_is_correct=bool(r["is_correct"]),
                             stored_n_tokens=r.get("cot_n_tokens")))
    return rows


def _item_regex(items: list[str]) -> tuple[str, dict[str, str]]:
    alts, canon = [], {}
    for it in items:
        for a in ALIASES.get(it, [it]):
            alts.append(re.escape(a)); canon[a.lower()] = it
    alts.sort(key=len, reverse=True)
    return "(" + "|".join(alts) + ")", canon


def statements(trace: str, items: list[str]) -> list[dict]:
    """Every relational statement in the trace: char span and the two items (canonical names).
    Matches do not overlap, so a chained statement ("A > B > C") yields only its first relation."""
    pat_items, canon = _item_regex(items)
    rx = re.compile(rf"(?<![A-Za-z]){pat_items}\s*(?P<op>>|<|{GT_WORDS}|{LT_WORDS})\s*{pat_items}(?![A-Za-z])", re.IGNORECASE)
    out = []
    for m in rx.finditer(trace):
        x, y = canon.get(m.group(1).lower()), canon.get(m.group(3).lower())
        if x and y and x != y:
            out.append(dict(start=m.start(), end=m.end(), x=x, y=y))
    return out


def premise_lines(reasoning_prompt: str, items: list[str]) -> tuple[tuple[int, int], list[dict]]:
    """Char span of the premise block in the reasoning-pass prompt and, per line, its span and the
    items it mentions."""
    u = reasoning_prompt.index("<|im_start|>user\n") + len("<|im_start|>user\n")
    q = reasoning_prompt.index("Is it true or false", u)
    block = reasoning_prompt[u:q]
    lines, pos = [], u
    for line in block.split("\n"):
        s, e = pos, pos + len(line)
        if line.strip():
            lines.append(dict(start=s, end=e, items=[it for it in items if re.search(r"(?<![A-Za-z])" + re.escape(it) + r"(?![A-Za-z])", line)]))
        pos = e + 1
    return (u, q), lines


def token_positions(offsets, s: int, e: int) -> list[int]:
    return [i for i, (a, b) in enumerate(offsets) if a < e and b > s]


def chain_record(tok, t: dict, max_seq: int) -> dict:
    """One chain's token ids and the positions scored: the premise block, the composed-inference
    tokens with their relevant premise tokens (items >= 2 context ranks apart), the restatement
    tokens (adjacent items) and the control tokens (every other chain token). A prompt and chain
    longer than ``max_seq`` tokens is an error, never cut."""
    text = t["stage1"] + t["trace"]
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, off = enc["input_ids"], enc["offset_mapping"]
    if len(ids) > max_seq:
        raise ValueError(f"{t['arm']} trial {t['trial_id']}: prompt and chain are {len(ids)} tokens, "
                         f"more than max_seq = {max_seq}")
    (pb_s, pb_e), lines = premise_lines(t["stage1"], t["items"])
    prem_all = token_positions(off, pb_s, pb_e)
    line_pos = [dict(items=L["items"], pos=token_positions(off, L["start"], L["end"])) for L in lines]
    base = len(t["stage1"]); n_s1 = len(token_positions(off, 0, base))
    # (query token position, relevant premise token positions);
    # derived = items >= 2 context ranks apart
    deriv, copy = [], []
    covered = set(); rank = {it: k for k, it in enumerate(t["items"])}
    for st in statements(t["trace"], t["items"]):
        qpos = token_positions(off, base + st["start"], base + st["end"])
        rel = sorted({p for L in line_pos if (st["x"] in L["items"] or st["y"] in L["items"]) for p in L["pos"]})
        bucket = deriv if abs(rank[st["x"]] - rank[st["y"]]) >= 2 else copy
        for p in qpos:
            if p >= n_s1 and p not in covered:
                bucket.append((p, rel)); covered.add(p)
    ctrl = [p for p in range(n_s1, len(ids)) if p not in covered]
    return dict(ids=ids, prem_all=prem_all, deriv=deriv, copy=copy, ctrl=ctrl, arm=t["arm"], trial_id=t["trial_id"], n_s1=n_s1)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def accumulate_masses(w, r: dict, l: int, rel_sum, all_sum, copy_sum, ctrl_sum) -> None:
    """Add one layer's attention weights ``w`` (``(H, T, T)``, float) for chain record ``r`` into
    the
    per-head sums: relevant-premise mass at the composed-inference tokens (``rel_sum``), whole-block
    mass there (``all_sum``), relevant-premise mass at the restatement tokens (``copy_sum``) and
    whole-block mass at the control tokens (``ctrl_sum``)."""
    import torch
    if r["deriv"]:
        q = torch.tensor([p for p, _ in r["deriv"]], device=w.device)
        wa = w[:, q, :]                                               # (H, nq, T)
        all_sum[l] += wa[:, :, r["prem_all"]].sum(-1).sum(-1).cpu().numpy()
        rel = torch.zeros(len(q), w.shape[-1], device=w.device)
        for i, (_, rp) in enumerate(r["deriv"]):
            if rp: rel[i, rp] = 1.0
        rel_sum[l] += (wa * rel[None]).sum(-1).sum(-1).cpu().numpy()
    if r["copy"]:
        q = torch.tensor([p for p, _ in r["copy"]], device=w.device)
        wc = w[:, q, :]
        relc = torch.zeros(len(q), w.shape[-1], device=w.device)
        for i, (_, rp) in enumerate(r["copy"]):
            if rp: relc[i, rp] = 1.0
        copy_sum[l] += (wc * relc[None]).sum(-1).sum(-1).cpu().numpy()
    if r["ctrl"]:
        c = torch.tensor(r["ctrl"], device=w.device)
        ctrl_sum[l] += w[:, c, :][:, :, r["prem_all"]].sum(-1).sum(-1).cpu().numpy()


def select_head_sets(ranked: list[tuple[int, int]], k_list=K_LIST, seeds=(0,)) -> dict[str, list[tuple[int, int]]]:
    """``top<k>``: the first ``k`` of the ordering; ``rand<k>_s<seed>``: ``k`` heads drawn without
    replacement from outside the top ``N_OUTSIDE`` with ``default_rng(100 k + seed)``, in ordering
    order."""
    top64 = set(ranked[:N_OUTSIDE]); rest = [x for x in ranked if x not in top64]
    sel = {}
    for k in k_list:
        sel[f"top{k}"] = ranked[:k]
        for s in seeds:
            rng = np.random.default_rng(100 * k + s); idx = rng.choice(len(rest), size=k, replace=False)
            sel[f"rand{k}_s{s}"] = [rest[i] for i in sorted(idx)]
    return sel


def score_heads(runs: Path, dim: str, model_cfg: ModelConfig, out_dir: Path, *, tests=SCORE_TESTS,
                n_orders: int = 3, max_seq: int = 1200) -> None:
    """One eager-attention pass per stored chain of ``tests``; writes ``head_scores.csv`` (per head:
    ``rel_mass_deriv`` -- the ordering statistic -- ``rel_mass_copy``, ``all_mass_deriv``,
    ``all_mass_ctrl``, ``specificity``), ``head_means.npz`` (``means[L, H, D]``) and
    ``selected_heads.json`` (:func:`select_head_sets`)."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cfg = model_cfg; model_short = cfg.short_name
    tok, _ = load_tokenizer(cfg.name, cfg.revision)
    trials = []
    for test in tests:
        rows = test_rows(runs, dim, model_short, test, n_orders)
        trials += rows; print(f"{test}: {len(rows)} chains", flush=True)
    recs = [chain_record(tok, t, max_seq) for t in trials]
    n_deriv = sum(len(r["deriv"]) for r in recs); n_copy = sum(len(r["copy"]) for r in recs); n_ctrl = sum(len(r["ctrl"]) for r in recs)
    print(f"{len(recs)} chains; {n_deriv} derivation tokens (composed, span >= 2), {n_copy} premise-restatement tokens (span 1), {n_ctrl} control tokens; "
          f"premise block {np.mean([len(r['prem_all']) for r in recs]):.0f} tokens; seq len mean {np.mean([len(r['ids']) for r in recs]):.0f}", flush=True)
    import torch
    model, tok = load_model_and_tokenizer(cfg); model.eval()
    try:
        model.set_attn_implementation("eager")
    except Exception:
        model.config._attn_implementation = "eager"
    L, H, D = model.config.num_hidden_layers, model.config.num_attention_heads, getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads)
    rel_sum = np.zeros((L, H)); all_sum = np.zeros((L, H)); ctrl_sum = np.zeros((L, H)); copy_sum = np.zeros((L, H)); n_d = 0; n_c = 0; n_cp = 0
    means = torch.zeros(L, H, D, dtype=torch.float64, device="cpu"); n_pos = 0
    cur = {}
    def attn_hook(l):
        def hook(mod, args, out):
            w = out[1]                      # (1, H, T, T) eager attention weights
            if w is None:
                raise RuntimeError("attention weights not returned: eager attention is required for scoring")
            w = w[0].float()
            accumulate_masses(w, cur["rec"], l, rel_sum, all_sum, copy_sum, ctrl_sum)
        return hook
    def oproj_hook(l):
        def hook(mod, args):
            x = args[0][0].float()                                            # (T, H*D)
            means[l] += x.view(-1, H, D).sum(0).double().cpu()
        return hook
    handles = []
    for l, blk in enumerate(model.model.layers):
        handles.append(blk.self_attn.register_forward_hook(attn_hook(l)))
        handles.append(blk.self_attn.o_proj.register_forward_pre_hook(oproj_hook(l)))
    t0 = time.time()
    with torch.no_grad():
        for i, r in enumerate(recs):
            cur["rec"] = r
            ids = torch.tensor([r["ids"]], device=model.device)
            model(input_ids=ids, use_cache=False)
            n_d += len(r["deriv"]); n_c += len(r["ctrl"]); n_cp += len(r["copy"]); n_pos += len(r["ids"])
            if i % 10 == 0:
                print(f"  {i}/{len(recs)}  {time.time() - t0:.0f}s", flush=True)
    for h in handles: h.remove()
    import pandas as pd
    rows = []
    for l in range(L):
        for h in range(H):
            rows.append(dict(layer=l, head=h, rel_mass_deriv=rel_sum[l, h] / max(1, n_d), rel_mass_copy=copy_sum[l, h] / max(1, n_cp),
                             all_mass_deriv=all_sum[l, h] / max(1, n_d),
                             all_mass_ctrl=ctrl_sum[l, h] / max(1, n_c), specificity=all_sum[l, h] / max(1, n_d) - ctrl_sum[l, h] / max(1, n_c)))
    df = pd.DataFrame(rows).sort_values("rel_mass_deriv", ascending=False); df.to_csv(out_dir / "head_scores.csv", index=False)
    np.savez_compressed(out_dir / "head_means.npz", means=(means / max(1, n_pos)).float().numpy())
    ranked = [(int(r.layer), int(r.head)) for r in df.itertuples()]
    sel = select_head_sets(ranked)
    (out_dir / "selected_heads.json").write_text(json.dumps(dict(model=cfg.name, L=L, H=H, D=D, n_chains=len(recs), n_deriv_tokens=n_d, n_ctrl_tokens=n_c,
                                                                   arms=list(tests), n_copy_tokens=n_cp, ranking="rel_mass_deriv (relevant-premise mass at composed-inference tokens)", selected=sel), indent=1), encoding="utf-8")
    print(df.head(12).to_string(index=False)); print("wrote", out_dir / "head_scores.csv", out_dir / "selected_heads.json")


# ---------------------------------------------------------------------------
# Mean ablation
# ---------------------------------------------------------------------------


class HeadAblator:
    """Forward pre-hooks on every layer's o_proj: the ablated heads' slices are replaced by their
    mean vectors."""
    def __init__(self, model, means: np.ndarray, H: int, D: int):
        self.model, self.H, self.D = model, H, D
        import torch
        self.means = torch.as_tensor(means)                      # (L, H, D)
        self.sel: dict[int, list[int]] = {}                       # layer -> heads
        self.n_edits = 0
        self.handles = [blk.self_attn.o_proj.register_forward_pre_hook(self._hook(l)) for l, blk in enumerate(model.model.layers)]
    def set_heads(self, heads: list[tuple[int, int]]):
        self.sel = {}
        for l, h in heads:
            self.sel.setdefault(int(l), []).append(int(h))
    def _hook(self, l):
        def hook(mod, args):
            hs = self.sel.get(l)
            if not hs:
                return None
            x = args[0]
            B, T, _ = x.shape
            xv = x.view(B, T, self.H, self.D).clone()
            m = self.means[l, hs].to(device=x.device, dtype=x.dtype)        # (k, D)
            xv[:, :, hs, :] = m[None, None]
            self.n_edits += B * T * len(hs)
            return (xv.view(B, T, self.H * self.D),) + tuple(args[1:])
        return hook
    def remove(self):
        for h in self.handles: h.remove()


def conditions_for(test: str) -> tuple[str, ...]:
    """The head sets a test was scored under."""
    return CHAIN_CONDITIONS if test.startswith("cot") else DIRECT_CONDITIONS


def run_ablation(runs: Path, dim: str, model_cfg: ModelConfig, heads_path: Path, out_dir: Path, *,
                 tests: list[str] = TESTS, conditions: list[str] | None = None, means_path: Path | None = None,
                 n_orders: int = 3, budget: int = 1024, score_budgets: list[int] = (256, 512, 1024),
                 batch: int = 16, items_dir: str | Path = DEFAULT_ITEMS_DIR) -> int:
    """Score every test under every head set of ``selected_heads.json`` named in ``conditions``
    (default: :func:`conditions_for` the test) with the ablator installed; the files and records are
    those of :func:`geo_alg_ti_llms.represent.run_intervention`, with ``k`` (heads ablated) in place
    of ``eta``."""
    a_out = Path(out_dir); a_out.mkdir(parents=True, exist_ok=True)
    heads_path = Path(heads_path)
    sel = json.loads(heads_path.read_text(encoding="utf-8")); heads = {k: [tuple(x) for x in v] for k, v in sel["selected"].items()}; heads["intact"] = []
    cond_of = {test: list(conditions or conditions_for(test)) for test in tests}
    for c in {c for cs in cond_of.values() for c in cs}:
        assert c in heads, f"condition {c} not in {heads_path}"
    cfg = model_cfg; model_short = cfg.short_name
    trials = {test: load_trials(runs, dim, model_short, test, n_orders) for test in tests}
    for test, tr in trials.items():
        attach_written_ordering(tr, items_dir)
        print(f"{test}: {len(tr)} trials", flush=True)
    print("conditions:", {test: {c: len(heads[c]) for c in cs} for test, cs in cond_of.items()})
    from geo_alg_ti_llms.scoring import resolve_family_candidates, score_yes_no_ids, eos_token_ids, generate_new_token_ids, answer_pass_input, strip_trace_ids
    model, tok = load_model_and_tokenizer(cfg); model.eval()
    cands = resolve_family_candidates("true_false", tok); eos = eos_token_ids(model, tok)
    means = np.load(means_path or (heads_path.parent / "head_means.npz"))["means"]
    abl = HeadAblator(model, means, sel["H"], sel["D"])
    manifest = dict(dim=dim, model=cfg.name, heads=str(heads_path), arms=list(tests), conditions=cond_of, n_perms=n_orders, budget=budget, batch=batch,
                    ctxonly=True, started=time.strftime("%Y-%m-%dT%H:%M:%S"), job=os.environ.get("SLURM_JOB_ID"))
    mf = a_out / "run_manifest.json"
    if not mf.exists(): mf.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    with open(a_out / "run_manifest_history.jsonl", "a", encoding="utf-8") as fh: fh.write(json.dumps(dict(manifest, event="started")) + "\n")
    def score_ids(ids, trial):
        lp_yes, lp_no, _ = score_yes_no_ids(model, ids, cands); pred = lp_yes > lp_no
        return dict(predicted=bool(pred), lp_yes=lp_yes, lp_no=lp_no, is_correct=bool(pred == trial["correct"]),
                    agree_world=(bool(pred == trial["world_truth"]) if trial["world_truth"] is not None else None), agree_written=bool(pred == trial["written_truth"]))
    for test, tr in trials.items():
        for cond in cond_of[test]:
            abl.set_heads(heads[cond])
            outf = a_out / f"{test}__{cond}.jsonl"
            if outf.exists() and sum(1 for _ in open(outf, encoding="utf-8")) >= len(tr) * (len(score_budgets) if test.startswith("cot") else 1):
                print(f"skip {outf.name} (complete)", flush=True); continue
            t0 = time.time(); n = 0
            with open(outf, "w", encoding="utf-8") as fh:
                if not test.startswith("cot"):
                    for t in tr:
                        ids = tok.encode(t["prompt"], add_special_tokens=False)
                        rec = {k: v for k, v in t.items() if k not in ("prompt", "z_new")}
                        rec.update(condition=cond, k=len(heads[cond]), budget=0, **score_ids(ids, t)); fh.write(json.dumps(rec) + "\n"); n += 1
                else:
                    for s in range(0, len(tr), batch):
                        chunk = tr[s:s + batch]
                        gen = generate_new_token_ids(model, tok, [t["prompt"] for t in chunk], max_new_tokens=budget, thinking_mode=bool(cfg.thinking_mode), batch_size=batch, eos_ids=eos, thinking_stop_string=cfg.thinking_stop_string)
                        for t, (trace_ids, n_in) in zip(chunk, gen):
                            s1_ids = tok.encode(t["prompt"], add_special_tokens=False)
                            for b in score_budgets:
                                tr_b = trace_ids[:b]
                                text = t["prompt"] + tok.decode(strip_trace_ids(tok, tr_b, eos_ids=eos)) + "\n" + CUE
                                ids2, src, _ = answer_pass_input(tok, t["prompt"], s1_ids, tr_b, text, eos_ids=eos, cue=CUE)
                                rec = {k: v for k, v in t.items() if k not in ("prompt", "z_new")}
                                rec.update(condition=cond, k=len(heads[cond]), budget=b, trace_n_tokens=len(trace_ids), cut=len(trace_ids) > b, stage2_source=src, **score_ids(ids2, t))
                                if b == score_budgets[-1]: rec["trace_text"] = tok.decode(trace_ids, skip_special_tokens=True)
                                fh.write(json.dumps(rec) + "\n"); n += 1
                        print(f"  {test}/{cond}: {min(s + batch, len(tr))}/{len(tr)} chains  {time.time() - t0:.0f}s", flush=True)
            print(f"wrote {outf.name}: {n} rows in {time.time() - t0:.0f}s", flush=True)
    abl.remove()
    with open(a_out / "run_manifest_history.jsonl", "a", encoding="utf-8") as fh: fh.write(json.dumps(dict(manifest, event="finished", finished=time.strftime("%Y-%m-%dT%H:%M:%S"))) + "\n")
    return 0


__all__ = [
    "K_LIST", "N_OUTSIDE", "SCORE_TESTS", "DIRECT_CONDITIONS", "CHAIN_CONDITIONS", "GT_WORDS", "LT_WORDS",
    "test_rows", "statements", "premise_lines", "token_positions", "chain_record", "accumulate_masses",
    "select_head_sets", "score_heads", "HeadAblator", "conditions_for", "run_ablation",
]
