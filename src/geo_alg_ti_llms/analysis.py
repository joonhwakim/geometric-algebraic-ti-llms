"""The estimators behind every LLM panel and every number the paper reports about the models.

Trials are the only replication unit: a pretrained model decoded greedily has no seeds, so an
accuracy carries a Wilson interval over trials and the reasoning budget a cluster bootstrap
over trials (a resampled trial keeps its outcome at every budget). Pure numpy and pandas.

- Loading: ``direct_trials``, ``ladder_trials``, ``spacing_trials`` and the small tables read
  the committed tidy tables and add two columns: ``dimension`` (the dimension of an item set,
  ``countries`` for ``countries_rep2``) and ``grid_set`` (the rows a cross-model comparison
  draws: the first two IWL item sets of every dimension, every row of every other condition).
- Per panel: ``trial_accuracy_by``, ``accuracy_at_budget`` (accuracy at one reasoning
  budget, pooled over distances), ``running_best`` (each budget's best accuracy at that budget
  or below), ``budget_to_criterion`` (the smallest budget whose accuracy reaches 90% of the
  maximum over budgets; no budget when that maximum's 95% Wilson lower bound is at or below
  chance).
- Per model: ``accuracy_slope``, ``budget_summary``, ``competence``.
- Over the roster: ``distance_slope_points``, ``budget_points``, ``prior_strength_points``,
  ``recovery_points``, one row per point of the four cross-model scatters, and the counts the
  paper quotes about the per-model appendix figures.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from geo_alg_ti_llms import roster

RESULTS_ROOT: Path = Path(__file__).resolve().parents[2] / "results"
DISTANCES: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
CHANCE: float = 0.5
FRACTION: float = 0.9               # the reasoning budget's criterion: 90% of the maximum
ABOVE_CHANCE_Z: float = 1.96        # the best accuracy must beat chance at this Wilson z
BAND_Z: float = 1.0                 # 68% Wilson intervals on an accuracy
N_BOOT: int = 1000
BOOT_SEED: int = 42
MIN_USABLE: int = 3                 # distances with a budget a model's budget summary needs
GRID_SETS: int = 2                  # IWL item sets per dimension a cross-model comparison draws
COMPETENCE_ALPHA: float = 0.05
BUDGET_UNIT: float = 1000.0         # the recovery slope is per 1,000 tokens
SLOPE_ZERO_ATOL: float = 1e-12      # a fitted slope this small is exactly zero
CONTEXT_TRIALS: int = 420           # innermost trials of one context on one item set
IWL_TRIALS: int = 42                # innermost IWL trials of one item set

_COLUMNS = ["model", "reasoning_kind", "condition", "item_pool", "dim", "d", "innermost",
            "trial_key", "budget", "is_correct", "predicted_true", "truncated"]
_SET_SUFFIX = re.compile(r"_rep\d+$")
# The axes a panel must pin, group by, or name as pooled before it averages over trials.
SWEPT: tuple[str, ...] = ("model", "dimension", "condition", "item_pool", "budget", "innermost")


# ── Loading ──────────────────────────────────────────────────────────────────────────────────
def _read(experiment: str, table: str, results_root: Path, *, columns=None,
          innermost_only: bool = False) -> pd.DataFrame:
    path = Path(results_root) / experiment / "tables" / f"{table}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"no {experiment}/{table} table at {path}")
    filters = [("innermost", "==", True)] if innermost_only else None
    return pd.read_parquet(path, columns=columns, filters=filters)


def _finish(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the dimension and the grid-set flag; store the text columns as categories."""
    frame = frame.reset_index(drop=True)
    frame["dimension"] = frame["dim"].astype(str).str.replace(_SET_SUFFIX, "", regex=True)
    first = {dim: sorted({s for s in frame["dim"].astype(str).unique()
                          if _SET_SUFFIX.sub("", s) == dim}, key=_set_order)[:GRID_SETS]
             for dim in frame["dimension"].unique()}
    keep = {s for sets in first.values() for s in sets}
    frame["grid_set"] = (frame["condition"] != roster.IWL_CONDITION) | frame["dim"].isin(keep)
    for col in ("model", "reasoning_kind", "condition", "item_pool", "dim", "dimension"):
        frame[col] = frame[col].astype(str).astype("category")
    return frame


def _set_order(name: str) -> tuple[int, str]:
    """The main item set first, then its replicates in order."""
    return (0, name) if not _SET_SUFFIX.search(name) else (1, name)


def direct_trials(results_root: Path = RESULTS_ROOT) -> pd.DataFrame:
    """Every direct answer of the IWL and ICL tests and of the five contexts, all ranks."""
    return _finish(pd.concat([_read("symbolic_distance", "direct", results_root, columns=_COLUMNS),
                              _read("content_effects", "direct", results_root, columns=_COLUMNS)],
                             ignore_index=True))


def ladder_trials(results_root: Path = RESULTS_ROOT) -> pd.DataFrame:
    """Every reasoning-budget ladder at the innermost seven ranks."""
    return _finish(pd.concat(
        [_read("symbolic_distance", "ladders", results_root, columns=_COLUMNS, innermost_only=True),
         _read("content_effects", "ladders", results_root, columns=_COLUMNS, innermost_only=True)],
        ignore_index=True))


def spacing_trials(results_root: Path = RESULTS_ROOT) -> pd.DataFrame:
    """The direct answers on the 19 item sets of other spacings, at the innermost ranks."""
    frame = _read("prior_strength", "direct", results_root, columns=_COLUMNS, innermost_only=True)
    return frame.reset_index(drop=True)


def chain_steps(results_root: Path = RESULTS_ROOT) -> pd.DataFrame:
    """The inferences each reasoning chain had composed by each budget."""
    return _read("reasoning_traces", "chain_steps", results_root)


def intervention_trials(results_root: Path = RESULTS_ROOT) -> dict[str, pd.DataFrame]:
    """The trials of the rank-direction run and of the head run, keyed ``rank`` and ``heads``."""
    return {"rank": _read("representation", "trials", results_root),
            "heads": _read("components", "trials", results_root)}


def probe_table(results_root: Path = RESULTS_ROOT) -> pd.DataFrame:
    """The rank direction's held-out ordering accuracy at every hidden state."""
    return _read("representation", "probe", results_root)


# ── Selection ────────────────────────────────────────────────────────────────────────────────
def pick(df: pd.DataFrame, **filters) -> pd.DataFrame:
    """Rows matching every filter (equality, or membership for a tuple, list or set)."""
    out = df
    for col, value in filters.items():
        if isinstance(value, (list, tuple, set, frozenset)):
            out = out[out[col].isin(list(value))]
        else:
            out = out[out[col] == value]
    if out.empty:
        wanted = ", ".join(f"{k}={v!r}" for k, v in filters.items())
        raise ValueError(f"pick: no rows match {wanted}")
    return out


def guard(df: pd.DataFrame, group_cols: Iterable[str], *, pooled: Iterable[str] = (),
          fname: str) -> None:
    """Refuse to average over a swept axis that is neither grouped nor named as pooled."""
    allowed = set(group_cols) | set(pooled)
    for col in SWEPT:
        if col in allowed or col not in df.columns:
            continue
        values = df[col].dropna().unique()
        if len(values) > 1:
            shown = sorted(map(str, values))[:8]
            raise ValueError(f"{fname}: {col!r} takes {len(values)} values {shown}; pin it, "
                             "group by it, or name it as pooled")


# ── Accuracy ─────────────────────────────────────────────────────────────────────────────────
def wilson_ci(k, n, *, z: float = BAND_Z) -> tuple[np.ndarray, np.ndarray]:
    """The Wilson score interval of k successes in n trials (NaN where n is 0)."""
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = k / n
        denom = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return np.where(n > 0, center - half, np.nan), np.where(n > 0, center + half, np.nan)


def trial_accuracy_by(df: pd.DataFrame, group_cols: Sequence[str], *, z: float = BAND_Z,
                      pooled: Iterable[str] = (), fname: str = "trial_accuracy_by") -> pd.DataFrame:
    """Accuracy over trials per group, with its Wilson interval at ``z``."""
    group_cols = list(group_cols)
    guard(df, group_cols, pooled=pooled, fname=fname)
    g = df.groupby(group_cols, observed=True, sort=True)
    out = pd.DataFrame({"n": g.size(), "n_correct": g["is_correct"].sum().astype(int)}).reset_index()
    out["accuracy"] = out["n_correct"] / out["n"]
    out["ci_lo"], out["ci_hi"] = wilson_ci(out["n_correct"], out["n"], z=z)
    return out


def accuracy_at_budget(ladder: pd.DataFrame, *, budget: int, min_distance: int = 2,
                       fname: str = "accuracy_at_budget") -> dict:
    """One ladder's accuracy at one reasoning budget, pooled over trials, item sets and the
    distances from ``min_distance``, at the innermost ranks.

    Returns ``n``, ``n_correct``, ``accuracy``, its Wilson interval (``ci_lo``, ``ci_hi``) and
    the number of item sets pooled (``n_item_sets``).
    """
    rows = ladder[ladder["innermost"] & (ladder["budget"] == int(budget))
                  & (ladder["d"] >= min_distance)]
    if rows.empty:
        raise ValueError(f"{fname}: no innermost rows at budget {int(budget)} and distances "
                         f"from {min_distance}")
    row = trial_accuracy_by(rows.assign(_all=0), ["_all"], pooled=("item_pool", "d"),
                            fname=fname).iloc[0]
    return {"n": int(row["n"]), "n_correct": int(row["n_correct"]),
            "accuracy": float(row["accuracy"]), "ci_lo": float(row["ci_lo"]),
            "ci_hi": float(row["ci_hi"]), "n_item_sets": int(rows["item_pool"].nunique())}


def running_best(table: pd.DataFrame) -> pd.DataFrame:
    """Each budget's row replaced by the row of the best accuracy at that budget or below.

    ``table`` holds one row per budget (``trial_accuracy_by`` grouped by ``budget``). A budget
    keeps the row (accuracy, interval, counts) of the first budget that reached the best
    accuracy so far, so a tie keeps the earlier budget; the ``budget`` column is unchanged.
    """
    ordered = table.sort_values("budget").reset_index(drop=True)
    best, picks = 0, []
    for i, acc in enumerate(ordered["accuracy"]):
        if acc > ordered.loc[best, "accuracy"]:
            best = i
        picks.append(best)
    out = ordered.iloc[picks].reset_index(drop=True)
    out["budget"] = ordered["budget"].to_numpy()
    return out


def binomial_se(k, n) -> np.ndarray:
    """The binomial standard error sqrt(p(1 - p) / n) at p = k / n."""
    k = np.asarray(k, float)
    n = np.asarray(n, float)
    p = k / n
    return np.sqrt(p * (1.0 - p) / n)


def pooled_accuracy(df: pd.DataFrame, measure: str = "is_correct", *, d_min: int | None = None,
                    d_max: int | None = None) -> dict:
    """k, n, accuracy and binomial standard error of ``measure`` over distances d_min..d_max."""
    rows = df
    if d_min is not None:
        rows = rows[rows["d"] >= d_min]
    if d_max is not None:
        rows = rows[rows["d"] <= d_max]
    kept = rows[measure].dropna().astype(bool)
    k, n = int(kept.sum()), int(len(kept))
    if n == 0:
        raise ValueError(f"pooled_accuracy: no {measure} trials at distances {d_min}-{d_max}")
    return {"k": k, "n": n, "acc": k / n, "se": float(binomial_se(k, n))}


# ── The reasoning budget ─────────────────────────────────────────────────────────────────────
def budget_matrix(df: pd.DataFrame, *, pooled: Iterable[str] = (),
                  fname: str = "budget_matrix") -> tuple[np.ndarray, np.ndarray]:
    """One ladder as a (trial x budget) 0/1 matrix and its budgets, trials in key order."""
    guard(df, ("budget",), pooled=pooled, fname=fname)
    outcome = df["is_correct"].astype(float)
    matrix = outcome.groupby([df["trial_key"], df["budget"]], observed=True).mean().unstack("budget")
    return matrix.to_numpy(dtype=float), matrix.columns.to_numpy(dtype=float)


def reasoning_budget(budgets, accs) -> float:
    """The smallest budget whose accuracy reaches ``FRACTION`` of the maximum over budgets."""
    b = np.asarray(list(budgets), dtype=float)
    a = np.asarray(list(accs), dtype=float)
    ok = ~np.isnan(a)
    if not ok.any():
        return float("nan")
    b, a = b[ok], a[ok]
    order = np.argsort(b)
    b, a = b[order], a[order]
    hit = np.flatnonzero(a >= FRACTION * np.nanmax(a))
    return float(b[hit[0]]) if hit.size else float("nan")


def budget_to_criterion(df: pd.DataFrame, *, n_boot: int | None = None,
                        pooled: Iterable[str] = (), fname: str = "budget_to_criterion") -> dict:
    """The reasoning budget of one ladder at one distance, with its bootstrap SD over trials.

    ``boot_sd`` is the standard deviation of the estimate over ``n_boot`` resamples of the
    trials (each keeping its outcome at every budget; None: ``N_BOOT``; 0 skips it).
    ``above_chance`` says whether the best accuracy beats ``CHANCE`` (Wilson lower bound at
    ``ABOVE_CHANCE_Z`` over the trials scored at that budget); where it does not, there is no
    budget to report.
    """
    matrix, budgets = budget_matrix(df, pooled=pooled, fname=fname)
    acc = np.nanmean(matrix, axis=0)
    point = reasoning_budget(budgets, acc)
    n = matrix.shape[0]
    sd = float("nan")
    n_boot = N_BOOT if n_boot is None else n_boot
    if n_boot:
        rng = np.random.default_rng(BOOT_SEED)
        draws = np.empty(n_boot)
        for i in range(n_boot):
            idx = rng.integers(0, n, n)
            draws[i] = reasoning_budget(budgets, np.nanmean(matrix[idx], axis=0))
        good = draws[~np.isnan(draws)]
        sd = float(np.std(good)) if good.size else float("nan")
    best = float(np.nanmax(acc))
    n_best = int(np.isfinite(matrix[:, int(np.nanargmax(acc))]).sum())
    lo_best = float(wilson_ci(best * n_best, n_best, z=ABOVE_CHANCE_Z)[0])
    return dict(budget=point, boot_sd=sd, n_trials=int(n), max_acc=best,
                above_chance=bool(lo_best > CHANCE),
                acc_by_budget={int(b): float(a) for b, a in zip(budgets, acc)})


def budget_summary(ladder: pd.DataFrame, *, pooled: Iterable[str] = (),
                   fname: str = "budget_summary") -> dict:
    """One ladder reduced to the mean budget and its least-squares slope on distance.

    A distance whose best accuracy is not above chance has no budget and is left out; with
    fewer than ``MIN_USABLE`` distances left the ladder is ``omitted``.
    """
    per_distance = {}
    for d in [d for d in DISTANCES if (ladder["d"] == d).any()]:
        got = budget_to_criterion(pick(ladder, d=d), n_boot=0, pooled=pooled, fname=fname)
        per_distance[d] = (got["budget"], got["above_chance"])
    usable = [d for d, (_, ok) in per_distance.items() if ok]
    budgets = [per_distance[d][0] for d in usable]
    omitted = len(usable) < MIN_USABLE
    return dict(usable=usable, omitted=omitted,
                mean_budget=float(np.mean(budgets)) if usable else float("nan"),
                slope=float("nan") if omitted else
                float(np.polyfit(np.asarray(usable, float), np.asarray(budgets, float), 1)[0]))


# ── One number per model ─────────────────────────────────────────────────────────────────────
def snapped(value: float) -> float:
    """A slope within ``SLOPE_ZERO_ATOL`` of zero, as exactly zero (it lies on the line)."""
    number = float(value)
    return 0.0 if abs(number) <= SLOPE_ZERO_ATOL else number


def accuracy_slope(df: pd.DataFrame, *, pooled: Iterable[str] = (),
                   fname: str = "accuracy_slope") -> float:
    """The least-squares slope of accuracy on symbolic distance, one point per distance."""
    rows = df[df["d"].isin(DISTANCES)]
    table = trial_accuracy_by(rows, ["d"], pooled=pooled, fname=fname)
    return snapped(np.polyfit(table["d"].to_numpy(float), table["accuracy"].to_numpy(float), 1)[0])


def binomial_sf(k: int, n: int, p: float = 0.5) -> float:
    """The exact one-sided p value P(X >= k) for X ~ Binomial(n, p), summed in the log domain."""
    n, k = int(n), int(k)
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    log_p, log_q = math.log(p), math.log1p(-p)
    terms = [math.log(math.comb(n, i)) + i * log_p + (n - i) * log_q for i in range(k, n + 1)]
    top = max(terms)
    return float(min(1.0, max(0.0, math.exp(top) * sum(math.exp(t - top) for t in terms))))


def spearman_rho(x, y) -> float:
    """Spearman's rank correlation: Pearson's r on average ranks (pairs with a NaN dropped)."""
    xs = pd.Series(np.asarray(x, dtype=float))
    ys = pd.Series(np.asarray(y, dtype=float))
    ok = xs.notna() & ys.notna()
    rx, ry = xs[ok].rank().to_numpy(), ys[ok].rank().to_numpy()
    if len(rx) < 2 or rx.std() == 0.0 or ry.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def competence(direct: pd.DataFrame) -> pd.DataFrame:
    """Does each model answer the IWL test above chance at all?

    Pools every innermost IWL trial at distances one to six over the grid sets of the five
    dimensions and asks whether the correct count beats chance by an exact one-sided binomial
    test at ``COMPETENCE_ALPHA``. One row per model, in roster order.
    """
    rows = direct[(direct["condition"] == roster.IWL_CONDITION) & direct["innermost"]
                  & direct["d"].isin(DISTANCES) & direct["dimension"].isin(roster.DIMENSIONS)
                  & direct["grid_set"]]
    g = rows.groupby("model", observed=True)["is_correct"]
    table = pd.DataFrame({"n": g.size(), "n_correct": g.sum().astype(int)})
    table = table.reindex([m for m in roster.MODELS if m in table.index])
    table["p_one_sided"] = [binomial_sf(k, n) for k, n in zip(table["n_correct"], table["n"])]
    table["competent"] = table["p_one_sided"] < COMPETENCE_ALPHA
    return table.rename_axis("model").reset_index()


# ── The cross-model scatters ─────────────────────────────────────────────────────────────────
def _window(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["innermost"] & df["d"].isin(DISTANCES)]


def distance_slope_points(direct: pd.DataFrame) -> pd.DataFrame:
    """One point per model: the IWL test's accuracy slope on distance against the ICL test's.

    x is the mean slope over the five dimensions (their grid sets pooled), with the standard
    error over the five; y the mean over the two sets of novel items, with their two values.
    """
    window = _window(direct)
    rows = []
    for model in roster.MODELS:
        own = window[window["model"] == model]
        xs = [accuracy_slope(own[(own["condition"] == roster.IWL_CONDITION) & own["grid_set"]
                                 & (own["dimension"] == dim)], pooled=("item_pool",),
                             fname=f"slope {model}/{dim}") for dim in roster.DIMENSIONS]
        ys = [accuracy_slope(own[own["condition"] == c], fname=f"slope {model}/{c}")
              for c in roster.ICL_CONDITIONS]
        rows.append({"model": model, "x": snapped(np.mean(xs)), "y": snapped(np.mean(ys)),
                     "x_se": snapped(np.std(xs, ddof=1) / np.sqrt(len(xs))),
                     "y_lo": snapped(min(ys)), "y_hi": snapped(max(ys))})
    return pd.DataFrame(rows)


def budget_points(ladder: pd.DataFrame) -> pd.DataFrame:
    """One point per model: the extra reasoning budget the ICL test needs, and its slope.

    Each set of novel items and each dimension's IWL ladder (every item set pooled) is reduced
    by ``budget_summary``. x is the mean over the two sets of novel items of their mean budget
    minus the IWL budget (the mean over the dimensions whose ladder is not omitted); y the mean
    of their slopes. A model is ``omitted`` when either set of novel items is, or when no
    dimension's IWL ladder is left; the bars span the two sets' own values.
    """
    window = _window(ladder)
    rows = []
    for model in roster.MODELS:
        own = window[window["model"] == model]
        icl = {c: budget_summary(own[own["condition"] == c], fname=f"budget {model}/{c}")
               for c in roster.ICL_CONDITIONS}
        iwl = {dim: budget_summary(own[(own["condition"] == roster.IWL_CONDITION)
                                       & (own["dimension"] == dim)],
                                   pooled=("item_pool",), fname=f"budget {model}/{dim}")
               for dim in roster.DIMENSIONS}
        usable = [dim for dim in roster.DIMENSIONS if not iwl[dim]["omitted"]]
        omitted = any(s["omitted"] for s in icl.values()) or not usable
        row = {"model": model, "omitted": omitted, "n_usable_dimensions": len(usable)}
        if omitted:
            row.update(x=np.nan, y=np.nan, x_lo=np.nan, x_hi=np.nan, y_lo=np.nan, y_hi=np.nan)
        else:
            iwl_mean = float(np.mean([iwl[dim]["mean_budget"] for dim in usable]))
            diffs = [icl[c]["mean_budget"] - iwl_mean for c in roster.ICL_CONDITIONS]
            slopes = [snapped(icl[c]["slope"]) for c in roster.ICL_CONDITIONS]
            row.update(x=snapped(np.mean(diffs)), y=snapped(np.mean(slopes)),
                       x_lo=snapped(min(diffs)), x_hi=snapped(max(diffs)),
                       y_lo=snapped(min(slopes)), y_hi=snapped(max(slopes)))
        rows.append(row)
    return pd.DataFrame(rows)


def prior_strength_points(direct: pd.DataFrame, spacing: pd.DataFrame) -> pd.DataFrame:
    """One point per model and item set: IWL accuracy against reversed-context accuracy.

    The item sets are each dimension's main set and the 19 sets of other spacings; x is the
    accuracy over a set's 42 innermost IWL trials, y over its 420 innermost reversed-context
    trials. ``is_main`` marks the main sets.
    """
    window = _window(direct)
    sets = [(roster.PRIMARY_SETS[dim], True) for dim in roster.DIMENSIONS]
    sets += [(item_set, False) for item_set in sorted(spacing["item_pool"].unique())]
    frames = {True: window, False: spacing[spacing["d"].isin(DISTANCES)]}
    rows = []
    for model in roster.MODELS:
        for item_set, is_main in sets:
            block = frames[is_main]
            block = block[(block["model"] == model) & (block["item_pool"] == item_set)]
            x = _accuracy_of(block[block["condition"] == roster.IWL_CONDITION], IWL_TRIALS,
                             f"{model}/{item_set}/IWL")
            y = _accuracy_of(block[block["condition"] == "reversed"], CONTEXT_TRIALS,
                             f"{model}/{item_set}/reversed")
            rows.append({"model": model, "item_set": item_set, "is_main": is_main, "x": x, "y": y})
    return pd.DataFrame(rows)


def _accuracy_of(rows: pd.DataFrame, expect: int, label: str) -> float:
    if len(rows) != expect:
        raise ValueError(f"prior strength: {label} has {len(rows)} innermost trials, not {expect}")
    return float(rows["is_correct"].mean())


def recovery_points(ladder: pd.DataFrame) -> pd.DataFrame:
    """One point per model and dimension: the content effect against the recovery slope.

    x is congruent minus reversed accuracy averaged over the distance x budget combinations;
    y the least-squares slope, per 1,000 tokens, of reversed accuracy (pooled over distances)
    on the reasoning budget.
    """
    window = _window(ladder)
    rows = []
    for model in roster.MODELS:
        for dim in roster.DIMENSIONS:
            base = window[(window["model"] == model) & (window["dimension"] == dim)]
            mats, budgets = {}, None
            for condition in ("congruent", "reversed"):
                sub = pick(base, condition=condition)
                for d in DISTANCES:
                    matrix, rungs = budget_matrix(pick(sub, d=d), fname=f"recovery {model}/{dim}")
                    if budgets is not None and not np.array_equal(budgets, rungs):
                        raise ValueError(f"recovery: {model}/{dim} ladders disagree on budgets")
                    mats[(condition, d)], budgets = matrix, rungs
            gaps = np.concatenate([np.nanmean(mats[("congruent", d)], axis=0)
                                   - np.nanmean(mats[("reversed", d)], axis=0) for d in DISTANCES])
            totals = sum(np.nansum(mats[("reversed", d)], axis=0) for d in DISTANCES)
            counts = sum(np.isfinite(mats[("reversed", d)]).sum(axis=0) for d in DISTANCES)
            slope = np.polyfit(budgets / BUDGET_UNIT, totals / counts, 1)[0]
            rows.append({"model": model, "dimension": dim, "x": snapped(np.mean(gaps)),
                         "y": snapped(slope)})
    return pd.DataFrame(rows)


def quadrant_counts(xs, ys, *, x_ref: float = 0.0, y_ref: float = 0.0) -> dict[str, int]:
    """Points per corner about the reference values; a point on a reference line is ``on_ref``."""
    counts = {"upper_right": 0, "upper_left": 0, "lower_right": 0, "lower_left": 0, "on_ref": 0}
    for x, y in zip(np.asarray(xs, float), np.asarray(ys, float)):
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        if x == x_ref or y == y_ref:
            counts["on_ref"] += 1
            continue
        counts[f"{'upper' if y > y_ref else 'lower'}_{'right' if x > x_ref else 'left'}"] += 1
    return counts


# ── The per-model appendix figures, summarized over the roster ───────────────────────────────
def accuracy_change(direct: pd.DataFrame, condition: str, *, dimensions: Sequence[str]) -> pd.Series:
    """Per model and dimension, accuracy at the farthest distance minus at the nearest (grid sets
    pooled)."""
    low, high = DISTANCES[0], DISTANCES[-1]
    rows = _window(direct)
    rows = rows[rows["grid_set"] & (rows["condition"] == condition)
                & rows["dimension"].isin(list(dimensions))]
    by_d = rows.groupby(["model", "dimension", "d"], observed=True)["is_correct"].mean().unstack("d")
    return by_d[high] - by_d[low]


def budget_change(ladder: pd.DataFrame, condition: str, *, dimensions: Sequence[str],
                  pooled: Iterable[str] = ("item_pool",)) -> pd.Series:
    """Per model and dimension, the budget at the farthest distance minus at the nearest, where
    both exist."""
    low, high = DISTANCES[0], DISTANCES[-1]
    rows = _window(ladder)
    rows = rows[(rows["condition"] == condition) & rows["dimension"].isin(list(dimensions))]
    out = {}
    for (model, dim), block in rows.groupby(["model", "dimension"], observed=True):
        got = {d: budget_to_criterion(pick(block, d=d), n_boot=0, pooled=pooled,
                                      fname=f"budget change {model}/{dim}") for d in (low, high)}
        if all(g["above_chance"] for g in got.values()):
            out[(model, dim)] = got[high]["budget"] - got[low]["budget"]
    return pd.Series(out, dtype=float)


def context_accuracy(direct: pd.DataFrame, condition: str) -> pd.Series:
    """Per model and dimension, direct accuracy in one context over its innermost trials."""
    rows = _window(direct)
    rows = rows[(rows["condition"] == condition) & rows["dimension"].isin(roster.DIMENSIONS)]
    return rows.groupby(["model", "dimension"], observed=True)["is_correct"].mean()


def ladder_accuracy(ladder: pd.DataFrame, condition: str, *, by_distance: bool = False) -> pd.DataFrame:
    """Accuracy per model, dimension and budget in one context (and distance, when asked)."""
    rows = _window(ladder)
    rows = rows[(rows["condition"] == condition) & rows["dimension"].isin(roster.DIMENSIONS)]
    keys = ["model", "dimension", "budget", *(["d"] if by_distance else [])]
    return rows.groupby(keys, observed=True)["is_correct"].mean().reset_index()


def composed_counts(chains: pd.DataFrame, model: str, condition: str, budget: int) -> pd.DataFrame:
    """Per distance two to six, the inferences composed by ``budget`` and the inferences needed,
    summed over the trials: whole numbers, so their ratio can be compared with a bound exactly."""
    rows = chains[(chains["model"] == model) & (chains["condition"] == condition)
                  & chains["innermost"] & chains["d"].between(2, 6) & (chains["budget"] == budget)]
    if rows["dim"].nunique() != 1:
        raise ValueError(f"composed_counts: {model}/{condition} spans the dimensions "
                         f"{sorted(rows['dim'].unique())}; pin one")
    counts = rows[["d", "inferences_composed", "inferences_needed"]].astype("int64")
    return counts.groupby("d")[["inferences_composed", "inferences_needed"]].sum()


def composed_fraction(chains: pd.DataFrame, model: str, condition: str, budget: int) -> pd.Series:
    """Per distance two to six, the share of the needed inferences composed by ``budget``."""
    counts = composed_counts(chains, model, condition, budget)
    return counts["inferences_composed"] / counts["inferences_needed"]
