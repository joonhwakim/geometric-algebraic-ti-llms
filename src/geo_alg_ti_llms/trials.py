"""One transitive-inference trial and the generator that builds a trial set of them.

A trial is a hierarchy of N items under one rank assignment, the N-1
adjacent-pair premises (the context, shuffled once per premise order) and a
query about two items at a fixed symbolic distance (``hop``). The paper's
geometry is one trial set per ``(N=11, hop, context mode, assignment)``: every
non-terminal rank pair at that distance is queried in both polarities (True
when the higher-ranked item is named first), on ``n_premise_orders``
shared-context premise orders.

Context modes (``surface_form_mode``):

- ``fixed_gt``: the premises are shown, every one as ``higher > lower`` (the
  IWL/ICL test with context and the congruent / reversed / shuffled content
  conditions; the assignment selects which).
- ``zero_context``: the IWL test. No premise is shown; the trial still carries
  the (natural) rank assignment so the gold answer is defined.
- ``disjoint_context``: the uninformative context. A well-formed premise chain
  over the item set's items ``[N:2N]``, disjoint from the queried items.

Seeding: each premise order's RNG is ``SeedSequence([master_seed,
md5(cell_id), premise_order])``, so a trial set regenerates byte for byte from
its ``cell_id`` and ``master_seed`` (42 in every published run).
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np

from geo_alg_ti_llms.items import AssignmentStrategy, ItemSet


SurfaceForm = Literal["high_first", "low_first"]
"""How a context pair is rendered:

- ``"high_first"`` → ``"{higher} > {lower}"``
- ``"low_first"``  → ``"{lower} < {higher}"``
"""

SurfaceFormMode = Literal["fixed_gt", "zero_context", "disjoint_context"]
"""The context mode of a trial set (see the module docstring)."""


@dataclass(frozen=True, slots=True)
class ContextPair:
    """One adjacent-pair premise shown to the model in the context.

    Attributes
    ----------
    higher : str
        The item that is *truly* higher in the underlying rank hierarchy.
    lower : str
        The truly lower item. Always adjacent to ``higher`` in rank
        (rank-distance == 1).
    surface_form : SurfaceForm
        Determines how this pair is rendered in the prompt; see SurfaceForm.
    """

    higher: str
    lower: str
    surface_form: SurfaceForm

    def render(self) -> str:
        """Return the prompt-ready string for this premise."""
        if self.surface_form == "high_first":
            return f"{self.higher} > {self.lower}"
        return f"{self.lower} < {self.higher}"


@dataclass(frozen=True, slots=True)
class Query:
    """The forced-choice question posed to the model after the context.

    The model must answer whether ``a`` is higher than ``b`` (True / False).

    Attributes
    ----------
    a : str
        First item shown in the query.
    b : str
        Second item shown in the query.
    correct : bool
        True iff ``a`` is truly higher than ``b`` in the rank hierarchy.
    symbolic_distance : int
        |rank(a) - rank(b)|; always positive, always in [1, N-1].
    signed_distance : int
        rank(a) - rank(b); positive when a is higher (correct = True),
        negative when b is higher (correct = False). Useful for analysis.
    """

    a: str
    b: str
    correct: bool
    symbolic_distance: int
    signed_distance: int

    def __post_init__(self) -> None:
        # |signed| must equal symbolic; sign of signed must match correct.
        if abs(self.signed_distance) != self.symbolic_distance:
            raise ValueError(
                f"|signed_distance|={abs(self.signed_distance)} != "
                f"symbolic_distance={self.symbolic_distance}"
            )
        if self.symbolic_distance < 1:
            raise ValueError(
                f"symbolic_distance must be >= 1 (got {self.symbolic_distance})"
            )
        if (self.signed_distance > 0) != self.correct:
            raise ValueError(
                f"sign of signed_distance ({self.signed_distance}) inconsistent "
                f"with correct={self.correct}"
            )

    def render(self) -> str:
        """Return the prompt-ready query string."""
        return f"Is {self.a} greater than {self.b}?"


@dataclass(frozen=True, slots=True)
class Trial:
    """A single in-context transitive-inference problem.

    Attributes
    ----------
    trial_id : int
        Unique within a generated dataset. Stable across regenerations of
        the same config + seed.
    cell_id : str
        Deterministic id of the experimental trial set (:func:`make_cell_id`).
    master_seed : int
        The dataset-level master seed used to construct this trial.
    seed : int
        Per-trial seed drawn from the premise order's RNG (provenance only).
    N : int
        Hierarchy size.
    item_pool : str
        Name of the item set (e.g. ``"countries_by_population_11"``).
    assignment_strategy : str
        How items were mapped to ranks: ``"random"``, ``"natural"``, or
        ``"reversed"`` (the latter two only valid for item sets that declare
        a natural order).
    rank_to_item : tuple[str, ...]
        Length-N tuple. Index = rank (0 = lowest, N-1 = highest).
        ``rank_to_item[k]`` is the item assigned to rank k for this trial.
    context_pairs : tuple[ContextPair, ...]
        Length-(N-1) tuple of premises, **already shuffled** in the order
        the model sees them. Together they fully specify the hierarchy.
    query : Query
        The question the model must answer.
    extra : dict
        Optional metadata bag for downstream extensions. Keys must be
        JSON-serializable.
    """

    trial_id: int
    cell_id: str
    master_seed: int
    seed: int
    N: int
    item_pool: str
    assignment_strategy: str
    rank_to_item: tuple[str, ...]
    context_pairs: tuple[ContextPair, ...]
    query: Query
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Invariant checks: cheap, only run at construction; catches bugs early.
        if len(self.rank_to_item) != self.N:
            raise ValueError(
                f"rank_to_item has {len(self.rank_to_item)} items but N={self.N}"
            )
        if len(self.context_pairs) != self.N - 1:
            raise ValueError(
                f"context_pairs has {len(self.context_pairs)} entries "
                f"but expected N-1={self.N - 1}"
            )
        if self.query.symbolic_distance < 1 or self.query.symbolic_distance > self.N - 1:
            raise ValueError(
                f"query symbolic_distance={self.query.symbolic_distance} "
                f"outside [1, {self.N - 1}]"
            )
        if abs(self.query.signed_distance) != self.query.symbolic_distance:
            raise ValueError(
                f"|signed_distance|={abs(self.query.signed_distance)} != "
                f"symbolic_distance={self.query.symbolic_distance}"
            )

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable representation. Tuples become lists."""
        d = asdict(self)
        d["rank_to_item"] = list(self.rank_to_item)
        d["context_pairs"] = [asdict(p) for p in self.context_pairs]
        d["query"] = asdict(self.query)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Trial:
        """Inverse of :meth:`to_dict`. Round-trips losslessly."""
        return cls(
            trial_id=d["trial_id"],
            cell_id=d["cell_id"],
            master_seed=d["master_seed"],
            seed=d["seed"],
            N=d["N"],
            item_pool=d["item_pool"],
            assignment_strategy=d["assignment_strategy"],
            rank_to_item=tuple(d["rank_to_item"]),
            context_pairs=tuple(ContextPair(**p) for p in d["context_pairs"]),
            query=Query(**d["query"]),
            extra=dict(d.get("extra", {})),
        )


def make_cell_id(
    *, N: int, item_set: str, hop: int, surface_form_mode: str,
    assignment_strategy: AssignmentStrategy = "random",
) -> str:
    """Deterministic trial set identifier; it seeds the trial set's RNG streams.

    The ``assign=`` axis is appended **only** when the strategy is not the
    default ``"random"``, so the shuffled trial sets and the novel-item trial sets share
    one id format; ``natural`` / ``reversed`` trial sets get a
    distinct suffix and so never collide with the shuffled trial set in a shared
    results directory. The literal ``phaseD`` prefix is load-bearing:
    it is hashed into every RNG seed and stored in every row, so the published trials
    regenerate only with this exact string.
    """
    base = f"phaseD|N={N}|pool={item_set}|hop={hop}|surface={surface_form_mode}"
    if assignment_strategy != "random":
        base += f"|assign={assignment_strategy}"
    return base


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------


def _hash_cell_id(cell_id: str) -> int:
    """Deterministic 32-bit hash of a cell_id for RNG seeding.

    md5 is fine here: only stability across runs and good mixing across
    trial sets matter, not cryptographic properties.
    """
    return int(hashlib.md5(cell_id.encode("utf-8")).hexdigest(), 16) % (2**32)


def _trial_rng(master_seed: int, cell_id: str, trial_id: int) -> np.random.Generator:
    """Independent per-trial RNG. Derived deterministically; reproducible."""
    ss = np.random.SeedSequence(
        [int(master_seed), _hash_cell_id(cell_id), int(trial_id)]
    )
    return np.random.default_rng(ss)


def _draw_surface_form(
    rng: np.random.Generator, mode: str
) -> SurfaceForm:
    """Return one ``SurfaceForm`` according to ``mode``.

    ``"random"`` consumes one ``rng.random()`` per pair; the fixed mode
    consumes nothing, so the per-trial RNG state diverges between modes by
    design (a fixed mode leaves more RNG budget for downstream draws like the
    context-pair shuffle).
    """
    if mode == "random":
        return "high_first" if rng.random() < 0.5 else "low_first"
    if mode == "fixed_gt":
        return "high_first"
    raise ValueError(f"unknown surface_form_mode: {mode!r}")


def _build_disjoint_pairs(
    filler_items: tuple[str, ...],
    rng: np.random.Generator,
) -> tuple[ContextPair, ...]:
    """Build an ``M-1``-premise chain over ``filler_items`` (real, non-queried).

    Same shape as a normal premise context (a coherent adjacent chain over the
    filler items in their given order, per-pair 50/50 surface form, Fisher-Yates
    shuffled), but over items that are disjoint from the queried ones — so the
    context is on-topic and truthful yet carries no usable ordering signal about
    the query. Consumes RNG in the same idiom as :func:`_build_context_pairs`
    (one ``rng.random()`` per pair, then a shuffle), so trial streams stay
    reproducible.
    """
    pairs: list[ContextPair] = []
    for idx in range(len(filler_items) - 1):
        higher, lower = filler_items[idx + 1], filler_items[idx]
        surface = _draw_surface_form(rng, "random")
        pairs.append(ContextPair(higher=higher, lower=lower, surface_form=surface))
    rng.shuffle(pairs)
    return tuple(pairs)


def _build_context_pairs(
    rank_to_item: tuple[str, ...],
    rng: np.random.Generator,
    surface_form_mode: SurfaceFormMode = "fixed_gt",
    filler_items: tuple[str, ...] | None = None,
) -> tuple[ContextPair, ...]:
    """Build the (N-1) adjacent-pair premises for one trial.

    ``disjoint_context`` short-circuits to :func:`_build_disjoint_pairs`.
    ``fixed_gt`` renders every pair ``"high_first"``. Context-pair order is
    Fisher-Yates shuffled in-place regardless of surface form mode. The
    shuffle uses the same per-trial RNG.
    """
    if surface_form_mode == "disjoint_context":
        if not filler_items or len(filler_items) < 2:
            raise ValueError(
                "surface_form_mode='disjoint_context' needs >= 2 filler_items "
                "(items of the item set disjoint from the queried ones); the caller must "
                "pass them (the item set needs at least 2*N items)."
            )
        return _build_disjoint_pairs(tuple(filler_items), rng)
    # zero_context: the model is shown NO premises, but the ``Trial`` invariant
    # requires N-1 context_pairs, so a normal, well-formed premise set is still
    # built here (the trial stays valid + reproducible) and every prompt function
    # (direct and reasoning pass, through prompts.render_grid_shown_context)
    # suppresses it at render time when ``extra['surface_form_mode'] ==
    # 'zero_context'``. The hidden premises' surface form is immaterial (never
    # rendered); draw them as 'random' so _draw_surface_form has a defined mode.
    effective_mode = "random" if surface_form_mode == "zero_context" else surface_form_mode
    N = len(rank_to_item)
    pairs: list[ContextPair] = []
    for rank_low in range(N - 1):
        rank_high = rank_low + 1
        higher, lower = rank_to_item[rank_high], rank_to_item[rank_low]
        surface = _draw_surface_form(rng, effective_mode)
        pairs.append(
            ContextPair(higher=higher, lower=lower, surface_form=surface)
        )
    rng.shuffle(pairs)  # Fisher-Yates in-place
    return tuple(pairs)


def _build_query(
    rank_to_item: tuple[str, ...],
    rank_high: int,
    rank_low: int,
    correct_is_yes: bool,
) -> Query:
    """Build the query for a given rank-pair and answer polarity.

    If ``correct_is_yes`` is True, the higher-ranked item is shown first
    in the query (so the correct answer is True). Otherwise the lower-
    ranked item is first (correct is False).
    """
    item_high = rank_to_item[rank_high]
    item_low = rank_to_item[rank_low]
    distance = rank_high - rank_low

    if correct_is_yes:
        return Query(
            a=item_high,
            b=item_low,
            correct=True,
            symbolic_distance=distance,
            signed_distance=distance,
        )
    return Query(
        a=item_low,
        b=item_high,
        correct=False,
        symbolic_distance=distance,
        signed_distance=-distance,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate_dataset_constrained_hop(
    *,
    N: int,
    item_set: ItemSet,
    hop: int,
    n_premise_orders: int = 20,
    master_seed: int,
    surface_form_mode: SurfaceFormMode = "fixed_gt",
    starting_trial_id: int = 0,
    all_pairs: bool = True,
    assignment_strategy: AssignmentStrategy = "random",
) -> list[Trial]:
    """Fixed-hop trial generator: query distance ``hop``, both query items
    non-endpoint, with **N and hop decoupled** (any ``N >= hop + 3``).

    Item-to-rank assignment (``assignment_strategy``)
    ------------------------------------------------
    - ``"random"`` (default): a fresh random order, per premise order, of the first
      ``N`` items of the item set (the shuffled context, and the novel-item test).
    - ``"natural"`` / ``"reversed"`` (require ``item_set.natural_order=True``):
      the item set's real-world ordering becomes the trial's rank ordering
      (``natural`` = congruent) or its reverse (``reversed`` = incongruent).
      Both use the SAME first-``N`` items, differing only in rank assignment,
      so a congruent/incongruent contrast isolates semantic interference from
      item identity. These produce a DISTINCT trial set id, so a congruency run
      never collides with a ``random`` run in the same dir.

    Geometry
    --------
    - The chain has ``N`` items at ranks ``0..N-1``.
    - The context shows all ``N-1`` adjacent-rank premises (Fisher-Yates
      shuffled per premise order).
    - Each query is at hop distance ``d == hop`` and BOTH query items
      are non-endpoints, i.e. ``rank_a, rank_b ∈ (0, N-1)``. This
      guarantees each query item appears in exactly two premises in the
      context (once as the greater, once as the lesser).
    - The non-endpoint constraint forces **N ≥ d + 3**. The valid lower
      ranks are ``k ∈ [1, N-2-d]`` (``N-2-d`` distinct pairs).

    Per ``(N, hop, surface_form_mode)`` trial set, for each of
    ``n_premise_orders`` independent premise orders (seeded by the trial set, so a
    re-run reproduces the same trial set, and the context — item assignment
    + premise rendering + shuffle — is built **once per premise order** and
    shared across every query emitted from it), every valid pair is emitted
    in both polarities on the shared per-premise order context:
    ``2 * (N-2-hop) * n_premise_orders`` trials per trial set. The True/False
    polarities of a given pair share an identical context (matched-pair
    comparison), and at fixed premise order every queried pair shares the same
    context too (isolates the query-position effect).

    Asserts
    -------
    - ``N >= hop + 3`` (non-endpoint feasibility).
    - ``len(item_set.items) >= N``.
    - For every emitted trial: ``0 < rank_a < N-1`` and
      ``0 < rank_b < N-1``.
    """
    if not all_pairs:
        raise ValueError("only the all_pairs=True geometry is supported")
    if hop < 1:
        raise ValueError(f"hop must be >= 1 (got {hop})")
    if N < hop + 3:
        raise ValueError(
            f"N={N} must be >= hop+3={hop + 3} "
            "(both query items must be non-endpoints in the chain)."
        )
    if N > item_set.size:
        raise ValueError(
            f"item set {item_set.name!r} has {item_set.size} items but N={N} requested"
        )
    if assignment_strategy in ("natural", "reversed") and not item_set.natural_order:
        raise ValueError(
            f"assignment_strategy={assignment_strategy!r} requires item set with "
            f"natural_order=True, but item set {item_set.name!r} is order-meaningless"
        )

    # disjoint_context: the premise chain runs over the item set's items[N:2N],
    # which must exist and must not overlap the queried items[:N].
    filler_items: tuple[str, ...] | None = None
    if surface_form_mode == "disjoint_context":
        if item_set.size < 2 * N:
            raise ValueError(
                f"surface_form_mode='disjoint_context' needs an item set of at least 2*N items "
                f"(item set {item_set.name!r} has {item_set.size} items, N={N})"
            )
        filler_items = tuple(item_set.items[N:2 * N])

    cell_id = make_cell_id(
        N=N, item_set=item_set.name, hop=hop,
        surface_form_mode=surface_form_mode,
        assignment_strategy=assignment_strategy,
    )

    # Pre-compute the valid (rank_high, rank_low) pairs at this hop. The
    # non-endpoint constraint requires rank_low > 0 (k > 0 → k ≥ 1) AND
    # rank_high < N-1 (k + hop < N-1 → k ≤ N-2-hop). So k ∈ [1, N-2-hop],
    # i.e. range(1, N-1-hop).
    valid_pairs = [
        (k + hop, k) for k in range(1, N - 1 - hop)
    ]
    if not valid_pairs:  # belt-and-braces; should be unreachable given the N check
        raise RuntimeError(
            f"no valid (rank_high, rank_low) pairs for N={N}, hop={hop}"
        )

    trials: list[Trial] = []
    for order_idx in range(n_premise_orders):
        # Per-premise order RNG, derived from the trial set's master seed in the
        # standard way. The context (item assignment + premise rendering +
        # shuffle) is built ONCE here and reused for every query emitted from
        # this premise order, so the True/False polarities of a pair — and
        # every queried pair — share an identical context.
        order_rng = _trial_rng(master_seed, cell_id, order_idx)
        rank_to_item = item_set.assign(
            N, order_rng, strategy=assignment_strategy
        )
        context_pairs = _build_context_pairs(
            rank_to_item, order_rng, surface_form_mode,
            filler_items=filler_items,
        )

        for rank_high, rank_low in valid_pairs:
            # Belt-and-braces: re-assert the rank constraints per trial.
            assert 0 < rank_high < N - 1, (rank_high, N)
            assert 0 < rank_low < N - 1, (rank_low, N)

            for correct in (True, False):
                query = _build_query(
                    rank_to_item=rank_to_item,
                    rank_high=rank_high,
                    rank_low=rank_low,
                    correct_is_yes=correct,
                )
                trial_id = starting_trial_id + len(trials)
                extra = {
                    "hop": hop,
                    "permutation_idx": order_idx,
                    "surface_form_mode": surface_form_mode,
                    "assignment_strategy": assignment_strategy,
                    "rank_high": rank_high,
                    "rank_low": rank_low,
                }
                trials.append(Trial(
                    trial_id=trial_id,
                    cell_id=cell_id,
                    master_seed=master_seed,
                    seed=int(order_rng.integers(0, 2**31 - 1)),
                    N=N,
                    item_pool=item_set.name,
                    assignment_strategy=assignment_strategy,
                    rank_to_item=rank_to_item,
                    context_pairs=context_pairs,
                    query=query,
                    extra=extra,
                ))
    return trials
