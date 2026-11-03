"""Tests for the trial data structures and the fixed-hop, all-pairs generator."""

from __future__ import annotations

import json

import pytest

from geo_alg_ti_llms.items import ItemSet
from geo_alg_ti_llms.trials import (
    ContextPair,
    Query,
    Trial,
    generate_dataset_constrained_hop,
    make_cell_id,
)


# A trivial 26-item order-meaningless item set inline so generator tests don't
# depend on the YAML loader (pure-Python, fast):
LETTERS = ItemSet(
    name="abstract_letters",
    description="A..Z",
    items=tuple(chr(ord("A") + i) for i in range(26)),
    natural_order=False,
)
RANKED = ItemSet(
    name="ranked_letters", description="A..Z, natural order",
    items=LETTERS.items, natural_order=True,
)


def _make_minimal_trial(N: int = 3) -> Trial:
    items = tuple(f"item{i}" for i in range(N))
    pairs = tuple(
        ContextPair(higher=items[k + 1], lower=items[k], surface_form="high_first")
        for k in range(N - 1)
    )
    query = Query(
        a=items[N - 1], b=items[0], correct=True,
        symbolic_distance=N - 1, signed_distance=N - 1,
    )
    return Trial(
        trial_id=0, cell_id="test", master_seed=0, seed=0, N=N,
        item_pool="test", assignment_strategy="random",
        rank_to_item=items, context_pairs=pairs, query=query,
    )


# --- data structures --------------------------------------------------------


def test_context_pair_render_high_first():
    p = ContextPair(higher="A", lower="B", surface_form="high_first")
    assert p.render() == "A > B"


def test_context_pair_render_low_first():
    p = ContextPair(higher="A", lower="B", surface_form="low_first")
    assert p.render() == "B < A"


def test_trial_invariant_rank_to_item_length():
    with pytest.raises(ValueError, match="rank_to_item"):
        Trial(
            trial_id=0, cell_id="x", master_seed=0, seed=0, N=3,
            item_pool="x", assignment_strategy="random",
            rank_to_item=("a", "b"),  # wrong length: 2 != N=3
            context_pairs=(
                ContextPair("b", "a", "high_first"),
                ContextPair("c", "b", "high_first"),
            ),
            query=Query("a", "b", True, 1, 1),
        )


def test_trial_invariant_context_length():
    with pytest.raises(ValueError, match="context_pairs"):
        Trial(
            trial_id=0, cell_id="x", master_seed=0, seed=0, N=3,
            item_pool="x", assignment_strategy="random",
            rank_to_item=("a", "b", "c"),
            context_pairs=(ContextPair("b", "a", "high_first"),),  # only 1, expected 2
            query=Query("a", "b", True, 1, 1),
        )


def test_trial_invariant_signed_vs_symbolic_distance():
    with pytest.raises(ValueError, match="signed_distance"):
        Query(a="x", b="y", correct=True, symbolic_distance=2, signed_distance=3)


def test_query_sign_must_match_correct():
    with pytest.raises(ValueError, match="inconsistent"):
        Query(a="x", b="y", correct=False, symbolic_distance=2, signed_distance=2)


def test_trial_round_trip_via_dict():
    t = _make_minimal_trial(N=4)
    assert Trial.from_dict(t.to_dict()) == t


def test_trial_jsonifiable():
    """to_dict() output must be plain JSON types — no tuples, no dataclass instances."""
    t = _make_minimal_trial(N=4)
    s = json.dumps(t.to_dict())  # would raise on tuple / dataclass
    assert "item3" in s


# --- trial set ids ---------------------------------------------------------------


def test_cell_id_format_matches_the_published_runs():
    """The literal format of every stored row's cell_id (it seeds the RNG)."""
    assert make_cell_id(N=11, item_set="countries_by_population_11", hop=3,
                        surface_form_mode="fixed_gt") == \
        "phaseD|N=11|pool=countries_by_population_11|hop=3|surface=fixed_gt"
    assert make_cell_id(N=11, item_set="countries_by_population_11", hop=3,
                        surface_form_mode="fixed_gt", assignment_strategy="reversed") == \
        "phaseD|N=11|pool=countries_by_population_11|hop=3|surface=fixed_gt|assign=reversed"
    # random is the unsuffixed default
    assert make_cell_id(N=11, item_set="p", hop=1, surface_form_mode="zero_context",
                        assignment_strategy="random") == \
        "phaseD|N=11|pool=p|hop=1|surface=zero_context"


# --- generator: feasibility and geometry -------------------------------------


def test_rejects_N_less_than_hop_plus_3():
    """The non-endpoint constraint requires N >= hop + 3."""
    with pytest.raises(ValueError, match=r"N=4 must be >= hop\+3=5"):
        generate_dataset_constrained_hop(
            N=4, item_set=LETTERS, hop=2,
            n_premise_orders=5, master_seed=0,
        )


def test_rejects_hop_below_1():
    with pytest.raises(ValueError, match="hop"):
        generate_dataset_constrained_hop(N=5, item_set=LETTERS, hop=0, master_seed=0)


def test_rejects_single_pair_mode():
    with pytest.raises(ValueError, match="all_pairs"):
        generate_dataset_constrained_hop(N=5, item_set=LETTERS, hop=1, master_seed=0,
                                         all_pairs=False)


def test_rejects_natural_assignment_on_order_meaningless_item_set():
    with pytest.raises(ValueError, match="natural_order"):
        generate_dataset_constrained_hop(N=5, item_set=LETTERS, hop=1, master_seed=0,
                                         assignment_strategy="natural")


def _valid_nonterminal_pairs(N: int, hop: int) -> set[tuple[int, int]]:
    """The (rank_high, rank_low) pairs at this hop with both ranks non-terminal."""
    return {(k + hop, k) for k in range(1, N - 1 - hop)}


def test_all_pairs_enumerates_every_valid_k():
    """Every non-terminal pair, both polarities, per premise order."""
    N, hop, K = 11, 1, 4
    trials = generate_dataset_constrained_hop(
        N=N, item_set=LETTERS, hop=hop, n_premise_orders=K, master_seed=3,
    )
    valid = _valid_nonterminal_pairs(N, hop)        # 8 pairs
    # 8 pairs × 2 polarities × 4 orders = 64 trials.
    assert len(trials) == 2 * len(valid) * K == 64
    by_order: dict[int, set] = {}
    for t in trials:
        by_order.setdefault(t.extra["permutation_idx"], set()).add(
            (t.extra["rank_high"], t.extra["rank_low"])
        )
    assert set(by_order) == set(range(K))
    for order_pairs in by_order.values():
        assert order_pairs == valid


def test_trial_count_formula():
    """Trial count = 2 * (N-2-hop) * n_premise_orders across the grid."""
    K = 5
    for N, hop in [(7, 1), (7, 3), (11, 8), (15, 8), (23, 10)]:
        trials = generate_dataset_constrained_hop(
            N=N, item_set=LETTERS, hop=hop, n_premise_orders=K, master_seed=1,
        )
        assert len(trials) == 2 * (N - 2 - hop) * K


def test_shared_context_within_premise_order():
    """All trials from one premise order share an identical context + assignment,
    so only the queried pair / polarity varies."""
    trials = generate_dataset_constrained_hop(
        N=9, item_set=LETTERS, hop=2, n_premise_orders=3, master_seed=9,
    )
    by_order: dict[int, list] = {}
    for t in trials:
        by_order.setdefault(t.extra["permutation_idx"], []).append(t)
    assert len(by_order) == 3
    for order_trials in by_order.values():
        ref = order_trials[0]
        for t in order_trials:
            assert t.rank_to_item == ref.rank_to_item
            assert t.context_pairs == ref.context_pairs


def test_balanced_true_false():
    trials = generate_dataset_constrained_hop(
        N=12, item_set=LETTERS, hop=3, n_premise_orders=6, master_seed=4,
    )
    n_true = sum(t.query.correct for t in trials)
    assert n_true == sum(not t.query.correct for t in trials) == len(trials) // 2


def test_every_query_non_terminal_at_the_hop():
    trials = generate_dataset_constrained_hop(
        N=15, item_set=LETTERS, hop=5, n_premise_orders=3, master_seed=5,
    )
    for t in trials:
        rank_a = t.rank_to_item.index(t.query.a)
        rank_b = t.rank_to_item.index(t.query.b)
        assert 0 < rank_a < t.N - 1 and 0 < rank_b < t.N - 1
        assert abs(rank_a - rank_b) == 5
        assert t.query.symbolic_distance == 5


def test_fixed_gt_surface_form():
    trials = generate_dataset_constrained_hop(
        N=6, item_set=LETTERS, hop=2, n_premise_orders=5, master_seed=11,
        surface_form_mode="fixed_gt",
    )
    for t in trials:
        for p in t.context_pairs:
            assert p.surface_form == "high_first"


def test_context_is_full_chain_and_shuffled():
    """Context = N-1 adjacent-rank premises (full chain), every link once."""
    trials = generate_dataset_constrained_hop(
        N=7, item_set=LETTERS, hop=2, n_premise_orders=3, master_seed=0,
    )
    for t in trials:
        assert len(t.context_pairs) == t.N - 1
        rank_pairs = set()
        for p in t.context_pairs:
            high = t.rank_to_item.index(p.higher)
            low = t.rank_to_item.index(p.lower)
            assert high == low + 1
            rank_pairs.add((high, low))
        assert rank_pairs == {(k + 1, k) for k in range(t.N - 1)}


def test_extra_fields_populated():
    trials = generate_dataset_constrained_hop(
        N=5, item_set=LETTERS, hop=1, n_premise_orders=3, master_seed=0,
    )
    for t in trials:
        assert t.extra["hop"] == 1
        assert t.extra["surface_form_mode"] == "fixed_gt"
        assert t.extra["assignment_strategy"] == "random"
        assert 0 <= t.extra["permutation_idx"] < 3
        assert "phase" not in t.extra


def test_reproducible_under_same_seed():
    args = dict(N=6, item_set=LETTERS, hop=2, n_premise_orders=10, master_seed=33)
    assert generate_dataset_constrained_hop(**args) == generate_dataset_constrained_hop(**args)


def test_natural_and_reversed_use_the_same_items():
    nat = generate_dataset_constrained_hop(N=6, item_set=RANKED, hop=2, n_premise_orders=2,
                                           master_seed=1, assignment_strategy="natural")
    rev = generate_dataset_constrained_hop(N=6, item_set=RANKED, hop=2, n_premise_orders=2,
                                           master_seed=1, assignment_strategy="reversed")
    assert nat[0].rank_to_item == RANKED.items[:6]
    assert rev[0].rank_to_item == tuple(reversed(RANKED.items[:6]))
    assert nat[0].cell_id.endswith("|assign=natural")
    assert rev[0].cell_id.endswith("|assign=reversed")


# --- generator: context modes -----------------------------------------------


def test_zero_context_keeps_hidden_premises_and_marks_the_mode():
    """The IWL test: the trial stays a valid Trial (hidden premises) and the
    renderers suppress the context via extra['surface_form_mode']."""
    trials = generate_dataset_constrained_hop(
        N=6, item_set=RANKED, hop=2, n_premise_orders=1, master_seed=42,
        surface_form_mode="zero_context", assignment_strategy="natural",
    )
    for t in trials:
        assert t.extra["surface_form_mode"] == "zero_context"
        assert len(t.context_pairs) == 5
        assert t.rank_to_item == RANKED.items[:6]


def test_disjoint_context_premises_never_mention_queried_items():
    """The uninformative context = a real, well-formed chain over the item set's items[N:2N]."""
    trials = generate_dataset_constrained_hop(
        N=6, item_set=LETTERS, hop=2, n_premise_orders=3, master_seed=13,
        surface_form_mode="disjoint_context",
    )
    queried = set(LETTERS.items[:6])
    filler = set(LETTERS.items[6:12])
    assert trials
    for t in trials:
        assert len(t.context_pairs) == 5  # still an N-1 chain
        ctx = {p.higher for p in t.context_pairs} | {p.lower for p in t.context_pairs}
        assert ctx == filler
        assert not (ctx & queried)
        assert t.query.a in queried and t.query.b in queried


def test_disjoint_context_requires_an_item_set_of_at_least_2N():
    small = ItemSet(
        name="tiny", description="7 items", natural_order=False,
        items=tuple(chr(ord("A") + i) for i in range(7)),
    )
    with pytest.raises(ValueError, match="2\\*N"):
        generate_dataset_constrained_hop(
            N=6, item_set=small, hop=2, n_premise_orders=1, master_seed=0,
            surface_form_mode="disjoint_context",
        )


def test_disjoint_context_preserves_yes_no_balance():
    trials = generate_dataset_constrained_hop(
        N=6, item_set=LETTERS, hop=2, n_premise_orders=4, master_seed=5,
        surface_form_mode="disjoint_context",
    )
    n_true = sum(t.query.correct for t in trials)
    assert n_true * 2 == len(trials)


def test_unknown_surface_mode_raises():
    with pytest.raises(ValueError, match="surface_form_mode"):
        generate_dataset_constrained_hop(N=5, item_set=LETTERS, hop=1, master_seed=0,
                                         surface_form_mode="fixed_lt")
