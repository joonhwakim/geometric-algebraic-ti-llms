"""Tests for item-set loading and the three rank assignments."""

from __future__ import annotations

import numpy as np
import pytest

from geo_alg_ti_llms.items import DEFAULT_ITEMS_DIR, ItemSet, list_item_sets, load_item_set


def test_default_items_dir_is_the_repository_item_sets():
    assert DEFAULT_ITEMS_DIR.is_dir()
    assert (DEFAULT_ITEMS_DIR / "countries_by_population_11.yaml").is_file()


def test_load_item_set_known(items_dir):
    item_set = load_item_set("countries_by_population_11", items_dir)
    assert item_set.name == "countries_by_population_11"
    assert item_set.natural_order is True
    assert item_set.size == 11


def test_every_committed_item_set_loads(items_dir):
    """Every item set of the paper is a static list with the expected keys."""
    names = list_item_sets(items_dir)
    assert len(names) == 41
    for name in names:
        item_set = load_item_set(name, items_dir)
        assert item_set.size >= 11
        assert len(set(item_set.items)) == item_set.size


def test_novel_item_sets_are_order_meaningless(items_dir):
    for name in ("novel_names_11", "novel_cities_11"):
        item_set = load_item_set(name, items_dir)
        assert item_set.natural_order is False
        assert item_set.size == 11


def test_disjoint_item_sets_hold_two_chains(items_dir):
    """The uninformative context runs over items[11:22], so a _disjoint item set has 22."""
    for stem in ("countries_by_population_11", "cities_by_population_11", "animals_by_mass_11",
                 "us_states_by_population_11", "countries_by_area_11", "cities_by_population_11_linear"):
        item_set = load_item_set(f"{stem}_disjoint", items_dir)
        assert item_set.size == 22
        assert item_set.items[:11] == load_item_set(stem, items_dir).items


def test_load_item_set_unknown_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_item_set("nonexistent", tmp_path)


def test_load_item_set_rejects_duplicate_items(tmp_path):
    p = tmp_path / "dupes.yaml"
    p.write_text(
        "name: dupes\ndescription: x\nnatural_order: false\nitems: [a, b, a, c]\n",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_item_set("dupes", tmp_path)


def test_load_item_set_rejects_name_mismatch(tmp_path):
    p = tmp_path / "x.yaml"
    p.write_text("name: y\ndescription: x\nnatural_order: false\nitems: [a, b]\n")
    with pytest.raises(ValueError, match="doesn't match"):
        load_item_set("x", tmp_path)


def test_assign_random_uses_first_N(items_dir):
    """The random strategy permutes the first N items, not a random N-item subset,
    so congruent and incongruent conditions on the same item set operate on *the same
    items*, just rank-flipped."""
    item_set = load_item_set("countries_by_population_11_disjoint", items_dir)
    rng = np.random.default_rng(0)
    out = item_set.assign(N=5, rng=rng, strategy="random")
    assert set(out) == set(item_set.items[:5])
    assert len(out) == 5


def test_assign_natural_returns_item_set_order(items_dir):
    item_set = load_item_set("animals_by_mass_11", items_dir)
    rng = np.random.default_rng(0)
    out = item_set.assign(N=5, rng=rng, strategy="natural")
    assert out == item_set.items[:5]


def test_assign_reversed_returns_reversed_item_set_order(items_dir):
    item_set = load_item_set("animals_by_mass_11", items_dir)
    rng = np.random.default_rng(0)
    out = item_set.assign(N=5, rng=rng, strategy="reversed")
    assert out == tuple(reversed(item_set.items[:5]))


def test_assign_natural_rejected_for_order_meaningless_item_set(items_dir):
    item_set = load_item_set("novel_names_11", items_dir)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="order-meaningless"):
        item_set.assign(N=5, rng=rng, strategy="natural")


def test_assign_random_is_deterministic_under_same_seed(items_dir):
    item_set = load_item_set("animals_by_mass_11", items_dir)
    out1 = item_set.assign(N=7, rng=np.random.default_rng(123), strategy="random")
    out2 = item_set.assign(N=7, rng=np.random.default_rng(123), strategy="random")
    assert out1 == out2


def test_assign_too_many_raises(items_dir):
    item_set = load_item_set("countries_by_population_11", items_dir)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="N=12"):
        item_set.assign(N=12, rng=rng)


def test_item_set_dataclass_is_frozen():
    p = ItemSet(name="x", description="y", items=("a", "b"), natural_order=False)
    with pytest.raises(Exception):  # FrozenInstanceError or similar
        p.name = "z"  # type: ignore[misc]
