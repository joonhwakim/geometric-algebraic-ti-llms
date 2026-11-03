"""Item sets: the ordered familiar and novel items a trial is built from.

An item set is a named, ordered sequence of item names stored as YAML under ``data/items/``.
The familiar-item sets (countries, cities, animals, US states by population,
mass or area; their replicate ``_rep`` item sets, their disjoint ``_disjoint``
item sets for the uninformative context, and the item-spacing ``_scale`` item sets)
are listed lowest-rank first and carry ``natural_order: true``, so the
congruent (``natural``), reversed and shuffled (``random``) rank assignments
are all valid. The two novel-item sets (given names, invented cities) are
static lists with ``natural_order: false`` and admit only the shuffled
assignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import yaml


AssignmentStrategy = Literal["random", "natural", "reversed"]

DEFAULT_ITEMS_DIR: Path = Path(__file__).resolve().parents[2] / "data" / "items"
"""The repository's ``data/items`` directory (the item sets the paper used)."""


@dataclass(frozen=True, slots=True)
class ItemSet:
    """An ordered, named list of items.

    Attributes
    ----------
    name : str
        Item set identifier (e.g. ``"countries_by_population_11"``). Must be unique
        across the project; used in budget_ids and JSONL rows.
    description : str
        One-line human-readable description (purpose, ordering dimension).
    items : tuple[str, ...]
        For order-meaningful item sets, items are listed **lowest-rank first**
        (rank 0 is at index 0). For order-meaningless item sets, the order is
        immaterial to the science; YAML order is still preserved for
        reproducibility.
    natural_order : bool
        If True, ``natural`` and ``reversed`` assignment strategies are
        valid (items have a stable real-world magnitude ordering).
    """

    name: str
    description: str
    items: tuple[str, ...]
    natural_order: bool

    @property
    def size(self) -> int:
        return len(self.items)

    def assign(
        self,
        N: int,
        rng: np.random.Generator,
        strategy: AssignmentStrategy = "random",
    ) -> tuple[str, ...]:
        """Return a length-N tuple mapping rank → item.

        Index 0 is rank 0 (lowest); index N-1 is rank N-1 (highest).

        Strategies
        ----------
        - ``"random"``: take the first N items of the item set and permute
          THEM (the same N items on every trial, in a fresh random order).
          Defines a per-trial assignment unrelated to the item set's natural
          order. Valid for all item sets.
        - ``"natural"``: take the first N items in the item set's order. The item set's
          natural ordering becomes the trial's rank ordering (semantic
          *congruent*). Requires ``natural_order=True``.
        - ``"reversed"``: take the first N items in the item set's order, then
          reverse them before assigning to ranks. The item set's natural
          ordering becomes the *opposite* of the trial's rank ordering
          (semantic *incongruent*). Requires ``natural_order=True``.

        Notes
        -----
        The first N items of the item set are always used (not a random
        N-item subset) so that congruent and incongruent conditions
        operate on the *same* items — they differ only in rank
        assignment, isolating semantic interference from item identity.
        """
        if N > self.size:
            raise ValueError(
                f"Item set {self.name!r} has {self.size} items but N={N} requested"
            )
        if strategy in ("natural", "reversed") and not self.natural_order:
            raise ValueError(
                f"Item set {self.name!r} is order-meaningless (natural_order=False); "
                f"only the 'random' assignment strategy is valid"
            )

        head = self.items[:N]
        match strategy:
            case "random":
                idx = rng.permutation(N)
                return tuple(head[i] for i in idx)
            case "natural":
                return tuple(head)
            case "reversed":
                return tuple(reversed(head))
            case _:  # pragma: no cover  -- caught by Literal at type-check time
                raise ValueError(f"Unknown assignment strategy: {strategy!r}")


def _load_yaml(name: str, items_dir: str | Path) -> tuple[Path, dict]:
    """Read the YAML for ``name`` and return (path, parsed dict)."""
    p = Path(items_dir) / f"{name}.yaml"
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{p}: top-level YAML must be a mapping")
    if raw.get("name") != name:
        raise ValueError(
            f"{p}: file basename {name!r} doesn't match name field "
            f"{raw.get('name')!r}"
        )
    return p, raw


def load_item_set(name: str, items_dir: str | Path = DEFAULT_ITEMS_DIR) -> ItemSet:
    """Load a static :class:`ItemSet` from ``{items_dir}/{name}.yaml``.

    Expected YAML schema::

        name: animals_by_mass_11         # must equal the file's basename
        description: ...
        natural_order: true              # bool
        items:                           # list of strings, low-rank first
          - wolf
          - cheetah
          - ...
    """
    p, raw = _load_yaml(name, items_dir)

    expected_keys = {"name", "description", "natural_order", "items"}
    if set(raw) != expected_keys:
        raise ValueError(
            f"{p}: expected keys {sorted(expected_keys)}, got {sorted(raw)}"
        )
    if not isinstance(raw["items"], list) or not all(isinstance(x, str) for x in raw["items"]):
        raise ValueError(f"{p}: 'items' must be a list of strings")
    if len(set(raw["items"])) != len(raw["items"]):
        dupes = [x for x in raw["items"] if raw["items"].count(x) > 1]
        raise ValueError(f"{p}: duplicate items: {sorted(set(dupes))}")

    return ItemSet(
        name=raw["name"],
        description=raw["description"],
        items=tuple(raw["items"]),
        natural_order=bool(raw["natural_order"]),
    )


def list_item_sets(items_dir: str | Path = DEFAULT_ITEMS_DIR) -> list[str]:
    """Return the names of all item sets available under ``items_dir``."""
    return sorted(p.stem for p in Path(items_dir).glob("*.yaml"))
