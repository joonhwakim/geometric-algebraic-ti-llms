"""Test fixtures shared across the suite (CPU only; no torch / transformers needed)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make src/ imports work in tests run via plain pytest (no editable install needed):
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))


class MockTokenizer:
    """Stand-in for a HF tokenizer in unit tests.

    Implements the minimum surface used by candidate resolution: ``name_or_path``
    and ``encode`` (with ``add_special_tokens=False``). Tokenization is
    deterministic and word-based so small assertion-tested cases can run
    without a real model.
    """

    name_or_path = "MockTokenizer"

    def __init__(self) -> None:
        self._next_id = 0
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}

    def _intern(self, tok: str) -> int:
        if tok not in self._token_to_id:
            self._token_to_id[tok] = self._next_id
            self._id_to_token[self._next_id] = tok
            self._next_id += 1
        return self._token_to_id[tok]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        # Split on whitespace; each whitespace-delimited word is one token.
        # Leading-space variants (" True" vs "True") encode differently because
        # the leading space becomes part of the first token's surface form.
        if not text:
            return []
        if text.startswith(" "):
            tokens = [" " + t for t in text[1:].split()]
        else:
            tokens = text.split()
        return [self._intern(t) for t in tokens]

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [self._id_to_token[i] for i in ids]


@pytest.fixture
def mock_tokenizer() -> MockTokenizer:
    return MockTokenizer()


@pytest.fixture
def items_dir() -> Path:
    return _ROOT / "data" / "items"


@pytest.fixture
def models_dir() -> Path:
    return _ROOT / "configs" / "models"
