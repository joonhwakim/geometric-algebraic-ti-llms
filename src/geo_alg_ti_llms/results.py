"""One scored trial per row, and the JSONL files the runs are stored in.

A :class:`TrialResult` is one row of a run's JSONL: the trial's provenance
(trial and trial set ids, item set, assignment, signed and symbolic distance, gold
answer), the model and prompt format, the True/False log-probabilities, the
prediction, and -- for the reasoning pass -- the trace, its length and
finish reason, and both prompts. The field names are those of the published
runs; the committed tables are exported from these rows.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


Mode = Literal["direct", "cot"]


@dataclass(frozen=True, slots=True)
class TrialResult:
    """Outcome of scoring one trial under one (model, prompt_spec)."""

    # Trial provenance:
    trial_id: int
    cell_id: str
    master_seed: int
    N: int
    item_pool: str
    assignment_strategy: str
    signed_distance: int
    symbolic_distance: int
    correct: bool          # ground-truth True/False

    # Model / prompt config — fully describes what was run:
    model: str             # HuggingFace id
    prompt_spec: dict[str, Any]   # variant ids: instructions, trigger, budget, ...
    mode: Mode

    # Model output:
    predicted: bool        # True == predicted "True"
    is_correct: bool       # predicted == correct
    logprob_yes: float     # logsumexp over True-candidate token logprobs
    logprob_no: float      # logsumexp over False-candidate token logprobs
    logit_gap: float       # logprob_yes - logprob_no  (signed; 0 = uncertain)

    # Reasoning-only fields (None for ``mode == "direct"``):
    cot_trace: str | None = None
    cot_n_tokens: int | None = None
    cot_finish_reason: str | None = None
    """One of: ``"eos"`` (model emitted an end-of-sequence id), ``"stop"`` (ended on a
    stop string: the native ``</think>`` close, gpt-oss ``<|end|>``), ``"length"`` (hit
    the budget; ``length`` outranks ``eos``/``stop`` when a chain ends exactly at the
    budget), ``"unknown"``."""

    # Prompt-provenance fields, so each result row is fully self-contained —
    # the analysis never needs to recompute the prompt from ``trial`` + ``spec``
    # to verify what the model actually saw.
    prompt: str = ""
    """The exact prompt string scored against True / False (i.e., for direct
    mode this is the only prompt; for the reasoning pass this is the *answer-pass*
    prompt — reasoning-pass prompt + generated trace + extract cue)."""

    prompt_n_tokens: int = 0
    """Token count of ``prompt`` under the model's tokenizer."""

    cot_stage1_prompt: str | None = None
    """For the reasoning pass: the *reasoning-pass* prompt that produced the trace
    (None for direct mode)."""

    cot_stage1_n_tokens: int | None = None
    """For the reasoning pass: token count of ``cot_stage1_prompt``."""

    # Run metadata:
    timestamp_iso: str = ""
    duration_seconds: float = 0.0

    extra: dict[str, Any] = field(default_factory=dict)
    """Per-row provenance: the rank-to-item map, the item set's items, the rank
    pair, and the ladder's generation provenance."""

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrialResult:
        return cls(
            trial_id=d["trial_id"],
            cell_id=d["cell_id"],
            master_seed=d["master_seed"],
            N=d["N"],
            item_pool=d["item_pool"],
            assignment_strategy=d["assignment_strategy"],
            signed_distance=d["signed_distance"],
            symbolic_distance=d["symbolic_distance"],
            correct=d["correct"],
            model=d["model"],
            prompt_spec=dict(d["prompt_spec"]),
            mode=d["mode"],
            predicted=d["predicted"],
            is_correct=d["is_correct"],
            logprob_yes=d["logprob_yes"],
            logprob_no=d["logprob_no"],
            logit_gap=d["logit_gap"],
            cot_trace=d.get("cot_trace"),
            cot_n_tokens=d.get("cot_n_tokens"),
            cot_finish_reason=d.get("cot_finish_reason"),
            prompt=d.get("prompt", ""),
            prompt_n_tokens=d.get("prompt_n_tokens", 0),
            cot_stage1_prompt=d.get("cot_stage1_prompt"),
            cot_stage1_n_tokens=d.get("cot_stage1_n_tokens"),
            timestamp_iso=d.get("timestamp_iso", ""),
            duration_seconds=d.get("duration_seconds", 0.0),
            extra=dict(d.get("extra", {})),
        )


# ---------------------------------------------------------------------------
# JSONL
# ---------------------------------------------------------------------------
#
# One JSON object per line, UTF-8, no trailing comma. Files have a `.jsonl`
# extension; compressed `.jsonl.gz` is accepted on read (transparent
# decompression) but not produced.


def write_jsonl(path: str | Path, items: Iterable[dict[str, Any]]) -> int:
    """Write an iterable of dicts as JSONL. Returns the number of records written.

    Creates parent directories as needed. Overwrites any existing file at
    `path`.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False))
            f.write("\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield dicts from a JSONL file. Transparently decompresses ``.jsonl.gz``."""
    p = Path(path)
    opener = gzip.open if str(p).endswith(".gz") else open
    with opener(p, mode="rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_results(path: str | Path, results: Iterable[TrialResult]) -> int:
    """Write TrialResults as JSONL. Returns number of records written."""
    return write_jsonl(path, (r.to_dict() for r in results))


def read_results(path: str | Path) -> Iterator[TrialResult]:
    """Yield TrialResults from JSONL."""
    for d in read_jsonl(path):
        yield TrialResult.from_dict(d)
