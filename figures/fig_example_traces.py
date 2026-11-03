"""The appendix's "Example reasoning traces": the four traces (reversed context, US state
population, distance four, 1,024-token budget), one correct and one incorrect trial each of
Qwen3-32B and Qwen2.5-32B-Instruct, each prompt verbatim with its chat template, then the chain
as generated. Written as text to ``figures/out/example_traces.txt``.
Tables: reasoning_traces/trace_examples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _shared import OUT_DIR, RESULTS_ROOT
from geo_alg_ti_llms import roster

NAME = "example_traces"
ORDER = ("qwen3-32b-correct", "qwen3-32b-incorrect", "qwen2.5-32b-instruct-correct",
         "qwen2.5-32b-instruct-incorrect")
RULE = "-" * 100


def build(args: argparse.Namespace) -> str:
    """The four traces as one text, each headed by its model and outcome."""
    path = Path(args.results) / "reasoning_traces" / "tables" / "trace_examples.json"
    examples = {e["id"]: e for e in json.loads(path.read_text(encoding="utf-8"))["examples"]}
    parts = []
    for i, key in enumerate(ORDER, 1):
        e = examples[key]
        outcome = "correct" if e["is_correct"] else "incorrect"
        parts += [f"Trace {i}. {roster.MODEL_LABELS[e['model']]}, {outcome} answer.", RULE,
                  "Prompt", RULE, e["reasoning_prompt"], RULE,
                  f"Reasoning chain ({e['cot_n_tokens']:,} tokens)", RULE, e["trace"], RULE, ""]
    return "\n".join(parts)


def main(argv=None) -> list[Path]:
    """Parse ``--results`` and ``--out`` and write the traces."""
    parser = argparse.ArgumentParser(description="write the four example reasoning traces")
    parser.add_argument("--results", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{NAME}.txt"
    path.write_text(build(args), encoding="utf-8")
    print(path)
    return [path]


if __name__ == "__main__":
    main()
