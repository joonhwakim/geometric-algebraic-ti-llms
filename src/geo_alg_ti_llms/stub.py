"""A stub model: deterministic answers and chains computed from the prompt text, no torch.

The smoke test and the runner tests drive the real runners, prompt builders, stored-row
writers and exports with it, so the whole pipeline runs on a laptop in seconds. Its
tokenizer is character level with Qwen-like chat markers and a thinking toggle, and the
answer candidates are single ids above the character range.
"""

from __future__ import annotations

from pathlib import Path

from geo_alg_ti_llms.models import STUB_NAME, ModelConfig

STUB_SHORT_NAME = "stub"


class StubTokenizer:
    """Character-level tokenizer with a Qwen-like chat template."""

    name_or_path = STUB_NAME
    chat_template = "plain"
    eos_token_id = 7
    pad_token_id = 7
    padding_side = "right"
    all_special_ids = [7]

    # the answer candidates are single ids (above the character range); all else is chars
    _WORDS = {w: 100000 + i for i, w in enumerate(("True", " True", "true", " true",
                                                  "False", " False", "false", " false"))}

    def encode(self, s, add_special_tokens=False):
        if s in self._WORDS:
            return [self._WORDS[s]]
        return [ord(c) for c in s]

    def decode(self, ids, skip_special_tokens=False):
        words = {v: k for k, v in self._WORDS.items()}
        return "".join(words.get(i, chr(i)) for i in ids
                       if not (skip_special_tokens and i in self.all_special_ids))

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True,
                            enable_thinking=None):
        out = "".join(f"<|{m['role']}|>{m['content']}<|end|>\n" for m in messages)
        out += "<|assistant|>\n"
        if enable_thinking is False:
            out += "<think>\n\n</think>\n\n"
        return out


class StubModel:
    """Stands in for a loaded model; the scoring functions recognize it by type."""

    generation_config = None
    device = "cpu"


def score(model, tokenizer, prompt: str, candidates, *, add_special_tokens: bool = False):
    """``(logprob_true, logprob_false, n_tokens)``: True when the prompt's length is even."""
    n = len(tokenizer.encode(prompt))
    return (-0.5, -1.5, n) if n % 2 == 0 else (-1.5, -0.5, n)


def score_ids(model, ids: list[int], candidates):
    return (-0.2, -1.2, len(ids))


def generate(model, tokenizer, prompts: list[str], *, max_new_tokens: int, **_):
    """One short chain per prompt, cut at the budget, ending with the eos id."""
    out = []
    for p in prompts:
        ids = tokenizer.encode("Let me think.\nA > B\n" + ("x" * 40)) + [tokenizer.eos_token_id]
        out.append((ids[:max_new_tokens], len(tokenizer.encode(p))))
    return out


def model_config() -> ModelConfig:
    """The stub as a model config (a native-thinking model with the default stop string)."""
    return ModelConfig(
        name=STUB_NAME, revision=None, short_name=STUB_SHORT_NAME, dtype="bfloat16",
        chat_template=True, thinking_mode=True, expected_vram_gb=None,
        chat_generation_suffix=None, thinking_suppress_prefix=None,
        thinking_stop_string="</think>", n_gpus=1, device_map="auto", experts_implementation=None,
    )


def write_config(models_dir: Path) -> Path:
    """``<models_dir>/stub.yaml``, so the exports can look the stub up by short name."""
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"{STUB_SHORT_NAME}.yaml"
    path.write_text(f"name: {STUB_NAME}\nshort_name: {STUB_SHORT_NAME}\ndtype: bfloat16\n"
                    "chat_template: true\nthinking_mode: true\n", encoding="utf-8")
    return path
