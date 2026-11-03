"""The 22 models and the five dimensions as the figures name and order them.

Pure data. Keys are the values the tidy tables carry in their ``model`` and ``dim`` columns.

- ``MODELS`` is the roster in a fixed order; ``MODEL_LABELS`` gives each model its published
  name and ``MODEL_ROW_LABELS`` the same name broken over at most two lines (at a hyphen,
  which the first line keeps) for the left gutter of a per-model page.
- ``PARAMETERS_B`` gives each model's size in billions of parameters.
- ``REASONING_KIND``: "native" models reason in their own thinking block, "prompted" models
  are asked to ("Let's think step by step."); the tables carry the same value.
- ``PAGES`` splits the roster into the four pages every per-model appendix figure shares:
  the native models first, then the prompted ones, each page by family with the largest model
  of a family first. The pages partition the roster.
- ``DIMENSIONS`` are the five real-world dimensions the figures draw, in column order, with
  ``DIMENSION_LABELS`` and the wrapped ``DIMENSION_HEADERS``; ``PRIMARY_SETS`` names the main
  item set of each. ``cities`` (constant-ratio spacing) is the city population set of the
  mechanism runs only.
"""

from __future__ import annotations

MODELS: tuple[str, ...] = (
    "qwen3-32b", "qwen3-14b", "qwen3-8b", "qwen3-4b", "qwen3-1.7b",
    "qwen2.5-32b-instruct", "qwen2.5-7b-instruct", "glm-4-32b-0414",
    "deepseek-r1-distill-llama-8b", "phi-4-reasoning", "llama-3.1-70b-instruct",
    "llama-3.1-8b-instruct", "llama-3.2-3b", "llama-3.2-1b", "gemma-2-9b-it", "glm-4-9b-0414",
    "mistral-7b-instruct-v0.3", "olmo-2-32b-instruct", "olmo-2-7b-instruct",
    "deepseek-r1-distill-qwen-32b", "deepseek-r1-distill-qwen-14b", "gpt-oss-20b",
)

MODEL_ROW_LABELS: dict[str, str] = {
    "qwen3-32b": "Qwen3-32B",
    "qwen3-14b": "Qwen3-14B",
    "qwen3-8b": "Qwen3-8B",
    "qwen3-4b": "Qwen3-4B",
    "qwen3-1.7b": "Qwen3-1.7B",
    "qwen2.5-32b-instruct": "Qwen2.5-32B-\nInstruct",
    "qwen2.5-7b-instruct": "Qwen2.5-7B-\nInstruct",
    "glm-4-32b-0414": "GLM-4-32B-0414",
    "deepseek-r1-distill-llama-8b": "DeepSeek-R1-\nDistill-Llama-8B",
    "phi-4-reasoning": "Phi-4-reasoning",
    "llama-3.1-70b-instruct": "Llama-3.1-70B-\nInstruct",
    "llama-3.1-8b-instruct": "Llama-3.1-8B-\nInstruct",
    "llama-3.2-3b": "Llama-3.2-3B",
    "llama-3.2-1b": "Llama-3.2-1B",
    "gemma-2-9b-it": "Gemma-2-9B-it",
    "glm-4-9b-0414": "GLM-4-9B-0414",
    "mistral-7b-instruct-v0.3": "Mistral-7B-\nInstruct-v0.3",
    "olmo-2-32b-instruct": "OLMo-2-32B-\nInstruct",
    "olmo-2-7b-instruct": "OLMo-2-7B-\nInstruct",
    "deepseek-r1-distill-qwen-32b": "DeepSeek-R1-\nDistill-Qwen-32B",
    "deepseek-r1-distill-qwen-14b": "DeepSeek-R1-\nDistill-Qwen-14B",
    "gpt-oss-20b": "gpt-oss-20b",
}
# A row label closes up into the one-line name (the break follows a hyphen it keeps).
MODEL_LABELS: dict[str, str] = {m: label.replace("\n", "") for m, label in MODEL_ROW_LABELS.items()}

# Parameters, in billions, as each model's release states them.
PARAMETERS_B: dict[str, float] = {
    "qwen3-32b": 32.0, "qwen3-14b": 14.0, "qwen3-8b": 8.0, "qwen3-4b": 4.0, "qwen3-1.7b": 1.7,
    "qwen2.5-32b-instruct": 32.0, "qwen2.5-7b-instruct": 7.0, "glm-4-32b-0414": 32.0,
    "deepseek-r1-distill-llama-8b": 8.0, "phi-4-reasoning": 14.0, "llama-3.1-70b-instruct": 70.0,
    "llama-3.1-8b-instruct": 8.0, "llama-3.2-3b": 3.0, "llama-3.2-1b": 1.0, "gemma-2-9b-it": 9.0,
    "glm-4-9b-0414": 9.0, "mistral-7b-instruct-v0.3": 7.0, "olmo-2-32b-instruct": 32.0,
    "olmo-2-7b-instruct": 7.0, "deepseek-r1-distill-qwen-32b": 32.0,
    "deepseek-r1-distill-qwen-14b": 14.0, "gpt-oss-20b": 21.0,
}

_NATIVE = {"qwen3-32b", "qwen3-14b", "qwen3-8b", "qwen3-4b", "qwen3-1.7b",
           "deepseek-r1-distill-llama-8b", "deepseek-r1-distill-qwen-32b",
           "deepseek-r1-distill-qwen-14b", "phi-4-reasoning", "gpt-oss-20b"}
REASONING_KIND: dict[str, str] = {m: "native" if m in _NATIVE else "prompted" for m in MODELS}

PAGES: dict[str, tuple[str, ...]] = {
    "qwen3": ("qwen3-32b", "qwen3-14b", "qwen3-8b", "qwen3-4b", "qwen3-1.7b"),
    "deepseek_phi_gptoss": ("deepseek-r1-distill-qwen-32b", "deepseek-r1-distill-qwen-14b",
                            "deepseek-r1-distill-llama-8b", "phi-4-reasoning", "gpt-oss-20b"),
    "qwen25_glm_gemma_mistral": ("qwen2.5-32b-instruct", "qwen2.5-7b-instruct", "glm-4-32b-0414",
                                 "glm-4-9b-0414", "gemma-2-9b-it", "mistral-7b-instruct-v0.3"),
    "llama_olmo": ("llama-3.1-70b-instruct", "llama-3.1-8b-instruct", "llama-3.2-3b",
                   "llama-3.2-1b", "olmo-2-32b-instruct", "olmo-2-7b-instruct"),
}

DIMENSIONS: tuple[str, ...] = ("countries", "citieslinear", "animals", "usstates", "countryarea")
DIMENSION_LABELS: dict[str, str] = {
    "countries": "Country population",
    "citieslinear": "City population",
    "animals": "Animal body mass",
    "usstates": "US state population",
    "countryarea": "Country surface area",
    "cities": "City population",
}
DIMENSION_HEADERS: dict[str, str] = {
    "countries": "Country\npopulation",
    "citieslinear": "City population",
    "animals": "Animal body mass",
    "usstates": "US state\npopulation",
    "countryarea": "Country surface\narea",
}
PRIMARY_SETS: dict[str, str] = {
    "countries": "countries_by_population_11",
    "citieslinear": "cities_by_population_11_linear",
    "animals": "animals_by_mass_11",
    "usstates": "us_states_by_population_11",
    "countryarea": "countries_by_area_11",
}

# The ICL test's two sets of novel items; the given names are the single-model panels' set.
ICL_CONDITIONS: tuple[str, ...] = ("icl_people", "icl_cities")
ICL_CONDITION: str = "icl_people"
IWL_CONDITION: str = "no_context"
# The ICL test was run with the country population runs; its rows carry that dimension.
ICL_DIMENSION: str = "countries"

assert len(set(MODELS)) == 22
assert set(PARAMETERS_B) == set(MODELS)
assert sorted(m for page in PAGES.values() for m in page) == sorted(MODELS)
assert all(len({REASONING_KIND[m] for m in page}) == 1 for page in PAGES.values())
