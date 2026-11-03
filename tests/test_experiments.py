"""The experiment declarations resolve to the configs of the published runs."""

from __future__ import annotations

import pytest

from geo_alg_ti_llms import experiments
from geo_alg_ti_llms.items import load_item_set
from geo_alg_ti_llms.models import load_model_config, model_config_path
from geo_alg_ti_llms.run import load_geometry, load_ladder_budgets

QWEN3 = load_model_config(model_config_path("qwen3-32b"))
GEMMA = load_model_config(model_config_path("gemma-2-9b-it"))
BUDGETS = [32, 64, 128, 256, 384, 512, 640, 768, 896, 1024]


def test_three_experiments_are_declared():
    assert experiments.list_experiments() == ["content_effects", "prior_strength", "symbolic_distance"]


def test_symbolic_distance_tests_and_item_sets():
    exp = experiments.load_experiment("symbolic_distance")
    assert list(exp["tests"]) == ["iwl", "icl", "icl_cities", "budget_iwl", "budget_icl", "budget_icl_cities"]
    assert experiments.test_set_keys(exp, "iwl") == {
        "countries": ("countries", "countries_by_population_11"),
        "countries_rep2": ("countries", "countries_by_population_11_rep2"),
        "countries_rep3": ("countries", "countries_by_population_11_rep3"),
        "cities": ("cities", "cities_by_population_11"),
        "cities_rep2": ("cities", "cities_by_population_11_rep2"),
        "citieslinear": ("citieslinear", "cities_by_population_11_linear"),
        "animals": ("animals", "animals_by_mass_11"),
        "animals_rep2": ("animals", "animals_by_mass_11_rep2"),
        "usstates": ("usstates", "us_states_by_population_11"),
        "usstates_rep2": ("usstates", "us_states_by_population_11_rep2"),
        "usstates_rep3": ("usstates", "us_states_by_population_11_rep3"),
        "countryarea": ("countryarea", "countries_by_area_11"),
        "countryarea_rep2": ("countryarea", "countries_by_area_11_rep2"),
        "countryarea_rep3": ("countryarea", "countries_by_area_11_rep3"),
    }
    assert experiments.test_set_keys(exp, "icl") == {"countries": ("countries", "novel_names_11")}
    assert experiments.test_set_keys(exp, "budget_icl_cities") == {"countries": ("countries", "novel_cities_11")}
    assert len(experiments.jobs(exp)) == 14 + 1 + 1 + 14 + 1 + 1


def test_content_effects_and_prior_strength_jobs():
    ce = experiments.load_experiment("content_effects")
    assert list(ce["tests"]) == ["content", "uninformative", "budget_natural", "budget_reversed", "budget_shuffled"]
    assert len(experiments.jobs(ce)) == 5 * 6
    ps = experiments.load_experiment("prior_strength")
    assert list(ps["tests"]) == ["iwl", "content"]
    keys = experiments.test_set_keys(ps, "content")
    assert len(keys) == 19
    assert keys["countries_scale3p16"] == ("countries", "countries_by_population_scale3p16")
    assert keys["countryarea_scale56p2"] == ("countryarea", "countries_by_area_scale56p2")


def test_every_declared_item_set_exists():
    for name in experiments.list_experiments():
        exp = experiments.load_experiment(name)
        for test in exp["tests"]:
            for key, (dimension, item_set) in experiments.test_set_keys(exp, test).items():
                spec = experiments.resolve(exp, key, test, QWEN3)
                load_item_set(spec.geometry["prompt"]["item_pool"])


def test_iwl_resolves_to_the_published_geometry():
    exp = experiments.load_experiment("symbolic_distance")
    spec = experiments.resolve(exp, "countries", "iwl", QWEN3)
    assert spec.runner == "direct" and spec.ladder is None
    assert spec.conditions == {"natural": "no_context"}
    assert spec.published_as == "iwl"
    assert spec.geometry == {
        "master_seed": 42,
        "prompt": {"template_id": "T2", "item_pool": "countries_by_population_11",
                   "instructions_v1_id": "iwl_population_T2",
                   "query_wrapper_id": "is_it_true_or_false", "separator": "\n"},
        "grid": {"N_values": [11], "hop_values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
        "all_pairs": True, "n_premise_orders": 1,
        "assignment_strategies": ["natural"], "surface_form_modes": ["zero_context"],
    }
    load_geometry(spec.geometry)


def test_content_resolves_to_the_context_only_instruction():
    exp = experiments.load_experiment("content_effects")
    spec = experiments.resolve(exp, "animals", "content", QWEN3)
    assert spec.conditions == {"natural": "congruent", "reversed": "reversed", "random": "shuffled"}
    assert spec.geometry["prompt"]["instructions_v1_id"] == "iwl_mass_T2_ctxonly"
    assert spec.geometry["assignment_strategies"] == ["natural", "reversed", "random"]
    assert spec.geometry["surface_form_modes"] == ["fixed_gt"]
    assert spec.geometry["n_premise_orders"] == 10
    assert spec.published_as == "content_ctxonly"
    un = experiments.resolve(exp, "countryarea", "uninformative", QWEN3)
    assert un.geometry["prompt"] == {"template_id": "T2", "item_pool": "countries_by_area_11_disjoint",
                                     "instructions_v1_id": "iwl_area_T2_ctxonly",
                                     "query_wrapper_id": "is_it_true_or_false", "separator": "\n"}
    assert un.geometry["surface_form_modes"] == ["disjoint_context"]


def test_novel_item_tests_use_their_own_item_set_and_instruction():
    exp = experiments.load_experiment("symbolic_distance")
    icl = experiments.resolve(exp, "countries", "icl", QWEN3)
    assert icl.geometry["prompt"]["item_pool"] == "novel_names_11"
    assert icl.geometry["prompt"]["instructions_v1_id"] == "iwl_height_T2"
    assert icl.geometry["assignment_strategies"] == ["random"]
    cities = experiments.resolve(exp, "countries", "icl_cities", QWEN3)
    assert cities.geometry["prompt"]["item_pool"] == "novel_cities_11"
    assert cities.geometry["prompt"]["instructions_v1_id"] == "iwl_population_T2"


def test_ladder_trigger_depends_on_the_model():
    exp = experiments.load_experiment("content_effects")
    q3 = experiments.resolve(exp, "countries", "budget_reversed", QWEN3)
    gm = experiments.resolve(exp, "countries", "budget_reversed", GEMMA)
    assert q3.ladder["defaults"]["trigger_id"] == "none"
    assert gm.ladder["defaults"]["trigger_id"] == "passive"
    assert list(q3.ladder["budgets"]) == [f"b{b}" for b in BUDGETS]
    trial_sets = load_ladder_budgets(q3.ladder)
    assert trial_sets["b32"] == {"trigger_id": "none", "max_new_tokens": 32,
                            "instructions_v1_id": None, "answer_family": None}
    assert q3.geometry["assignment_strategies"] == ["reversed"]
    with pytest.raises(ValueError, match="model config"):
        experiments.resolve(exp, "countries", "budget_reversed", None)


def test_budget_iwl_uses_one_premise_order_like_iwl():
    exp = experiments.load_experiment("symbolic_distance")
    assert experiments.resolve(exp, "cities_rep2", "budget_iwl", QWEN3).geometry["n_premise_orders"] == 1
    assert experiments.resolve(exp, "countries", "budget_icl", QWEN3).geometry["n_premise_orders"] == 10


def test_unknown_test_or_item_set_raise():
    exp = experiments.load_experiment("symbolic_distance")
    with pytest.raises(ValueError, match="no test"):
        experiments.resolve(exp, "countries", "content", QWEN3)
    with pytest.raises(ValueError, match="does not run on item set"):
        experiments.resolve(exp, "cities", "icl", QWEN3)
