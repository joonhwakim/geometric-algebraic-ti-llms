"""The reasoning-trace analysis: the relation reader and the composed-inference counter on
hand-built chains, and the committed chain-steps table and example traces."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from geo_alg_ti_llms import tables, traces
from geo_alg_ti_llms.traces import (
    SELF_TEST,
    SELF_TEST_ITEMS,
    _chains,
    _placeholder_subs,
    composed_inferences,
    normalize_tracked,
    pick,
    self_test,
)

_ROOT = Path(__file__).resolve().parent.parent
TABLES = _ROOT / "results" / traces.EXPERIMENT / "tables"

# The ordering shown in the reversed countries context of the methods notes (rank 0 lowest):
# the premises read "Jamaica > Panama > Singapore > Belgium > Netherlands > Ukraine > Spain >
# Germany > Mexico > Brazil > United States". The query Belgium > Spain is ranks 7 vs 4 and
# needs two inferences.
ITEMS = SELF_TEST_ITEMS
BELGIUM, NETHERLANDS, UKRAINE, SPAIN = 7, 6, 5, 4


# --- the relation reader -----------------------------------------------------------------


def _runs(text, items=ITEMS):
    return [(its, ops) for its, ops, _s, _e in _chains(normalize_tracked(text, items).text)]


def test_placeholders_take_the_longest_alias_first():
    norm = normalize_tracked("the United States of America > Brazil; the US > Mexico; USA > Spain",
                             ITEMS).text
    assert norm.count("⟦0⟧") == 3 and "America" not in norm and "US" not in norm
    assert [its for its, _ in _runs("USA > Brazil")] == [[0, 1]]
    # A short upper-case alias needs letter boundaries only, but the full stop after it is
    # not whitespace, so "US. > Brazil" is no run.
    assert "\u27e60\u27e7" in normalize_tracked("US. > Brazil", ITEMS).text
    assert _runs("US. > Brazil") == []


def test_placeholders_do_not_match_inside_words():
    assert _runs("Spainish > Ukraine") == []
    assert _runs("Spain > Ukraine") == [([SPAIN, UKRAINE], [">"])]


def test_markup_and_verbal_comparatives_become_operators():
    assert _runs("**Belgium** has a larger population than Ukraine.") == [([BELGIUM, UKRAINE], [">"])]
    assert _runs("Belgium is smaller than Ukraine") == [([BELGIUM, UKRAINE], ["<"])]
    assert _runs("Belgium is above Ukraine, Ukraine ranks below Spain") == \
        [([BELGIUM, UKRAINE], [">"]), ([UKRAINE, SPAIN], ["<"])]
    assert _runs(r"$\text{Belgium} \gt \text{Ukraine}$") == [([BELGIUM, UKRAINE], [">"])]
    assert _runs("Belgium &gt; Ukraine ≥ Spain") == [([BELGIUM, UKRAINE, SPAIN], [">", ">"])]
    assert _runs("<think>Belgium > Ukraine</think>") == [([BELGIUM, UKRAINE], [">"])]


def test_a_run_needs_only_whitespace_between_its_tokens():
    assert _runs("Belgium > Netherlands > Ukraine > Spain") == \
        [([BELGIUM, NETHERLANDS, UKRAINE, SPAIN], [">", ">", ">"])]
    # A word between two relations splits the run; a negation blocks the operator.
    assert _runs("Belgium > Netherlands and Netherlands > Ukraine") == \
        [([BELGIUM, NETHERLANDS], [">"]), ([NETHERLANDS, UKRAINE], [">"])]
    assert _runs("Netherlands does not have a larger population than Spain.") == []
    assert _runs("Belgium is more populous.") == []


def test_tracked_offsets_map_back_into_the_raw_text():
    text = "So **Belgium** has a larger population than Ukraine."
    t = normalize_tracked(text, ITEMS)
    assert t.map[len(t.text)] == len(text)
    assert all(a <= b for a, b in zip(t.map, t.map[1:]))
    (_its, _ops, start, end), = list(_chains(t.text))
    # A replacement's characters spread linearly over the span they replace, so the run's
    # ends map inside the item names it starts and ends with.
    lo, hi = text.index("Belgium"), text.index("Ukraine")
    assert lo <= t.map[start] <= lo + len("Belgium")
    assert hi <= t.map[end] <= hi + len("Ukraine")


def test_placeholder_substitutions_are_longest_alias_first():
    subs = _placeholder_subs(["United States", "Brazil"])
    first = subs[0][0].pattern
    assert "United\\ States\\ of\\ America" in first


# --- the composed-inference counter ----------------------------------------------------------


def test_self_test_sentences_give_their_stated_counts():
    assert len(self_test()) == len(SELF_TEST) == 11


@pytest.mark.parametrize("text,lo,hi,want", [
    # The methods notes' test sentences, query Belgium > Spain (ranks 7 vs 4; needs 2).
    ("Ukraine > Spain", SPAIN, BELGIUM, 0),                         # a premise restated
    ("Spain < Ukraine", SPAIN, BELGIUM, 0),
    ("Ukraine has a larger population than Spain.", SPAIN, BELGIUM, 0),
    ("Netherlands > Spain", SPAIN, BELGIUM, 1),                     # one inference, one short
    ("Spain > Netherlands", SPAIN, BELGIUM, 0),                     # against the ordering
    ("Netherlands > Ukraine > Spain", SPAIN, BELGIUM, 1),           # a run links all its pairs
    ("Belgium > Netherlands > Ukraine > Spain", SPAIN, BELGIUM, 2),
    ("Spain < Ukraine < Netherlands < Belgium", SPAIN, BELGIUM, 2),
    ("Belgium > Netherlands > Spain", SPAIN, BELGIUM, 2),
    ("Belgium > Ukraine > Spain", SPAIN, BELGIUM, 2),
    ("Therefore, the statement Spain > Belgium is false.", SPAIN, BELGIUM, 0),
    ("So it is false that Spain > Belgium.", SPAIN, BELGIUM, 0),
    ("So Belgium > Spain.", SPAIN, BELGIUM, 2),                     # the queried relation, derived
    ("Belgium has a larger population than Spain.", SPAIN, BELGIUM, 2),
    ("Netherlands does not have a larger population than Spain.", SPAIN, BELGIUM, 0),
    ("Belgium is more populous.", SPAIN, BELGIUM, 0),
    ("Spain > Belgium", SPAIN, BELGIUM, 0),
    ("Jamaica > Panama > Singapore > Belgium > Netherlands > Ukraine > Spain > Germany > Mexico "
     "> Brazil > United States", SPAIN, BELGIUM, 2),                # the whole ordering
    ("**Mexico > Brazil:** This means Brazil has a smaller population than Mexico.",
     SPAIN, BELGIUM, 0),                                            # unrelated to the query
    ("We know Netherlands > Ukraine, and Ukraine > Spain, so Netherlands > Spain.",
     SPAIN, BELGIUM, 1),
    ("Belgium > Ukraine > Netherlands > Spain", SPAIN, BELGIUM, 1),  # a wrong link voids the run
    # The queried pair after a question cue is a restatement, not an inference.
    ("We need to determine whether Belgium > Spain. Belgium > Netherlands.", SPAIN, BELGIUM, 0),
    ("Is Belgium > Spain? Belgium > Netherlands. Netherlands > Spain.", SPAIN, BELGIUM, 2),
    # Pieces far apart in the chain combine; a distance-1 query needs nothing.
    ("First Belgium > Ukraine. ... Much later Ukraine > Germany.", 3, BELGIUM, 2),
    ("Belgium > Netherlands", NETHERLANDS, BELGIUM, 0),
])
def test_composed_inferences_on_hand_written_chains(text, lo, hi, want):
    assert composed_inferences(text, ITEMS, lo, hi) == want


def test_composed_inferences_bounds():
    assert composed_inferences("", ITEMS, SPAIN, BELGIUM) == 0
    long = " ".join(f"{ITEMS[i + 1]} > {ITEMS[i]}" for i in range(10))     # every premise
    assert composed_inferences(long, ITEMS, 1, 9) == 0
    assert composed_inferences("Panama > Brazil", ITEMS, 1, 9) == 7         # needs 7, one link


def test_pick_takes_the_chain_nearest_the_median_then_the_lower_trial_id():
    group = [{"trial_id": 3, "cot_n_tokens": 100.0}, {"trial_id": 1, "cot_n_tokens": 300.0},
             {"trial_id": 2, "cot_n_tokens": 200.0}, {"trial_id": 0, "cot_n_tokens": 240.0},
             {"trial_id": 4, "cot_n_tokens": 180.0}]
    r, median = pick(group)
    assert median == 200.0 and r["trial_id"] == 2
    even = group[:4]                       # median 220: 200 and 240 tie, the lower id wins
    assert pick(even) == (group[3], 220.0)
    tie = [{"trial_id": 5, "cot_n_tokens": 10.0}, {"trial_id": 4, "cot_n_tokens": 30.0}]
    assert pick(tie)[0]["trial_id"] == 4


# --- the committed tables -----------------------------------------------------------------


@pytest.fixture(scope="module")
def manifest():
    return tables.load_manifest(TABLES / "MANIFEST.yaml")


@pytest.fixture(scope="module")
def chain_steps():
    return traces.read_chain_steps()


@pytest.fixture(scope="module")
def examples():
    return traces.read_trace_examples()


def test_committed_tables_match_their_manifest(manifest):
    assert tables.verify_manifest(TABLES) == []
    assert manifest["package_version"] == tables.PACKAGE_VERSION
    assert manifest["experiment"] == traces.EXPERIMENT
    assert {t["table"] for t in manifest["tables"]} == {"chain_steps", "trace_examples"}
    assert set(manifest["derived_from"]) == {test for _e, test, _d in traces.LADDERS}
    assert manifest["models"] == list(traces.MODELS)


def test_chain_steps_table(manifest, chain_steps):
    df = chain_steps
    entry = next(t for t in manifest["tables"] if t["table"] == "chain_steps")
    assert list(df.columns) == traces.COLUMNS == list(entry["columns"])
    assert {c: str(df[c].dtype) for c in df.columns} == traces.DTYPES
    budgets = manifest["ladder_budgets"]
    assert len(df) == len(traces.MODELS) * len(traces.LADDERS) * traces.TRIALS_PER_LADDER * len(budgets)
    assert sorted(df["condition"].unique()) == ["icl_people", "reversed"]
    assert sorted(df["model"].unique()) == sorted(traces.MODELS)
    assert not df.duplicated(["model", "condition", "trial_key", "budget"]).any()
    assert (df["inferences_needed"] == df["d"] - 1).all()
    assert (df.loc[df["budget"] == 0, "inferences_composed"] == 0).all()
    assert (df["inferences_composed"] >= 0).all() and (df["inferences_composed"] <= df["inferences_needed"]).all()
    per_trial = df.groupby(["model", "condition", "trial_key"])["budget"].apply(lambda s: sorted(s))
    assert all(b == budgets for b in per_trial)
    assert entry["checks"]["self_test"] == self_test()
    assert sum(n for n, _ in entry["trial_sets"].values()) == len(df)
    for key, (n, digest) in entry["trial_sets"].items():
        model, condition, dim = key.split("|")
        g = df[(df["model"] == model) & (df["condition"] == condition) & (df["dim"] == dim)]
        assert len(g) == n and tables.row_digest(g) == digest


def test_chain_steps_innermost_counts_are_the_trial_grid(chain_steps):
    inner = chain_steps[chain_steps["innermost"] & (chain_steps["budget"] == 1024)]
    n = inner.groupby(["model", "condition", "d"]).size()
    for (_m, _c, d), count in n.items():
        assert count == 2 * 10 * (7 - d)       # two polarities x ten premise orders x pairs


def test_trace_examples_are_the_selected_trials(manifest, chain_steps, examples):
    entry = next(t for t in manifest["tables"] if t["table"] == "trace_examples")
    ids = [e["id"] for e in examples["examples"]]
    assert ids == [f"{m}-{o}" for m in traces.MODELS for o in ("correct", "incorrect")]
    assert [e["id"] for e in entry["examples"]] == ids
    assert examples["selection"]["budget"] == traces.EXAMPLE_BUDGET == 1024
    assert examples["selection"]["distance"] == traces.EXAMPLE_DISTANCE == 4
    for e, summary in zip(examples["examples"], entry["examples"]):
        assert e["condition"] == "reversed" and e["test"] == "budget_reversed"
        assert e["d"] == 4 and e["inferences_needed"] == 3
        assert e["rank_low"] >= tables.DROP_LO and e["rank_high"] <= tables.DROP_HI
        assert e["cot_finish_reason"] != "length" and e["cot_n_tokens"] <= 1024
        assert e["is_correct"] == (e["scored_answer_true"] == e["query_true_in_ordering_shown"])
        assert e["is_correct"] == e["id"].endswith("-correct")
        if e["is_correct"]:
            assert e["inferences_composed"] == 3
        # The strings are verbatim: their digests, the scored prompt's composition, the query.
        assert hashlib.sha256(e["trace"].encode()).hexdigest() == e["sha256_trace"] == summary["sha256_trace"]
        scored = e["reasoning_prompt"] + e["trace"] + e["scoring_suffix"]
        assert hashlib.sha256(scored.encode()).hexdigest() == e["sha256_scored_prompt"]
        assert "Therefore the answer is:" in e["scoring_suffix"]
        assert tables.CTXONLY_SENTENCE in e["reasoning_prompt"]
        assert f"Is it true or false that {e['query']}?" in e["reasoning_prompt"]
        left, right = e["query"].split(" > ")
        order = e["context_ordering"]
        assert {left, right} == {order[e["rank_low"]], order[e["rank_high"]]}
        assert e["query_true_in_ordering_shown"] == (order.index(left) > order.index(right))
        # The count the table records is the count of the chain as copied.
        assert composed_inferences(e["trace"], order, e["rank_low"], e["rank_high"]) == e["inferences_composed"]
        row = chain_steps[(chain_steps["model"] == e["model"]) & (chain_steps["trial_key"] == e["trial_key"])
                          & (chain_steps["budget"] == 1024)]
        assert len(row) == 1 and int(row["inferences_composed"].iloc[0]) == e["inferences_composed"]
        assert bool(row["is_correct"].iloc[0]) == e["is_correct"]
    prompted = {e["model"]: e["reasoning_prompt"].split("<|im_end|>\n<|im_start|>assistant")[0]
                .rstrip().endswith(traces.TRIGGER) for e in examples["examples"]}
    assert prompted == {"qwen3-32b": False, "qwen2.5-32b-instruct": True}


# --- against the published runs (where present) ---------------------------------------------

PUBLISHED = os.environ.get(tables.RUNS_ENV)


@pytest.mark.skipif(not PUBLISHED, reason=f"{tables.RUNS_ENV} not set")
def test_examples_are_the_stored_rows(examples):
    for e in examples["examples"]:
        raw = json.loads(list(open(Path(PUBLISHED) / e["source_file"]))[e["source_line"]])
        assert raw["cot_trace"] == e["trace"] and raw["cot_stage1_prompt"] == e["reasoning_prompt"]
        assert raw["prompt"] == e["reasoning_prompt"] + e["trace"] + e["scoring_suffix"]
        assert int(raw["trial_id"]) == e["trial_id"] and bool(raw["is_correct"]) == e["is_correct"]
        assert list(raw["extra"]["rank_to_item"]) == e["context_ordering"]
