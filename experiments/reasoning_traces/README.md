# reasoning_traces

**Question.** Do the larger reasoning budgets needed at larger distances reflect more
inferences being composed in the chain? (The appendix figure "Composed inferences in LLM
reasoning chains", and the four traces of the appendix's "Example reasoning traces".)

**Design.** The chains of Qwen3-32B and Qwen2.5-32B-Instruct on the ICL test (given names;
`symbolic_distance/budget_icl`) and the reversed US-state-population context
(`content_effects/budget_reversed`). An inference combines two relations sharing an item; a
query at distance d needs d − 1. At each budget, the relations in the chain so far are read and
the number of required inferences they compose is counted, keeping only relations consistent
with the ordering shown and ignoring restatements of the query (`traces.composed_inferences`).
The four example traces are reversed-context trials at distance 4: one correct and one
incorrect per model, the trial whose chain length is closest to the median among chains that
ended within 1,024 tokens and, for a correct trial, composed all three inferences.

No runs of its own: a derived table over the two ladders above.

**Tables.** `results/reasoning_traces/tables/chain_steps.parquet` (31,680 rows),
`trace_examples.json`, `MANIFEST.yaml`.
