# prior_strength

**Question.** Does a stronger prior (items spaced further apart in value) cause more
interference when the context reverses the ordering? (The prior-strength panel of the
content-effects figure.)

**Design.** For each dimension, item sets with the same 11-item geometry but a different
value ratio between the ends of a distance-6 pair (2 to 56.2 instead of the calibrated
ratio of the main item set): 19 item sets in all. Direct answers only: the query alone
(`iwl`) and the three context conditions (`content`), every model.

**Feeds.** The prior-strength panel of the figure "Content effects in transitive inference"
(no-context accuracy against reversed-context accuracy, one point per model and item set).

**Tables.** `results/prior_strength/tables/direct.parquet` (932,976 rows), `pools.parquet`
(one row per item set: value ratio, spacing, provenance file), `MANIFEST.yaml`.

**Provenance.** The same software stack and GPU as `symbolic_distance`.
