"""Large language model experiments for the paper "Geometric and algebraic inference in humans and
neural networks".

One prompt format (an instruction naming the dimension, ``A > B``
premise lines, ``Is it true or false that A > B?``), one trial geometry
(11 items, every non-terminal pair at every symbolic distance, both
polarities), and a direct True/False read-out or a reasoning pass under a
token budget. The experiments are the IWL and ICL tests across symbolic
distance, the content effects of a congruent, reversed, shuffled or
uninformative context, and the item-spacing (prior-strength) item sets; the
representation and component analyses build on the same runs.
"""

__version__ = "0.1.0"
