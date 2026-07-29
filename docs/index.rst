sortition
=========

Counterfactual evaluation for LLM routing policies.

Every LLM router picks a model. None of them can tell you whether it picked
well, because answering that requires the counterfactual: what would the other
model have cost and scored? Sortition treats routing as policy learning. It logs
propensities, explores on a small traffic slice, and ships the estimators that
turn those logs into valid claims about policies that were never deployed.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   readme

The two halves
--------------

``sortition.eval``
   The estimators, diagnostics and confidence intervals. Works on any log that
   carries a propensity, from any gateway. This is the part usable without
   changing the router.

``sortition.decide``
   A reference policy that produces such logs, shipped as an in-process LiteLLM
   routing plugin. Sub-millisecond, holds no credentials, executes no calls.

Why propensities
----------------

Randomization without recorded propensities is wasted entropy. A router that
samples across models perturbs production traffic and buys no inferential value
unless it records the probability with which it made each choice. That one
number is what makes a log answerable.

Indices
-------

* :ref:`genindex`
* :ref:`search`
