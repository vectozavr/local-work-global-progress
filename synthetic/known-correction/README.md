# Exact correction: convergence and the private-work horizon


Run `python3 synthetic/known-correction/run.py` from the project root with NumPy installed. It writes results.json in this directory. It uses fixed seed 20260925, one six-block/eight-bit task, and 2,000 random solver trajectories per horizon. No LLM calls are made.

The exact global score is a weighted binary target mismatch plus pairwise disagreements on a path. Alternate agents' cheap private scores prefer the wrong target. A cached coefficient correction exactly aligns their score with the global conditional score while neighbors remain frozen. The correction depends on candidate state; a constant offset cannot fix the local preference. All outputs undergo global singleton verification and best-color merging.

Outputs include the task, weights, activity probabilities, exact finite-state error-bound certificate, correction audits, per-horizon mean completion epochs and standard errors, full mean-gap trajectories, a failed uncorrected ablation, and modeled costs for shared overheads 0, 5, 20 and 100. Zero overhead is an idealized diagnostic. One local slot costs 1 in parallel, shared overhead C includes correction preparation and transfer, verification and integration, and its actual hardware cost is not measured. Total private work scales with all six agents' slots, so no token-saving claim is made.

At overhead 20, costs decrease from 1296.4 at H=1 to 176.7 at H=64 and then rise to 296.0 at H=128. This is the observed minimum over the eight tested horizons, not a population optimum. The same runtime random seed is reset with a deterministic horizon offset; these are seeded within-instance comparisons, not new independent task draws. The uncorrected H=8 case solves none of the 2,000 runs by 128 epochs and leaves mean objective gap 30.979.

Important limitation: the constructed objective is simple enough to solve centrally. These measurements illustrate the correction mechanism and communication trade-off for a stipulated randomized local-search oracle; they do not establish competitiveness with centralized optimization or effectiveness for LLM agents.
