# Approximate-correction study (error floor)

Run `python3 synthetic/approximate-correction/run.py` from the project root with NumPy
installed. It takes about 40 seconds and writes `results.json` in this directory. No LLM
calls are made; all quantities are modeled.

The task, solver, singleton verification and best-color merge are those of
`synthetic/known-correction/run.py`; the task (targets, weights, private targets, activity
rates) is read from that study's `results.json`, whose SHA-256 is recorded. The only change is
the correction: agents search the exact conditional coefficients plus a fixed bias
`eps_bj = eta * s_bj` with random signs `s_bj` drawn once per run and unknown to the agents.
The corrected objective stays anchored.

Settings: `eta` in {0, 0.25, ..., 2}, `H` in {8, 32}, 2,000 runs per point, 160 epochs, seed
20261009 (offset by `H` and `eta`). For every search the script records the two-point errors of
Theorem 2 (at a true block minimizer and at the returned proposal), the realized fraction `Q` of
the corrected improvement and the exact block improvement. It asserts, for every run and epoch,
the pathwise local-gain inequality `g_b >= max(0, Q (D_b - e*_b) - e^U_b)`, the merge inequality,
and that the observed gap never exceeds the propagated recurrence envelope.

Result: below the smallest exact coefficient magnitude (about 0.61) the bias never changes a
decision and all runs reach the optimum. Larger biases leave a floor that grows with `eta`
(H=8: 0.30 at 0.75, 3.08 at 1.0, 30.0 at 2.0) and is the same for H=32 (0.27, 2.79, 29.8). The
recurrence is exact for `eta=0` and upper-bounds the floor in every run.
