# Reusing proposal measurements with observable stopping

Run `python3 synthetic/verification-reuse/run.py` from the project root. The fixed task is read from `../known-correction/results.json`, so run `known-correction` first. Results are written here. NumPy is the only external dependency.

We compare dedicated calibration, dedicated calibration plus verification reuse, and identify-once, at H=8,32,64 and p=1. Each configuration has 1,000 seeded trajectories. The seed convention matches the earlier learned-correction experiment. The controls reproduce the earlier completion costs.

For reuse, a proposal is informative when its displacement includes exactly one unobserved coefficient after contributions from identified coefficients have been removed. That coefficient is identified directly from the already-paid proposal value. Both accepted and rejected proposals are eligible. Multi-unknown measurements are skipped. Estimates are updated after the current proposals are fixed, so reuse affects later local searches. Already identified coordinates are removed from future deliberate calibration.

Stopping uses the paper's certified gap upper bound: sum the remaining coordinate radii and exact affine-surrogate improvements, then divide by the analytical dominance constant computed from known unary and coupling weights. The neighbor context is recomputed at the current committed base. The threshold is 1e-9. The true target and optimum enter only task construction, the verifier, and offline diagnostics, not the stopping decision. Tests assert that the actual gap never exceeds the certificate.

C=20 covers modeled synchronization, context loading, integration, residual processing, and stopping-certificate arithmetic. Each parallel private step costs 1 and each counted scalar verifier call costs 1. No additional verifier call is needed for reuse. This does not mean residual processing is computationally free; its hardware cost is not measured. Offline diagnostic calls are excluded from modeled protocol costs.

All 9,000 runs stop with zero certified gap. Reuse saves a small number of queries, with the same mean completion epochs as dedicated-only calibration. Identify-once wins at every tested horizon. This is an additional mechanism check on the same constructed task, not broader empirical validation.

Run `python3 synthetic/verification-reuse/checks.py` for independent finite-state checks of the stopping certificate and dominance bound, the finite-domain geometry characterization, and nonincrease of coefficient error under exact projections.
