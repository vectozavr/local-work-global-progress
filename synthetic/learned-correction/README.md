# Persistent correction learned from verifier residuals

Run `python3 synthetic/learned-correction/run.py` from the project root with NumPy. The script reads the fixed task in `../known-correction/results.json` and writes results.json here. Run `known-correction` first.

The learned policy starts with zero correction and coordinate bounds M=2*weights. It may query the exact scalar verifier on a one-bit flip, subtracts the known local and coupling changes, and applies the resulting residual update to one stored coefficient. It retains coefficients and never probes an identified one again. At most 48 calibration queries are used per trajectory. Each epoch refreshes only known coupling/context, executes H randomized private coordinate steps, verifies proposed singleton gains, and merges the best color class.

Baselines:
- Refresh: identify all 48 coefficients from scalar probes every epoch. No true analytic correction is handed to the baseline without charging its acquisition.
- Identify once: identify all 48 initially, then retain them. This is the stronger static-model baseline and outperforms gradual learning on the reported task.
- No learned unary correction: no residual calibration, but the known coupling term is retained. This differs from the first experiment's purely private-score ablation.

There are 1,000 runs per configuration on one constructed task, seed 20260926 plus a configuration offset. We test H=8,32,64 and learned probe probability p=1,0.5,0.25. No independent task population or AI-model evaluation is claimed. H=8,p=1 has learned modeled cost 444.8, refresh 811.6, identify-once 367.9. All calibrated runs reach the optimum; no uncorrected run does by 256 epochs.

Modeled cost: one parallel private slot costs 1; communication/context/integration costs 20 per epoch; each scalar verifier call costs an additional unit. The base objective is cached, initial evaluation charged once, singleton validation counts only changed candidates, and best-color additivity supplies the committed value. Offline diagnostic calls are not protocol queries. No measured speedup is claimed.

Runs are stopped at the offline-known optimum to report first-hit statistics. This is not a stopping rule available to a general agent. Radius traces describe that stopped policy and are not compared with the continuing joint-contraction theorem. Separate numerical audits verify the projection identity and the coefficient inequalities used in both joint proofs. Model parameters and truth are used by the verifier and offline checks, not to update the learned estimator without a query.
