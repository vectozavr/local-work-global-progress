# Records of the robot-routing runs of the paper

`routing-runs/` has one folder per campaign of Section 8.1 and Appendix E (maps 1 and 2, two floors). Each holds the
campaign plan and map, the final result of every run and the summaries written by the independent replay
(`summary.json`, or `analysis-v2/summary.json` for teams). Map 1 seeds 10–19 were run later with the same code and
settings and have their own folders (`...-seeds10-19-...`).

`routing-theory/` holds the constants of the theory measured on the team campaigns by
`routing/analysis/theory/audit.py`: per-round coefficients and the round-by-round bound (`trajectory-rate.json`) and
the summaries of the exact block minima, progress fractions and correction errors (`assumptions-*.json`).

These records are what `routing/analysis/run_all.py` reads. The full records, with every prompt, model answer,
returned reasoning and raw API response (3.4 GB), are attached to the [v1.0 release](https://github.com/vectozavr/local-work-global-progress/releases/tag/v1.0) as
`routing-runs-v2.tar.gz`; unpack it into `routing-runs/` to replay the runs or repeat the audit of the constants.
