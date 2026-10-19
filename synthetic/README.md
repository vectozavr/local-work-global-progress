# Synthetic task with certified constants

Section 8.3 and Appendix G of the paper. Six blocks of eight binary variables interact along a path; the private
score of blocks 1, 3 and 5 prefers the wrong target, and the correction removes this mismatch. Every constant of
Theorem 2 can be certified on this task, so the bound can be compared with the observed convergence exactly. The
agents are randomized local searches, not language models, and nothing here calls an API.

| Folder | What it runs | Paper | Time |
|---|---|---|---|
| `known-correction/` | exact correction, private horizons H = 1 to 128, 2,000 runs each, and the uncorrected ablation | Figure 5(b, c), Table G.2 | 5 s |
| `approximate-correction/` | correction with a persistent bias ν, H = 8 and 32: the error floor | Figure 5(d), Table G.3 | 40 s |
| `learned-correction/` | correction learned from scalar verifier queries | Figure G.1(a, b), Table G.4 | 10 s |
| `verification-reuse/` | reusing already-paid verifications, with an observable stopping certificate | Figure G.1(c, d), Table G.5 | 5 s |

Run them from the repository root; `known-correction` first, because the other three read its fixed task:

```bash
python synthetic/known-correction/run.py
python synthetic/approximate-correction/run.py
python synthetic/learned-correction/run.py
python synthetic/verification-reuse/run.py
python synthetic/verification-reuse/checks.py   # finite-state checks of the stopping certificate
```

Each script writes `results.json` in its folder; the seeds are fixed, so every rerun gives the same results. The
README of each folder describes the settings and the checks the script performs.
