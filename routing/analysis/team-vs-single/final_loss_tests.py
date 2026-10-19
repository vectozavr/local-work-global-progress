"""Permutation tests on final losses (maps 1 and 2) and Fisher's exact test on threshold counts; no model calls.

Same two-sided permutation test as compare.py (100,000 relabelings, numpy seed 2), applied to the final loss F of
the twenty runs of each protocol. Writes final-loss-tests.json next to this script."""
import json, math
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
read = lambda p: json.loads(Path(p).read_text())


def perm(a, b, stat, rng, B=100000):
    allv = np.concatenate([a, b]); n = len(a); obs = stat(a, b); k = 0
    for _ in range(B):
        p = rng.permutation(allv); k += abs(stat(p[:n], p[n:])) >= abs(obs) - 1e-12
    return round(float(obs), 2), k / B


def fisher_two_sided(a, b, c, d):
    """2x2 table [[a, b], [c, d]]; sum of probabilities of tables no more likely than the observed one."""
    n1, n2, k = a + b, c + d, a + c
    pmf = lambda x: math.comb(n1, x) * math.comb(n2, k - x) / math.comb(n1 + n2, k)
    p0 = pmf(a)
    return sum(pmf(x) for x in range(max(0, k - n2), min(k, n1) + 1) if pmf(x) <= p0 * (1 + 1e-9))


out = {}
for name, folder in (('map1', 'pooled'), ('map2', 'pooled-map2')):
    runs = {k: read(HERE/folder/f'summary-{k}.json')['runs'] for k in ('single1', 'best3', 'team')}
    F = {k: np.array([r['final']['F'] for r in v], float) for k, v in runs.items()}
    target = {k: sum(bool(r['reached_target']) for r in v) for k, v in runs.items()}
    rng = np.random.default_rng(2); res = {}
    for k in ('single1', 'best3'):
        a, b = F[k], F['team']; n = len(a)
        res[k] = dict(mean_diff=perm(a, b, lambda u, v: u.mean() - v.mean(), rng),
                      median_diff=perm(a, b, lambda u, v: np.median(u) - np.median(v), rng),
                      threshold=[target[k], target['team']],
                      threshold_fisher_p=round(fisher_two_sided(target[k], n - target[k], target['team'], len(b) - target['team']), 3))
    res['mean_F'] = {k: round(float(v.mean()), 2) for k, v in F.items()}; res['median_F'] = {k: float(np.median(v)) for k, v in F.items()}
    out[name] = res; print(name, json.dumps(res))
(HERE/'final-loss-tests.json').write_text(json.dumps(out, indent=2) + '\n')
