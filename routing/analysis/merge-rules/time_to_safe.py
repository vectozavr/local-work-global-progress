"""Search time until the best plan so far is within 100 of the optimum, for every protocol of Figure 3 except the team
without correction; no model calls. A collision or an undelivered robot adds at least 100 above F*, so such a plan is
safe (the converse fails: a safe plan with long detours can be 100 or more above F*). The criterion uses only the loss,
which every trace records. Runs that never get there count as infinite time. Writes time-to-safe.json."""
import json, math, statistics
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; RES = HERE.parents[2]/'data'/'routing-runs'; V2 = HERE.parent/'team-vs-single'; TF = HERE.parent/'two-floors'
read = lambda p: json.loads(Path(p).read_text())
FS = {1: 144., 2: 147., 3: 291.}
rng = np.random.default_rng(2)


def checked(paths):
    runs = []
    for p in paths:
        s = read(p); assert s.get('all_planned_completed', True) and s.get('independent_replay_passed', True), p; runs += s['runs']
    assert len(runs) == 20, paths; return runs


def sources(m):
    if m in (1, 2):
        pool = V2/('pooled' if m == 1 else 'pooled-map2')
        return dict(single=[pool/'summary-single1.json'], best=[pool/'summary-best3.json'], team=[pool/'summary-team.json'],
                    best_subset=[RES/f'team-best-combination-map{m}-20261017/analysis-v2/summary.json'], all=[RES/f'team-unsafe-map{m}-20261017/analysis-v2/summary.json'])
    return dict(single=[RES/'twofloor-single-v2-one-update-20261017/summary.json'],
                best=[RES/'twofloor-single-v2-best6-20261017/summary.json', RES/'twofloor-single-v2-best6-seeds10-19-20261017/summary.json'],
                team=[TF/'composed/summary-team.json'], best_subset=[HERE/'composed/summary-best_subset.json'], all=[HERE/'composed/summary-all.json'])


def first_safe(run, fs):
    return next((t['checked_seconds'] for t in run['trace'] if t['F'] - fs < 100), math.inf)


def perm_median(a, b, B=100000):
    a, b = np.asarray(a, float), np.asarray(b, float); big = 1e9
    a, b = np.where(np.isinf(a), big, a), np.where(np.isinf(b), big, b); allv = np.concatenate([a, b]); n = len(a)
    obs = np.median(a) - np.median(b); k = 0
    for _ in range(B):
        p = rng.permutation(allv); k += abs(np.median(p[:n]) - np.median(p[n:])) >= abs(obs) - 1e-12
    return k / B


def main():
    out = {}
    for m in (1, 2, 3):
        T = {arm: [first_safe(r, FS[m]) for r in checked(ps)] for arm, ps in sources(m).items()}
        out[m] = {arm: dict(reached=sum(math.isfinite(x) for x in t), median=statistics.median(t),
                            mean_reached=statistics.mean(x for x in t if math.isfinite(x)),
                            p_median_vs_best_subset=None if arm == 'best_subset' else perm_median(t, T['best_subset'])) for arm, t in T.items()}
        print({1: 'map 1', 2: 'map 2', 3: 'two floors'}[m], {a: (v['reached'], round(v['median']), v['p_median_vs_best_subset']) for a, v in out[m].items()})
    (HERE/'time-to-safe.json').write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
