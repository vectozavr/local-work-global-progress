"""Two-floor versions of the merge rules and their comparisons with Algorithm 1; no model calls. Two floors: seed s of map 1 and seed s of map 2 played in lockstep, exactly as for the team of
the main text (two-floors/compose_two_floor_team.py); best combination over the six blocks of two
independent floors is the best combination of each floor, and unsafe installs all changed blocks of both floors.
Writes composed/ and comparison.json."""
import json, math, statistics, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; RES = HERE.parents[2]/'data'/'routing-runs'; TF = HERE.parent/'two-floors'; V2 = HERE.parent/'team-vs-single'
sys.path.insert(0, str(TF))
import compose_two_floor_team as C
read = lambda p: json.loads(Path(p).read_text())
MODES = {'best_subset': 'team-best-combination', 'all': 'team-unsafe'}
FS = {1: 144., 2: 147., 3: 291.}
rng = np.random.default_rng(2)


def campaign(mode, m):
    s = read(RES/f'{MODES[mode]}-map{m}-20261017/analysis-v2/summary.json')
    assert s['all_planned_completed'] and s['independent_replay_passed']; return sorted(s['runs'], key=lambda r: r['trial']['replicate'])


def alg1(m):
    if m == 3: return sorted(read(TF/'composed/summary-team.json')['runs'], key=lambda r: r['replicate'])
    return sorted(read(V2/('pooled' if m == 1 else 'pooled-map2')/'summary-team.json')['runs'], key=lambda r: r['trial']['replicate'])


def perm(a, b, stat, B=100000):
    a, b = np.asarray(a, float), np.asarray(b, float); allv = np.concatenate([a, b]); n = len(a); obs = stat(a, b); k = 0
    for _ in range(B):
        p = rng.permutation(allv); k += abs(stat(p[:n], p[n:])) >= abs(obs) - 1e-12
    return float(obs), k / B


def fisher(a, n1, b, n2):
    N, K = n1 + n2, a + b
    p = lambda x: math.comb(K, x) * math.comb(N - K, n1 - x) / math.comb(N, n1); p0 = p(a)
    return sum(p(x) for x in range(max(0, K - n2), min(K, n1) + 1) if p(x) <= p0 + 1e-12)


def describe(runs, n_robots):
    F = [r['final']['F'] for r in runs]; T = [r['checked_seconds'] for r in runs]
    return dict(n=len(runs), safe=sum(r['final']['delivered'] == n_robots and r['final']['collisions'] == 0 for r in runs),
                target=sum(bool(r['reached_target']) for r in runs), F_mean=statistics.mean(F), F_sd=statistics.stdev(F), F_median=statistics.median(F),
                time_mean=statistics.mean(T), time_sd=statistics.stdev(T), time_median=statistics.median(T),
                rounds_mean=statistics.mean(r['rounds'] for r in runs), answers_mean=statistics.mean(r['answers'] for r in runs),
                sequential_mean=statistics.mean(r['sequential_seconds'] for r in runs), cost_mean=statistics.mean(r['reported_cost_usd'] for r in runs))


def main():
    out = HERE/'composed'; out.mkdir(exist_ok=True); comp = {}; curves3 = {}
    runs = {(mode, m): campaign(mode, m) for mode in MODES for m in (1, 2)}
    for mode in MODES:
        a, b = runs[(mode, 1)], runs[(mode, 2)]
        assert [r['trial']['replicate'] for r in a] == [r['trial']['replicate'] for r in b] == list(range(20))
        reps = [C.compose(x, y) for x, y in zip(a, b)]; runs[(mode, 3)] = reps
        (out/f'summary-{mode}.json').write_text(json.dumps(dict(runs=reps), indent=1) + '\n')
        curves3[mode] = C.step_curve(reps, 'checked_seconds')
    (out/'curves-checked_seconds.json').write_text(json.dumps(curves3) + '\n')
    for m in (1, 2, 3):
        n = 18 if m == 3 else 9; base = alg1(m); key = lambda r: r['trial']['replicate'] if 'trial' in r else r['replicate']
        assert [key(r) for r in base] == list(range(20))
        comp[m] = {'alg1': describe(base, n)}
        for mode in MODES:
            r = runs[(mode, m)]; comp[m][mode] = describe(r, n)
            F0, F1 = [x['final']['F'] for x in base], [x['final']['F'] for x in r]; T0, T1 = [x['checked_seconds'] for x in base], [x['checked_seconds'] for x in r]
            comp[m][mode]['vs_alg1'] = dict(F_mean=perm(F1, F0, lambda u, v: u.mean() - v.mean()), F_median=perm(F1, F0, lambda u, v: np.median(u) - np.median(v)),
                                            time_mean=perm(T1, T0, lambda u, v: u.mean() - v.mean()), time_median=perm(T1, T0, lambda u, v: np.median(u) - np.median(v)),
                                            target_fisher=fisher(comp[m][mode]['target'], 20, comp[m]['alg1']['target'], 20),
                                            lower_F_seeds=sum(x < y for x, y in zip(F1, F0)), faster_seeds=sum(x < y for x, y in zip(T1, T0)))
        if m in (1, 2):
            un = runs[('all', m)]
            comp[m]['all']['rounds_with_F_increase'] = sum(sum(t['installed_F'] > p for t, p in zip(rr['trace'], [B_init(m)] + [q['installed_F'] for q in rr['trace'][:-1]])) for rr in un)
            comp[m]['all']['last_installed_worse_than_best'] = sum(rr['trace'][-1]['installed_F'] > rr['final']['F'] for rr in un)
            for mode in MODES:
                root = RES/f'{MODES[mode]}-map{m}-20261017'; sizes = []
                for t in read(root/'plan.json')['trials']:
                    h = read(root/t['path']/f"seed-{t['api_seed_base']}-learned/protocol/result.json")['history']
                    sizes += [len(x['accepted']) for x in h if x['accepted']]
                comp[m][mode]['blocks_per_installing_round'] = statistics.mean(sizes); comp[m][mode]['installing_rounds'] = len(sizes)
    (HERE/'comparison.json').write_text(json.dumps(comp, indent=1) + '\n')
    for m in (1, 2, 3):
        name = {1: 'map 1', 2: 'map 2', 3: 'two floors'}[m]
        for k in ('alg1', 'best_subset', 'all'):
            c = comp[m][k]; v = c.get('vs_alg1')
            print(f"{name:10s} {k:11s} safe {c['safe']}/20 target {c['target']}/20 F {c['F_mean']:.2f}+-{c['F_sd']:.2f} (median {c['F_median']:.1f}) time {c['time_mean']:.0f}+-{c['time_sd']:.0f} (median {c['time_median']:.0f}) rounds {c['rounds_mean']:.1f}"
                  + (f" | vs Alg1: F p={v['F_mean'][1]:.3f}/{v['F_median'][1]:.3f}, time p={v['time_mean'][1]:.3f}/{v['time_median'][1]:.3f}, target p={v['target_fisher']:.2f}, lower F on {v['lower_F_seeds']}, faster on {v['faster_seeds']}" if v else ''))


def B_init(m): return {1: 1476., 2: 1479.}[m]


if __name__ == '__main__':
    main()
