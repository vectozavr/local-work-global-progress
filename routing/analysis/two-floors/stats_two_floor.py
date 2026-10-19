"""Numbers and tests for the two-floor experiment, from audited files only; no model calls.

Single agent and best-of-six: their campaigns (independently replayed). Team, sequential replay and team without
correction: composed/ (compose_two_floor_team.py). Two-sided permutation tests (100,000 permutations, as in
team-vs-single/compare.py) on means, medians and log SD ratios; Fisher's exact test on threshold counts.
Writes numbers-two-floor.json.
"""
import json, math, statistics
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE.parents[2]/'data'/'routing-runs'
read = lambda p: json.loads(Path(p).read_text())
FS = 291.
rng = np.random.default_rng(2)


def load():
    single = read(RES/'twofloor-single-v2-one-update-20261017/summary.json')['runs']
    best6 = read(RES/'twofloor-single-v2-best6-20261017/summary.json')['runs'] + read(RES/'twofloor-single-v2-best6-seeds10-19-20261017/summary.json')['runs']
    for name in ('twofloor-single-v2-one-update-20261017', 'twofloor-single-v2-best6-20261017', 'twofloor-single-v2-best6-seeds10-19-20261017'):
        s = read(RES/name/'summary.json'); assert s['all_planned_completed'] and s['independent_replay_passed'], name
    key = lambda r: r['trial']['replicate'] if 'trial' in r else r['replicate']
    single, best6 = sorted(single, key=key), sorted(best6, key=key)
    team = sorted(read(HERE/'composed/summary-team.json')['runs'], key=key)
    nocorr = sorted(read(HERE/'composed/summary-nocorr.json')['runs'], key=key)
    assert [key(r) for r in single] == [key(r) for r in best6] == [key(r) for r in team] == list(range(20))
    assert [key(r) for r in nocorr] == list(range(10))
    return dict(single=single, best6=best6, team=team, nocorr=nocorr)


def perm(a, b, stat, B=100000):
    allv = np.concatenate([a, b]); n = len(a); obs = stat(a, b); k = 0
    for _ in range(B):
        p = rng.permutation(allv); k += abs(stat(p[:n], p[n:])) >= abs(obs) - 1e-12
    return float(obs), k / B


def fisher(a, n1, b, n2):
    N, K = n1 + n2, a + b
    p = lambda x: math.comb(K, x) * math.comb(N - K, n1 - x) / math.comb(N, n1); p0 = p(a)
    return sum(p(x) for x in range(max(0, K - n2), min(K, n1) + 1) if p(x) <= p0 + 1e-12)


def main():
    R = load(); out = {}
    safe = lambda r: r['final']['delivered'] == 18 and r['final']['collisions'] == 0
    for k, rs in R.items():
        F = [r['final']['F'] for r in rs]; T = [r['checked_seconds'] for r in rs]; A = [r['answers'] for r in rs]
        C = [r['reported_cost_usd'] for r in rs]
        o = dict(n=len(rs), safe=sum(map(safe, rs)), target=sum(bool(r['reached_target']) for r in rs),
                 F_mean=statistics.mean(F), F_sd=statistics.stdev(F), F_median=statistics.median(F), F_min=min(F), F_max=max(F),
                 time_mean=statistics.mean(T), time_sd=statistics.stdev(T), time_median=statistics.median(T), time_min=min(T), time_max=max(T),
                 answers_mean=statistics.mean(A), answers_sd=statistics.stdev(A), cost_mean=statistics.mean(C), cost_sd=statistics.stdev(C), cost_total=sum(C),
                 per_seed_F=F, per_seed_time=T, per_seed_answers=A, per_seed_floors=[r['final']['F_floors'] for r in rs],
                 per_seed_target=[bool(r['reached_target']) for r in rs], over_1000s=sum(t > 1000 for t in T))
        if 'rounds' in rs[0]:
            o['rounds_mean'] = statistics.mean(r['rounds'] for r in rs); o['per_seed_rounds'] = [r['rounds'] for r in rs]
        if 'sequential_seconds' in rs[0]:
            S = [r['sequential_seconds'] for r in rs]; o.update(sequential_mean=statistics.mean(S), sequential_sd=statistics.stdev(S),
                                                              sequential_median=statistics.median(S), per_seed_sequential=S)
        if k in ('single', 'best6'):
            o['stops'] = {s: sum(r['stopping'] == s for r in rs) for s in ('target_reached', 'no_improvement_patience_exhausted', 'round_budget_exhausted')}
            o['floor1_target'] = sum(r['final']['F_floors'][0] <= 153 for r in rs); o['floor2_target'] = sum(r['final']['F_floors'][1] <= 156 for r in rs)
        out[k] = o
    tF = np.array(out['team']['per_seed_F']); tT = np.array(out['team']['per_seed_time'])
    for k in ('single', 'best6'):
        F = np.array(out[k]['per_seed_F']); T = np.array(out[k]['per_seed_time'])
        out[f'team_vs_{k}'] = dict(
            F_mean_diff=perm(F, tF, lambda u, v: u.mean() - v.mean()), F_median_diff=perm(F, tF, lambda u, v: np.median(u) - np.median(v)),
            time_mean_diff=perm(T, tT, lambda u, v: u.mean() - v.mean()), time_median_diff=perm(T, tT, lambda u, v: np.median(u) - np.median(v)),
            time_sd_ratio=perm(T, tT, lambda u, v: np.log(u.std(ddof=1) / v.std(ddof=1))),
            target_fisher=fisher(out['team']['target'], 20, out[k]['target'], 20),
            team_F_at_most=int((tF <= F).sum()), team_F_lower=int((tF < F).sum()), team_faster=int((tT < T).sum()))
        r = out[f'team_vs_{k}']; r['time_sd_ratio'] = (float(np.exp(r['time_sd_ratio'][0])), r['time_sd_ratio'][1])
    th = read(HERE/'composed/theory-two-floor.json')
    out['theory'] = dict(rounds_total=th['rounds_total'], max_rounds=th['max_rounds'], final_observed_gap=th['observed_mean_F'][-1] - FS,
                         final_bound_gap=th['upper_mean_F'][-1] - FS,
                         ratio_first_rounds=[(u - FS) / (o - FS) for u, o in zip(th['upper_mean_F'][1:5], th['observed_mean_F'][1:5])],
                         mean_alpha=statistics.mean(r['alpha'] for p in th['pairs'] for r in p['rounds']),
                         mean_B=statistics.mean(r['B'] for p in th['pairs'] for r in p['rounds']))
    (HERE/'numbers-two-floor.json').write_text(json.dumps(out, indent=1) + '\n')
    brief = {k: {x: (round(v, 3) if isinstance(v, float) else v) for x, v in out[k].items() if not x.startswith('per_seed')} for k in R}
    print(json.dumps(dict(arms=brief, team_vs_single=out['team_vs_single'], team_vs_best6=out['team_vs_best6'], theory=out['theory']), indent=1))


if __name__ == '__main__':
    main()
