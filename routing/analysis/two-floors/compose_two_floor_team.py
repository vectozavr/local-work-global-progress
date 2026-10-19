"""Two-floor team arms from the single-floor team runs; no model calls.

The floors do not interact and every team agent sees only its own floor, so a two-floor team run is a run on
floor 1 and a run on floor 2 played in lockstep: in round k both floors install their best block (one color class of
the two disjoint triangles), the round lasts as long as its slowest agent, i.e. max(T1_k, T2_k), and a floor that has
stopped (target or five rounds without improvement) makes no further calls. Seed s of the first map is paired with
seed s of the second. The sequential replay plays all six agents in turn: its time is seq1 + seq2.

Theory: with 6 blocks and kappa = 3, alpha_k = sum over the six blocks of Q D / (3 Delta_k) with Delta_k = Delta1_k +
Delta2_k, and B_k = B1_k + B2_k (a stopped floor contributes neither); the one-step inequality of the combined problem
is checked in every round, and E_{k+1} = min(E_k, (1 - alpha_k) E_k + B_k) is propagated per pair.
Writes ./composed/{summary-team,summary-nocorr}.json, curves-checked_seconds.json and theory-two-floor.json.
"""
import json, statistics
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
V2 = HERE.parent/'team-vs-single'
RES = HERE.parents[2]/'data'/'routing-runs'
read = lambda p: json.loads(Path(p).read_text())
F1, F2 = 144., 147.
INITIAL = 1476. + 1479.
SOURCES = dict(team=(V2/'pooled/summary-team.json', V2/'pooled-map2/summary-team.json'),
               nocorr=(RES/'multiseed-ninecars-team-v2-no-correction-20261016-p5/analysis-v2/summary.json',
                       RES/'map2-team-v2-no-correction-20261016-p5/analysis-v2/summary.json'))
THEORY = (HERE.parents[2]/'data/routing-theory/map1-team/trajectory-rate.json', HERE.parents[2]/'data/routing-theory/map2-team/trajectory-rate.json')


def per_round(run):
    """Per-round (time, sequential time) increments and installed F after each round."""
    t = s = 0.; out = []
    for row in run['trace']:
        out.append((row['checked_seconds'] - t, row['sequential_seconds'] - s, row['F'], row['delivered'], row['collisions']))
        t, s = row['checked_seconds'], row['sequential_seconds']
    assert abs(t - run['checked_seconds']) < 1e-6 and abs(s - run['sequential_seconds']) < 1e-6
    assert out[-1][2] == run['final']['F']
    return out


def compose(a, b):
    ra, rb = per_round(a), per_round(b); K = max(len(ra), len(rb))
    hold = lambda r, k: r[min(k, len(r) - 1)]
    t = s = 0.; trace = []
    for k in range(K):
        ta = ra[k][0] if k < len(ra) else 0.; tb = rb[k][0] if k < len(rb) else 0.
        sa = ra[k][1] if k < len(ra) else 0.; sb = rb[k][1] if k < len(rb) else 0.
        t += max(ta, tb); s += sa + sb
        x, y = hold(ra, k), hold(rb, k)
        trace.append(dict(round=k + 1, checked_seconds=t, sequential_seconds=s, F=x[2] + y[2], F_floors=[x[2], y[2]],
                          delivered=x[3] + y[3], collisions=x[4] + y[4]))
    fa, fb = a['final'], b['final']
    final = dict(F=fa['F'] + fb['F'], F_floors=[fa['F'], fb['F']], delivered=fa['delivered'] + fb['delivered'],
                 collisions=fa['collisions'] + fb['collisions'])
    assert final['F'] == trace[-1]['F'] and abs(s - a['sequential_seconds'] - b['sequential_seconds']) < 1e-6
    return dict(pair=[a['trial']['id'], b['trial']['id']], replicate=a['trial']['replicate'], final=final,
                safe=final['delivered'] == 18 and final['collisions'] == 0, reached_target=bool(a['reached_target'] and b['reached_target']),
                rounds=K, answers=a['answers'] + b['answers'], checked_seconds=t, sequential_seconds=s,
                reported_cost_usd=a['reported_cost_usd'] + b['reported_cost_usd'], trace=trace)


def stats(v):
    return dict(n=len(v), mean=statistics.mean(v), sample_sd=statistics.stdev(v) if len(v) > 1 else 0., median=statistics.median(v),
                minimum=min(v), maximum=max(v))


def step_curve(reports, clock):
    x = np.array(sorted({0.} | {r[clock] for rep in reports for r in rep['trace']}))
    m = []
    for rep in reports:
        t = np.array([0.] + [r[clock] for r in rep['trace']]); y = np.array([INITIAL] + [r['F'] for r in rep['trace']])
        m.append(y[np.searchsorted(t, x, side='right') - 1])
    m = np.array(m)
    return dict(seconds=x.tolist(), mean=m.mean(0).tolist(), minimum=m.min(0).tolist(), maximum=m.max(0).tolist(),
                per_run=m.tolist(), runs=len(reports), terminal_extension='Hold final installed loss; no further API work')


def theory(team_pairs):
    t1, t2 = read(THEORY[0]), read(THEORY[1])
    assert t1['optimum'] == F1 and t2['optimum'] == F2
    rows = [{}, {}]
    for f, t in enumerate((t1, t2)):
        for r in t['per_round_coefficients']:
            rows[f][(r['seed'], r['round'])] = r
    pairs = []
    for rep in team_pairs:
        s = rep['replicate']; gap = [1476. - F1, 1479. - F2]; e = sum(gap); obs = [INITIAL]; env = [INITIAL + 0.]
        env[0] = F1 + F2 + e; rec = []
        for k, row in enumerate(rep['trace'], 1):
            progress = B = 0.
            for f in (0, 1):
                r = rows[f].get((s, k))
                if r is None: continue            # floor stopped: no search, no error term
                assert abs(r['initial_F'] - ([F1, F2][f] + gap[f])) < 1e-9
                progress += sum(b['Q'] * b['D'] for b in r['blocks']); B += r['B']
            delta = sum(gap); alpha = progress / (3 * delta) if delta > 0 else 0.
            new_gap = [row['F_floors'][0] - F1, row['F_floors'][1] - F2]
            assert sum(new_gap) <= (1 - alpha) * delta + B + 1e-8, (s, k)
            e = min(e, (1 - alpha) * e + B); gap = new_gap
            obs.append(row['F']); env.append(F1 + F2 + e)
            assert row['F'] <= env[-1] + 1e-8
            rec.append(dict(round=k, alpha=alpha, B=B, gap_before=delta, gap_after=sum(new_gap)))
        pairs.append(dict(replicate=s, observed=obs, upper=env, rounds=rec))
    K = max(len(p['observed']) for p in pairs)
    hold = lambda v: v + [v[-1]] * (K - len(v))
    O = np.array([hold(p['observed']) for p in pairs]); U = np.array([hold(p['upper']) for p in pairs])
    return dict(kind='Two-floor retrospective envelopes composed from the single-floor coefficients; no model calls.',
                optimum=F1 + F2, kappa=3, blocks=6, initial_F=INITIAL, max_rounds=K - 1, rounds=list(range(K)),
                observed_mean_F=O.mean(0).tolist(), observed_minimum_F=O.min(0).tolist(), observed_maximum_F=O.max(0).tolist(),
                upper_mean_F=U.mean(0).tolist(), upper_minimum_F=U.min(0).tolist(), upper_maximum_F=U.max(0).tolist(),
                one_step_inequality_verified_all_rounds=True, envelope_verified_all_rounds=True,
                rounds_total=sum(len(p['rounds']) for p in pairs), pairs=pairs)


def main():
    out = HERE/'composed'; out.mkdir(exist_ok=True); summaries = {}; reports = {}
    for arm, (p1, p2) in SOURCES.items():
        a = sorted(read(p1)['runs'], key=lambda r: r['trial']['replicate']); b = sorted(read(p2)['runs'], key=lambda r: r['trial']['replicate'])
        assert [r['trial']['replicate'] for r in a] == [r['trial']['replicate'] for r in b] == list(range(len(a)))
        reps = [compose(x, y) for x, y in zip(a, b)]; reports[arm] = reps
        fields = {k: [r[k] for r in reps] for k in ('checked_seconds', 'sequential_seconds', 'rounds', 'answers', 'reported_cost_usd')}
        fields['final_F'] = [r['final']['F'] for r in reps]
        summaries[arm] = dict(sources=[str(p1.relative_to(HERE.parents[2])), str(p2.relative_to(HERE.parents[2]))], pairing='seed s of the first map with seed s of the second',
                              runs_n=len(reps), safe_complete=sum(r['safe'] for r in reps), strict_target=sum(r['reached_target'] for r in reps),
                              statistics={k: stats(v) for k, v in fields.items()}, runs=reps)
        (out/f'summary-{arm}.json').write_text(json.dumps(summaries[arm], indent=1) + '\n')
    curves = dict(team=step_curve(reports['team'], 'checked_seconds'), sequential=step_curve(reports['team'], 'sequential_seconds'),
                  team_no_correction=step_curve(reports['nocorr'], 'checked_seconds'))
    (out/'curves-checked_seconds.json').write_text(json.dumps(curves) + '\n')
    th = theory(reports['team']); (out/'theory-two-floor.json').write_text(json.dumps(th) + '\n')
    for arm, s in summaries.items():
        st = s['statistics']
        print(arm, 'runs', s['runs_n'], 'safe', s['safe_complete'], 'target', s['strict_target'],
              'F %.2f+-%.2f median %.1f' % (st['final_F']['mean'], st['final_F']['sample_sd'], st['final_F']['median']),
              'time %.1f+-%.1f median %.1f max %.1f' % (st['checked_seconds']['mean'], st['checked_seconds']['sample_sd'],
                                                     st['checked_seconds']['median'], st['checked_seconds']['maximum']),
              'seq %.1f' % st['sequential_seconds']['mean'], 'rounds %.1f' % st['rounds']['mean'])
    print('theory: rounds', th['max_rounds'], 'final observed mean gap %.2f, bound %.2f' % (th['observed_mean_F'][-1] - 291, th['upper_mean_F'][-1] - 291),
          'checked rounds', th['rounds_total'])


if __name__ == '__main__':
    main()
