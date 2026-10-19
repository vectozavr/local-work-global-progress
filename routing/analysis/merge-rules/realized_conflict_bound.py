"""Bounds for the best-combination merge (Algorithm 2) with the realized and the effective number of colors; no model calls.

Per epoch k with improving-block gains g_b and realized gain F(x^k) - F(x^{k+1}):
  realized number of colors   kappa*_k = sum_b g_b / (F(x^k) - F(x^{k+1}))   (1 if no block improves);
  effective number of colors  kappa_k  = chromatic number of the conflicts among improving blocks, where b and b'
                              conflict if xi_bb' = F(both) - F(x) - delta_b - delta_b' > 0 (the routing loss is pairwise).
The merge gains exactly sum_b g_b / kappa*_k >= sum_b lb_b / kappa*_k with lb_b = Q D_b - Q e*_b - e^U_b, so the
recurrence of Proposition dev:prop:recorded holds with kappa replaced by kappa*_k (Corollary cor:realized-rate);
with pairwise factors kappa*_k <= kappa_k <= min(kappa, |I^k|) <= kappa (Proposition prop:realized-conflicts).
For Algorithm 1 the same ratio is sum_b g_b / max_b g_b (best block per floor on two floors). Two floors: floors are
independent, so the realized ratio is computed from the summed gains and kappa_k per floor. Every inequality is
checked in every round; envelopes are propagated per run with alpha capped at 1, held after the run stops, and averaged.
Also records the interaction classes and kappa statistics. Writes realized-conflict-bound.json."""
import itertools, json, statistics, sys
from collections import Counter
from pathlib import Path
HERE = Path(__file__).resolve().parent; RES = HERE.parents[2]/'data'/'routing-runs'; THEORY = HERE.parents[2]/'data'/'routing-theory'
read = lambda p: json.loads(Path(p).read_text())
FS = {1: 144., 2: 147.}; INIT = {1: 1476., 2: 1479.}
ALG1 = {1: (THEORY/'map1-team/trajectory-rate.json', ['multiseed-ninecars-team-v2-20261016-p5', 'multiseed-ninecars-team-v2-seeds10-19-20261016-p5']),
        2: (THEORY/'map2-team/trajectory-rate.json', ['map2-team-v2-20261016-p5'])}


def chrom(nodes, edges):
    for k in range(1, len(nodes) + 1):
        for col in itertools.product(range(k), repeat=len(nodes)):
            c = dict(zip(nodes, col))
            if all(c[a] != c[b] for a, b in edges): return k
    return 1


def independent(S, edges): return all(not (a in S and b in S) for a, b in edges)


def lower(r):
    QD = sum(x['Q'] * x['D'] for x in r['blocks']); err = sum(x['Q'] * x['true_minimizer_error'] + x['returned_error'] for x in r['blocks'])
    return QD, err


def best_combination_rounds(m):
    camp = RES/f'team-best-combination-map{m}-20261017'; th = read(THEORY/f'map{m}-team-best-combination/trajectory-rate.json'); fs = th['optimum']
    coef = {(r['seed'], r['round']): r for r in th['per_round_coefficients']}; out = {}; stats = Counter(); inter = Counter(); chis = Counter(); nimp = Counter(); kst = []; multi = []
    sys.path.insert(0, str(Path(__file__).resolve().parents[2])); from robot_routing.world import World; sys.path.pop(0)
    W = World(read(camp/'instance.json'))
    for t in read(camp/'plan.json')['trials']:
        s = t['replicate']; H = read(camp/t['path']/f"seed-{t['api_seed_base']}-learned/protocol/result.json")['history']
        for h in H:
            k = h['epoch'] + 1; r = coef[(s, k)]; before = r['initial_F']
            combos = {tuple(c['blocks']): c['F'] for c in h['combinations']}
            single = {b: combos.get((b,), before) for b in range(3)}; delta = {b: single[b] - before for b in range(3)}
            g = {b: max(0., -delta[b]) for b in range(3)}; G = sum(g.values()); gain = before - h['F']
            xi = {(b, c): combos[(b, c)] - before - delta[b] - delta[c] for b, c in itertools.combinations(range(3), 2) if (b, c) in combos}
            if (0, 1, 2) in combos:                                   # pairwise factors: the triple follows from singles and pairs
                assert abs(combos[(0, 1, 2)] - (before + sum(delta.values()) + sum(xi.values()))) < 1e-6, (m, s, k); stats['triples_exact'] += 1
            I = [b for b in range(3) if g[b] > 0]; nimp[len(I)] += 1
            edges = [(b, c) for b, c in itertools.combinations(I, 2) if xi[(b, c)] > 1e-9]
            for b, c in itertools.combinations(I, 2):
                inter['harmful' if xi[(b, c)] > 1e-9 else 'helpful' if xi[(b, c)] < -1e-9 else 'none'] += 1
                if xi[(b, c)] > 1e-9:                                 # xi/100 = change in collisions between the robots of b and c
                    def cross(S):
                        y = list(h['base'])
                        for q in S:
                            for i in W.groups[q]: y[i] = h['proposals'][q][i]
                        return {(tuple(e['pair']), e['time'], e['kind']) for e in W.evaluate_programs(y)['events'] if {W.owner[e['pair'][0]], W.owner[e['pair'][1]]} == {b, c}}
                    E0, Eb, Ec, Ebc = cross([]), cross([b]), cross([c]), cross([b, c])
                    assert abs(xi[(b, c)] - W.M * (len(Ebc) - len(Eb) - len(Ec) + len(E0))) < 1e-9
                    if Ebc - Eb - Ec: inter['harmful_new_collision'] += 1
                    else: inter['harmful_same_collision_removed_twice' if (E0 - Eb) & (E0 - Ec) else 'harmful_other'] += 1
            chi = chrom(I, edges); chis[chi] += 1
            if len(I) >= 2: stats['rounds_two_or_more_improving'] += 1; stats['of_which_no_conflict'] += not edges
            mwis = max([sum(g[b] for b in S) for n in range(len(I) + 1) for S in itertools.combinations(I, n) if independent(S, edges)])
            deg = {b: sum(b in e for e in edges) for b in I}
            assert gain >= mwis - 1e-9 and mwis >= G / chi - 1e-9 and mwis >= sum(g[b] / (1 + deg[b]) for b in I) - 1e-9 and gain >= max(g.values()) - 1e-9, (m, s, k)
            kstar = G / gain if gain > 0 else 1.
            assert kstar <= chi + 1e-9 and kstar <= 3 + 1e-9
            if gain > 0: kst.append(kstar)
            if len(I) >= 2: multi.append((kstar, chi))
            lb = [x['Q'] * x['D'] - x['Q'] * x['true_minimizer_error'] - x['returned_error'] for x in r['blocks']]
            assert all(l <= g[b] + 1e-9 for b, l in enumerate(lb))
            QD, err = lower(r)
            for d in (chi, kstar): assert r['next_F'] - fs <= before - fs - sum(lb) / d + 1e-6, (m, s, k)
            out[(s, k)] = dict(QD=QD, err=err, eff=chi, imp=max(1, min(3, len(I))), star=kstar, G=G, gain=gain, before=before, after=r['next_F'],
                               alpha_kappa=r['alpha'], B_kappa=r['B'])
    info = dict(rounds=sum(chis.values()), chi=dict(sorted(chis.items())), improving_blocks=dict(sorted(nimp.items())), improving_pairs=dict(inter),
                triples_checked=stats['triples_exact'], rounds_two_or_more_improving=stats['rounds_two_or_more_improving'], of_which_no_conflict=stats['of_which_no_conflict'],
                kstar_mean=statistics.mean(kst), kstar_max=max(kst), kstar_below_one=sum(x < 1 - 1e-9 for x in kst), rounds_with_gain=len(kst),
                multi_kstar_mean=statistics.mean(a for a, _ in multi), multi_keff_mean=statistics.mean(c for _, c in multi), multi_kstar_below_keff=sum(a < c - 1e-9 for a, c in multi))
    return out, info, th


def algorithm1_rounds(m):
    th = read(ALG1[m][0]); coef = {(r['seed'], r['round']): r for r in th['per_round_coefficients']}; out = {}; fac = []
    for name in ALG1[m][1]:
        for t in read(RES/name/'plan.json')['trials']:
            s = t['replicate']; H = read(RES/name/t['path']/f"seed-{t['api_seed_base']}-learned/protocol/result.json")['history']
            for h in H:
                r = coef[(s, h['epoch'] + 1)]; before = r['initial_F']; g = [max(0., before - v) for v in h['verified_F']]; G = sum(g); gain = before - r['next_F']
                assert abs(gain - max(g)) < 1e-9                      # one map: the best color is the best single block
                kstar = G / gain if gain > 0 else 1.
                if gain > 0: fac.append(kstar)
                QD, err = lower(r); nI = sum(x > 0 for x in g)
                out[(s, h['epoch'] + 1)] = dict(QD=QD, err=err, star=kstar, imp=max(1, min(3, nI)), G=G, gain=gain, before=before, after=r['next_F'], alpha_kappa=r['alpha'], B_kappa=r['B'])
    gain_rounds = [v for v in out.values() if v['gain'] > 0]     # rounds that improve the plan
    discarded = sum(v['G'] > v['gain'] + 1e-9 for v in gain_rounds)  # ... in which another improving proposal is discarded
    return out, dict(realized_mean=statistics.mean(fac), realized_max=max(fac), improving_rounds=len(gain_rounds), rounds_discarding_improvement=discarded), th


def propagate(steps, gap0, K):
    """steps: (alpha, B, gap_before, gap_after) for one run; returns the envelope held to K rounds."""
    E = gap0; path = [E]
    for a, B, before, after in steps:
        assert 0 <= a <= 1 and after <= (1 - a) * before + B + 1e-6
        E = min(E, (1 - a) * E + B); assert after <= E + 1e-6; path.append(E)
    return path + [path[-1]] * (K + 1 - len(path))


def step(QD, err, d, gap, after): return (min(1., QD / (d * gap)) if gap > 0 else 0., err / d, gap, after)


def single_map(R, th, m, keys):
    K = th['max_rounds']; seeds = sorted({s for s, _ in R}); P = {k: [] for k in ('kappa', 'observed') + keys}
    for s in seeds:
        rows = [R[(s, k)] for k in sorted(k for q, k in R if q == s)]; gap = lambda F: F - FS[m]
        P['kappa'].append(propagate([(r['alpha_kappa'], r['B_kappa'], gap(r['before']), gap(r['after'])) for r in rows], INIT[m] - FS[m], K))
        for key in keys: P[key].append(propagate([step(r['QD'], r['err'], r[key], gap(r['before']), gap(r['after'])) for r in rows], INIT[m] - FS[m], K))
        obs = [INIT[m] - FS[m]] + [gap(r['after']) for r in rows]; P['observed'].append(obs + [obs[-1]] * (K + 1 - len(obs)))
    return {k: [statistics.mean(p[i] for p in v) for i in range(K + 1)] for k, v in P.items()}


def two_floors(R1, R2, keys):
    """keys: per-floor divisors ('imp', 'eff') or the combined realized ratio ('star')."""
    paths = {k: [] for k in ('kappa', 'observed') + keys}; n = 0
    for s in range(20):
        gap = [INIT[1] - FS[1], INIT[2] - FS[2]]; steps = {k: [] for k in ('kappa',) + keys}; obs = [sum(gap)]
        for k in range(1, 40):
            rows = [R.get((s, k)) for R in (R1, R2)]
            if not any(rows): break
            new = [rows[f]['after'] - FS[f + 1] if rows[f] else gap[f] for f in (0, 1)]; D = sum(gap); live = [r for r in rows if r]
            QD = sum(r['QD'] for r in live); err = sum(r['err'] for r in live)
            steps['kappa'].append((QD / (3 * D) if D > 0 else 0., err / 3, D, sum(new)))
            for key in keys:
                if key == 'star':
                    G = sum(r['G'] for r in live); gain = sum(r['gain'] for r in live); d = G / gain if gain > 0 else 1.
                    assert sum(new) <= D - (QD - err) / d + 1e-6; steps[key].append(step(QD, err, d, D, sum(new)))
                else:
                    q = sum(r['QD'] / r[key] for r in live); e = sum(r['err'] / r[key] for r in live)
                    steps[key].append((min(1., q / D) if D > 0 else 0., e, D, sum(new)))
            gap = new; obs.append(sum(new)); n += 1
        for key in steps: paths[key].append(steps[key])
        paths['observed'].append(obs)
    K = max(len(p) for p in paths['kappa']); g0 = INIT[1] + INIT[2] - FS[1] - FS[2]
    env = {k: [propagate(p, g0, K) for p in paths[k]] for k in ('kappa',) + keys}
    env['observed'] = [o + [o[-1]] * (K + 1 - len(o)) for o in paths['observed']]
    return n, {k: [statistics.mean(p[i] for p in v) for i in range(K + 1)] for k, v in env.items()}


if __name__ == '__main__':
    out = {'best_combination': {}, 'algorithm1': {}}
    BC, A1 = {}, {}
    for m in (1, 2):
        R, info, th = best_combination_rounds(m); BC[m] = R
        out['best_combination'][str(m)] = dict(info, **{f'envelope_{k}': v for k, v in single_map(R, th, m, ('imp', 'eff', 'star')).items()})
        R, info, th = algorithm1_rounds(m); A1[m] = R
        out['algorithm1'][str(m)] = dict(info, **{f'envelope_{k}': v for k, v in single_map(R, th, m, ('imp', 'star')).items()})
    n, env = two_floors(BC[1], BC[2], ('imp', 'eff', 'star')); out['best_combination']['3'] = dict(rounds=n, **{f'envelope_{k}': v for k, v in env.items()})
    n, env = two_floors(A1[1], A1[2], ('imp', 'star')); out['algorithm1']['3'] = dict(rounds=n, **{f'envelope_{k}': v for k, v in env.items()})
    (HERE/'realized-conflict-bound.json').write_text(json.dumps(out, indent=1) + '\n')
    for alg, d in out.items():
        for m, o in d.items():
            fin = {k[9:]: round(v[-1], 1) for k, v in o.items() if k.startswith('envelope_')}
            extra = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in o.items() if not k.startswith('envelope_') and k not in ('improving_pairs', 'chi', 'improving_blocks')}
            print(alg, m, fin, extra)
