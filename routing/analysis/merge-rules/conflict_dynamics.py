"""How the realized coupling changes during the best-combination runs; no model calls. For every round: the number of
improving proposals |I^k|, the conflicting pairs among them (xi_bc > 0), the number kappa_k of colors the conflicts
need, and the number of blocks installed. Rounds are grouped by round index and by the state of the base: the
all-waiting start (round 1), an unsafe base (an undelivered robot or a collision; the collision-repair phase) and a
safe base (refinement). Writes conflict-dynamics.json."""
import itertools, json, statistics, sys
from collections import Counter, defaultdict
from pathlib import Path
HERE = Path(__file__).resolve().parent; RES = HERE.parents[2]/'data'/'routing-runs'
read = lambda p: json.loads(Path(p).read_text())
FS = {1: 144., 2: 147.}
CATS = ('none', 'one', 'several_no_conflict', 'kappa2', 'kappa3')


def chrom(nodes, edges):
    for k in range(1, len(nodes) + 1):
        for col in itertools.product(range(k), repeat=len(nodes)):
            c = dict(zip(nodes, col))
            if all(c[a] != c[b] for a, b in edges): return k
    return 1


def rounds(m):
    camp = RES/f'team-best-combination-map{m}-20261017'
    sys.path.insert(0, str(Path(__file__).resolve().parents[2])); from robot_routing.world import World; sys.path.pop(0)
    W = World(read(camp/'instance.json')); out = []
    for t in read(camp/'plan.json')['trials']:
        H = read(camp/t['path']/f"seed-{t['api_seed_base']}-learned/protocol/result.json")['history']
        for h in H:
            ev = W.evaluate_programs(h['base']); before = ev['F']
            combos = {tuple(c['blocks']): c['F'] for c in h['combinations']}
            single = {b: combos.get((b,), before) for b in range(3)}
            g = {b: max(0., before - single[b]) for b in range(3)}; I = [b for b in range(3) if g[b] > 0]
            edges = [(b, c) for b, c in itertools.combinations(I, 2) if combos[(b, c)] - single[b] - single[c] + before > 1e-9]
            chi = chrom(I, edges); k = h['epoch'] + 1
            def cross(S, b, c):                                       # collision events between the robots of b and c
                y = list(h['base'])
                for q in S:
                    for i in W.groups[q]: y[i] = h['proposals'][q][i]
                return {(tuple(e['pair']), e['time'], e['kind']) for e in W.evaluate_programs(y)['events'] if {W.owner[e['pair'][0]], W.owner[e['pair'][1]]} == {b, c}}
            new = sum(bool(cross([b, c], b, c) - cross([b], b, c) - cross([c], b, c)) for b, c in edges)
            acc = set(h['accepted']); installed_conflict = any(b in acc and c in acc for b, c in edges)
            assert before - h['F'] >= max(g.values()) - 1e-9
            cat = 'none' if not I else 'one' if len(I) == 1 else 'several_no_conflict' if chi == 1 else f'kappa{chi}'
            phase = 'start' if k == 1 else 'repair' if ev['delivered'] < len(W.ids) or ev['collisions'] else 'refine'
            out.append(dict(seed=t['replicate'], round=k, gap=before - FS[m], phase=phase, base_collisions=ev['collisions'], improving=len(I), pairs=len(I) * (len(I) - 1) // 2,
                            conflicts=len(edges), conflicts_new_collision=new, kappa=chi, installed=len(h['accepted']), installed_conflicting_pair=installed_conflict, category=cat))
    return out


def summary(R):
    sev = [r for r in R if r['improving'] >= 2]
    return dict(rounds=len(R), base_collisions_mean=statistics.mean(r['base_collisions'] for r in R), improving_mean=statistics.mean(r['improving'] for r in R), kappa_mean=statistics.mean(r['kappa'] for r in R),
                installed_mean=statistics.mean(r['installed'] for r in R), conflicting_pairs=sum(r['conflicts'] for r in R), pairs=sum(r['pairs'] for r in R),
                conflicting_pairs_new_collision=sum(r['conflicts_new_collision'] for r in R), rounds_installing_conflicting_pair=sum(r['installed_conflicting_pair'] for r in R),
                rounds_with_conflict=sum(r['conflicts'] > 0 for r in R),
                rounds_two_or_more_improving=len(sev), kappa_mean_two_or_more=statistics.mean(r['kappa'] for r in sev) if sev else None,
                categories={c: sum(r['category'] == c for r in R) for c in CATS})


def main():
    out = {}
    for m in (1, 2):
        R = rounds(m); by_round = defaultdict(list)
        for r in R: by_round[r['round']].append(r)
        assert len(by_round[1]) == 20
        out[str(m)] = dict(by_round={k: summary(v) for k, v in sorted(by_round.items())},
                           by_phase={p: summary([r for r in R if r['phase'] == p]) for p in ('start', 'repair', 'refine')}, all=summary(R), rounds=R)
        print(f'map {m}')
        for p, s in out[str(m)]['by_phase'].items():
            print(f"  {p:7s} rounds {s['rounds']:3d} |I| {s['improving_mean']:.2f} kappa {s['kappa_mean']:.2f} installed {s['installed_mean']:.2f} conflicts {s['conflicting_pairs']}/{s['pairs']} "
                  f"kappa(|I|>=2) {s['kappa_mean_two_or_more']} n={s['rounds_two_or_more_improving']} {s['categories']} new-collision {s['conflicting_pairs_new_collision']} "
                  f"rounds with conflict {s['rounds_with_conflict']}, installing a conflicting pair {s['rounds_installing_conflicting_pair']}")
        for k, s in out[str(m)]['by_round'].items():
            print(f"  round {k:2d} n {s['rounds']:2d} base collisions {s['base_collisions_mean']:.2f} {s['categories']} kappa {s['kappa_mean']:.2f} installed {s['installed_mean']:.2f}")
    (HERE/'conflict-dynamics.json').write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
