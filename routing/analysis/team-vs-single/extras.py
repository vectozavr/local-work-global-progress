"""Further numbers quoted in the text, from audited files only; no model calls.

  coupling      among returned block proposals that deliver all three own robots without internal collisions, the
                share that collides with another agent's robot (full plan)
  best_block    per-seed bound Delta_{k+1} <= max(0, Delta_k - max_b(Q D - Q e* - e^U)), held after stop
  error_by_round  mean B_k over the runs still active in round k
  bound_ratio   mean upper-envelope gap / mean observed gap per round
Usage: extras.py --team <campaign> --theory <dir with trajectory-rate.json and assumptions-*.json> --out <json>
"""
import argparse, json, statistics, sys
from pathlib import Path

read = lambda p: json.loads(Path(p).read_text())
F_STAR = 144


def coupling(team_roots):
    team_roots = [team_roots] if isinstance(team_roots, Path) else list(team_roots); team_root = team_roots[0]
    spec = read(team_root/'instance.json')
    ids = [r['id'] for r in spec['robots']]; index = {r: i for i, r in enumerate(ids)}
    groups = [{index[r] for r in g} for g in spec['groups']]; owner = {i: b for b, g in enumerate(groups) for i in g}
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from robot_routing.world import World
    w = World(spec)
    local_safe = local_safe_ext = returned = returned_ext = 0
    for troot, t in [(r, t) for r in team_roots for t in read(r/'plan.json')['trials']]:
        if t.get('arm', 'team') != 'team': continue
        p = troot/t['path']/f"seed-{t['api_seed_base']}-learned/protocol/result.json"
        if not p.exists(): continue
        for h in read(p)['history']:
            for b, prog in enumerate(h['proposals']):
                ev = w.evaluate_programs(prog)['events']; own = groups[b]
                tr = [w.trace(i, prog[i]) for i in own]
                intern = [e for e in ev if set(e['pair']) <= own]
                ext = [e for e in ev if set(e['pair']) & own and not set(e['pair']) <= own]
                returned += 1; returned_ext += bool(ext)
                if all(x.arrival is not None for x in tr) and not intern:
                    local_safe += 1; local_safe_ext += bool(ext)
    return dict(returned=returned, returned_with_external=returned_ext, local_safe=local_safe,
                local_safe_with_external=local_safe_ext, percent=100 * local_safe_ext / local_safe)


def bounds(theory):
    data = read(theory/'trajectory-rate.json'); rounds = data['per_round_coefficients']; K = data['max_rounds']
    F_STAR = data['optimum']
    seeds = sorted({r['seed'] for r in rounds}); obs = data['observed_per_seed_F']
    per_seed = []
    for s in seeds:
        g = obs[seeds.index(s)][0] - F_STAR; path = [g]
        for r in sorted([r for r in rounds if r['seed'] == s], key=lambda r: r['round']):
            lb = max(b['Q'] * b['D'] - b['Q'] * b['true_minimizer_error'] - b['returned_error'] for b in r['blocks'])
            g = max(0., g - lb); path.append(g)
            assert r['next_F'] - F_STAR <= g + 1e-8
        per_seed.append(path + [path[-1]] * (K + 1 - len(path)))
    best_final = statistics.mean(p[-1] for p in per_seed)
    up = [u - F_STAR for u in data['upper_mean_F']]; ob = [o - F_STAR for o in data['observed_mean_F']]
    by_round = {}
    for r in rounds: by_round.setdefault(r['round'], []).append(r['B'])
    return dict(best_block_final_mean_gap=best_final, best_block_mean_path=[statistics.mean(p[k] for p in per_seed) for k in range(K + 1)],
                mean_block_final_mean_gap=up[-1], observed_final_mean_gap=ob[-1],
                ratio_upper_to_observed=[u / o for u, o in zip(up, ob)],
                error_by_round={k: statistics.mean(v) for k, v in sorted(by_round.items())},
                runs_active_by_round={k: len(v) for k, v in sorted(by_round.items())})


def assumptions(theory):
    s = read(theory/'assumptions-summary.json')
    return dict(mu=s['observed_mu_certificate'], q_mean=s['q_positive_gap']['mean'],
                returned_error_mean=s['returned_error']['mean'], uniform_delta_mean=s['uniform_delta']['mean'],
                uniform_delta_max=s['uniform_delta']['maximum'], q_by_round={k: (v['mean'] if v else None) for k, v in s['q_by_round'].items()},
                searches=s['searches'], rounds=s['global_rounds'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--team', type=Path, nargs='+', required=True); p.add_argument('--theory', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True); a = p.parse_args()
    out = dict(coupling=coupling([t.resolve() for t in a.team]), bounds=bounds(a.theory.resolve()), assumptions=assumptions(a.theory.resolve()))
    a.out.write_text(json.dumps(out, indent=1) + '\n')
    c, b, s = out['coupling'], out['bounds'], out['assumptions']
    print(json.dumps(dict(coupling_percent=round(c['percent'], 1), coupling=f"{c['local_safe_with_external']}/{c['local_safe']}",
                          best_block_final=round(b['best_block_final_mean_gap'], 2), mean_block_final=round(b['mean_block_final_mean_gap'], 2),
                          observed_final=round(b['observed_final_mean_gap'], 2), ratio=[round(x, 2) for x in b['ratio_upper_to_observed']],
                          error_by_round={k: round(v, 1) for k, v in b['error_by_round'].items()}, mu=s['mu'], q=round(s['q_mean'], 3),
                          returned_error=round(s['returned_error_mean'], 1), uniform_delta_mean=round(s['uniform_delta_mean']))))
