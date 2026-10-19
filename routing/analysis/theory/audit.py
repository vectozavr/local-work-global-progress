"""Audit of the theory's constants on a team campaign; no model calls.

For every round of every run: the exact block minima (certified by conflict-based search), the realized fraction Q of
the corrected improvement, the exact block improvements D_b, the correction errors at a true block minimizer and at the
returned proposal, and the round-by-round bound of the paper (Proposition A.2), checked against the observed gap.
Needs the full run records (every attempt and calls.jsonl), i.e. a new run or the data archive.

    python routing/analysis/theory/audit.py --root runs/map1-team --out runs/map1-team/theory
    python routing/analysis/theory/audit.py --root <unsafe campaign> --out <dir> --audit-only   # the loss can increase

Definitions:
  Q_kb   = (surrogate_initial - returned) / (surrogate_initial - corrected block minimum)
  D_b    = true block loss at x_k - true block minimum (all pairs)
  alpha_k = sum_b Q_kb D_b / (3 (F(x_k) - F*)),  B_k = sum_b (Q_kb |eps_kb(u*)| + |eps_kb(U_kb)|) / 3
  E_0 = F(x_0) - F*,  E_{k+1} = min(E_k, (1 - alpha_k) E_k + B_k).
"""
import argparse, copy, hashlib, json, statistics, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
MAP2_REFERENCE = HERE.parents[1]/'maps/map2'
REFERENCE = HERE.parents[1]/'maps/map1'
sys.path.insert(0, str(HERE)); sys.dont_write_bytecode = True
import audit_assumptions as AA                      # Certificates, score, independent_joint, parse_final, replay
from audit_assumptions import World, independent_joint, score, parse_final, replay, external_count

read = lambda p: json.loads(Path(p).read_text())
save = lambda p, d: Path(p).write_text(json.dumps(d, indent=1) + '\n')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
F_STAR = 144
FROZEN_CHECK = [True]


def stats(v):
    return dict(n=len(v), mean=statistics.mean(v), sample_sd=statistics.stdev(v) if len(v) > 1 else None,
                minimum=min(v), maximum=max(v)) if v else None


def audit(roots, out, allow_partial=False):
    """Certified exact block minima and measured search quantities at every recorded state (cf. AA.main).
    roots: one or more campaign roots with the same instance (e.g. seeds 0-9 and 10-19); seeds must be distinct."""
    global F_STAR
    roots = [roots] if isinstance(roots, Path) else list(roots)
    for root in roots:
        if FROZEN_CHECK[0] and (root/'FROZEN-SHA256.json').exists():
            for name, digest in read(root/'FROZEN-SHA256.json').items():
                assert sha(root/name) == digest, name
        assert sha(root/'instance.json') == sha(roots[0]/'instance.json'), 'roots use different maps'
    root = roots[0]
    spec = read(root/'instance.json'); w = World(spec)
    if (root/'private-reference.json').exists():          # map with its own certified reference (e.g. map 2)
        reference_file = root/'private-reference.json'
    elif sha(root/'instance.json') == sha(MAP2_REFERENCE/'instance.json'):
        reference_file = MAP2_REFERENCE/'private-reference.json'
    else:                                                 # map 1: the original certified reference
        assert sha(root/'instance.json') == sha(REFERENCE/'instance.json'), 'map differs from the certified reference'
        reference_file = REFERENCE/'private-reference.json'
    F_STAR = read(reference_file)['optimum']
    reference = read(reference_file)
    witness = independent_joint(spec, reference['programs'])
    assert witness['F'] == reference['optimum'] == F_STAR and witness['delivered'] == 9 and witness['collisions'] == 0
    certs = AA.Certificates(w); allpairs = {(i, j) for i in range(9) for j in range(i + 1, 9)}
    searches, attempts, bases = [], [], []
    trials = [(r, t) for r in roots for t in read(r/'plan.json')['trials'] if t.get('arm', 'team') == 'team']
    assert len({t['replicate'] for _, t in trials}) == len(trials), 'duplicate seeds across roots'
    for troot, trial in sorted(trials, key=lambda x: x[1]['replicate']):
        protocol = troot/trial['path']/f"seed-{trial['api_seed_base']}-learned/protocol"
        if allow_partial and not (protocol/'result.json').exists():
            continue
        result = read(protocol/'result.json')
        calls = [json.loads(s) for s in (protocol/'calls.jsonl').read_text().splitlines()]
        history = result['history']
        states = [(h['base'], set(map(tuple, h['known_before'])), h) for h in history]
        states.append((history[-1]['next'], set(map(tuple, history[-1]['known_after'])), None))
        for k, (base, known, h) in enumerate(states):
            actual = independent_joint(spec, base); assert actual['F'] == w.evaluate_programs(base)['F']
            ds, blocks = [], []
            for b, owned in enumerate(w.groups):
                initial = score(spec, owned, base, known)[0]; true_initial = score(spec, owned, base, allpairs)[0]
                local_ref = certs.minimum(b, base, known); true_ref = certs.minimum(b, base, allpairs)
                minimum, true_minimum = local_ref['exact_minimum'], true_ref['exact_minimum']
                D = true_initial - true_minimum; assert D >= 0; ds.append(D)
                delta = certs.uniform_error(b, base, known)
                blocks.append(dict(block=b, surrogate_initial=initial, surrogate_minimum=minimum, true_initial=true_initial,
                                   true_minimum=true_minimum, D=D, uniform_error=delta,
                                   surrogate_minimum_certificate=local_ref, true_minimum_certificate=true_ref))
                if h is None:
                    continue
                returned = score(spec, owned, h['proposals'][b], known)[0]
                gap, gain = initial - minimum, initial - returned
                assert 0 <= gain <= gap
                q = None if gap == 0 else gain / gap
                group_calls = sorted([c for c in calls if c['epoch'] == k and c['block'] == b], key=lambda c: c['step'])
                best = initial
                for j, c in enumerate(group_calls):
                    programs = parse_final(c['text']); errors = {}
                    assert set(programs) == {w.ids[i] for i in owned}
                    for i in owned:
                        try: replay(spec, spec['robots'][i], programs[w.ids[i]])
                        except (ValueError, TypeError, KeyError) as exc: errors[w.ids[i]] = str(exc)
                    assert bool(errors) == (not c['valid'])
                    row = dict(seed=trial['replicate'], round=k + 1, block=b, step=j + 1, route_valid=not errors, errors=errors)
                    if not errors:
                        cand = copy.deepcopy(base)
                        for i in owned: cand[i] = programs[w.ids[i]]
                        assert cand == c['candidate']
                        ev = independent_joint(spec, cand); value = score(spec, owned, cand, known)[0]
                        assert value - initial == c['score']
                        f = ev['F'] - actual['F']; s = value - initial; error = s - f
                        lo, hi = delta['interval_of_surrogate_minus_true_change']
                        assert lo <= error <= hi
                        best = min(best, value)
                        row.update(candidate_F=ev['F'], true_change=f, surrogate_change=s, error=error, absolute_error=abs(error))
                    row.update(best_score=best, realized_q=None if gap == 0 else (initial - best) / gap)
                    attempts.append(row)
                assert best == returned
                ev = independent_joint(spec, h['proposals'][b]); error = (returned - initial) - (ev['F'] - actual['F'])
                searches.append(dict(seed=trial['replicate'], round=k + 1, block=b, initial_gap=gap, surrogate_gain=gain,
                    realized_q=q, already_optimal=gap == 0, attempts=len(group_calls), reached_surrogate_optimum=returned == minimum,
                    globally_accepted=b in h['accepted'], true_D=D, verified_gain=max(0, actual['F'] - ev['F']),
                    true_gain_fraction=None if D == 0 else max(0, actual['F'] - ev['F']) / D,
                    returned_error=error, absolute_returned_error=abs(error), exact_uniform_delta=delta['exact_uniform_delta']))
                assert q is None or max(0, actual['F'] - ev['F']) >= q * D - abs(error) - 1e-9
            gap_F = actual['F'] - F_STAR; assert gap_F >= 0 and (gap_F > 0 or h is None), 'only a terminal state may be optimal'
            bases.append(dict(seed=trial['replicate'], state=k, terminal=h is None, base=base,
                              known_pairs=[list(p) for p in sorted(known)], F=actual['F'], global_gap=gap_F, D=ds,
                              sum_D=sum(ds), observed_mu_ratio=sum(ds) / gap_F if gap_F > 0 else None, blocks=blocks))
        print(json.dumps(dict(seed=trial['replicate'], states=len(states))), flush=True)
    meaningful = [s for s in searches if s['realized_q'] is not None]
    valid = [a for a in attempts if a['route_valid']]
    K = max(s['round'] for s in searches)
    summary = dict(model_requests_made=0, campaign='+'.join(r.name for r in roots), team_runs=len({s['seed'] for s in searches}), global_rounds=len(searches) // 3,
        max_rounds=K, reached_bases=len(bases), searches=len(searches), checked_attempts=len(attempts),
        q_positive_gap=stats([s['realized_q'] for s in meaningful]),
        q_by_block={str(b): stats([s['realized_q'] for s in meaningful if s['block'] == b]) for b in range(3)},
        q_by_round={str(k): stats([s['realized_q'] for s in meaningful if s['round'] == k]) for k in range(1, K + 1)},
        weighted_gap_fraction=sum(s['surrogate_gain'] for s in meaningful) / sum(s['initial_gap'] for s in meaningful),
        q_zeros=sum(s['realized_q'] == 0 for s in meaningful), q_ones=sum(s['realized_q'] == 1 for s in meaningful),
        returned_surrogate_optima=sum(s['reached_surrogate_optimum'] for s in searches),
        observed_mu=stats([v['observed_mu_ratio'] for v in bases if v['observed_mu_ratio'] is not None]),
        observed_mu_certificate=min(1., min(v['observed_mu_ratio'] for v in bases if v['observed_mu_ratio'] is not None)),
        optimal_states=sum(v['global_gap'] == 0 for v in bases),
        suboptimal_single_block_traps=sum(v['sum_D'] == 0 and v['global_gap'] > 0 for v in bases),
        uniform_delta=stats([b['uniform_error']['exact_uniform_delta'] for v in bases for b in v['blocks']]),
        valid_attempts=len(valid), route_invalid_attempts=len(attempts) - len(valid),
        evaluated_error=stats([a['absolute_error'] for a in valid]),
        returned_error=stats([s['absolute_returned_error'] for s in searches]),
        returned_nonzero_errors=sum(s['absolute_returned_error'] != 0 for s in searches),
        true_gain_fraction=stats([s['true_gain_fraction'] for s in searches if s['true_gain_fraction'] is not None]),
        local_improvement_without_global_improvement=sum(s['surrogate_gain'] > 0 and s['verified_gain'] == 0 for s in searches))
    out.mkdir(parents=True, exist_ok=True)
    for name, data in [('summary', summary), ('searches', searches), ('attempts', attempts), ('bases', bases)]:
        save(out/f'assumptions-{name}.json', data)
    return summary, bases, searches


def trajectory(roots, out, bases, searches):
    """Per-round alpha/B, one-step inequality, and per-seed envelopes held after stop (cf. PRC.trajectory_rate_data)."""
    roots = [roots] if isinstance(roots, Path) else list(roots); root = roots[0]
    spec = read(root/'instance.json'); w = World(spec)
    lookup = {(v['seed'], v['round'], v['block']): v for v in searches}
    allpairs = {(i, j) for i in range(9) for j in range(i + 1, 9)}
    seeds = sorted({v['seed'] for v in bases})
    rounds, observed = [], {}
    for seed in seeds:
        states = sorted([v for v in bases if v['seed'] == seed], key=lambda v: v['state']); assert states[-1]['terminal']
        observed[seed] = [v['F'] for v in states]
        for k, base in enumerate(states[:-1]):
            progress = err = 0.; blocks = []; gains = []
            for block in base['blocks']:
                b = block['block']; s = lookup[seed, k + 1, b]; owned = w.groups[b]
                cert = block['true_minimum_certificate']; assert cert['minimum_certified']
                cand = copy.deepcopy(base['base'])
                for i, p in zip(owned, cert['hard_routing']['programs']): cand[i] = p
                private = score(spec, owned, cand, set(map(tuple, base['known_pairs'])))[0]
                true = score(spec, owned, cand, allpairs)[0]; assert true == block['true_minimum'] == cert['exact_minimum']
                opt_err = abs((private - block['surrogate_initial']) - (true - block['true_initial']))
                q = s['realized_q']
                if q is None:
                    assert s['initial_gap'] == 0; q = 0.
                e = q * opt_err + s['absolute_returned_error']
                assert s['verified_gain'] + 1e-8 >= q * block['D'] - e
                progress += q * block['D']; err += e; gains.append(s['verified_gain'])
                blocks.append(dict(block=b, Q=q, D=block['D'], true_minimizer_error=opt_err, returned_error=s['absolute_returned_error']))
            alpha = progress / (3 * base['global_gap']); B = err / 3; nxt = states[k + 1]
            assert 0 <= alpha <= 1 and base['F'] - nxt['F'] >= max(gains) - 1e-9
            assert nxt['global_gap'] <= (1 - alpha) * base['global_gap'] + B + 1e-8
            rounds.append(dict(seed=seed, round=k + 1, alpha=alpha, B=B, blocks=blocks, initial_F=base['F'], next_F=nxt['F'],
                               one_step_gap_rhs=(1 - alpha) * base['global_gap'] + B, one_step_inequality_verified=True))
    K = max(len(v) for v in observed.values()) - 1
    obs = np.array([observed[s] + [observed[s][-1]] * (K + 1 - len(observed[s])) for s in seeds], dtype=float)
    E0 = float(obs[0, 0]) - F_STAR; assert np.all(obs[:, 0] == obs[0, 0])
    env = []
    for i, seed in enumerate(seeds):
        e = E0; path = [F_STAR + e]
        for r in [v for v in rounds if v['seed'] == seed]:
            r['previous_envelope_gap'] = e
            e = min(e, (1 - r['alpha']) * e + r['B']); path.append(F_STAR + e)
            assert obs[i, len(path) - 1] <= path[-1] + 1e-8
            r['next_envelope_F'] = path[-1]; r['envelope_inequality_verified'] = True
        env.append(path + [path[-1]] * (K + 1 - len(path)))
    env = np.array(env); assert np.all(env + 1e-8 >= obs)
    data = dict(kind='Verified retrospective upper envelopes on all recorded trajectories (team v2).', campaign='+'.join(r.name for r in roots),
        averaging='Propagate each seed with its measured round coefficients, hold after stopping, then average the envelopes.',
        H=3, kappa=3, optimum=F_STAR, initial_F=float(obs[0, 0]), max_rounds=K,
        alpha_definition='sum_b Q_kb D_b(x_k) / (3 (F(x_k)-144))',
        B_definition='sum_b (Q_kb |epsilon_kb(u_star)| + |epsilon_kb(U_kb)|) / 3',
        descriptive_mean_alpha=statistics.mean(v['alpha'] for v in rounds), descriptive_mean_B=statistics.mean(v['B'] for v in rounds),
        descriptive_means_used_for_propagation=False,
        recurrence=f'E_s,0={E0:g}; E_s,k+1=min(E_s,k,(1-alpha_s,k)*E_s,k+B_s,k); U_K=144+mean_s E_s,min(K,K_s)',
        coefficients_measured_on_same_runs=True, fitted_to_loss_curve=False, model_requests_made=0,
        rounds=list(range(K + 1)), upper_per_seed_F=env.tolist(), upper_mean_F=env.mean(0).tolist(),
        upper_minimum_F=env.min(0).tolist(), upper_maximum_F=env.max(0).tolist(),
        observed_per_seed_F=obs.tolist(), observed_mean_F=obs.mean(0).tolist(), observed_minimum_F=obs.min(0).tolist(),
        observed_maximum_F=obs.max(0).tolist(), final_upper_mean_F=float(env[:, -1].mean()),
        final_observed_mean_F=float(obs[:, -1].mean()), bound_verified_per_seed_per_round=True,
        all_true_minimizer_errors_zero=all(b['true_minimizer_error'] == 0 for r in rounds for b in r['blocks']),
        rounds_total=len(rounds), rounds_per_seed={str(s): sum(r['seed'] == s for r in rounds) for s in seeds},
        mean_Q_by_block=[statistics.mean(v['realized_q'] for v in searches if v['block'] == b and v['realized_q'] is not None) for b in range(3)],
        mean_returned_absolute_error_by_block=[statistics.mean(v['absolute_returned_error'] for v in searches if v['block'] == b) for b in range(3)],
        terminal_hold='Both loss and upper envelope are held after stopping.',
        per_round_coefficients=rounds)
    save(out/'trajectory-rate.json', data)
    return data


def heldout(data, out):
    """Calibrate per-round coefficients on half the seeds, predict the other half's mean gap (cf. heldout.py)."""
    gap = [[f - F_STAR for f in row] for row in data['observed_per_seed_F']]; K = data['max_rounds']
    coef = {(r['seed'], r['round']): r for r in data['per_round_coefficients']}
    seeds = sorted({r['seed'] for r in data['per_round_coefficients']}); index = {s: i for i, s in enumerate(seeds)}

    def per_round(cal):
        res = []
        for k in range(1, K + 1):
            rows = [coef[(s, k)] for s in cal if (s, k) in coef]
            if not rows: res.append(None); continue
            rates = []
            for r in rows:
                g = gap[index[r['seed']]][k - 1]
                lb = max(b['Q'] * b['D'] - b['Q'] * b['true_minimizer_error'] - b['returned_error'] for b in r['blocks'])
                rates.append(max(0., lb) / g if g > 0 else 0.)
            res.append(dict(alpha=statistics.mean(r['alpha'] for r in rows), B=statistics.mean(r['B'] for r in rows),
                            rate=statistics.mean(rates), n=len(rows)))
        return res

    def predict(st, mode):
        g = [gap[0][0]]
        for k in range(K):
            s = st[k]
            if s is None: g.append(g[-1]); continue
            g.append(min(g[-1], (1 - s['alpha']) * g[-1] + s['B'] if mode == 'theorem' else (1 - s['rate']) * g[-1]))
        return g

    half = len(seeds) // 2; folds = [(seeds[:half], seeds[half:]), (seeds[half:], seeds[:half])]
    res = []
    for cal, test in folds:
        st = per_round(cal)
        ob = [statistics.mean(gap[index[s]][k] for s in test) for k in range(K + 1)]
        th, bb = predict(st, 'theorem'), predict(st, 'best')
        res.append(dict(calibration=cal, test=test, observed=ob, theorem_form=th, best_block_form=bb,
                        theorem_above_observed_every_round=all(t + 1e-9 >= o for t, o in zip(th, ob))))
    save(out/'heldout.json', res)
    return res


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, nargs='+', required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--allow-partial', action='store_true'); p.add_argument('--skip-frozen-check', action='store_true')
    p.add_argument('--audit-only', action='store_true', help='team unsafe: the loss can increase, so no envelope')
    a = p.parse_args(); t0 = time.time()
    if a.skip_frozen_check: FROZEN_CHECK[0] = False
    roots = [r.resolve() for r in a.root]
    summary, bases, searches = audit(roots, a.out.resolve(), a.allow_partial)
    if a.audit_only:
        print(json.dumps(dict(mu_certificate=summary['observed_mu_certificate'], q_mean=summary['q_positive_gap']['mean'],
                              rounds=summary['global_rounds'], seconds=round(time.time() - t0, 1)))); sys.exit(0)
    data = trajectory(roots, a.out.resolve(), bases, searches)
    ho = heldout(data, a.out.resolve())
    print(json.dumps(dict(rounds=data['rounds_total'], max_rounds=data['max_rounds'],
        final_observed_mean_F=round(data['final_observed_mean_F'], 2), final_upper_mean_F=round(data['final_upper_mean_F'], 2),
        mean_alpha=round(data['descriptive_mean_alpha'], 3), mean_B=round(data['descriptive_mean_B'], 2),
        mu_certificate=summary['observed_mu_certificate'], q_mean=summary['q_positive_gap']['mean'],
        heldout=[(round(h['theorem_form'][-1], 1), round(h['observed'][-1], 1), h['theorem_above_observed_every_round']) for h in ho],
        seconds=round(time.time() - t0, 1))))
