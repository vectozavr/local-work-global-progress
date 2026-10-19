"""Independent replay and summary of a finished campaign, from its saved answers alone; no model calls.

summarize_single, summarize_two_floors and summarize_team re-check every round of every trial (proposals, losses,
the accepted plan, search time) and write summary.json with the statistics of the campaign and mean loss curves.
run_campaign.py calls them after every trial.
"""
import copy
from itertools import combinations
import json
import time

from .files import read, rows, save, stats, step_curves
from .llm import checkable_programs, parse_program_response
from .two_floors import TwoFloors
from .world import World

def _replay_single(root, trial, plan):
    directory = root/trial['path']; protocol = directory/'protocol'
    result = read(protocol/'result.json'); spec = read(protocol/'instance.json'); w = World(spec)
    K, W = plan['max_updated_robots'], plan['width']
    assert result['max_updated_robots'] == K and result['width'] == W
    base = [[] for _ in w.ids]; seconds = 0.; trace = []
    for row in result['history']:
        r = row['round']; current = w.evaluate_programs(base)
        assert current['F'] == row['base_F']
        best = None; times = []
        for c in range(W):
            a = read(protocol/'attempts'/f'r{r:03d}-c{c}.json')
            assert a['round'] == r and a['copy'] == c and a['base'] == base
            exact = plan.get('exact_updates', False)
            updates = checkable_programs(a['text'], w.ids, max_updates=K, min_updates=K if exact else 1)
            assert (len(updates) == K) if exact else (1 <= len(updates) <= K)
            programs = dict(zip(w.ids, copy.deepcopy(base))); programs.update(updates)
            try:
                for i, rid in enumerate(w.ids): w.trace(i, programs[rid])
                candidate = [programs[rid] for rid in w.ids]; F = w.evaluate_programs(candidate)['F']
                assert row['valid'][c] and row['candidate_F'][c] == F
                if F < current['F'] and (best is None or F < best[0]): best = (F, c, candidate)
            except (ValueError, TypeError):
                assert not row['valid'][c]
            times.append(a['seconds'])
        assert row['accepted'] == (best is not None) and row['best_copy'] == (best[1] if best else None)
        if best: base = best[2]
        assert abs(max(times) - row['round_seconds']) < 1e-9
        seconds += max(times); after = w.evaluate_programs(base)
        assert after['F'] == row['F'] and abs(seconds - row['checked_seconds']) < 1e-6
        trace.append(dict(round=r, checked_seconds=seconds, F=after['F']))
    final = w.evaluate_programs(base)
    assert final['F'] == result['F'] and final['delivered'] == result['delivered'] and final['collisions'] == result['collisions']
    assert base == [result['final_programs'][rid] for rid in w.ids]
    assert abs(seconds - result['checked_response_seconds']) < 1e-6
    journal = rows(directory/'api-requests.jsonl')
    cost = sum(((x.get('response') or {}).get('usage') or {}).get('cost') or 0. for x in journal)
    return dict(trial=trial, final=dict(F=final['F'], delivered=final['delivered'], collisions=final['collisions']),
                checked_seconds=seconds, rounds=result['checked_rounds'], answers=result['checked_attempts'],
                wall_seconds=result['wall_seconds'], stopping=result['stopping'], reached_target=result['reached_target'],
                requests=len(journal), http_errors=sum(1 for x in journal if x.get('error')),
                reported_cost_usd=cost, trace=trace)

def summarize_single(root):
    """Replay every finished trial of a single-agent or best-of-W campaign; write summary.json and mean curves."""
    plan = read(root/'plan.json'); reports = []
    for trial in plan['trials']:
        if (root/trial['path']/'protocol/result.json').exists():
            reports.append(_replay_single(root, trial, plan))
    safe = lambda r: r['final']['delivered'] == 9 and r['final']['collisions'] == 0
    fields = {'final_F': [r['final']['F'] for r in reports], 'checked_seconds': [r['checked_seconds'] for r in reports],
              'answers': [r['answers'] for r in reports], 'rounds': [r['rounds'] for r in reports],
              'wall_seconds': [r['wall_seconds'] for r in reports], 'reported_cost_usd': [r['reported_cost_usd'] for r in reports],
              'time_to_target_successes_only': [r['checked_seconds'] for r in reports if r['reached_target']]}
    summary = dict(updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), planned=len(plan['trials']),
                   completed=len(reports), all_planned_completed=len(reports) == len(plan['trials']),
                   safe_complete=sum(map(safe, reports)), strict_target=sum(r['reached_target'] for r in reports),
                   statistics={k: stats(v) for k, v in fields.items()},
                   reported_total_cost_usd=sum(fields['reported_cost_usd']), runs=reports,
                   independent_replay_passed=True)
    save(root/'summary.json', summary)
    if reports:
        arm = plan.get('arm', f"single_best{plan['width']}")
        save(root/'curves-checked_seconds.json', {arm: step_curves(reports, initial=plan['initial_F'])})
    print(json.dumps(dict(completed=len(reports), safe=summary['safe_complete'], target=summary['strict_target'],
                          final_F=summary['statistics']['final_F'])))


def _replay_two_floors(root, trial, plan):
    directory = root/trial['path']; protocol = directory/'protocol'
    result = read(protocol/'result.json'); spec = read(protocol/'instance.json'); tf = TwoFloors(spec)
    assert spec == read(root/'instance.json')
    W = plan['width']; targets = plan['target_F']
    assert result['width'] == W and result['targets'] == targets
    base = [[] for _ in tf.ids]; seconds = 0.; trace = []
    for row in result['history']:
        r = row['round']; current = tf.evaluate_programs(base)
        assert current['F'] == row['base_F'] and not tf.reached(base, targets)
        best = None; times = []
        for c in range(W):
            a = read(protocol/'attempts'/f'r{r:03d}-c{c}.json')
            assert a['round'] == r and a['copy'] == c and a['base'] == base
            updates = checkable_programs(a['text'], tf.ids, max_updates=1)
            assert len(updates) == 1 and row['proposed_floor'][c] == tf.floor_of(next(iter(updates)))
            programs = dict(zip(tf.ids, copy.deepcopy(base))); programs.update(updates)
            try:
                for k, rid in enumerate(tf.ids): tf.trace(k, programs[rid])
                candidate = [programs[rid] for rid in tf.ids]; F = tf.evaluate_programs(candidate)['F']
                assert row['valid'][c] and row['candidate_F'][c] == F
                if F < current['F'] and (best is None or F < best[0]): best = (F, c, candidate)
            except (ValueError, TypeError):
                assert not row['valid'][c]
            times.append(a['seconds'])
        assert row['accepted'] == (best is not None) and row['best_copy'] == (best[1] if best else None)
        if best: base = best[2]
        assert abs(max(times) - row['round_seconds']) < 1e-9
        seconds += max(times); after = tf.evaluate_programs(base)
        assert after['F'] == row['F'] and [v['F'] for v in after['floors']] == row['F_floors']
        assert abs(seconds - row['checked_seconds']) < 1e-6
        trace.append(dict(round=r, checked_seconds=seconds, F=after['F'], F_floors=[v['F'] for v in after['floors']],
                          delivered=after['delivered'], collisions=after['collisions']))
    final = tf.evaluate_programs(base)
    assert final['F'] == result['F'] and final['delivered'] == result['delivered'] and final['collisions'] == result['collisions']
    assert base == [result['final_programs'][rid] for rid in tf.ids]
    assert abs(seconds - result['checked_response_seconds']) < 1e-6
    assert result['reached_target'] == tf.reached(base, targets)
    journal = rows(directory/'api-requests.jsonl')
    cost = sum(((x.get('response') or {}).get('usage') or {}).get('cost') or 0. for x in journal)
    return dict(trial=trial, final=dict(F=final['F'], F_floors=[v['F'] for v in final['floors']], delivered=final['delivered'],
                                        collisions=final['collisions'], collisions_floors=[v['collisions'] for v in final['floors']]),
                checked_seconds=seconds, rounds=result['checked_rounds'], answers=result['checked_attempts'],
                wall_seconds=result['wall_seconds'], stopping=result['stopping'], reached_target=result['reached_target'],
                requests=len(journal), http_errors=sum(1 for x in journal if x.get('error')),
                reported_cost_usd=cost, trace=trace)


def summarize_two_floors(root):
    plan = read(root/'plan.json'); reports = []
    n = len(TwoFloors(read(root/'instance.json')).ids)
    for trial in plan['trials']:
        if (root/trial['path']/'protocol/result.json').exists():
            reports.append(_replay_two_floors(root, trial, plan))
    safe = lambda r: r['final']['delivered'] == n and r['final']['collisions'] == 0
    fields = {'final_F': [r['final']['F'] for r in reports], 'checked_seconds': [r['checked_seconds'] for r in reports],
              'answers': [r['answers'] for r in reports], 'rounds': [r['rounds'] for r in reports],
              'wall_seconds': [r['wall_seconds'] for r in reports], 'reported_cost_usd': [r['reported_cost_usd'] for r in reports],
              'time_to_target_successes_only': [r['checked_seconds'] for r in reports if r['reached_target']]}
    summary = dict(updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), planned=len(plan['trials']),
                   completed=len(reports), all_planned_completed=len(reports) == len(plan['trials']),
                   safe_complete=sum(map(safe, reports)), strict_target=sum(r['reached_target'] for r in reports),
                   statistics={k: stats(v) for k, v in fields.items() if v},
                   reported_total_cost_usd=sum(fields['reported_cost_usd']), runs=reports,
                   independent_replay_passed=True)
    save(root/'summary.json', summary)
    if reports:
        save(root/'curves-checked_seconds.json', {plan.get('arm', plan.get('campaign', 'two_floors')): step_curves(reports, initial=plan['initial_F'])})
    print(json.dumps(dict(completed=len(reports), safe=summary['safe_complete'], target=summary['strict_target'],
                          final_F=summary['statistics'].get('final_F'))))


def _step_curve(reports, clock, initial):
    import numpy as np
    x = np.array(sorted({0.} | {r[clock] for rep in reports for r in rep['trace']} | {rep[clock] for rep in reports}))
    m = []
    for rep in reports:
        t = np.array([0.] + [r[clock] for r in rep['trace']]); y = np.array([initial] + [r['F'] for r in rep['trace']])
        m.append(y[np.searchsorted(t, x, side='right') - 1])
    m = np.array(m)
    return dict(seconds=x.tolist(), mean=m.mean(0).tolist(), minimum=m.min(0).tolist(), maximum=m.max(0).tolist(),
                per_run=m.tolist(), runs=len(reports), terminal_extension='Hold final installed loss; no further API work')


def block_value(w, b, programs, known):
    own = set(w.groups[b])
    need = own | {j if i in own else i for i, j in known if (i in own) != (j in own)}
    tr = {i: w.trace(i, programs[i]) for i in need}; ev = []
    for i, j in combinations(range(len(w.ids)), 2):
        if {i, j} <= own or ((i, j) in known and (i in own) != (j in own)):
            ev += w.collisions(tr[i].positions, tr[j].positions, i, j)
    return sum(tr[i].cost for i in own) + w.M * len(ev)


def _replay_team(run_dir, seed, correction, World, parse, merge='best_single'):
    proto = run_dir/f'seed-{seed}-learned/protocol'
    result = read(proto/'result.json'); spec = read(proto/'instance.json'); w = World(spec)
    assert result['cross_block_correction'] == correction if 'cross_block_correction' in result else correction
    base = [[] for _ in w.ids]; known = set(); par = seq = 0.; trace = []; calls_seen = 0
    best_plan = (w.evaluate_programs(base)['F'], copy.deepcopy(base))
    for h in result['history']:
        e = h['epoch']; assert h['base'] == base, (run_dir, e, 'base')
        assert set(map(tuple, h['known_before'])) == known, (run_dir, e, 'known')
        before = w.evaluate_programs(base)['F']; proposals = []; times = []
        for b, group in enumerate(w.groups):
            baseline = block_value(w, b, base, known); best = 0.; proposal = copy.deepcopy(base); t = 0.
            steps = sorted(proto.glob(f'attempts/e{e:02d}-b{b}-s*.json'))
            assert len(steps) == h['local_attempts'][b], (run_dir, e, b, 'attempt count')
            for f in steps:
                a = read(f); t += a['seconds']; calls_seen += 1
                try:
                    d = parse(a['text']); owned = {w.ids[i] for i in group}
                    if set(d) != owned: raise ValueError('ids')
                    cand = copy.deepcopy(base)
                    for i in group: w.trace(i, d[w.ids[i]]); cand[i] = d[w.ids[i]]
                except (ValueError, TypeError, KeyError):
                    assert not a['valid'], (f, 'should be invalid'); continue
                assert a['valid'], (f, 'should be valid')
                score = block_value(w, b, cand, known) - baseline
                assert abs(score - a['score']) < 1e-9, (f, score, a['score'])
                if score < best: best = score; proposal = cand
            assert proposal == h['proposals'][b], (run_dir, e, b, 'proposal')
            proposals.append(proposal); times.append(t)
        checked = [w.evaluate_programs(p) for p in proposals]
        assert [c['F'] for c in checked] == h['verified_F'], (run_dir, e, 'verified F')
        gains = [max(0, before - c['F']) for c in checked]; combos = []
        if merge == 'best_single':
            sel = max(range(len(gains)), key=lambda b: (gains[b], -b))
            nxt = proposals[sel] if gains[sel] > 0 else base; acc = [sel] if gains[sel] > 0 else []
        else:
            changed = [b for b in range(len(w.groups)) if any(proposals[b][i] != base[i] for i in w.groups[b])]
            def merged(S):
                x = copy.deepcopy(base)
                for b in S:
                    for i in w.groups[b]: x[i] = copy.deepcopy(proposals[b][i])
                return x
            for r in range(1, len(changed) + 1):
                for S in combinations(changed, r):
                    v = w.evaluate_programs(merged(S)); combos.append((list(S), v))
            assert [c['blocks'] for c in h['combinations']] == [c[0] for c in combos] and [c['F'] for c in h['combinations']] == [c[1]['F'] for c in combos]
            if merge == 'best_subset':
                good = [c for c in combos if c[1]['F'] < before - 1e-9]
                choice = min(good, key=lambda c: (c[1]['F'], len(c[0]), c[0])) if good else None
                nxt, acc = (merged(choice[0]), choice[0]) if choice else (base, [])
                assert w.evaluate_programs(nxt)['F'] <= min([before] + [c['F'] for c in checked]) + 1e-9
            else:
                nxt, acc = (merged(changed), changed) if changed else (base, [])
        assert h['accepted'] == acc and h['next'] == nxt, (run_dir, e, 'selection')
        cur = w.evaluate_programs(nxt); assert cur['F'] == h['F'], (run_dir, e, 'F')
        if correction:
            for _, c in combos:
                for ev in c['events']:
                    i, j = ev['pair']
                    if w.owner[i] != w.owner[j]: known.add((i, j))
            for c in checked:
                for ev in c['events']:
                    i, j = ev['pair']
                    if w.owner[i] != w.owner[j]: known.add((i, j))
        assert set(map(tuple, h['known_after'])) == known, (run_dir, e, 'known after')
        par += max(times); seq += sum(times); base = nxt
        if cur['F'] < best_plan[0] - 1e-9: best_plan = (cur['F'], copy.deepcopy(nxt))
        bestv = w.evaluate_programs(best_plan[1])
        trace.append(dict(round=e + 1, checked_seconds=par, sequential_seconds=seq, F=bestv['F'], installed_F=cur['F'],
                          delivered=bestv['delivered'], collisions=bestv['collisions'], interference=(sum(gains) - (before - cur['F'])) if len(acc) > 1 else 0.))
    best_programs = base if merge != 'all' or w.evaluate_programs(base)['F'] <= best_plan[0] else best_plan[1]
    final = w.evaluate_programs(best_programs)
    assert best_programs == [result['final_programs'][i] for i in w.ids] and final['F'] == result['F']
    assert result.get('merge', 'best_single') == merge
    assert abs(par - result['checked_parallel_response_seconds']) < 1e-6, (run_dir, par, result['checked_parallel_response_seconds'])
    assert calls_seen == result['model_calls']
    journal = rows(run_dir/f'seed-{seed}-learned/api-requests.jsonl')
    cost = sum(((x.get('response') or {}).get('usage') or {}).get('cost') or 0. for x in journal)
    safe = final['delivered'] == len(w.ids) and final['collisions'] == 0
    return dict(final=dict(F=final['F'], delivered=final['delivered'], collisions=final['collisions']), safe=safe,
                reached_target=bool(result['reached_target']), termination=result['termination'], rounds=len(trace),
                answers=calls_seen, checked_seconds=par, sequential_seconds=seq, wall_seconds=result['wall_seconds'],
                http_requests=len(journal), reported_cost_usd=cost, trace=trace,
                first_safe_target_seconds=next((r['checked_seconds'] for r in trace if r['delivered'] == len(w.ids)
                                                and not r['collisions'] and r['F'] <= result['target_F']), None))


def summarize_team(root, allow_partial=True):
    """Replay every finished trial of a team campaign (any merge rule, with or without correction) from its saved
    answers; write analysis-v2/summary.json and mean curves, with the sequential replay of the same answers."""
    plan = read(root/'plan.json')
    correction = plan.get('cross_block_correction', True) is not False
    reports = []
    for tr in plan['trials']:
        d = root/tr['path']
        if (d/f'seed-{tr["api_seed_base"]}-learned/protocol/result.json').exists():
            rep = _replay_team(d, tr['api_seed_base'], correction, World, parse_program_response, plan.get('merge', 'best_single')); rep['trial'] = tr; reports.append(rep)
    assert allow_partial or len(reports) == len(plan['trials']), f'{len(reports)} of {len(plan["trials"])} trials complete'
    fields = {k: [r[k] for r in reports] for k in ('checked_seconds', 'sequential_seconds', 'rounds', 'answers', 'wall_seconds', 'reported_cost_usd')}
    fields['final_F'] = [r['final']['F'] for r in reports]
    fields['time_to_safe_target_successes_only'] = [r['first_safe_target_seconds'] for r in reports if r['first_safe_target_seconds'] is not None]
    out = root/'analysis-v2'; out.mkdir(exist_ok=True)
    summary = dict(campaign=root.name, correction=correction, planned=len(plan['trials']), completed=len(reports),
                   all_planned_completed=len(reports) == len(plan['trials']), safe_complete=sum(r['safe'] for r in reports),
                   strict_target=sum(r['reached_target'] for r in reports), statistics={k: stats(v) for k, v in fields.items()},
                   reported_total_cost_usd=sum(fields['reported_cost_usd']), independent_replay_passed=True, runs=reports)
    save(out/'summary.json', summary)
    init = plan.get('initial_F', 1476.)
    arm = 'team_' + plan.get('merge', 'best_single')
    curves = {arm: _step_curve(reports, 'checked_seconds', init), arm + '_sequential': _step_curve(reports, 'sequential_seconds', init)}
    if plan.get('merge') == 'all':
        inst = [dict(r, trace=[dict(t, F=t['installed_F']) for t in r['trace']]) for r in reports]
        curves[arm + '_installed'] = _step_curve(inst, 'checked_seconds', init)
    save(out/'curves-checked_seconds.json', curves)
    s = summary['statistics']
    print(json.dumps(dict(campaign=root.name, completed=len(reports), safe=summary['safe_complete'], target=summary['strict_target'],
                          final_F=(round(s['final_F']['mean'], 2), round(s['final_F']['sample_sd'] or 0, 2)),
                          search=(round(s['checked_seconds']['mean'], 1), round(s['checked_seconds']['sample_sd'] or 0, 1)),
                          sequential=round(s['sequential_seconds']['mean'], 1), rounds=s['rounds']['mean'],
                          cost=round(summary['reported_total_cost_usd'], 2))))
