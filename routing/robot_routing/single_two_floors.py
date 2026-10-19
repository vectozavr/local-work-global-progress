"""A single agent and best-of-W on the two-floor task (eighteen robots, one route per answer)."""
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import time

from .files import read, save
from .llm import checkable_programs
from .two_floors import TwoFloors, build_two_floor_prompt
from .world import digest

PROMPT_VERSION = 'two-floor-single-v2-k1-best{W}'


def run_best_of(spec, model, root, *, targets, rounds, seed, width, no_improvement_patience):
    W = width
    mode = f'two-floor-single-v2-k1-best{W}'
    tf = TwoFloors(spec)
    protocol = root/'protocol'
    (protocol/'attempts').mkdir(parents=True, exist_ok=True)
    statefile = root/'STATE.json'
    if not statefile.exists():
        save(protocol/'instance.json', spec)
        save(statefile, dict(state='RUNNING', started_unix=time.time(), round=0,
             base=[[] for _ in tf.ids], log=[], accepted=[], history=[], checked_seconds=0.))
    state = read(statefile)
    if state['state'] == 'COMPLETED':
        return read(protocol/'result.json')
    nonimproving = 0
    for row in reversed(state['history']):
        if row['accepted']:
            break
        nonimproving += 1
    while state['round'] < rounds:
        base = copy.deepcopy(state['base'])
        current = tf.evaluate_programs(base)
        if tf.reached(base, targets):
            break
        if nonimproving >= no_improvement_patience:
            break
        r = state['round']
        prompt = build_two_floor_prompt(tf, base, iteration=r, targets=targets, log=state['log'],
                                        accepted=state['accepted'])
        prompts = [prompt] * W

        def ask(c):
            file = protocol/'attempts'/f'r{r:03d}-c{c}.json'
            if file.exists():
                record = read(file)
                if record['prompt'] != prompts[c]:
                    raise ValueError('Resume prompt differs from saved attempt')
                return record
            answer = model.one(prompts[c], seed + 1000*r + c, call_context=dict(mode=mode, round=r, copy=c))
            record = dict(round=r, copy=c, prompt=prompts[c], base=base, **answer)
            save(file, record)
            return record

        with ThreadPoolExecutor(max_workers=W) as pool:
            records = list(pool.map(ask, range(W)))
        candidates, attempts = [], []
        for c, record in enumerate(records):
            updates = checkable_programs(record['text'], tf.ids, max_updates=1)
            floor = tf.floor_of(next(iter(updates)))
            programs = dict(zip(tf.ids, copy.deepcopy(base)))
            programs.update(updates)
            errors = []
            for k, rid in enumerate(tf.ids):
                try:
                    tf.trace(k, programs[rid])
                except (ValueError, TypeError) as exc:
                    errors.append(f'{rid}: {exc}')
            entry = dict(iteration=r, copy=c if W > 1 else None, updates=updates, floor=floor)
            candidate = [programs[rid] for rid in tf.ids]
            if errors:
                entry.update(valid=False, error='; '.join(errors))
                candidates.append(dict(copy=c, valid=False, error=entry['error'], proposed_robot_ids=list(updates),
                                       floor=floor))
            else:
                checked = tf.evaluate_programs(candidate)
                entry.update(valid=True, candidate=candidate, F=checked['F'],
                             F_floors=[v['F'] for v in checked['floors']], events=checked['floors'][floor]['events'])
                candidates.append(dict(copy=c, valid=True, F=checked['F'], candidate=candidate,
                                       proposed_robot_ids=list(updates), floor=floor))
            attempts.append(entry)
        improving = [x for x in candidates if x['valid'] and x['F'] < current['F']]
        best = min(improving, key=lambda x: (x['F'], x['copy'])) if improving else None
        if best is not None:
            state['base'] = best['candidate']
            state['accepted'].append(dict(iteration=r, robots=best['proposed_robot_ids'],
                                          F_before=current['F'], F_after=best['F']))
            state['log'] = []
        else:
            state['log'].extend(attempts)
        round_seconds = max(rec['seconds'] for rec in records)
        state['checked_seconds'] += round_seconds
        actual = tf.evaluate_programs(state['base'])
        row = dict(round=r, base_F=current['F'], F=actual['F'], F_floors=[v['F'] for v in actual['floors']],
                   accepted=best is not None, best_copy=best['copy'] if best else None,
                   candidate_F=[x.get('F') for x in candidates], valid=[x['valid'] for x in candidates],
                   proposed_robot_ids=[x['proposed_robot_ids'] for x in candidates],
                   proposed_floor=[x['floor'] for x in candidates],
                   delivered=actual['delivered'], collisions=actual['collisions'],
                   collisions_floors=[v['collisions'] for v in actual['floors']],
                   response_seconds=[rec['seconds'] for rec in records], round_seconds=round_seconds,
                   checked_seconds=state['checked_seconds'], elapsed_seconds=time.time() - state['started_unix'])
        state['history'].append(row)
        nonimproving = 0 if best is not None else nonimproving + 1
        state.update(round=r + 1)
        save(statefile, state)
        (protocol/'rounds.jsonl').write_text(''.join(json.dumps(h) + '\n' for h in state['history']))
        print(json.dumps(dict(event='best_of_round', **{k: v for k, v in row.items()})), flush=True)
    current = tf.evaluate_programs(state['base'])
    reached = tf.reached(state['base'], targets)
    answers = sorted((protocol/'attempts').glob('*.json'))
    result = dict(mode=mode, width=W, max_updated_robots=1, prompt_version=PROMPT_VERSION.format(W=W),
        spec_sha256=digest(spec), targets=list(targets), reached_target=bool(reached),
        stopping=('target_reached' if reached else 'no_improvement_patience_exhausted'
                  if nonimproving >= no_improvement_patience else 'round_budget_exhausted'),
        no_improvement_patience_rounds=no_improvement_patience, round_budget=rounds,
        checked_rounds=state['round'], checked_attempts=len(answers),
        checked_response_seconds=state['checked_seconds'],
        summed_answer_seconds=sum(read(p)['seconds'] for p in answers),
        wall_seconds=time.time() - state['started_unix'],
        F=current['F'], F_floors=[v['F'] for v in current['floors']], delivered=current['delivered'],
        collisions=current['collisions'], collisions_floors=[v['collisions'] for v in current['floors']],
        history=state['history'], final_programs=dict(zip(tf.ids, state['base'])), model=model.metadata,
        note='Search time adds, over rounds, the slowest of the W answers; schema/transport retries excluded.')
    save(protocol/'result.json', result)
    state['state'] = 'COMPLETED'
    save(statefile, state)
    return result
