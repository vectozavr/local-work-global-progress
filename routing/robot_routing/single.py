"""A single agent that controls all robots (W=1), and best-of-W: W single agents asked in parallel from the same
plan, the best improving answer accepted. Each answer may replace at most K routes (exactly K with exact_updates).
The run resumes from saved answers after an interruption.
"""
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import time

from .files import read, save
from .llm import checkable_programs
from .prompt_single import build_single_prompt
from .world import World, digest

PROMPT_VERSION = 'single-v2-k{K}-best{W}'


def run_best_of(spec, model, root, *, target_F, rounds, seed, width, max_updated_robots,
                no_improvement_patience, exact_updates=False):
    K, W = max_updated_robots, width
    mode = f'single-v2-k{K}{"exact" if exact_updates else ""}-best{W}'
    spec = copy.deepcopy(spec)
    spec['groups'] = [[r['id'] for r in spec['robots']]]
    w = World(spec)
    protocol = root/'protocol'
    (protocol/'attempts').mkdir(parents=True, exist_ok=True)
    statefile = root/'STATE.json'
    if not statefile.exists():
        save(protocol/'instance.json', spec)
        save(statefile, dict(state='RUNNING', started_unix=time.time(), round=0,
             base=[[] for _ in w.ids], log=[], accepted=[], history=[], checked_seconds=0.))
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
        current = w.evaluate_programs(base)
        if current['delivered'] == len(w.ids) and current['collisions'] == 0 and current['F'] <= target_F:
            break
        if nonimproving >= no_improvement_patience:
            break
        r = state['round']
        prompt = build_single_prompt(w, base, iteration=r, target_F=target_F, max_updates=K,
                                     log=state['log'], accepted=state['accepted'], exact=exact_updates)
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
            updates = checkable_programs(record['text'], w.ids, max_updates=K, min_updates=K if exact_updates else 1)
            programs = dict(zip(w.ids, copy.deepcopy(base)))
            programs.update(updates)
            errors = []
            for i, rid in enumerate(w.ids):
                try:
                    w.trace(i, programs[rid])
                except (ValueError, TypeError) as exc:
                    errors.append(f'{rid}: {exc}')
            entry = dict(iteration=r, copy=c if W > 1 else None, updates=updates)
            candidate = [programs[rid] for rid in w.ids]
            if errors:
                entry.update(valid=False, error='; '.join(errors))
                candidates.append(dict(copy=c, valid=False, error=entry['error'], proposed_robot_ids=list(updates)))
            else:
                checked = w.evaluate_programs(candidate)
                entry.update(valid=True, candidate=candidate, F=checked['F'], events=checked['events'])
                candidates.append(dict(copy=c, valid=True, F=checked['F'], candidate=candidate,
                                       proposed_robot_ids=list(updates)))
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
        actual = w.evaluate_programs(state['base'])
        row = dict(round=r, base_F=current['F'], F=actual['F'], accepted=best is not None,
                   best_copy=best['copy'] if best else None,
                   candidate_F=[x.get('F') for x in candidates], valid=[x['valid'] for x in candidates],
                   proposed_robot_ids=[x['proposed_robot_ids'] for x in candidates],
                   delivered=actual['delivered'], collisions=actual['collisions'],
                   response_seconds=[rec['seconds'] for rec in records], round_seconds=round_seconds,
                   checked_seconds=state['checked_seconds'], elapsed_seconds=time.time() - state['started_unix'])
        state['history'].append(row)
        nonimproving = 0 if best is not None else nonimproving + 1
        state.update(round=r + 1)
        save(statefile, state)
        (protocol/'rounds.jsonl').write_text(''.join(json.dumps(h) + '\n' for h in state['history']))
        print(json.dumps(dict(event='best_of_round', **{k: v for k, v in row.items()})), flush=True)
    current = w.evaluate_programs(state['base'])
    reached = current['F'] <= target_F and current['delivered'] == len(w.ids) and not current['collisions']
    answers = sorted((protocol/'attempts').glob('*.json'))
    result = dict(mode=mode, width=W, max_updated_robots=K, exact_updates=bool(exact_updates),
        prompt_version=PROMPT_VERSION.format(K=K, W=W) + ('-exact' if exact_updates else ''),
        spec_sha256=digest(spec), target_F=target_F, reached_target=bool(reached),
        stopping=('target_reached' if reached else 'no_improvement_patience_exhausted'
                  if nonimproving >= no_improvement_patience else 'round_budget_exhausted'),
        no_improvement_patience_rounds=no_improvement_patience, round_budget=rounds,
        checked_rounds=state['round'], checked_attempts=len(answers),
        checked_response_seconds=state['checked_seconds'],
        summed_answer_seconds=sum(read(p)['seconds'] for p in answers),
        wall_seconds=time.time() - state['started_unix'],
        F=current['F'], delivered=current['delivered'], collisions=current['collisions'],
        history=state['history'], final_programs=dict(zip(w.ids, state['base'])), model=model.metadata,
        note='Search time adds, over rounds, the slowest of the W answers; schema/transport retries excluded.')
    save(protocol/'result.json', result)
    state['state'] = 'COMPLETED'
    save(statefile, state)
    return result
