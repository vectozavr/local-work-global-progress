"""Run a robot-routing campaign of the paper with live LLM agents.

    python routing/run_campaign.py --list
    python routing/run_campaign.py map1-team --seeds 0-2            # prepare runs/map1-team, no model calls
    python routing/run_campaign.py map1-team --seeds 0-2 --execute  # run seeds 0, 1 and 2

A campaign (routing/campaigns/<name>.json) fixes the map, the protocol, the model and its sampling, and the stopping
rule. Each seed is one run; runs are executed one after another, each followed by an independent replay that writes
the campaign's summary. --execute reads OPENROUTER_API_KEY from the environment or from .env at the repository root.
Every request and answer is saved, so an interrupted campaign continues where it stopped when the command is rerun.
"""
import argparse
from argparse import Namespace
import hashlib
import json
import sys
import time
from pathlib import Path

ROUTING = Path(__file__).resolve().parent
REPO = ROUTING.parent
sys.path.insert(0, str(ROUTING))

from robot_routing import single, single_two_floors, summary, team  # noqa: E402
from robot_routing.files import read, save                         # noqa: E402
from robot_routing.llm import load_api_key, openrouter_model        # noqa: E402


def seed_list(text):
    out = []
    for part in text.split(','):
        lo, _, hi = part.partition('-')
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def prepare(campaign, root, seeds):
    """Write the run folder: plan (settings and one trial per seed) and map. Refuses a folder of another campaign."""
    code = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROUTING/'robot_routing').glob('*.py'))}
    trials = [dict(id=f"{campaign['campaign']}-{s:02d}", replicate=s, api_seed_base=4000000 + 100000 * s,
                   path=campaign['trial_path'].format(seed=s)) for s in seeds]
    if (root/'plan.json').exists():
        plan = read(root/'plan.json')
        if plan['campaign'] != campaign['campaign'] or plan['settings'] != campaign['settings']:
            sys.exit(f'{root} holds another campaign or other settings; choose another --out')
        known = {t['id'] for t in plan['trials']}
        plan['trials'] += [t for t in trials if t['id'] not in known]
        if plan['code_sha256'] != code:
            print('note: routing/robot_routing changed since this folder was created; new runs use the current code')
    else:
        root.mkdir(parents=True)
        (root/'instance.json').write_text((ROUTING/'maps'/campaign['map']/'instance.json').read_text())
        plan = dict(campaign=campaign['campaign'], description=campaign['description'], runner=campaign['runner'],
                    settings=campaign['settings'], code_sha256=code, trials=trials,
                    created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    # flat copies of the settings, as read by the summaries
    plan.update({k: v for k, v in campaign['settings'].items() if k not in plan})
    save(root/'plan.json', plan)
    return plan


def run_trial(plan, trial, root, spec, key):
    s = plan['settings']; seed = trial['api_seed_base']; directory = root/trial['path']
    resume = directory.exists(); directory.mkdir(parents=True, exist_ok=True)
    if plan['runner'] == 'team':
        protocol_dir = directory/f'seed-{seed}-learned'; protocol_dir.mkdir(exist_ok=True)
        model = openrouter_model(s, key, journal=protocol_dir/'api-requests.jsonl')
        args = Namespace(H=s['team_H'], epochs=s['round_budget'], seed=seed, no_improvement_patience=s['patience_rounds'],
                         target_F=s['target_F'], merge=s.get('merge', 'best_single'),
                         no_correction=s.get('cross_block_correction', True) is False, resume=resume,
                         transport_attempts=s['transport_attempts'], format_attempts=s['format_attempts'])
        return team.run(args, spec, model, directory)
    model = openrouter_model(s, key, journal=directory/'api-requests.jsonl')
    if plan['runner'] == 'single':
        return single.run_best_of(spec, model, directory, target_F=s['target_F'], rounds=s['round_budget'], seed=seed,
                                  width=s['width'], max_updated_robots=s['max_updated_robots'],
                                  no_improvement_patience=s['patience_rounds'], exact_updates=s.get('exact_updates', False))
    return single_two_floors.run_best_of(spec, model, directory, targets=s['target_F'], rounds=s['round_budget'], seed=seed,
                                         width=s['width'], no_improvement_patience=s['patience_rounds'])


def finished(plan, trial, root):
    d = root/trial['path']
    return (d/f"seed-{trial['api_seed_base']}-learned/protocol/result.json" if plan['runner'] == 'team' else d/'protocol/result.json').exists()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('campaign', nargs='?', help='a file name in routing/campaigns/, without .json')
    p.add_argument('--seeds', help='e.g. 0-19 or 0,3,5 (default: the seeds of the paper)')
    p.add_argument('--out', type=Path, help='run folder (default: runs/<campaign>)')
    p.add_argument('--execute', action='store_true', help='send requests to the model')
    p.add_argument('--list', action='store_true', help='list the campaigns')
    a = p.parse_args()
    if a.list or not a.campaign:
        for f in sorted((ROUTING/'campaigns').glob('*.json')):
            print(f'{f.stem:28s} {read(f)["paper"]}')
        return
    campaign = read(ROUTING/'campaigns'/f'{a.campaign}.json')
    root = (a.out or REPO/'runs'/a.campaign).resolve()
    seeds = seed_list(a.seeds or campaign['paper_seeds'])
    plan = prepare(campaign, root, seeds)
    todo = [t for t in plan['trials'] if t['replicate'] in seeds and not finished(plan, t, root)]
    print(f"{a.campaign}: {len(seeds)} run{'s' * (len(seeds) != 1)} in {root}, {len(todo)} to do")
    if not a.execute:
        print('add --execute to run them'); return
    try:
        key = load_api_key(REPO/'.env')
    except ValueError:
        sys.exit('Set OPENROUTER_API_KEY in the environment or in a .env file at the repository root (see .env.example).')
    spec = read(root/'instance.json')
    summarize = dict(team=summary.summarize_team, single=summary.summarize_single,
                     two_floors=summary.summarize_two_floors)[plan['runner']]
    for trial in todo:
        print(f"run {trial['id']}", flush=True)
        r = run_trial(plan, trial, root, spec, key)
        print(json.dumps(dict(run=trial['id'], F=r['F'], delivered=r['delivered'], collisions=r['collisions'])), flush=True)
        summarize(root)


if __name__ == '__main__':
    main()
