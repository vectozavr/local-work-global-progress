"""Print the first prompt that a campaign sends to the model, built by robot_routing; no model calls.

    python routing/show_prompt.py map1-team            # agent 1 of the team
    python routing/show_prompt.py map1-team --agent 3
    python routing/show_prompt.py two-floors-single

Useful after editing a prompt module (robot_routing/prompt_team.py, prompt_single.py or two_floors.py).
"""
import argparse
import copy
import json
import sys
from pathlib import Path

ROUTING = Path(__file__).resolve().parent
sys.path.insert(0, str(ROUTING))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('campaign'); p.add_argument('--agent', type=int, default=1, help='team campaigns: agent 1, 2 or 3')
    a = p.parse_args()
    c = json.loads((ROUTING/'campaigns'/f'{a.campaign}.json').read_text()); s = c['settings']
    spec = json.loads((ROUTING/'maps'/c['map']/'instance.json').read_text())
    if c['runner'] == 'team':
        from robot_routing import team
        w = team.World(spec); b = a.agent - 1; base = [[] for _ in w.ids]
        value, events = team.available(w, b, base, set())
        print(team.build_team_prompt(w, b, base, set(), value=value, events=events, log=[], global_feedback=None,
                                     epoch=0, step=0, H=s['team_H'], merge=s.get('merge', 'best_single'),
                                     correction=s.get('cross_block_correction', True) is not False))
    elif c['runner'] == 'two_floors':
        from robot_routing.two_floors import TwoFloors, build_two_floor_prompt
        tf = TwoFloors(spec)
        print(build_two_floor_prompt(tf, [[] for _ in tf.ids], iteration=0, targets=s['target_F'], log=[], accepted=[]))
    else:
        from robot_routing.world import World
        from robot_routing.prompt_single import build_single_prompt
        spec = copy.deepcopy(spec); spec['groups'] = [[r['id'] for r in spec['robots']]]; w = World(spec)
        print(build_single_prompt(w, [[] for _ in w.ids], iteration=0, target_F=s['target_F'],
                                  max_updates=s['max_updated_robots'], log=[], accepted=[], exact=s.get('exact_updates', False)))


if __name__ == '__main__':
    main()
