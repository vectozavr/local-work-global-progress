"""An independent implementation of the routing rules (route replay, collisions and loss), written separately from
robot_routing.world, used by the theory audit to cross-check every recorded plan."""
from collections import deque
from itertools import combinations
import json

from robot_routing.world import MOVES


def replay(spec, robot, program):
    if not isinstance(program, list) or len(program) > spec['horizon']:
        raise ValueError('Invalid command array length')
    cells = {tuple(p) for p in spec['cells']}
    goal = tuple(robot['goal'])
    pos = tuple(robot['start'])
    arrival = 0 if pos == goal else None
    positions = [pos]
    for tick in range(1, spec['horizon'] + 1):
        command = program[tick - 1] if tick <= len(program) else 'wait'
        if not isinstance(command, str) or command not in MOVES:
            raise ValueError('Invalid command')
        if arrival is not None and command != 'wait':
            raise ValueError('Moving after arrival')
        dx, dy = MOVES[command]
        pos = pos[0] + dx, pos[1] + dy
        if pos not in cells:
            raise ValueError('Wall or grid boundary')
        positions.append(pos)
        if pos == goal and arrival is None:
            arrival = tick
    distance = {goal: 0}
    queue = deque([goal])
    while queue:
        x, y = queue.popleft()
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            point = x + dx, y + dy
            if point in cells and point not in distance:
                distance[point] = distance[x, y] + 1
                queue.append(point)
    loss = ((spec['weights']['unfinished'] if arrival is None else 0)
            + (spec['horizon'] if arrival is None else arrival)
            + spec['weights']['distance'] * distance[pos])
    return dict(loss=loss, arrival=arrival, end=list(pos), positions=positions,
                optimum=distance[tuple(robot['start'])])


def parse_final(text):
    found = []
    decoder = json.JSONDecoder()
    i = 0
    while i < len(text):
        if text[i] != '{':
            i += 1
            continue
        try:
            data, n = decoder.raw_decode(text[i:])
        except json.JSONDecodeError:
            i += 1
            continue
        i += n
        if isinstance(data, dict) and isinstance(data.get('programs'), dict):
            found.append(data['programs'])
    if len(found) != 1:
        raise ValueError('Missing or ambiguous final programs object')
    return found[0]


def independent_joint(spec, programs):
    traces = [replay(spec, r, p) for r, p in zip(spec['robots'], programs)]
    events = []
    for i, j in combinations(range(len(traces)), 2):
        a, b = traces[i]['positions'], traces[j]['positions']
        for t in range(spec['horizon'] + 1):
            if a[t] == b[t]:
                events.append(dict(pair=[i, j], time=t, kind='vertex', cell=list(a[t])))
            elif t and a[t - 1] == b[t] and b[t - 1] == a[t] and a[t - 1] != a[t]:
                events.append(dict(pair=[i, j], time=t, kind='edge', edge=[list(a[t - 1]), list(a[t])]))
    return dict(F=sum(t['loss'] for t in traces) + spec['weights']['collision'] * len(events),
                delivered=sum(t['arrival'] is not None for t in traces), collisions=len(events), events=events)


def score(spec, owned, candidate, known):
    traces=[replay(spec,r,p) for r,p in zip(spec['robots'],candidate)]
    events=independent_joint(spec,candidate)['events']
    own=set(owned)
    visible=[e for e in events if set(e['pair'])<=own or
             (tuple(e['pair']) in known and bool(set(e['pair']) & own) and not set(e['pair'])<=own)]
    external=[e for e in visible if not set(e['pair'])<=own]
    value=sum(traces[i]['loss'] for i in own)+spec['weights']['collision']*len(visible)
    return value,visible,external
