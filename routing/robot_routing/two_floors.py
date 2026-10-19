"""Two independent floors as one routing task, and the single-agent prompt for it.

The instance holds two ordinary single-floor specs ('floors') with distinct robot IDs; robots on
different floors never interact, and F = F_1 + F_2. A plan lists programs in floor order (all
floor-1 robots, then all floor-2 robots). The prompt shows, for each floor, exactly what the
single-floor prompt shows (robot_routing/prompt_single.py): the map with both axes and its
wall list, every program and cost, the tick-by-tick timeline, every current collision with maps
of the tick before and the tick of the collision. Accepted changes and rejected attempts are
listed once for the whole plan. No route, distance to the optimum or repair strategy is supplied.
"""
import copy
import json

from .world import World
from . import prompt_single as P


class TwoFloors:
    def __init__(self, spec):
        if spec.get('family') != 'two-floors-v1' or len(spec.get('floors', [])) != 2:
            raise ValueError('Expected a two-floors-v1 instance with two floors')
        self.spec = spec
        self.floors = []
        for f in spec['floors']:
            f = copy.deepcopy(f)
            f['groups'] = [[r['id'] for r in f['robots']]]
            self.floors.append(World(f))
        self.ids = [rid for w in self.floors for rid in w.ids]
        if len(set(self.ids)) != len(self.ids):
            raise ValueError('Robot IDs must differ across floors')
        self.index = {rid: k for k, rid in enumerate(self.ids)}
        self.where = [(f, i) for f, w in enumerate(self.floors) for i in range(len(w.ids))]
        self.offsets = [0, len(self.floors[0].ids)]
        w0 = self.floors[0]
        if any(w.T != w0.T or w.weights != w0.weights for w in self.floors):
            raise ValueError('Floors must share horizon and weights')
        self.T, self.M, self.weights = w0.T, w0.M, w0.weights

    def split(self, programs):
        if len(programs) != len(self.ids):
            raise ValueError('One program per robot required')
        return [list(programs[o:o + len(w.ids)]) for o, w in zip(self.offsets, self.floors)]

    def trace(self, k, program):
        f, i = self.where[k]
        return self.floors[f].trace(i, program)

    def floor_of(self, rid):
        return self.where[self.index[rid]][0]

    def evaluate_programs(self, programs):
        per = [w.evaluate_programs(p) for w, p in zip(self.floors, self.split(programs))]
        return dict(F=sum(v['F'] for v in per), collisions=sum(v['collisions'] for v in per),
                    delivered=sum(v['delivered'] for v in per), floors=per)

    def reached(self, programs, targets):
        per = self.evaluate_programs(programs)['floors']
        return all(v['delivered'] == len(w.ids) and v['collisions'] == 0 and v['F'] <= t
                   for v, w, t in zip(per, self.floors, targets))


def _span(ids):
    return f'{ids[0]}-{ids[-1]}'


RULES = '''You plan routes for {n} robots on two floors, as one agent that sees and controls all of them. Each floor is its own grid with its own map, walls and robots: robots {ids1} are on floor 1 and robots {ids2} are on floor 2; a robot never changes floors. Below is the complete state computed by the checker for both floors: the maps, every robot's program and cost, where every robot is at every tick, every collision of the installed plan, and the attempts already rejected on this plan.
Grid (each floor): x increases to the right, y increases downward. Commands: right=(+1,0), left=(-1,0), down=(0,+1), up=(0,-1), wait=(0,0). Example: up from [2,8] reaches [2,7]. '#' cells are walls; robots never enter walls or leave the map.
Timing: at tick 0 every robot is at its start. Command k (k = 1, 2, ...) moves the robot from its tick k-1 position to its tick k position. A program has at most {T} commands and ends when the robot reaches its goal; from then on the robot stays on its goal through tick {T}, so that cell stays occupied. A robot whose program ends before its goal waits at its last cell.
Collisions: two robots on the same floor in the same cell at the same tick (also when one of them is parked on its goal), or two robots on the same floor exchanging cells between tick t-1 and tick t (reported at tick t). Robots on different floors never collide. Visiting the same cell at different ticks is safe. Each colliding pair counts once per tick.
Loss: F = F1 + F2. The loss of a floor is the sum of its robots' costs + {M:g} x number of collisions on that floor. A delivered robot costs its arrival tick; an undelivered robot costs {U:g} + {T} + its remaining shortest grid distance to the goal.'''


def floor_section(world, number, programs, digit_prefix):
    ids = world.ids; n = len(ids)
    traces = [world.trace(k, p) for k, p in enumerate(programs)]
    current = world.evaluate_programs(programs)
    events = current['events']
    out = [f'\n=== FLOOR {number}: robots {_span(ids)} ===']
    out.append(f'MAP of floor {number}: two header rows give x (tens digit, then units digit); lowercase letters are '
               'starts, the matching uppercase letters are goals (robot letters: '
               + ', '.join(f'{rid}={chr(97 + k)}/{chr(65 + k)}' for k, rid in enumerate(ids)) + ').')
    marks = {}
    for k, r in enumerate(world.robots):
        marks[tuple(r['start'])] = chr(97 + k); marks[tuple(r['goal'])] = chr(65 + k)
    out += P.map_lines(world, marks)
    x0, x1, y0, y1 = P._bounds(world)
    walls = [(x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1) if (x, y) not in world.cells]
    out.append(f'Wall cells of floor {number} [x,y] ({len(walls)}): ' + ' '.join(P._xy(c) for c in walls)
               + f'. Every other cell with {x0} <= x <= {x1} and {y0} <= y <= {y1} is free.')
    cost_sum = sum(t.cost for t in traces)
    out.append(f'\nINSTALLED PLAN on floor {number}: F{number} = {current["F"]:g} = robot costs {cost_sum:g} + '
               f'{world.M:g} x {current["collisions"]} collision{"s" if current["collisions"] != 1 else ""}; '
               f'{current["delivered"]} of {n} robots delivered.')
    out += P.robot_lines(world, programs, traces)
    lines, tail = P.timeline(world, traces, programs, events)
    out.append(f'\nTIMELINE of floor {number}: position x,y of every floor-{number} robot at every tick; G = the robot '
               'is on its goal (delivered).')
    out += lines
    out.append(tail.replace('no robot moves: every robot stays',
                            f'no floor-{number} robot moves: every floor-{number} robot stays'))
    if events:
        out.append(f'\nCOLLISIONS on floor {number} ({len(events)}):')
        for e in events:
            out.append('- ' + P.describe_event(world, traces, e))
        out.append(f'Maps of all floor-{number} robots at the tick before and the tick of each collision (digit k = '
                   f'robot {digit_prefix}k, + = several robots in one cell)'
                   + (f'; the first {P.MAX_FRAMES} collisions are drawn:' if len(events) > P.MAX_FRAMES else ':'))
        for e in events[:P.MAX_FRAMES]:
            out.append(f'collision at tick {e["time"]}, {ids[e["pair"][0]]} and {ids[e["pair"][1]]}:')
            out += P.frame(world, traces, e)
    else:
        out.append(f'\nCOLLISIONS on floor {number}: none.')
    return out


def build_two_floor_prompt(tf, base, *, iteration, targets, log, accepted):
    """base: installed programs (list, floor order); log: rejected attempts on this plan; accepted: past changes."""
    ids = tf.ids; n = len(ids)
    w1, w2 = tf.floors
    current = tf.evaluate_programs(base)
    per = current['floors']
    out = [RULES.format(n=n, ids1=_span(w1.ids), ids2=_span(w2.ids), T=tf.T, M=tf.M, U=tf.weights['unfinished'])]
    out.append(f'\nINSTALLED PLAN: F = F1 + F2 = {per[0]["F"]:g} + {per[1]["F"]:g} = {current["F"]:g}; '
               f'{current["delivered"]} of {n} robots delivered; {current["collisions"]} '
               f'collision{"s" if current["collisions"] != 1 else ""} ({per[0]["collisions"]} on floor 1, '
               f'{per[1]["collisions"]} on floor 2).')
    for number, (world, programs) in enumerate(zip(tf.floors, tf.split(base)), 1):
        prefix = world.ids[0].rstrip('0123456789')
        out += floor_section(world, number, programs, prefix)
    if accepted:
        shown = accepted[-P.MAX_ACCEPTED:]
        out.append('\nACCEPTED CHANGES so far (iteration: robot changed, F before -> after)'
                   + (f', last {len(shown)} of {len(accepted)}' if len(accepted) > len(shown) else '') + ': '
                   + '; '.join(f'{a["iteration"]}: {", ".join(a["robots"])} {a["F_before"]:g}->{a["F_after"]:g}' for a in shown))
    if log:
        shown = log[-P.MAX_LOG:]
        out.append(f'\nREJECTED ATTEMPTS on the installed plan ({len(log)}'
                   + (f', last {len(shown)} shown' if len(log) > len(shown) else '') + '; programs in compact '
                   'notation, "left x3" = three left commands; F is the loss of the whole plan):')
        for entry in shown:
            f = entry['floor']
            view = dict(entry)
            if entry['valid']:
                view['candidate'] = tf.split(entry['candidate'])[f]
            lines = P.attempt_lines(tf.floors[f], view, current['F'])
            lines[0] += f' (floor {f + 1})'
            out += lines
    else:
        out.append('\nREJECTED ATTEMPTS on the installed plan: none yet.')
    out.append(f'\nTASK: Choose exactly one robot, on either floor, and return a new complete program for it, starting '
               f'from its start at tick 0. The other {n - 1} robots keep the programs shown above. The checker replays '
               f'all {n} robots tick by tick on their floors, recomputes F, and installs your change only if F strictly '
               f'decreases. Goal: all robots delivered, no collisions, F1 <= {targets[0]:g} on floor 1 and '
               f'F2 <= {targets[1]:g} on floor 2; a delivered collision-free plan can still be improved by shortening '
               'routes.')
    task = dict(owned=list(ids), max_updated_robots=1, iteration=iteration, installed_F=current['F'],
                target_F=list(targets))
    out.append('TASK_JSON\n' + json.dumps(task, separators=(',', ':')))
    out.append('Final answer: exactly one JSON object {"programs": {"<robot id>": [<complete literal array of command '
               'names>]}} with exactly one robot ID. No shorthand (the "x3" notation above is only for reading), loops, '
               'code or alternative answers.')
    return '\n'.join(out)
