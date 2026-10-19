"""Prompt of a single agent: the complete checked state of the installed plan, without solution hints.

Everything shown is computed by the checker from the installed plan and from the attempts
already rejected on it: the map with both axes, every robot's program and cost, a tick-by-tick
timeline of all robots, every current collision with maps of the tick before and the tick of
the collision, and the rejected attempts with their outcomes. No route, distance to the optimum
or repair strategy is supplied.
"""
import json
import re

SYMBOLS = '0123456789ABCDEFGHJKLMNPQRSTUVWXYZ'
MAX_FRAMES = 3
MAX_LOG = 10
MAX_ACCEPTED = 12


def _bounds(world):
    xs = [x for x, _ in world.cells]; ys = [y for _, y in world.cells]
    return min(xs), max(xs), min(ys), max(ys)


def _xy(p):
    return f'[{p[0]},{p[1]}]'


def map_lines(world, marks):
    """Rows with a y label and two header rows giving x (tens digit, then units digit)."""
    x0, x1, y0, y1 = _bounds(world)
    pad = ' ' * 7
    lines = [pad + ''.join(str(x // 10) for x in range(x0, x1 + 1)),
             pad + ''.join(str(x % 10) for x in range(x0, x1 + 1))]
    for y in range(y0, y1 + 1):
        lines.append(f'y={y:02d}   ' + ''.join(marks.get((x, y), '.' if (x, y) in world.cells else '#')
                                             for x in range(x0, x1 + 1)))
    return lines


def side_by_side(left, right, gap=4):
    """Join two maps of equal height; the right one drops its 7-character y label."""
    width = max(len(s) for s in left)
    return [a.ljust(width) + ' ' * gap + b[7:] for a, b in zip(left, right)]


def rle(program):
    """Readable run-length summary; only for reading, never an accepted answer format."""
    out = []
    for cmd in program:
        if out and out[-1][0] == cmd:
            out[-1][1] += 1
        else:
            out.append([cmd, 1])
    return ', '.join(c if n == 1 else f'{c} x{n}' for c, n in out) or '(empty: waits at start)'


def describe_event(world, traces, event):
    i, j = event['pair']; t = event['time']
    a, b = world.ids[i], world.ids[j]
    pa, pb = traces[i].positions, traces[j].positions
    if event['kind'] == 'edge':
        return (f'tick {t}: {a} moves {_xy(pa[t-1])}->{_xy(pa[t])} while {b} moves {_xy(pb[t-1])}->{_xy(pb[t])} '
                f'(they exchange cells between tick {t-1} and tick {t})')
    parked = [f'{name} has been parked on its goal since tick {tr.arrival}'
              for name, tr in ((a, traces[i]), (b, traces[j])) if tr.arrival is not None and tr.arrival < t]
    return f'tick {t}: {a} and {b} are both at {_xy(pa[t])}' + (f' ({"; ".join(parked)})' if parked else '')


def frame(world, traces, event):
    t = event['time']
    maps = []
    for tick in (t - 1, t):
        occupants = {}
        for k, tr in enumerate(traces):
            occupants.setdefault(tr.positions[tick], []).append(SYMBOLS[k])
        marks = {cell: (labels[0] if len(labels) == 1 else '+') for cell, labels in occupants.items()}
        maps.append(map_lines(world, marks))
    title = ' ' * 7 + f'tick {t-1}'.ljust(len(maps[0][0]) - 7 + 4) + f'tick {t}'
    return [title] + side_by_side(*maps)


def timeline(world, traces, programs, events):
    last = max([len(p) for p in programs] + [0])
    ids = world.ids
    width = max(7, max(len(i) for i in ids) + 1)
    by_tick = {}
    for e in events:
        i, j = e['pair']
        kind = 'exchange' if e['kind'] == 'edge' else 'same cell'
        by_tick.setdefault(e['time'], []).append(f'{ids[i]}-{ids[j]} {kind}')
    lines = ['t   | ' + ''.join(i.ljust(width) for i in ids) + '| collisions']
    for t in range(last + 1):
        cells = []
        for k, tr in enumerate(traces):
            p = tr.positions[t]
            mark = 'G' if tr.arrival is not None and tr.arrival <= t else ''
            cells.append(f'{p[0]},{p[1]}{mark}'.ljust(width))
        lines.append(f'{t:<3} | ' + ''.join(cells) + '| ' + ', '.join(by_tick.get(t, [])))
    later = sorted({e['time'] for e in events if e['time'] > last})
    tail = (f'After tick {last} no robot moves: every robot stays where it is at tick {last} '
            f'through tick {world.T}.')
    if later:
        tail += ' Collisions at later ticks: ' + '; '.join(', '.join(by_tick[t]) + f' at tick {t}' for t in later) + '.'
    return lines, tail


def robot_lines(world, programs, traces):
    lines = []
    for k, rid in enumerate(world.ids):
        r = world.robots[k]; tr = traces[k]
        head = f'{rid} (map digit {SYMBOLS[k]}): start {_xy(r["start"])}, goal {_xy(r["goal"])}'
        if tr.arrival is not None:
            head += f'; delivered at tick {tr.arrival}, cost {tr.cost:g}'
        else:
            head += (f'; NOT delivered, ends at {_xy(tr.positions[-1])}, remaining distance {tr.remaining}, '
                     f'cost {tr.cost:g}')
        lines.append(head)
        lines.append(f'   program ({len(programs[k])} commands): ' + json.dumps(programs[k]))
    return lines


def attempt_lines(world, entry, installed_F):
    names = ', '.join(entry['updates'])
    lines = [f'iteration {entry["iteration"]}' + (f' (copy {entry["copy"]})' if entry.get('copy') is not None else '')
             + f': changed {names}']
    for rid, program in entry['updates'].items():
        lines.append(f'   {rid} -> {rle(program)} ({len(program)} commands)')
    if not entry['valid']:
        lines.append('   rejected, invalid program: ' + re.sub(r'\((-?\d+), (-?\d+)\)', r'[\1,\2]', entry['error']))
        return lines
    traces = [world.trace(k, p) for k, p in enumerate(entry['candidate'])]
    change = entry['F'] - installed_F
    res = f'   rejected: F={entry["F"]:g} ({change:+g} against the installed plan)'
    arrivals = []
    for rid in entry['updates']:
        k = world.index[rid]; tr = traces[k]
        arrivals.append(f'{rid} ' + (f'arrives at tick {tr.arrival}' if tr.arrival is not None else 'not delivered'))
    res += '; ' + ', '.join(arrivals)
    events = entry['events']
    if events:
        shown = [describe_event(world, traces, e) for e in events[:4]]
        res += f'; {len(events)} collision' + ('s' if len(events) != 1 else '') + ': ' + ' | '.join(shown)
        if len(events) > 4:
            res += ' | ...'
    else:
        res += '; no collisions'
    lines.append(res)
    return lines


RULES = '''You plan routes for {n} robots on a grid, as one agent that sees and controls all of them. Below is the complete state computed by the checker: the map, every robot's program and cost, where every robot is at every tick, every collision of the installed plan, and the attempts already rejected on this plan.
Grid: x increases to the right, y increases downward. Commands: right=(+1,0), left=(-1,0), down=(0,+1), up=(0,-1), wait=(0,0). Example: up from [2,8] reaches [2,7]. '#' cells are walls; robots never enter walls or leave the map.
Timing: at tick 0 every robot is at its start. Command k (k = 1, 2, ...) moves the robot from its tick k-1 position to its tick k position. A program has at most {T} commands and ends when the robot reaches its goal; from then on the robot stays on its goal through tick {T}, so that cell stays occupied. A robot whose program ends before its goal waits at its last cell.
Collisions: two robots in the same cell at the same tick (also when one of them is parked on its goal), or two robots exchanging cells between tick t-1 and tick t (reported at tick t). Visiting the same cell at different ticks is safe. Each colliding pair counts once per tick.
Loss: F = sum of robot costs + {M:g} x number of collisions. A delivered robot costs its arrival tick; an undelivered robot costs {U:g} + {T} + its remaining shortest grid distance to the goal.'''


def build_single_prompt(world, base, *, iteration, target_F, max_updates, log, accepted, exact=False):
    """base: installed programs (list, world order); log: rejected attempts on this plan; accepted: past changes."""
    ids = world.ids; n = len(ids)
    traces = [world.trace(k, p) for k, p in enumerate(base)]
    current = world.evaluate_programs(base)
    events = current['events']
    out = [RULES.format(n=n, T=world.T, M=world.M, U=world.weights['unfinished'])]
    out.append('\nMAP: two header rows give x (tens digit, then units digit); lowercase letters are starts, '
               'the matching uppercase letters are goals (robot letters: '
               + ', '.join(f'{rid}={chr(97 + k)}/{chr(65 + k)}' for k, rid in enumerate(ids)) + ').')
    marks = {}
    for k, r in enumerate(world.robots):
        marks[tuple(r['start'])] = chr(97 + k); marks[tuple(r['goal'])] = chr(65 + k)
    out += map_lines(world, marks)
    x0, x1, y0, y1 = _bounds(world)
    walls = [(x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1) if (x, y) not in world.cells]
    out.append(f'Wall cells [x,y] ({len(walls)}): ' + ' '.join(_xy(c) for c in walls) + '. Every other cell with '
               f'{x0} <= x <= {x1} and {y0} <= y <= {y1} is free.')
    cost_sum = sum(t.cost for t in traces)
    out.append(f'\nINSTALLED PLAN: F = {current["F"]:g} = robot costs {cost_sum:g} + {world.M:g} x {current["collisions"]} '
               f'collision{"s" if current["collisions"] != 1 else ""}; {current["delivered"]} of {n} robots delivered.')
    out += robot_lines(world, base, traces)
    lines, tail = timeline(world, traces, base, events)
    out.append('\nTIMELINE of the installed plan: position x,y of every robot at every tick; G = the robot is on its goal '
               '(delivered).')
    out += lines
    out.append(tail)
    if events:
        out.append(f'\nCOLLISIONS of the installed plan ({len(events)}):')
        traces_full = traces
        for e in events:
            out.append('- ' + describe_event(world, traces_full, e))
        out.append('Maps of all robots at the tick before and the tick of each collision (digit k = robot rk, '
                   '+ = several robots in one cell)' + (f'; the first {MAX_FRAMES} collisions are drawn:' if len(events) > MAX_FRAMES else ':'))
        for e in events[:MAX_FRAMES]:
            out.append(f'collision at tick {e["time"]}, {ids[e["pair"][0]]} and {ids[e["pair"][1]]}:')
            out += frame(world, traces_full, e)
    else:
        out.append('\nCOLLISIONS of the installed plan: none.')
    if accepted:
        shown = accepted[-MAX_ACCEPTED:]
        out.append('\nACCEPTED CHANGES so far (iteration: robots changed, F before -> after)'
                   + (f', last {len(shown)} of {len(accepted)}' if len(accepted) > len(shown) else '') + ': '
                   + '; '.join(f'{a["iteration"]}: {", ".join(a["robots"])} {a["F_before"]:g}->{a["F_after"]:g}' for a in shown))
    if log:
        shown = log[-MAX_LOG:]
        out.append(f'\nREJECTED ATTEMPTS on the installed plan ({len(log)}'
                   + (f', last {len(shown)} shown' if len(log) > len(shown) else '') + '; programs in compact '
                   'notation, "left x3" = three left commands):')
        for entry in shown:
            out += attempt_lines(world, entry, current['F'])
    else:
        out.append('\nREJECTED ATTEMPTS on the installed plan: none yet.')
    if max_updates == 1:
        choose = ('Choose exactly one robot and return a new complete program for it, starting from its start at tick 0. '
                  f'The other {n - 1} robots keep the programs shown above.')
        final = 'with exactly one robot ID'
    elif exact:
        choose = (f'Choose exactly {max_updates} distinct robots and return a complete program for each, starting from '
                  'its start at tick 0 (repeat a program unchanged to keep it). All other robots keep the programs shown above.')
        final = f'with exactly {max_updates} robot IDs'
    else:
        choose = (f'Choose one to {max_updates} distinct robots and return a new complete program for each, starting '
                  'from its start at tick 0. All other robots keep the programs shown above.')
        final = f'with one to {max_updates} robot IDs'
    out.append(f'\nTASK: {choose} The checker replays all {n} robots tick by tick, recomputes F, and installs your change '
               f'only if F strictly decreases. Goal: all robots delivered, no collisions and F <= {target_F:g}; '
               'a delivered collision-free plan can still be improved by shortening routes.')
    task = dict(owned=list(ids), max_updated_robots=max_updates, iteration=iteration, installed_F=current['F'],
                target_F=target_F)
    if exact and max_updates > 1:
        task = dict(owned=task['owned'], max_updated_robots=max_updates, min_updated_robots=max_updates,
                    **{k: v for k, v in task.items() if k not in ('owned', 'max_updated_robots')})
    out.append('TASK_JSON\n' + json.dumps(task, separators=(',', ':')))
    out.append('Final answer: exactly one JSON object {"programs": {"<robot id>": [<complete literal array of command '
               f'names>]}}}} {final}. No shorthand (the "x3" notation above is only for reading), loops, code or '
               'alternative answers.')
    return '\n'.join(out)
