"""Prompt of a team agent: the presentation of prompt_single.py with the team's information structure.

The agent of block b sees its own robots and the known peers, i.e. robots of other blocks whose
collisions with its robots are already known to the coordinator, with their current fixed
timelines. Other robots are hidden: they appear in no table, drawing or collision list. Shown:
the map with both axes and the wall list, the block's programs and costs, a tick-by-tick timeline
of every visible robot, every visible collision of the installed plan with maps of the tick before
and the tick of the collision, the attempts already made in this search, and the collisions that
the coordinator's global check found for the block's previous proposal. No global loss, target,
route or repair strategy is supplied.
"""
import json
import re

from .prompt_single import SYMBOLS, _bounds, _xy, map_lines, side_by_side, rle

MAX_FRAMES = 3


def visible_peers(w, b, known):
    own = set(w.groups[b]); peers = {}
    for i, j in known:
        if (i in own) != (j in own):
            mine, peer = (i, j) if i in own else (j, i)
            peers.setdefault(peer, set()).add(mine)
    return {p: sorted(m) for p, m in sorted(peers.items())}


def describe(w, traces, e):
    i, j = e['pair']; t = e['time']
    a, b = w.ids[i], w.ids[j]
    pa, pb = traces[i].positions, traces[j].positions
    if e['kind'] == 'edge':
        return (f'tick {t}: {a} moves {_xy(pa[t-1])}->{_xy(pa[t])} while {b} moves {_xy(pb[t-1])}->{_xy(pb[t])} '
                f'(they exchange cells between tick {t-1} and tick {t})')
    parked = [f'{n} has been parked on its goal since tick {tr.arrival}'
              for n, tr in ((a, traces[i]), (b, traces[j])) if tr.arrival is not None and tr.arrival < t]
    return f'tick {t}: {a} and {b} are both at {_xy(pa[t])}' + (f' ({"; ".join(parked)})' if parked else '')


def frame(w, traces, e, visible):
    t = e['time']; maps = []
    for tick in (t - 1, t):
        occupants = {}
        for k in visible:
            occupants.setdefault(traces[k].positions[tick], []).append(SYMBOLS[k])
        maps.append(map_lines(w, {c: (l[0] if len(l) == 1 else '+') for c, l in occupants.items()}))
    title = ' ' * 7 + f'tick {t-1}'.ljust(len(maps[0][0]) - 7 + 4) + f'tick {t}'
    return [title] + side_by_side(*maps)


def timeline(w, traces, programs, visible, events):
    last = max([len(programs[k]) for k in visible] + [0])
    ids = w.ids; width = 7
    by_tick = {}
    for e in events:
        i, j = e['pair']
        by_tick.setdefault(e['time'], []).append(f'{ids[i]}-{ids[j]} {"exchange" if e["kind"] == "edge" else "same cell"}')
    lines = ['t   | ' + ''.join(ids[k].ljust(width) for k in visible) + '| visible collisions']
    for t in range(last + 1):
        cells = []
        for k in visible:
            p = traces[k].positions[t]; tr = traces[k]
            cells.append((f'{p[0]},{p[1]}' + ('G' if tr.arrival is not None and tr.arrival <= t else '')).ljust(width))
        lines.append(f'{t:<3} | ' + ''.join(cells) + '| ' + ', '.join(by_tick.get(t, [])))
    tail = f'After tick {last} none of these robots moves: each stays where it is at tick {last} through tick {w.T}.'
    later = sorted({e['time'] for e in events if e['time'] > last})
    if later:
        tail += ' Visible collisions at later ticks: ' + '; '.join(', '.join(by_tick[t]) + f' at tick {t}' for t in later) + '.'
    return lines, tail


VISIBILITY = {True: ("You see your robots and the known peers: robots of other agents whose collisions with your robots are "
                     "already known to the coordinator. Their current programs stay fixed during your search. Robots of "
                     "other agents that are not listed are hidden from you; collisions with them are found by the "
                     "coordinator's global check, and such robots then become known peers in later rounds."),
              False: "You see only your own robots; the robots of other agents are hidden from you."}

RULES = '''You are one agent of a team that plans routes for {n} robots on a grid. You control the robots of your block: {own}. Other agents control the other robots. A coordinator keeps the installed plan of all robots and, after each round, checks the full plan and installs the best block proposal of the round.
{visibility}
Grid: x increases to the right, y increases downward. Commands: right=(+1,0), left=(-1,0), down=(0,+1), up=(0,-1), wait=(0,0). Example: up from [2,8] reaches [2,7]. '#' cells are walls; robots never enter walls or leave the map.
Timing: at tick 0 every robot is at its start. Command k (k = 1, 2, ...) moves the robot from its tick k-1 position to its tick k position. A program has at most {T} commands and ends when the robot reaches its goal; from then on the robot stays on its goal through tick {T}, so that cell stays occupied. A robot whose program ends before its goal waits at its last cell.
Collisions: two robots in the same cell at the same tick (also when one of them is parked on its goal), or two robots exchanging cells between tick t-1 and tick t (reported at tick t). Visiting the same cell at different ticks is safe. Each colliding pair counts once per tick.
Block loss: sum of your robots' costs + {M:g} x {collision_scope}. A delivered robot costs its arrival tick; an undelivered robot costs {U:g} + {T} + its remaining shortest grid distance to the goal. Lower is better.'''


def attempt_lines(w, entry, base_value, traces_base):
    lines = [f'attempt {entry["step"] + 1}: programs ' + '; '.join(f'{rid} -> {rle(p)} ({len(p)} commands)'
                                                              for rid, p in entry['programs'].items())]
    if not entry['valid']:
        lines.append('   rejected, invalid: ' + re.sub(r'\((-?\d+), (-?\d+)\)', r'[\1,\2]', entry['error']))
        return lines
    traces = list(traces_base)
    for rid, p in entry['programs'].items():
        traces[w.index[rid]] = w.trace(w.index[rid], p)
    res = (f'   checked: block loss {entry["value"]:g} ({entry["value"] - base_value:+g} against the installed plan); '
           + ', '.join(f'{rid} ' + (f'arrives at tick {traces[w.index[rid]].arrival}' if traces[w.index[rid]].arrival
                                    is not None else 'not delivered') for rid in entry['programs']))
    ev = entry['events']
    res += (f'; {len(ev)} visible collision' + ('s' if len(ev) != 1 else '') + ': '
            + ' | '.join(describe(w, traces, e) for e in ev[:4]) + (' | ...' if len(ev) > 4 else '')) if ev else '; no visible collisions'
    lines.append(res)
    return lines


RULE_BEST_SINGLE = 'checks the full plan and installs the best block proposal of the round.'
RULES_BY_MERGE = {'best_single': RULE_BEST_SINGLE,
                  'best_subset': 'checks every combination of the block proposals of the round on the full plan and installs the combination that lowers the loss most.',
                  'all': 'installs the block proposals of all agents of the round together.'}


def build_team_prompt(w, b, base, known, *, value, events, log, global_feedback, epoch, step, H, merge='best_single', correction=True):
    """base: installed programs of all robots (world order); value/events: block loss and visible collisions of base."""
    own = list(w.groups[b]); peers = visible_peers(w, b, known); visible = own + list(peers)
    traces = [w.trace(k, p) for k, p in enumerate(base)]
    if not correction:
        assert not peers and not global_feedback, 'without correction no peer or global feedback may be shown'
    scope = ('(collisions among your robots + collisions between your robots and the known peers they are checked against)'
             if correction else 'collisions among your robots')
    rules = RULES if merge == 'best_single' else RULES.replace(RULE_BEST_SINGLE, RULES_BY_MERGE[merge])
    assert merge == 'best_single' or rules != RULES
    out = [rules.format(n=len(w.ids), own=', '.join(w.ids[k] for k in own), T=w.T, M=w.M, U=w.weights['unfinished'],
                        visibility=VISIBILITY[correction], collision_scope=scope)]
    out.append('\nMAP: two header rows give x (tens digit, then units digit); lowercase letters are the starts of your '
               'robots, the matching uppercase letters their goals (' + ', '.join(
                   f'{w.ids[k]}={chr(97 + n)}/{chr(65 + n)}' for n, k in enumerate(own)) + ').')
    marks = {}
    for n, k in enumerate(own):
        marks[tuple(w.robots[k]['start'])] = chr(97 + n); marks[tuple(w.robots[k]['goal'])] = chr(65 + n)
    out += map_lines(w, marks)
    x0, x1, y0, y1 = _bounds(w)
    walls = [(x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1) if (x, y) not in w.cells]
    out.append(f'Wall cells [x,y] ({len(walls)}): ' + ' '.join(_xy(c) for c in walls) + '. Every other cell with '
               f'{x0} <= x <= {x1} and {y0} <= y <= {y1} is free.')
    cost = sum(traces[k].cost for k in own)
    out.append(f'\nYOUR ROBOTS in the installed plan: block loss {value:g} = robot costs {cost:g} + {w.M:g} x {len(events)} '
               f'visible collision{"s" if len(events) != 1 else ""}.')
    for k in own:
        r = w.robots[k]; tr = traces[k]
        head = f'{w.ids[k]} (map digit {SYMBOLS[k]}): start {_xy(r["start"])}, goal {_xy(r["goal"])}; '
        head += (f'delivered at tick {tr.arrival}, cost {tr.cost:g}' if tr.arrival is not None else
                 f'NOT delivered, ends at {_xy(tr.positions[-1])}, remaining distance {tr.remaining}, cost {tr.cost:g}')
        out += [head, f'   program ({len(base[k])} commands): ' + json.dumps(base[k])]
    if not correction:
        pass
    elif peers:
        out.append('\nKNOWN PEERS (fixed during this search): ' + '; '.join(
            f'{w.ids[p]} (map digit {SYMBOLS[p]}), checked against {", ".join(w.ids[m] for m in mine)}'
            for p, mine in peers.items()) + '.')
    else:
        out.append('\nKNOWN PEERS: none yet; all robots of other agents are hidden from you.')
    lines, tail = timeline(w, traces, base, visible, events)
    out.append('\nTIMELINE of the installed plan for your robots' + (' and the known peers' if correction else '') + ': position x,y at every tick; '
               'G = the robot is on its goal (delivered).')
    out += lines + [tail]
    if events:
        out.append(f'\nVISIBLE COLLISIONS of the installed plan ({len(events)}):')
        out += ['- ' + describe(w, traces, e) for e in events]
        out.append('Maps of your robots' + (' and the known peers' if correction else '') + ' at the tick before and the tick of each collision (digit k = '
                   'robot rk, + = several robots in one cell)' + (f'; the first {MAX_FRAMES} are drawn:'
                                                                   if len(events) > MAX_FRAMES else ':'))
        for e in events[:MAX_FRAMES]:
            out.append(f'collision at tick {e["time"]}, {w.ids[e["pair"][0]]} and {w.ids[e["pair"][1]]}:')
            out += frame(w, traces, e, visible)
    else:
        out.append('\nVISIBLE COLLISIONS of the installed plan: none.')
    if global_feedback:
        out.append(f'\nGLOBAL CHECK after round {global_feedback["global_round"]}: the coordinator checked your block\'s '
                   'proposal from that round against all robots; it collided as listed (the robots involved are now '
                   'known peers if they belong to other agents). Proposal: ' + '; '.join(
                       f'{rid} -> {rle(p)}' for rid, p in global_feedback['programs'].items()) + '.')
        gtr = list(traces)
        for rid, p in global_feedback['programs'].items():
            gtr[w.index[rid]] = w.trace(w.index[rid], p)
        for e in global_feedback['collisions']:
            ev = dict(e, pair=[w.index[x] if isinstance(x, str) else x for x in e['pair']])
            out.append('- ' + describe(w, gtr, ev))
    if log:
        out.append(f'\nTHIS SEARCH, attempts so far (programs in compact notation, "left x3" = three left commands):')
        for entry in log:
            out += attempt_lines(w, entry, value, traces)
    out.append(f'\nTASK: this is attempt {step + 1} of {H} in your search of round {epoch + 1}. Return a new complete program '
               'for EVERY robot of your block, each starting from its start at tick 0 (repeat a program unchanged to keep '
               'it). Your proposal is scored by the block loss' + (' against the known peers' if correction else '') + '; your best checked attempt of '
               'this search that lowers the block loss becomes your block\'s proposal for the coordinator.')
    task = dict(owned=[w.ids[k] for k in own], round=epoch + 1, attempt=step + 1, attempt_budget=H, installed_block_loss=value)
    out.append('TASK_JSON\n' + json.dumps(task, separators=(',', ':')))
    out.append('Final answer: exactly one JSON object {"programs": {"<robot id>": [<complete literal array of command '
               'names>], ...}} with every robot ID of your block. No shorthand (the "x3" notation above is only for '
               'reading), loops, code or alternative answers.')
    return '\n'.join(out)
