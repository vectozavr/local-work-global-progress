"""Value certificates for joint robot blocks via conflict-based search.

Reference programs remain inside the coordinator and are never supplied to an
LLM. A hard-constraint optimum is used for a soft-penalty objective only after
its value is compared with bounds for collision and unfinished alternatives.
"""
from collections import deque
import copy
import heapq
import time
from .world import World, MOVES, commands


def constrained_route(world, robot, vertices, edges):
    start = tuple(world.robots[robot]['start']); goal = tuple(world.robots[robot]['goal'])
    if (0,start) in vertices:
        return None
    queue = deque([(0,start)]); parent = {(0,start):None}
    while queue:
        tick,pos=queue.popleft()
        if pos==goal and all((t,goal) not in vertices for t in range(tick,world.T+1)):
            key=(tick,pos);path=[]
            while key is not None:
                path.append(key[1]);key=parent[key]
            return commands(list(reversed(path)))
        if tick==world.T or pos==goal:
            continue
        for dx,dy in MOVES.values():
            nxt=(pos[0]+dx,pos[1]+dy);key=(tick+1,nxt)
            if nxt not in world.cells or key in parent or key in vertices or (tick+1,pos,nxt) in edges:
                continue
            parent[key]=(tick,pos);queue.append(key)
    return None


def reference(world, seconds=30., node_cap=10000, initial_vertices=None, initial_edges=None):
    started=time.perf_counter();n=len(world.ids)
    vertices=tuple(frozenset() for _ in range(n)) if initial_vertices is None else tuple(initial_vertices)
    edges=tuple(frozenset() for _ in range(n)) if initial_edges is None else tuple(initial_edges)
    programs=[constrained_route(world,i,vertices[i],edges[i]) for i in range(n)]
    if any(p is None for p in programs):
        return dict(programs=None,expanded=0,seconds=time.perf_counter()-started,
                    status='No delivered zero-collision route under the supplied hard constraints.')
    heap=[(sum(map(len,programs)),0,vertices,edges,programs)];seen={(vertices,edges)}
    serial=0;expanded=0
    while heap and expanded<node_cap and time.perf_counter()-started<seconds:
        cost,_,vertices,edges,programs=heapq.heappop(heap);expanded+=1
        actual=world.evaluate_programs(programs)
        if not actual['events']:
            assert actual['delivered']==n and actual['F']==cost
            assert not heap or heap[0][0] >= cost
            return dict(programs=programs,**actual,expanded=expanded,
                minimum_delivered_collision_free_cost=cost,
                minimum_remaining_frontier_bound=None if not heap else heap[0][0],
                seconds=time.perf_counter()-started)
        event=min(actual['events'],key=lambda e:e['time'])
        for robot in event['pair']:
            vs=list(vertices);es=list(edges)
            if event['kind']=='vertex':
                vs[robot]=vs[robot]|{(event['time'],tuple(event['cell']))}
            else:
                a,b=map(tuple,event['edge'])
                if robot==event['pair'][1]:a,b=b,a
                es[robot]=es[robot]|{(event['time'],a,b)}
            vs,es=tuple(vs),tuple(es)
            if (vs,es) in seen:continue
            seen.add((vs,es))
            route=constrained_route(world,robot,vs[robot],es[robot])
            if route is None:continue
            following=list(programs);following[robot]=route;serial+=1
            heapq.heappush(heap,(sum(map(len,following)),serial,vs,es,following))
    return dict(programs=None,expanded=expanded,seconds=time.perf_counter()-started,
                status='Budget reached or heap empty; no impossibility certificate claimed.')


def joint_block_minimum(world, owned, base, known, seconds=5.):
    """Certify the soft-penalty block optimum only when hard routing proves it.

    External known timelines become exact vertex and reverse-edge constraints.
    A sufficiently cheap optimum with zero collisions also beats every plan
    having a penalized collision or unfinished delivery. Otherwise no soft
    optimum is claimed.
    """
    spec=copy.deepcopy(world.spec)
    spec['robots']=[spec['robots'][i] for i in owned]
    spec['groups']=[[r['id'] for r in spec['robots']]]
    subworld=World(spec);vertices=[];edges=[]
    for i in owned:
        vs=set();es=set()
        peers=[j if i==k else k for k,j in known if i in (k,j) and not {k,j} <= set(owned)]
        for peer in peers:
            positions=world.trace(peer,base[peer]).positions
            vs.update((t,p) for t,p in enumerate(positions))
            es.update((t,positions[t],positions[t-1]) for t in range(1,world.T+1)
                      if positions[t]!=positions[t-1])
        vertices.append(frozenset(vs));edges.append(frozenset(es))
    answer=reference(subworld,seconds=seconds,initial_vertices=vertices,initial_edges=edges)
    lower=sum(world.dist[i][tuple(world.robots[i]['start'])] for i in owned)
    failed_bound=lower+min(world.weights['unfinished']+world.T-
        world.dist[i][tuple(world.robots[i]['start'])] for i in owned)
    certified=answer['programs'] is not None and answer['F'] <= min(lower+world.M,failed_bound)
    return dict(exact_minimum=answer['F'] if certified else None,
                minimum_certified=certified,hard_routing=answer,
                penalty_plan_lower_bound=lower+world.M,unfinished_plan_lower_bound=failed_bound,
                disclosure='Post-hoc reference only, never given to the model.')


# Exact single-robot best responses (oracle)
def best_response(world, programs, robot, peers=None):
    """Minimize Q_robot plus collisions with selected peers, exactly up to T.

    Each state is a time and a non-goal cell. Reaching the absorbing goal closes
    a branch with its arrival cost and every later collision at that goal.
    Non-arriving paths are scored with the same unfinished/distance penalties
    as World.trace. Peers' programs remain fixed for this whole computation.
    """
    peers = list(range(len(world.ids))) if peers is None else list(peers)
    peers = sorted(set(peers) - {robot})
    other = [world.trace(i, programs[i]).positions for i in peers]
    start = tuple(world.robots[robot]['start'])
    goal = tuple(world.robots[robot]['goal'])
    tail = [0.] * (world.T+1)
    for t in range(world.T-1, -1, -1):
        tail[t] = tail[t+1] + world.M * sum(path[t+1] == goal for path in other)
    initial = world.M * sum(path[0] == start for path in other)
    if start == goal:
        return dict(score=initial+tail[0], program=[], arrival=0,
                    delivered=True, states_expanded=0)

    current = {start: (initial, ())}
    best = float('inf'); best_program = None; expanded = 0
    for tick in range(1, world.T+1):
        following = {}
        for position, (cost, program) in current.items():
            expanded += 1
            for command, (dx, dy) in MOVES.items():
                nxt = (position[0]+dx, position[1]+dy)
                if nxt not in world.cells:
                    continue
                events = sum(
                    nxt == path[tick] or
                    (position == path[tick] and nxt == path[tick-1] and position != nxt)
                    for path in other)
                value = cost + world.M * events
                candidate = program + (command,)
                if nxt == goal:
                    total = value + tick + tail[tick]
                    if total < best:
                        best = total; best_program = candidate
                elif nxt not in following or value < following[nxt][0]:
                    following[nxt] = (value, candidate)
        current = following
    for position, (cost, program) in current.items():
        total = (cost + world.weights['unfinished'] + world.T
                 + world.weights['distance'] * world.dist[robot][position])
        if total < best:
            best = total; best_program = program
    if best_program is None:
        raise ValueError('No admissible response found')
    result = list(best_program)
    trace = world.trace(robot, result)
    replay = trace.cost + world.M * sum(
        len(world.collisions(trace.positions, path, robot, i))
        for i, path in zip(peers, other))
    if abs(best-replay) > 1e-9:
        raise AssertionError('Best-response score disagrees with direct replay')
    return dict(score=best, program=result, arrival=trace.arrival,
                delivered=trace.arrival is not None, states_expanded=expanded)


def minimum_score(world, programs, robot, peers):
    """Value-only interface for live stopping; no reference program is exposed."""
    result = best_response(world, programs, robot, peers)
    return dict(score=result['score'],states_expanded=result['states_expanded'])
