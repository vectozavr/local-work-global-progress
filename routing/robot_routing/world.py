"""The grid world: route replay, collisions and the loss. Deterministic; coordinates are (x, y), y grows down."""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from itertools import combinations
import hashlib
import json
import math

MOVES = {'up': (0, -1), 'down': (0, 1), 'left': (-1, 0), 'right': (1, 0), 'wait': (0, 0)}


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def distances(cells, goal):
    cells = set(map(tuple, cells)); goal = tuple(goal)
    out = {goal: 0}; queue = deque([goal])
    while queue:
        x, y = queue.popleft()
        for dx, dy in list(MOVES.values())[:4]:
            q = (x + dx, y + dy)
            if q in cells and q not in out:
                out[q] = out[(x, y)] + 1; queue.append(q)
    return out


def commands(path):
    inverse = {v: k for k, v in MOVES.items()}
    return [inverse[(b[0]-a[0], b[1]-a[1])] for a, b in zip(path, path[1:])]


@dataclass
class Trace:
    positions: list
    arrival: int | None
    remaining: int
    cost: float


class World:
    def __init__(self, spec):
        self.spec=spec
        if type(spec['horizon']) is not int or spec['horizon']<1:raise ValueError('Horizon must be a positive integer')
        if set(spec['weights'])!={'unfinished','distance','collision'} or any(not math.isfinite(float(v)) or float(v)<0 for v in spec['weights'].values()):raise ValueError('Weights must be finite and nonnegative')
        self.T=spec['horizon']; self.cells=set(map(tuple,spec['cells']))
        self.robots=spec['robots']; self.ids=[r['id'] for r in self.robots]
        if len(set(self.ids)) != len(self.ids): raise ValueError('Duplicate robot ID')
        self.index={r:i for i,r in enumerate(self.ids)}
        self.groups=[[self.index[r] for r in g] for g in spec['groups']]
        if sorted(sum(self.groups,[])) != list(range(len(self.ids))) or any(not g for g in self.groups):
            raise ValueError('Groups must partition robots into nonempty blocks')
        self.owner={i:b for b,g in enumerate(self.groups) for i in g}
        self.weights=spec['weights']; self.M=float(self.weights['collision'])
        self.dist=[distances(self.cells, r['goal']) for r in self.robots]
        for r in self.robots:
            if tuple(r['start']) not in self.cells or tuple(r['goal']) not in self.cells:
                raise ValueError('Start or goal outside the map')

    def trace(self,i,program):
        if not isinstance(program,list) or len(program)>self.T:
            raise ValueError('Program must be an array of at most T commands')
        pos=tuple(self.robots[i]['start']); goal=tuple(self.robots[i]['goal'])
        positions=[pos]; arrival=0 if pos==goal else None
        for tick in range(1,self.T+1):
            cmd=program[tick-1] if tick<=len(program) else 'wait'
            if not isinstance(cmd,str) or cmd not in MOVES: raise ValueError(f'Robot {self.ids[i]}: unknown command {cmd!r} at t={tick}')
            if arrival is not None and cmd!='wait': raise ValueError(f'Robot {self.ids[i]} must stay at its goal {goal}; non-wait command at t={tick}')
            dx,dy=MOVES[cmd]; previous=pos;pos=(pos[0]+dx,pos[1]+dy)
            if pos not in self.cells:
                legal=[c for c,(dx,dy) in MOVES.items() if (previous[0]+dx,previous[1]+dy) in self.cells]
                raise ValueError(f'Robot {self.ids[i]} enters a wall at t={tick}: {cmd} from {previous} to {pos}. Legal commands from {previous}: {legal}')
            positions.append(pos)
            if pos==goal and arrival is None: arrival=tick
        remaining=self.dist[i].get(pos)
        if remaining is None: raise ValueError('Final position cannot reach its goal')
        cost=(self.weights['unfinished']*(arrival is None)+(self.T if arrival is None else arrival)+self.weights['distance']*remaining)
        return Trace(positions,arrival,remaining,float(cost))

    @staticmethod
    def collisions(a,b,i,j):
        out=[]
        for t,(p,q) in enumerate(zip(a,b)):
            if p==q: out.append(dict(pair=[i,j],time=t,kind='vertex',cell=list(p)))
            elif t and a[t-1]==q and b[t-1]==p and a[t-1]!=p:
                out.append(dict(pair=[i,j],time=t,kind='edge',edge=[list(a[t-1]),list(p)]))
        return out

    def evaluate_programs(self,programs):
        if len(programs)!=len(self.ids): raise ValueError('One program per robot required')
        traces=[self.trace(i,p) for i,p in enumerate(programs)]
        events=[]
        for i,j in combinations(range(len(traces)),2):
            events.extend(self.collisions(traces[i].positions,traces[j].positions,i,j))
        return dict(F=sum(t.cost for t in traces)+self.M*len(events), collisions=len(events),events=events,
                    delivered=sum(t.arrival is not None for t in traces),unary=[t.cost for t in traces],
                    arrival=[t.arrival for t in traces],remaining=[t.remaining for t in traces])
