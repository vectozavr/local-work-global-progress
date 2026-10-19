"""Exact block minima and assumption certificates for recorded team runs (library used by the theory audits).

Reads frozen results only. Reference paths are never sent to a model and never
replace recorded routes. This analysis sends no network requests.
"""
import copy
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from robot_routing.world import World, MOVES
from robot_routing.reference import joint_block_minimum
from independent import score, independent_joint, parse_final, replay


def read(path):
    return json.loads(path.read_text())


def save(path, data):
    path.write_text(json.dumps(data, indent=2)+'\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    if not values:
        return None
    return dict(n=len(values),mean=statistics.mean(values),
        sample_sd=statistics.stdev(values) if len(values)>1 else None,
        minimum=min(values),maximum=max(values),
        quantiles=np.quantile(values,[0,.25,.5,.75,1]).tolist())


def external_count(events, owned, pairs):
    own=set(owned)
    return sum(tuple(e['pair']) in pairs and bool(set(e['pair']) & own)
               and not set(e['pair'])<=own for e in events)


class Certificates:
    def __init__(self, world):
        self.w=world
        self.cache={}
        self.oracle_calls=0
        self.cells=sorted(world.cells)
        self.index={p:i for i,p in enumerate(self.cells)}
        self.moves=list(MOVES)
        self.neighbors=np.array([[self.index.get((x+dx,y+dy),-1)
            for dx,dy in MOVES.values()] for x,y in self.cells])

    def minimum(self, b, base, known):
        owned=self.w.groups[b];own=set(owned)
        relevant={p for p in known if bool(set(p)&own) and not set(p)<=own}
        key=(b,tuple(tuple(p) for i,p in enumerate(base) if i not in own),tuple(sorted(relevant)))
        if key not in self.cache:
            answer=None
            for budget in [3.,15.,60.]:
                answer=joint_block_minimum(self.w,owned,base,relevant,seconds=budget)
                self.oracle_calls+=1
                if answer['minimum_certified']:
                    break
            assert answer['minimum_certified'], ('uncertified minimum',b,answer)
            candidate=copy.deepcopy(base)
            for i,p in zip(owned,answer['hard_routing']['programs']):
                candidate[i]=p
            actual,visible,_=score(self.w.spec,owned,candidate,relevant)
            assert actual==answer['exact_minimum'] and not visible
            assert answer['hard_routing']['minimum_delivered_collision_free_cost']==actual
            assert actual<=answer['penalty_plan_lower_bound']
            assert actual<=answer['unfinished_plan_lower_bound']
            unary=[self.soft_unary_minimum(i,base,
                [j if i==k else k for k,j in relevant if i in (k,j)]) for i in owned]
            lower=sum(v['score'] for v in unary)
            assert lower<=actual
            answer['independent_single_robot_lower_bound']=lower
            answer['independent_lower_bound_tight']=lower==actual
            answer['independent_robot_minima']=unary
            self.cache[key]=answer
        return self.cache[key]

    def soft_unary_minimum(self, robot, base, peers):
        """Independent time-expanded DP lower bound, relaxing internal conflicts."""
        w=self.w;n=len(self.cells)
        neighbors=self.neighbors.copy()
        goal=self.index[tuple(w.robots[robot]['goal'])]
        neighbors[goal,:]=-1;neighbors[goal,self.moves.index('wait')]=goal
        valid=neighbors>=0;clamped=np.maximum(neighbors,0)
        rows=np.arange(n)[:,None]
        positions={j:[self.index[p] for p in w.trace(j,base[j]).positions] for j in peers}
        value=np.array([w.weights['unfinished']+w.T+
            w.weights['distance']*w.dist[robot][p] for p in self.cells],dtype=float)
        value[goal]=0
        choices=[]
        for tick in range(w.T,0,-1):
            reward=np.zeros_like(neighbors)
            for path in positions.values():
                previous,current=path[tick-1],path[tick]
                reward+=(neighbors==current)
                if previous!=current:
                    reward+=((rows==current)&(neighbors==previous))
            arrival=((rows!=goal)&(neighbors==goal))*tick
            following=np.where(valid,w.M*reward+arrival+value[clamped],np.inf)
            choices.append(following.argmin(axis=1))
            value=following.min(axis=1)
        cell=self.index[tuple(w.robots[robot]['start'])]
        optimum=float(value[cell])+w.M*sum(path[0]==cell for path in positions.values())
        program=[]
        for choice in reversed(choices):
            j=int(choice[cell]);program.append(self.moves[j]);cell=int(neighbors[cell,j])
        candidate=copy.deepcopy(base);candidate[robot]=program
        known={tuple(sorted((robot,j))) for j in peers}
        actual,_,_=score(w.spec,[robot],candidate,known)
        assert actual==optimum
        return dict(robot=robot,score=optimum,program=program)

    def collision_extrema(self, robot, base, peers):
        """Exact min/max collision count over ALL legal T-tick paths.

        Time-expanded dynamic programming with absorbing goals. Shorter arrays
        are represented by their legal wait-padded T-tick equivalent.
        """
        w=self.w;n=len(self.cells)
        neighbors=self.neighbors.copy()
        goal=self.index[tuple(w.robots[robot]['goal'])]
        neighbors[goal,:]=-1
        neighbors[goal,self.moves.index('wait')]=goal
        valid=neighbors>=0;clamped=np.maximum(neighbors,0)
        rows=np.arange(n)[:,None]
        positions={j:[self.index[p] for p in w.trace(j,base[j]).positions] for j in peers}
        lo=np.zeros(n,dtype=int);hi=np.zeros(n,dtype=int)
        choices_lo=[];choices_hi=[]
        for tick in range(w.T,0,-1):
            reward=np.zeros_like(neighbors)
            for path in positions.values():
                previous,current=path[tick-1],path[tick]
                reward+=(neighbors==current)
                if previous!=current:
                    reward+=((rows==current)&(neighbors==previous))
            low=np.where(valid,reward+lo[clamped],10**9)
            high=np.where(valid,reward+hi[clamped],-10**9)
            choices_lo.append(low.argmin(axis=1))
            choices_hi.append(high.argmax(axis=1))
            lo=low.min(axis=1);hi=high.max(axis=1)
        start=self.index[tuple(w.robots[robot]['start'])]
        initial=sum(path[0]==start for path in positions.values())
        def program(choices):
            cell=start;commands=[]
            for choice in reversed(choices):
                j=int(choice[cell]);commands.append(self.moves[j])
                cell=int(neighbors[cell,j])
            trace=w.trace(robot,commands)
            count=sum(len(w.collisions(trace.positions,w.trace(j,base[j]).positions,robot,j)) for j in peers)
            return commands,count
        pmin,cmin=program(choices_lo);pmax,cmax=program(choices_hi)
        assert cmin==initial+int(lo[start]) and cmax==initial+int(hi[start])
        return dict(minimum=cmin,maximum=cmax,min_program=pmin,max_program=pmax)

    def uniform_error(self, b, base, known):
        owned=self.w.groups[b];own=set(owned)
        missing={(i,j) for i in range(9) for j in range(i+1,9)
            if bool({i,j}&own) and not {i,j}<=own and (i,j) not in known}
        base_events=independent_joint(self.w.spec,base)['events']
        current=external_count(base_events,owned,missing)
        extremes={}
        for i in owned:
            peers=[j if i==k else k for k,j in missing if i in (k,j)]
            extremes[i]=self.collision_extrema(i,base,peers)
        lower=sum(v['minimum'] for v in extremes.values())
        upper=sum(v['maximum'] for v in extremes.values())
        for field,total in [('min_program',lower),('max_program',upper)]:
            witness=copy.deepcopy(base)
            for i,v in extremes.items():
                witness[i]=v[field]
            events=independent_joint(self.w.spec,witness)['events']
            assert external_count(events,owned,missing)==total
        return dict(missing_pairs=[list(p) for p in sorted(missing)],
            missing_count=len(missing),base_collision_count=current,
            minimum_candidate_collision_count=lower,maximum_candidate_collision_count=upper,
            exact_uniform_delta=self.w.M*max(abs(lower-current),abs(upper-current)),
            interval_of_surrogate_minus_true_change=[
                -self.w.M*(upper-current),-self.w.M*(lower-current)],
            coarse_valid_bound=self.w.M*self.w.T*len(missing),
            witness_programs={self.w.ids[i]:v for i,v in extremes.items()})
