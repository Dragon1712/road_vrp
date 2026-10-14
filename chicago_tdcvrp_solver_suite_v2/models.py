"""VRP constraints and shared surrogate objective for ALL solvers."""
import functools
import math
import numpy as np


def verify(task,routes):
    k=int(task['vehicles']);n=len(task['customers'])
    if len(routes)!=k:raise ValueError(f'Expected {k} routes; found {len(routes)}')
    if task.get('require_all_vehicles',True) and any(len(r)==0 for r in routes):
        raise ValueError('Exactly-K convention: no empty routes allowed')
    ids=[int(c) for r in routes for c in r]
    if sorted(ids)!=list(range(1,n+1)):
        raise ValueError('Each customer must be visited exactly once')
    demand={int(c['id']):int(c['demand']) for c in task['customers']}
    loads=[sum(demand[c] for c in r) for r in routes]
    if max(loads,default=0)>int(task['capacity']):raise ValueError(f'Capacity violation {loads}')
    return loads


class Objective:
    def __init__(self, task, od, mode):
        self.task=task;self.od=od;self.mode=mode
        self.depart=od.start
        self.service=float(task['service_seconds'])
        self.dem={int(c['id']):int(c['demand']) for c in task['customers']}
        self.cap=int(task['capacity'])
        self.require_all=bool(task.get('require_all_vehicles',True))
        self.k=int(task['vehicles']);self.n=len(task['customers'])
        self.score_calls=0
        self.unconstrained_route_calls=0

    @functools.lru_cache(maxsize=250000)
    def route_score(self, route):
        self.unconstrained_route_calls+=1
        if not route:return 0.0
        t=self.depart;total=0.;previous=0
        for c in tuple(route)+(0,):
            w=self.od.cost(previous,c,t,self.mode)
            total+=w;t+=w
            if c:t+=self.service
            previous=c
        return float(total)

    def legal(self,routes):
        if len(routes)!=self.k:return False
        if self.require_all and any(not r for r in routes):return False
        return all(sum(self.dem[c] for c in r)<=self.cap for r in routes)

    def score(self,routes):
        self.score_calls+=1
        if not self.legal(routes):return math.inf
        return sum(self.route_score(tuple(r)) for r in routes)


def as_list(routes):return [list(map(int,r)) for r in routes]


def greedy(task,obj,seed=42):
    """Farthest-depot seed + cheapest feasible insertion. Fallback to task's saved feasible seed.

    Repaired greedy is a heuristic UPPER bound, not a mathematical lower bound.
    """
    k=obj.k;n=obj.n
    rng=np.random.default_rng(seed)
    order=list(range(1,n+1))
    # Seed farthest customers to different vehicles for exactly-K; at-most-K starts one route.
    seed_count=k if obj.require_all else 1
    seed_idx=sorted(order,key=lambda c:(-float(obj.od.static[0,c]),c))[:seed_count]
    routes=[[c] for c in seed_idx]+[[] for _ in range(k-seed_count)]
    pool=[c for c in order if c not in set(seed_idx)]
    # Randomized tie-breaking but no hidden future information.
    rng.shuffle(pool)
    while pool:
        candidates=[]
        for customer in pool:
            for i,route in enumerate(routes):
                if sum(obj.dem[x] for x in route)+obj.dem[customer]>obj.cap:continue
                base=obj.route_score(tuple(route))
                for pos in range(len(route)+1):
                    trial=route[:pos]+[customer]+route[pos:]
                    inc=obj.route_score(tuple(trial))-base
                    candidates.append((inc,customer,i,pos))
        if not candidates:
            feasible=as_list(task['seed_routes'])
            verify(task,feasible)
            return feasible,{'fallback_seed_routes':True,'greedy_insertion_steps':n-len(pool)}
        _,customer,i,pos=min(candidates,key=lambda x:(x[0],x[1],x[2],x[3]))
        routes[i].insert(pos,customer)
        pool.remove(customer)
    verify(task,routes)
    return routes,{'fallback_seed_routes':False,'greedy_insertion_steps':len(order)-seed_count}
