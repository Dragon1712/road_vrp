"""Budget-matched capacity-feasible simulated-annealing neighborhood search.

This is an honest heuristic pilot, not an optimal solver and not ALNS.
Same budget: fixed proposed local-search iterations. No time windows for CVRP30.
"""
from __future__ import annotations
import functools
import math
import time
import numpy as np


def validate(task, routes):
    nc=len(task['customers'])
    ids=[int(x) for r in routes for x in r]
    if len(routes)!=int(task['vehicles']): raise ValueError('Vehicle count mismatches')
    if any(not r for r in routes): raise ValueError('All vehicles must be used in this CVRP30 pilot')
    if sorted(ids)!=list(range(1,nc+1)): raise ValueError('Every customer must appear exactly once')
    demand={int(c['id']):int(c['demand']) for c in task['customers']}
    loads=[sum(demand[x] for x in r) for r in routes]
    if max(loads)>int(task['capacity']): raise ValueError('Capacity exceeded')
    return loads


class Objective:
    def __init__(self, task, surface, mode, departure_sec):
        self.task=task
        self.surf=surface
        self.mode=mode
        self.depart=float(departure_sec)
        self.demand=np.array([0]+[int(x['demand']) for x in task['customers']],dtype=int)
        self.capacity=int(task['capacity'])
        self.service=float(task['service_seconds'])

    @functools.lru_cache(maxsize=150000)
    def route_score(self, route):
        t=self.depart
        travel=0.
        last=0
        for nxt in tuple(route)+(0,):
            dt=self.surf.travel(last,nxt,t,self.mode)
            travel+=dt
            t+=dt
            if nxt: t+=self.service
            last=nxt
        return float(travel)

    def score(self, routes):
        if len(routes)!=len(self.task['seed_routes']) or any(not r for r in routes):return float('inf')
        if any(sum(self.demand[x] for x in r)>self.capacity for r in routes): return float('inf')
        return sum(self.route_score(tuple(r)) for r in routes)


def propose(rng,routes,demands,capacity):
    nr=len(routes)
    out=[r.copy() for r in routes]
    kind=int(rng.choice(3,p=[.35,.39,.26]))
    if kind==0:  # 2-opt reversal within one vehicle, order only
        eligible=[i for i,r in enumerate(out) if len(r)>=2]
        if not eligible:return out
        k=int(rng.choice(eligible)); n=len(out[k]); a,b=sorted(rng.choice(n,2,replace=False))
        out[k][a:b+1]=out[k][a:b+1][::-1]
    elif kind==1: # relocate customer; may change vehicle
        origin=int(rng.integers(nr))
        if len(out[origin])<=1: return out
        pos=int(rng.integers(len(out[origin])))
        cust=out[origin].pop(pos)
        dst=int(rng.integers(nr))
        if dst!=origin and sum(demands[x] for x in out[dst])+demands[cust]>capacity:
            return routes
        at=int(rng.integers(len(out[dst])+1))
        out[dst].insert(at,cust)
    else: # cross-vehicle or within-vehicle swap
        a,b=map(int,rng.integers(nr,size=2))
        p=int(rng.integers(len(out[a])));q=int(rng.integers(len(out[b])))
        x,y=out[a][p],out[b][q]
        if a!=b:
            ca=sum(demands[i] for i in out[a])-demands[x]+demands[y]
            cb=sum(demands[i] for i in out[b])-demands[y]+demands[x]
            if ca>capacity or cb>capacity:return routes
        out[a][p],out[b][q]=y,x
    return out


def solve(task, surface, departure_sec, mode, iterations=1500, seed=42, restarts=2):
    if iterations<1 or restarts<1: raise ValueError('Positive budget and restarts required')
    starter=[list(map(int,r)) for r in task['seed_routes']]
    validate(task,starter)
    objective=Objective(task,surface,mode,departure_sec)
    rng=np.random.default_rng(seed)
    demands=objective.demand
    best=[r.copy() for r in starter]
    best_cost=objective.score(best)
    start=time.perf_counter()
    trajectory=[]
    for k in range(restarts):
        current=[r.copy() for r in best if True] if k==0 else [r.copy() for r in starter]
        if k>0:
            for r in current: rng.shuffle(r)
        cur=objective.score(current)
        iters=iterations//restarts+(1 if k<iterations%restarts else 0)
        for i in range(iters):
            candidate=propose(rng,current,demands,objective.capacity)
            if candidate is current: continue
            nxt=objective.score(candidate)
            if not np.isfinite(nxt): continue
            progress=i/max(1,iters-1)
            t0=max(cur,1.)*0.02
            temp=t0*(0.02**progress)
            if nxt<cur-1e-9 or rng.random()<math.exp(min(0.,(cur-nxt)/max(temp,1e-8))):
                current,cur=candidate,nxt
            if cur<best_cost-1e-8:
                best=[r.copy() for r in current]
                best_cost=cur
            if (i+1)%250==0:
                trajectory.append({'restart':k,'iteration':i+1,'best_travel_sec':round(best_cost,3)})
    loads=validate(task,best)
    return {'routes':best,'loads':loads,'planned_travel_sec':float(best_cost),
            'search_wall_sec':time.perf_counter()-start,
            'iterations':int(iterations),'restarts':int(restarts),'mode':mode,
            'seed':int(seed),'history':trajectory,
            'objective_route_cache':objective.route_score.cache_info()._asdict()}


def evaluate_exact(road, task, routes, departure):
    """EX-POST evaluation: same historic scenario, TDSP reroutes each delivery leg.

    Compares vehicle assignments/visit order ONLY, not fixed road paths.
    """
    loads=validate(task,routes)
    cust={int(c['id']):str(c['node']) for c in task['customers']}
    depot=str(task['depot_node'])
    start=road.to_seconds(departure)
    total_travel=0.
    observed_count=0
    edge_count=0
    route_stats=[]
    for vi,sequence in enumerate(routes):
        t=start
        here=depot
        legs=[]
        for stop in list(sequence)+[0]:
            to=depot if stop==0 else cust[int(stop)]
            sec,edges=road.shortest(here,to,t,mode='dynamic',path=True)
            if not np.isfinite(sec): raise ValueError(f'No TDSP path {here}->{to}')
            n_obs=road.measure_path(edges,t)
            total_travel+=sec
            observed_count+=n_obs
            edge_count+=len(edges)
            t+=sec
            arrival=t
            if stop: t+=float(task['service_seconds'])
            legs.append({'customer_id':int(stop),'from_node':here,'to_node':to,
                         'road_tdsp_sec':round(float(sec),3),
                         'arrival_local':road.timestamp(arrival),
                         'measured_edge_entries':int(n_obs),'edge_traversals':len(edges)})
            here=to
        route_stats.append({'vehicle':vi,'customer_ids':[int(c) for c in sequence],
                            'load':loads[vi],'elapsed_sec':round(t-start,3),
                            'legs':legs})
    return {'realized_travel_sec':float(total_travel),
            'realized_total_elapsed_vehicle_sec':float(sum(x['elapsed_sec'] for x in route_stats)),
            'observed_edge_entries':int(observed_count),'edge_traversals':int(edge_count),
            'observed_entry_share':float(observed_count/max(1,edge_count)),
            'route_details':route_stats,
            'evaluation_protocol':'Ex-post TDSP road-leg rerouting; fixed customer order/assignment, not fixed road path'}
