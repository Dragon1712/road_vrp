"""Time-dependent customer-to-customer OD oracle sampled on a time grid.

Search uses an interpolated OD arrival-time surface (fast approximation).
Final reports MUST re-evaluate every leg on the original directed road graph.
"""
from __future__ import annotations
import numpy as np
from .road import RoadCosts


def precompute(road: RoadCosts, nodes, departure, horizon_hours=4, step_minutes=20, quiet=False):
    ids=list(map(str,nodes))
    if len(ids)!=len(set(ids)):
        raise ValueError('Duplicate depot/customer road nodes: sample different nodes')
    start=road.to_seconds(departure)
    step=float(step_minutes)*60
    if not (step>0 and horizon_hours>0): raise ValueError('Positive step and horizon required')
    # Include last end point; integer division prevents floating time drift.
    n_knots=int(np.ceil(horizon_hours*3600/step))+1
    knots=start+np.arange(n_knots,dtype=float)*step
    if knots[-1]+3600 >= len(road.times)*road.dt:
        raise ValueError('OD horizon too close to end of measured traffic data; choose earlier scenarios or shorter horizon')
    m=len(ids)
    static=np.empty((m,m),dtype=np.float64)
    for i,u in enumerate(ids):
        found=road.one_to_many(u,ids,0.0,mode='static')
        static[i,:]=[found.get(v,np.inf) for v in ids]
    if not np.isfinite(static).all():
        raise ValueError('Some customers/depot are not mutually reachable; sample from largest SCC')
    dyn=np.empty((len(knots),m,m),dtype=np.float64)
    for k,t in enumerate(knots):
        for i,u in enumerate(ids):
            found=road.one_to_many(u,ids,float(t),mode='dynamic')
            dyn[k,i,:]=[found.get(v,np.inf) for v in ids]
        if not quiet and (k==0 or (k+1)%5==0 or k+1==len(knots)):
            print(f'  OD oracle time grid: {k+1}/{len(knots)}; nodes={m}',flush=True)
    if not np.isfinite(dyn).all():
        raise ValueError('TD OD oracle returned disconnected graph nodes')
    # Time-dependent shortest-path arrival-time FIFO at every pair (allow roundoff).
    arrivals=knots[:,None,None]+dyn
    dA=np.diff(arrivals,axis=0)
    if dA.min() < -1e-5:
        raise ValueError(f'OD arrival-time knots not FIFO: min diff={dA.min()}')
    return ODSurface(ids,knots,dyn,static)


class ODSurface:
    def __init__(self, nodes, knots, dynamic, static):
        self.nodes=list(map(str,nodes))
        self.knots=np.asarray(knots,dtype=float)
        self.dynamic=np.asarray(dynamic,dtype=float)
        self.static=np.asarray(static,dtype=float)
        self.idx={x:i for i,x in enumerate(self.nodes)}
        self.step=self.knots[1]-self.knots[0]

    def travel(self, i, j, departure, mode):
        if i==j: return 0.0
        if mode=='static': return float(self.static[i,j])
        if mode!='oracle': raise ValueError(f'Unknown planner mode {mode}')
        u=(departure-self.knots[0])/self.step
        if u < -1e-7 or u > len(self.knots)-1+1e-7:
            raise ValueError('A route departure exceeds OD time horizon; increase --od-hours')
        k=int(max(0,min(np.floor(u),len(self.knots)-2)))
        alpha=float(np.clip(u-k,0.,1.))
        a0=self.knots[k]+self.dynamic[k,i,j]
        a1=self.knots[k+1]+self.dynamic[k+1,i,j]
        return max(0.,(1-alpha)*a0+alpha*a1-departure)
