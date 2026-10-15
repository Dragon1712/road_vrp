"""Search-enhanced and traffic-aware CVRP operators using N-Wouda/ALNS v7.0.

This module imports the official library; no upstream source is copied. Builds
against the *existing* v2 Objective / ODCost / exact road evaluation protocol.

IMPORTANT: When obj.mode == 'oracle', dynamic OD features know the historical
future; results are offline hindsight teachers, never a deployable predictor.
Static planner is forced to use static-only signals; no future traffic leakage.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import math
import time
import numpy as np

from models import greedy, verify, as_list
from solvers import relocate, swap, reverse, two_opt_star
from upstream_alns_adapter import VRPState, CombinedStop, canonical

TRAFFIC_METHODS = (
    'upstream_plus', 'upstream_traffic',
    'upstream_traffic_no_t_related', 'upstream_traffic_no_t_worst',
    'upstream_traffic_no_regret3', 'upstream_traffic_no_local',
    'upstream_traffic_uniform',
)


def get_config(method, mode):
    if method not in TRAFFIC_METHODS:
        raise ValueError('Unknown enhanced method: '+method)
    if mode not in ('static','oracle'):
        raise ValueError('Planning mode must be static/oracle')
    signal = 'traffic' if method!='upstream_plus' else 'static'
    if mode=='static':
        # Ensures that a static planner never consults historical future traffic.
        signal='static'
    methods = ['greedy','regret2','regret3']
    related='traffic_related' if signal=='traffic' and method!='upstream_traffic_no_t_related' else 'static_related'
    worst='traffic_worst' if signal=='traffic' and method!='upstream_traffic_no_t_worst' else 'static_worst'
    if method=='upstream_traffic_no_regret3': methods.remove('regret3')
    return {
        'destroy':['random','worst','related','segment',related,worst],
        'repair':methods,
        'signal':signal,
        'traffic_related_enabled':related=='traffic_related',
        'traffic_worst_enabled':worst=='traffic_worst',
        'local_search':method!='upstream_traffic_no_local',
        'adaptive_weights':method!='upstream_traffic_uniform',
        'initialization':'v2-greedy',
        'library':'N-Wouda/ALNS v7.0.0',
        'budget':'max_iterations and search_wall_sec both bounded; stop after proposal',
    }


class TrafficFeatures:
    """Small, precomputed traffic descriptors over 31x31 OD matrices.

    Relative delays and temporal variability summarize traffic information already
    contained in the sampled OD travel-time tensor (not actual GPS or measured
    edge-by-edge congestion). Traffic information is used ONLY in oracle mode.
    """
    def __init__(self, obj, enable_traffic):
        self.obj = obj
        self.dynamic_enabled = bool(enable_traffic and obj.mode == 'oracle')
        od = obj.od
        static = np.asarray(od.static,dtype=float)
        n=static.shape[0]
        assert static.shape == (n,n) and obj.n+1==n
        sym=(static+static.T)*0.5
        nz=sym[np.isfinite(sym)&(sym>1e-10)]
        self.scale=float(np.median(nz)) if len(nz) else 1.
        self.static_symmetric = sym / max(self.scale,1.)
        self.node_delay=np.zeros(n,float)
        self.node_variability=np.zeros(n,float)
        if self.dynamic_enabled:
            d=np.asarray(od.dynamic,dtype=float)
            if d.ndim!=3 or d.shape[1:]!=(n,n):raise ValueError('Unexpected dynamic OD tensor shape')
            near=np.argsort(sym,axis=1)[:,1:min(n,7)]
            base=np.maximum(static,1.)
            rel=np.clip((d-base[None,:,:])/base[None,:,:],-1.,5.)
            # For each customer, sample its K nearest static OD destinations;
            # avoid blending all far OD pairs that mostly use static fallback.
            for i in range(1,n):
                neigh=near[i]
                if not len(neigh):continue
                r=rel[:,i,neigh]
                self.node_delay[i]=float(np.mean(r[0]))
                self.node_variability[i]=float(np.mean(np.std(r,axis=0)))
            self.dynamic_symmetric=0.5*(d[0]+d[0].T)/max(self.scale,1.)
        else:
            self.dynamic_symmetric=self.static_symmetric.copy()
        self.max_demand=max(1,max(obj.dem.values()))

    def similarity(self,anchor,candidate,same_vehicle=False):
        # Lower means more related. Travel time is directional on road legs, so
        # symmetric averages are used ONLY for Shaw relatedness, never routing.
        base=(self.dynamic_symmetric[anchor,candidate] if self.dynamic_enabled
              else self.static_symmetric[anchor,candidate])
        value=(0.65*base +
               0.15*abs(self.obj.dem[anchor]-self.obj.dem[candidate])/self.max_demand)
        if self.dynamic_enabled:
            value += (0.13*abs(self.node_delay[anchor]-self.node_delay[candidate])+
                      0.07*abs(self.node_variability[anchor]-self.node_variability[candidate]))
        if same_vehicle:
            value-=0.1
        return float(value)

    def hotspot(self,seq,pos):
        """Traffic-specific marginal indicator at *actual* estimated visit times.

        Unlike static worst-removal, evaluates positive travel-time excess of
        incoming/outgoing arcs at their schedule-dependent departure times.
        Candidate still ranked jointly with exact surrogate removal saving.
        """
        if not self.dynamic_enabled:return 0.
        o=self.obj
        t=o.depart;last=0
        for j,c in enumerate(seq):
            if j==pos:
                incoming=o.od.cost(last,c,t,'oracle')
                next_c=seq[j+1] if j+1<len(seq) else 0
                after=t+incoming+o.service
                outgoing=o.od.cost(c,next_c,after,'oracle')
                sin=o.od.static[last,c]
                sout=o.od.static[c,next_c]
                return max(0.,incoming-sin)+max(0.,outgoing-sout)
            w=o.od.cost(last,c,t,'oracle'); t+=w+o.service;last=c
        return 0.


def removable(work, obj):
    return [(i,p,c) for i,r in enumerate(work)
            if len(r) > (1 if obj.require_all else 0)
            for p,c in enumerate(r)]


class EnhancedOperators:
    def __init__(self, obj, stop, cfg, seed=42):
        self.obj=obj;self.stop=stop;self.cfg=cfg
        self.features=TrafficFeatures(obj,cfg['signal']=='traffic')
        self.static_features=TrafficFeatures(obj,False)
        self.destroy_calls=Counter();self.repair_calls=Counter()
        self.failed_repairs=0;self.local_score_calls=0
        self.local_improvements=0;self.repair_ticks=0
        self.changed_proposals=0

    def destroy(self,name,state,rng):
        if state.removed:raise ValueError('Destroy must receive a complete state')
        self.destroy_calls[name]+=1
        work=as_list(state.routes)
        n=self.obj.n
        lo=max(2,n//10)
        hi=max(lo+1,n//3+1)
        # Wider random/segment neighborhoods; same for plus and traffic.
        if name in ('random','segment'):
            hi=max(hi,int(math.ceil(n*0.43))+1)
        q=int(rng.integers(lo,hi))
        removed=[]
        parent_routes=as_list(state.routes)
        parent_index={c:i for i,r in enumerate(parent_routes) for c in r}
        if name=='segment':
            # Demand-aware contiguous block removal: preserve one stop per
            # vehicle under Exactly-K. Blocks can span multiple routes.
            tries=0
            while len(removed)<q and tries<3*q+10:
                tries+=1
                options=[i for i,r in enumerate(work) if len(r)>(1 if self.obj.require_all else 0)]
                if not options:break
                i=int(rng.choice(options));r=work[i]
                max_take=min(q-len(removed),len(r)-(1 if self.obj.require_all else 0),max(1,len(r)//2))
                take=int(rng.integers(1,max_take+1))
                start=int(rng.integers(0,len(r)-take+1))
                removed.extend(r[start:start+take]);del r[start:start+take]
            return VRPState(canonical(work),self.obj,tuple(removed),state.routes)
        for _ in range(q):
            candidates=removable(work,self.obj)
            if not candidates:break
            if name=='random':
                pick=candidates[int(rng.integers(len(candidates)))]
            elif name in ('related','static_related','traffic_related'):
                if not removed:
                    pick=candidates[int(rng.integers(len(candidates)))]
                else:
                    anchor=int(rng.choice(removed))
                    if name=='related':
                        # v2 geometry-based relatedness, without future traffic.
                        def score(t):
                            c=t[2]
                            d=min(self.obj.od.static[anchor,c],self.obj.od.static[c,anchor])
                            return d+12*abs(self.obj.dem[anchor]-self.obj.dem[c])- (30 if parent_index[anchor]==parent_index[c] else 0)
                    else:
                        f=self.static_features if name=='static_related' else self.features
                        def score(t):
                            c=t[2]
                            return f.similarity(anchor,c,parent_index[anchor]==parent_index[c])
                    ordered=sorted(candidates,key=score)
                    pick=ordered[int(rng.integers(max(1,len(ordered)//4)))]
            elif name in ('worst','static_worst','traffic_worst'):
                gain=[]
                for i,p,c in candidates:
                    r=work[i]
                    if name=='static_worst':
                        # Pure STATIC local arc saving. Distinct from the
                        # surrogate objective's (possibly time-dependent) worst.
                        prev=r[p-1] if p else 0
                        nxt=r[p+1] if p+1<len(r) else 0
                        s=self.obj.od.static
                        margin=float(s[prev,c]+s[c,nxt]-s[prev,nxt])
                    else:
                        margin=(self.obj.route_score(tuple(r))-
                                self.obj.route_score(tuple(r[:p]+r[p+1:])))
                        if name=='traffic_worst' and self.features.dynamic_enabled:
                            margin += 0.45*self.features.hotspot(r,p)
                    gain.append((margin,i,p,c))
                gain.sort(reverse=True)
                top=max(1,int(math.ceil(len(gain)/3)))
                _,i,p,c=gain[int(rng.integers(top))]
                pick=(i,p,c)
            else:raise ValueError(name)
            i,p,c=pick
            work[i].pop(p);removed.append(c)
        return VRPState(canonical(work),self.obj,tuple(removed),state.routes)

    def _repair(self,work,removed,kind):
        """Regret-k insertion with feasible-capacity repair and exact surrogate delta.

        No time-dependent TSP arc-difference shortcut: evaluating a route's
        total travel is necessary because insertion shifts downstream clock.
        """
        routes=as_list(work);pool=list(removed);o=self.obj
        loads=[sum(o.dem[c] for c in r) for r in routes]
        # Existing exact-k vehicles always retain a client after destruction.
        while pool:
            if time.perf_counter()>=self.stop.deadline:
                return None
            choices=[]
            for c in pool:
                options=[]
                for ri,r in enumerate(routes):
                    if loads[ri]+o.dem[c]>o.cap:continue
                    old=o.route_score(tuple(r))
                    for p in range(len(r)+1):
                        inc=o.route_score(tuple(r[:p]+[c]+r[p:]))-old
                        options.append((inc,ri,p))
                if not options:continue
                options.sort(key=lambda t:(t[0],t[1],t[2]))
                if kind=='greedy':
                    key=(options[0][0],c)
                else:
                    kk=2 if kind=='regret2' else 3
                    # Regret uses 2nd / 3rd cheapest FEASIBLE insertion.
                    # If only one exists, high urgency is intentional.
                    regret=sum((options[h][0]-options[0][0] if h<len(options) else 2000.)
                               for h in range(1,kk))
                    key=(-regret,options[0][0],c)
                choices.append((key,c,options[0]))
            if not choices:return None
            _,c,(_,ri,p)=min(choices,key=lambda t:t[0])
            routes[ri].insert(p,c);loads[ri]+=o.dem[c];pool.remove(c)
        return routes if o.legal(routes) else None

    def _local_search(self,routes,rng):
        """Randomized bounded neighborhood improvement (not full VND)."""
        if time.perf_counter()>=self.stop.deadline:return routes
        curr=as_list(routes);value=self.obj.score(curr)
        moves=(relocate,swap,reverse,two_opt_star)
        # Randomized order avoids v2 first-improvement's front-loaded bias.
        budget=36
        for _ in range(budget):
            if time.perf_counter()>=self.stop.deadline:break
            move=moves[int(rng.integers(len(moves)))]
            trial=move(curr,rng,self.obj.dem,self.obj.cap,self.obj.require_all)
            if trial==curr or not self.obj.legal(trial):continue
            self.local_score_calls+=1
            cost=self.obj.score(trial)
            if cost<value-1e-8:
                curr=trial;value=cost;self.local_improvements+=1
                break
        return curr

    def repair(self,name,state,rng):
        self.repair_calls[name]+=1;self.repair_ticks+=1
        if not state.removed:return VRPState(state.routes,self.obj)
        output=self._repair(state.routes,state.removed,name)
        if output is None:
            self.failed_repairs+=1
            if state.previous is None:raise ValueError('Cannot restore failed repair without original')
            return VRPState(state.previous,self.obj)
        if self.cfg['local_search'] and self.repair_ticks%4==0:
            output=self._local_search(output,rng)
        verify(self.obj.task,output)
        if canonical(output)!=state.previous:self.changed_proposals+=1
        return VRPState(canonical(output),self.obj)


def run_traffic_solver(method, task, obj, *, seed=42, max_iterations=5000, wall_sec=5.):
    """Drop-in benchmark_suite output; uses official ALNS7 engine."""
    cfg=get_config(method,obj.mode)
    try:
        import alns
        from alns import ALNS
        from alns.select import RouletteWheel,RandomSelect
        from alns.accept import SimulatedAnnealing
    except ImportError as exc:
        raise RuntimeError('Requires official N-Wouda/ALNS==7.0.0 installed in your offline server environment') from exc
    begin=time.perf_counter()
    routes,meta=greedy(task,obj,seed)
    verify(task,routes)
    construct_wall=time.perf_counter()-begin
    start=VRPState(canonical(routes),obj)
    init_cost=start.objective()
    stop=CombinedStop(max_iterations,wall_sec)
    ops=EnhancedOperators(obj,stop,cfg,seed)
    engine=ALNS(np.random.default_rng(seed))
    for name in cfg['destroy']:
        def build_destroy(which):
            def fn(state,rng):return ops.destroy(which,state,rng)
            return fn
        engine.add_destroy_operator(build_destroy(name),name=name)
    for name in cfg['repair']:
        def build_repair(which):
            def fn(state,rng):return ops.repair(which,state,rng)
            return fn
        engine.add_repair_operator(build_repair(name),name=name)
    if cfg['adaptive_weights']:
        selector=RouletteWheel(scores=[8.,4.,1.,0.],decay=.8,
                               num_destroy=len(cfg['destroy']),num_repair=len(cfg['repair']))
    else:
        selector=RandomSelect(num_destroy=len(cfg['destroy']),num_repair=len(cfg['repair']))
    accept=SimulatedAnnealing.autofit(init_obj=max(init_cost,1e-6),worse=.05,
                                     accept_prob=.75,num_iters=int(max_iterations))
    tic=time.perf_counter()
    result=engine.iterate(start,selector,accept,stop)
    search_wall=time.perf_counter()-tic
    best=as_list(result.best_state.routes)
    verify(task,best)
    best_cost=obj.score(best)
    if best_cost>init_cost+1e-7:raise AssertionError('Best state worse than initial')
    stats=getattr(result,'statistics',None)
    sd=getattr(stats,'destroy_operator_counts',None)
    sr=getattr(stats,'repair_operator_counts',None)
    return {
        'method':method,'routes':best,'loads':verify(task,best),
        'surrogate_travel_sec':float(best_cost),'initial_surrogate_travel_sec':float(init_cost),
        'iterations':int(stop.calls),
        'accepted_moves':(sum(sum(row[:3]) for row in sd.values()) if sd is not None else 0),
        'best_improvements':(sum(row[0] for row in sd.values()) if sd is not None else 0),
        'failed_repairs':ops.failed_repairs,
        'construction_wall_sec':float(construct_wall),'search_wall_sec':float(search_wall),
        'score_calls':int(obj.score_calls),'route_score_calls':int(obj.unconstrained_route_calls),
        'termination_reason':stop.reason,'local_optimum_verified':False,
        'budget_unit':'search proposals + wall-clock upper bound',
        'method_config':cfg,'alns_version':getattr(alns,'__version__','not_reported'),
        'operator_usage':{'destroy':dict(ops.destroy_calls),'repair':dict(ops.repair_calls)},
        'operator_outcomes':{'destroy':dict(sd or {}),'repair':dict(sr or {})},
        'traffic_diagnostics':{'dynamic_signal_enabled':ops.features.dynamic_enabled,
                               'node_delay_mean':float(np.mean(ops.features.node_delay[1:])),
                               'node_delay_max':float(np.max(ops.features.node_delay[1:])),
                               'node_temporal_variability_mean':float(np.mean(ops.features.node_variability[1:])),
                               'failed_repairs':ops.failed_repairs,
                               'changed_proposals':ops.changed_proposals,
                               'local_improvements':ops.local_improvements,
                               'local_score_calls':ops.local_score_calls},
        'initial_fallback':bool(meta.get('fallback_seed_routes',False)),
    }
