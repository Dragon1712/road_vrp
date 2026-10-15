"""External open-source ALNS engine adapter for Chicago TD-CVRP30.

Upstream project: https://github.com/N-Wouda/ALNS (MIT), tested API target alns==7.0.0.
This is ORIGINAL integration code, not a vendored copy of upstream ALNS.

The ALNS engine handles iteration, adaptive operator selection, stopping and SA
acceptance; Chicago-specific operators/costs are imported from existing v2 code.
Objective() is evaluated on the same approx time-dependent OD surface as v2;
benchmark_suite re-evaluates resulting customer sequences on the original OSM
FIFO road network for the FINAL reported cost.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import math
import time
import numpy as np

from models import greedy, verify, as_list
from solvers import sample_destroy, repair, first_improvement

# A distinct label is important for correct matched-task benchmark accounting.
UPSTREAM_METHODS = (
    "upstream_alns",
    "upstream_no_related",
    "upstream_no_worst",
    "upstream_greedy_repair",
    "upstream_uniform",
    "upstream_no_local",
)


def config_for(method):
    if method not in UPSTREAM_METHODS:
        raise ValueError(f"Invalid upstream ALNS variant: {method}")
    destroy=["random","worst","related"]
    repairs=["greedy","regret2"]
    if method == "upstream_no_related": destroy.remove("related")
    if method == "upstream_no_worst": destroy.remove("worst")
    if method == "upstream_greedy_repair": repairs=["greedy"]
    return {
        "external_project":"N-Wouda/ALNS",
        "external_version":"7.0.0",
        "destroy":destroy,
        "repair":repairs,
        "adaptive_weights":method != "upstream_uniform",
        "local_search":method != "upstream_no_local",
        "selection":"RouletteWheel" if method != "upstream_uniform" else "RandomSelect",
        "acceptance":"SimulatedAnnealing",
    }


@dataclass(frozen=True)
class VRPState:
    """Immutable customer sequence state; only COMPLETE solutions are evaluated.

    Partial state is allowed only between destroy and repair. For partial states,
    objective() deliberately errors to catch incorrect library integration.
    """
    routes: tuple[tuple[int,...],...]
    obj: object
    removed: tuple[int,...]=()
    previous: tuple[tuple[int,...],...] | None=None

    def objective(self):
        if self.removed:
            raise RuntimeError('Partial CVRP state must be repaired before objective evaluation')
        return float(self.obj.score(self.routes))

    def get_context(self):
        return None  # reserved for future traffic-aware/contextual operator selection


def canonical(routes):
    return tuple(tuple(int(c) for c in r) for r in routes)


class CombinedStop:
    """ALNS7 stopping criterion protocol: (rng, best, current) -> bool.

    Bounds BOTH proposal count and walltime, like existing v2 solver. Note that
    Python-level checks happen only BETWEEN proposals, so walltime may overshoot.
    """
    def __init__(self, max_iterations, seconds):
        self.limit=int(max_iterations)
        self.seconds=float(seconds)
        if self.limit<1 or self.seconds<0:
            raise ValueError('Invalid ALNS search budget')
        self.calls=0
        self.begin=None
        self.reason='not_started'
        self.deadline=math.inf

    def __call__(self, rng, best, current):
        now=time.perf_counter()
        if self.begin is None:
            self.begin=now
            self.deadline=now+self.seconds if self.seconds>0 else math.inf
        if self.calls >= self.limit:
            self.reason='iteration_budget'
            return True
        if now >= self.deadline:
            self.reason='wall_budget'
            return True
        self.calls+=1
        return False


class ChicagoOperators:
    def __init__(self, obj, stop, local_search):
        self.obj=obj
        self.stop=stop
        self.local_search=bool(local_search)
        self.destroy_calls=Counter()
        self.repair_calls=Counter()
        self.failed_repairs=0
        self.repair_ticks=0

    def destroy(self, name, state, rng):
        if state.removed:
            raise RuntimeError('ALNS destroy received partial solution')
        self.destroy_calls[name]+=1
        # Up to 1/3 customers removed; from v2 operator implementation.
        n=self.obj.n
        lo=max(2,n//10)
        hi=max(lo+1,n//3+1)
        q=int(rng.integers(lo,hi))
        partial,removed=sample_destroy(state.routes,name,self.obj,rng,q)
        if not removed:
            # Fall back unchanged; repair returns same complete state.
            return VRPState(canonical(partial),self.obj,(),state.routes)
        return VRPState(canonical(partial),self.obj,tuple(removed),state.routes)

    def repair(self, name, state, rng):
        self.repair_calls[name]+=1
        self.repair_ticks+=1
        # Never let a repair failure introduce an infeasible candidate.
        if not state.removed:
            return VRPState(state.routes,self.obj)
        routes=repair(as_list(state.routes),list(state.removed),name,self.obj,rng)
        if routes is None:
            self.failed_repairs+=1
            if state.previous is None:
                raise RuntimeError('Repair failed without a feasible parent')
            return VRPState(state.previous,self.obj)
        if self.local_search and self.repair_ticks%5 == 0:
            # As in v2: bounded first improvement, not full VND.
            routes,_,_,_=first_improvement(routes,self.obj,self.stop.deadline,max_evals=60)
        if not self.obj.legal(routes):
            raise AssertionError('Repair produced an infeasible CVRP state')
        return VRPState(canonical(routes),self.obj)


def run_upstream_solver(method, task, obj, *, seed=42, max_iterations=5000, wall_sec=5.):
    """Drop-in v2 run_solver compatible output; official library required.

    MAJOR LIMITATION: SA cooling progression scales with max_iterations, even
    when time budget stops earlier; compare under the same fixed settings.
    """
    if method not in UPSTREAM_METHODS:
        raise ValueError(method)
    try:
        import alns
        from alns import ALNS
        from alns.select import RouletteWheel, RandomSelect
        from alns.accept import SimulatedAnnealing
    except ImportError as exc:
        raise RuntimeError('Official ALNS 7.0.0 is missing. On an online matching Python env, '
                           'run pip download -d alns_wheels alns==7.0.0; '
                           'install offline with pip install --no-index --find-links alns_wheels alns==7.0.0') from exc

    cfg=config_for(method)
    t0=time.perf_counter()
    init,meta=greedy(task,obj,seed=seed)
    init_routes=canonical(init)
    verify(task,init_routes)
    construct_sec=time.perf_counter()-t0
    initial_state=VRPState(init_routes,obj)
    init_cost=initial_state.objective()
    stop=CombinedStop(max_iterations,wall_sec)
    ops=ChicagoOperators(obj,stop,cfg['local_search'])
    engine=ALNS(np.random.default_rng(seed))

    def make_destroy(name):
        def fn(state,rng):return ops.destroy(name,state,rng)
        return fn
    def make_repair(name):
        def fn(state,rng):return ops.repair(name,state,rng)
        return fn
    for name in cfg['destroy']:
        engine.add_destroy_operator(make_destroy(name),name=name)
    for name in cfg['repair']:
        engine.add_repair_operator(make_repair(name),name=name)

    if cfg['adaptive_weights']:
        selector=RouletteWheel(scores=[8.,4.,1.,0.],decay=.8,
                               num_destroy=len(cfg['destroy']),num_repair=len(cfg['repair']))
    else:
        selector=RandomSelect(num_destroy=len(cfg['destroy']),num_repair=len(cfg['repair']))
    accept=SimulatedAnnealing.autofit(init_obj=max(init_cost,1e-6),worse=.05,
                                     accept_prob=.75,num_iters=int(max_iterations))
    # The start of search is measured after initial construction, as in v2.
    search_begin=time.perf_counter()
    result=engine.iterate(initial_state,selector,accept,stop)
    search_wall=time.perf_counter()-search_begin
    best=result.best_state
    final=as_list(best.routes)
    loads=verify(task,final)
    final_cost=obj.score(final)
    # Official ALNS records four outcomes: [BEST, BETTER, ACCEPTED, REJECTED].
    # The test double intentionally has no such statistics; do not fake counts.
    stats=getattr(result,'statistics',None)
    per_destroy=getattr(stats,'destroy_operator_counts',None)
    per_repair=getattr(stats,'repair_operator_counts',None)
    accepted_moves=(sum(sum(row[:3]) for row in per_destroy.values())
                    if per_destroy is not None else None)
    best_improvements=(sum(row[0] for row in per_destroy.values())
                       if per_destroy is not None else None)
    if final_cost>init_cost+1e-6:
        raise AssertionError('Library best state is worse than initial state')
    return {
        'method':method,'routes':final,'loads':loads,
        'surrogate_travel_sec':float(final_cost),
        'initial_surrogate_travel_sec':float(init_cost),
        'iterations':int(stop.calls),
        'accepted_moves':accepted_moves,
        'best_improvements':best_improvements,
        'failed_repairs':int(ops.failed_repairs),
        'construction_wall_sec':float(construct_sec),
        'search_wall_sec':float(search_wall),
        'score_calls':int(obj.score_calls),
        'route_score_calls':int(obj.unconstrained_route_calls),
        'termination_reason':stop.reason,
        'local_optimum_verified':False,
        'budget_unit':'ALNS iterations + search walltime (both bounded)',
        'method_config':cfg,
        'operator_usage':{'destroy':dict(ops.destroy_calls),'repair':dict(ops.repair_calls)},
        'operator_outcomes':{'destroy':dict(per_destroy or {}),'repair':dict(per_repair or {})},
        'alns_version':getattr(alns,'__version__','not_reported'),
        'initial_fallback':bool(meta.get('fallback_seed_routes',False)),
    }
