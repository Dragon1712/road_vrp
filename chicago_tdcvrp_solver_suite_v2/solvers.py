"""Four independent solver families plus ALNS operator ablations, zero training required.

Shared objective, random seed, capacities, OSM-based OD costs and exact road evaluation.
Budgets: `max_iterations` and/or `wall_sec`. Count full objective evaluations.
"""
from __future__ import annotations
import math
import time
from collections import defaultdict
import numpy as np
from models import greedy, verify, as_list

METHODS = ['greedy','sa','vnd','alns',
           'alns_no_related','alns_no_worst','alns_greedy_repair',
           'alns_uniform','alns_no_local']


def relocate(routes,rng,dem,cap,require_all):
    out=as_list(routes)
    eligible=[i for i,r in enumerate(out) if len(r)>(1 if require_all else 0)]
    if not eligible:return out
    i=int(rng.choice(eligible));j=int(rng.integers(len(out)))
    p=int(rng.integers(len(out[i])));c=out[i].pop(p)
    if sum(dem[x] for x in out[j])+dem[c] > cap:
        return as_list(routes)
    q=int(rng.integers(len(out[j])+1))
    out[j].insert(q,c)
    return out


def swap(routes,rng,dem,cap,require_all):
    out=as_list(routes)
    active=[i for i,r in enumerate(out) if r]
    if not active:return out
    i,j=map(int,rng.choice(active,2,replace=True))
    p=int(rng.integers(len(out[i])));q=int(rng.integers(len(out[j])))
    a,b=out[i][p],out[j][q]
    if i!=j:
        load_i=sum(dem[x] for x in out[i])-dem[a]+dem[b]
        load_j=sum(dem[x] for x in out[j])-dem[b]+dem[a]
        if load_i>cap or load_j>cap:return as_list(routes)
    out[i][p],out[j][q]=b,a
    return out


def reverse(routes,rng,dem,cap,require_all):
    out=as_list(routes)
    eligible=[i for i,r in enumerate(out) if len(r)>=2]
    if not eligible:return out
    i=int(rng.choice(eligible));a,b=sorted(map(int,rng.choice(len(out[i]),2,replace=False)))
    out[i][a:b+1]=out[i][a:b+1][::-1]
    return out


def two_opt_star(routes,rng,dem,cap,require_all):
    out=as_list(routes)
    if len(out)<2:return out
    i,j=map(int,rng.choice(len(out),2,replace=False))
    ai=int(rng.integers(0 if not require_all else 1,len(out[i])+1)) if out[i] else 0
    bj=int(rng.integers(0 if not require_all else 1,len(out[j])+1)) if out[j] else 0
    x,y=out[i][:ai]+out[j][bj:],out[j][:bj]+out[i][ai:]
    if require_all and (not x or not y):return as_list(routes)
    if sum(dem[z] for z in x)>cap or sum(dem[z] for z in y)>cap:return as_list(routes)
    out[i],out[j]=x,y
    return out

MOVES=[relocate,swap,reverse,two_opt_star]


def first_improvement(routes,obj,deadline,operator_names=None, max_evals=120, rng=None):
    """Bounded deterministic VND step; deadline checked between full evaluations.

    `operator_names` can remove specific local operators in controlled ablations.
    """
    allowed=set(operator_names or ['reverse','relocate','swap','two_opt_star'])
    current=as_list(routes)
    val=obj.score(current)
    n_eval=0
    def attempt(candidate):
        nonlocal current,val,n_eval
        if time.perf_counter()>=deadline or n_eval>=max_evals:return False,True
        n_eval+=1
        s=obj.score(candidate)
        if s<val-1e-7:
            current=as_list(candidate);val=s;return True,False
        return False,False
    # Small first-improvement neighborhoods in defined order.
    if 'reverse' in allowed:
        for i,r in enumerate(current):
            for a in range(len(r)):
                for b in range(a+1,len(r)):
                    c=as_list(current);c[i][a:b+1]=reversed(c[i][a:b+1])
                    found,stop=attempt(c)
                    if found or stop:return current,val,n_eval,found
    if 'relocate' in allowed:
        for i,r in enumerate(current):
            if obj.require_all and len(r)<=1:continue
            for a in range(len(r)):
                for j,rr in enumerate(current):
                    for p in range(len(rr)+1):
                        if i==j and p in (a,a+1):continue
                        c=as_list(current);node=c[i].pop(a)
                        dest=p if i!=j or p<=a else p-1
                        c[j].insert(dest,node)
                        found,stop=attempt(c)
                        if found or stop:return current,val,n_eval,found
    if 'swap' in allowed:
        for i,r in enumerate(current):
            for j in range(i,len(current)):
                for a in range(len(r)):
                    for b in range((a+1 if i==j else 0),len(current[j])):
                        c=as_list(current);c[i][a],c[j][b]=c[j][b],c[i][a]
                        found,stop=attempt(c)
                        if found or stop:return current,val,n_eval,found
    if 'two_opt_star' in allowed:
        for i in range(len(current)):
            for j in range(i+1,len(current)):
                for a in range(len(current[i])+1):
                    for b in range(len(current[j])+1):
                        c=as_list(current)
                        c[i]=current[i][:a]+current[j][b:]
                        c[j]=current[j][:b]+current[i][a:]
                        found,stop=attempt(c)
                        if found or stop:return current,val,n_eval,found
    return current,val,n_eval,False



# VND must distinguish "all neighbors exhausted" from "budget exhausted".
# The old implementation checked at most 200 proposals, then treated a truncated
# scan as a local optimum and terminated the entire run. The new generator and
# scan return an explicit budget status and restart from the first neighborhood
# after any genuine improvement.
VND_NEIGHBORHOODS = ('reverse', 'relocate', 'swap', 'two_opt_star')


def iter_neighborhood(routes, name, obj):
    """Enumerate candidates completely, in a deterministic order.

    Route assignments/visits are preserved; objective evaluates feasibility.
    No changes to caller-owned routes occur.
    """
    if name not in VND_NEIGHBORHOODS:
        raise ValueError(f'Unknown VND neighborhood: {name}')
    current=as_list(routes)
    if name=='reverse':
        for i,r in enumerate(current):
            for a in range(len(r)):
                for b in range(a+1,len(r)):
                    c=as_list(current);c[i][a:b+1]=reversed(c[i][a:b+1])
                    yield c
    elif name=='relocate':
        for i,r in enumerate(current):
            if obj.require_all and len(r)<=1:continue
            for a in range(len(r)):
                for j,rr in enumerate(current):
                    for pos in range(len(rr)+1):
                        if i==j and pos in (a,a+1):continue
                        c=as_list(current);node=c[i].pop(a)
                        insertion=pos if i!=j or pos<=a else pos-1
                        c[j].insert(insertion,node)
                        yield c
    elif name=='swap':
        for i,r in enumerate(current):
            for j in range(i,len(current)):
                for a in range(len(r)):
                    for b in range((a+1 if i==j else 0),len(current[j])):
                        c=as_list(current);c[i][a],c[j][b]=c[j][b],c[i][a]
                        yield c
    else:  # two_opt_star
        for i in range(len(current)):
            for j in range(i+1,len(current)):
                for a in range(len(current[i])+1):
                    for b in range(len(current[j])+1):
                        c=as_list(current)
                        c[i]=current[i][:a]+current[j][b:]
                        c[j]=current[j][:b]+current[i][a:]
                        yield c


def vnd_descent(initial, obj, *, deadline=math.inf, max_evaluations=None):
    """True first-improvement variable-neighborhood descent when budget permits.

    `max_evaluations` caps enumerated/evaluated candidates (unlike ALNS proposals).
    Return `local_optimum_verified=True` ONLY if every neighborhood was fully
    checked without finding an improvement. Time/eval budget exhaustion is
    reported as such, never mislabeled convergence.
    """
    routes=as_list(initial)
    current=obj.score(routes)
    best=as_list(routes)
    best_cost=current
    total_evals=0
    improvements=0
    k=0
    reason='local_optimum'
    completed_scans=0
    while k<len(VND_NEIGHBORHOODS):
        if time.perf_counter() >= deadline:
            reason='wall_budget';break
        name=VND_NEIGHBORHOODS[k]
        improved=False
        exhausted=True
        for candidate in iter_neighborhood(routes,name,obj):
            if time.perf_counter() >= deadline:
                reason='wall_budget';exhausted=False;break
            if max_evaluations is not None and total_evals>=max_evaluations:
                reason='evaluation_budget';exhausted=False;break
            total_evals+=1
            cost=obj.score(candidate)
            if cost<current-1e-8:
                routes=as_list(candidate);current=cost
                best=as_list(routes);best_cost=current
                improvements+=1;improved=True
                k=0
                break
        if not exhausted:break
        if not improved:
            completed_scans+=1
            k+=1
    return {'routes':best,'cost':best_cost,'evaluations':total_evals,
            'improvements':improvements,'termination_reason':reason,
            'local_optimum_verified':reason=='local_optimum',
            'completed_neighborhood_scans':completed_scans}


def alns_config(method):
    """Explicit configuration, persisted with every solve for ablation auditing."""
    if method not in METHODS or not method.startswith('alns'):
        raise ValueError('Not an ALNS variant')
    destroy=['random','worst','related']
    repair_methods=['greedy','regret2']
    if method=='alns_no_related':destroy.remove('related')
    if method=='alns_no_worst':destroy.remove('worst')
    if method=='alns_greedy_repair':repair_methods=['greedy']
    return {'destroy':destroy,'repair':repair_methods,
            'adaptive_weights':method!='alns_uniform',
            'local_search':method!='alns_no_local'}


def sample_destroy(routes,kind,obj,rng,q):
    work=as_list(routes)
    removed=[]
    for _ in range(q):
        legal=[(i,c) for i,r in enumerate(work) if len(r)>(1 if obj.require_all else 0) for c in r]
        if not legal:break
        if kind=='random':
            i,c=legal[int(rng.integers(len(legal)))]
        elif kind=='worst':
            candidates=[]
            for i,c in legal:
                route=work[i];p=route.index(c)
                gain=obj.route_score(tuple(route))-obj.route_score(tuple(route[:p]+route[p+1:]))
                candidates.append((gain,i,c))
            candidates.sort(reverse=True)
            # stochastic among the most costly 1/3
            _,i,c=candidates[int(rng.integers(max(1,len(candidates)//3)))]
        elif kind=='related':
            if not removed:
                i,c=legal[int(rng.integers(len(legal)))]
            else:
                anchor=int(removed[int(rng.integers(len(removed)))])
                def relation(ic):
                    ix,c=ic
                    oa=obj.od.static
                    near=min(oa[anchor,c],oa[c,anchor])
                    demand_delta=abs(obj.dem[anchor]-obj.dem[c])
                    same_vehicle=any(anchor in r and c in r for r in routes)
                    return (near + 12*demand_delta - (30 if same_vehicle else 0))
                legal.sort(key=relation)
                i,c=legal[int(rng.integers(max(1,len(legal)//4)))]
        else:raise ValueError(kind)
        work[i].remove(c)
        removed.append(c)
    return work,removed


def repair(work,removed,kind,obj,rng):
    """Greedy/regret-2 insertion; returns feasible complete solution or None."""
    work=as_list(work);pool=list(removed)
    while pool:
        choices=[]
        for c in pool:
            options=[]
            for ri,r in enumerate(work):
                if sum(obj.dem[z] for z in r)+obj.dem[c]>obj.cap:continue
                base=obj.route_score(tuple(r))
                for pos in range(len(r)+1):
                    test=r[:pos]+[c]+r[pos:]
                    options.append((obj.route_score(tuple(test))-base,ri,pos))
            options.sort()
            if not options:continue
            if kind=='greedy':
                choices.append((options[0][0],c,options[0]))
            elif kind=='regret2':
                regret=(options[1][0]-options[0][0]) if len(options)>1 else 10000.
                choices.append((-regret,c,options[0]))
            else:raise ValueError(kind)
        if not choices:return None
        if kind=='greedy':
            _,c,(_,i,p)=min(choices,key=lambda z:(z[0],z[1]))
        else:
            _,c,(_,i,p)=min(choices,key=lambda z:(z[0],z[2][0],z[1]))
        work[i].insert(p,c)
        pool.remove(c)
    if not obj.legal(work):return None
    return work


def run_solver(method,task,obj,seed=42,max_iterations=1000,wall_sec=3.0):
    if method not in METHODS:raise ValueError(f'Unknown solver: {method}')
    rng=np.random.default_rng(seed)
    tic=time.perf_counter()
    initial,meta=greedy(task,obj,seed=seed)
    constructed_wall=time.perf_counter()-tic
    initial_score=obj.score(initial)
    current=as_list(initial);cur=initial_score
    best=as_list(current);best_val=cur
    start_search=time.perf_counter()
    deadline=start_search+wall_sec if wall_sec>0 else math.inf
    steps=0;accepted=0;improvements=0;failure_repairs=0
    stats=defaultdict(lambda:[0.,0.,0.]) # uses,rewards,accept-improve
    if method=='greedy':
        result=pack(best,best_val,initial_score,meta,obj,method,seed,0,0,0,0,0,constructed_wall,0.,stats)
        result.update({'termination_reason':'construction_only','local_optimum_verified':False,
                       'budget_unit':'construction_only','method_config':{'construction':'greedy_insertion'}})
        return result
    if method=='vnd':
        result=vnd_descent(current,obj,deadline=deadline,max_evaluations=max_iterations)
        elapsed=time.perf_counter()-start_search
        verify(task,result['routes'])
        packed=pack(result['routes'],result['cost'],initial_score,meta,obj,method,seed,
                    result['evaluations'],result['improvements'],result['improvements'],
                    0,0,constructed_wall,elapsed,stats)
        packed.update({'termination_reason':result['termination_reason'],
                       'local_optimum_verified':result['local_optimum_verified'],
                       'completed_neighborhood_scans':result['completed_neighborhood_scans'],
                       'budget_unit':'neighborhood_candidate_evaluations',
                       'method_config':{'neighborhoods':list(VND_NEIGHBORHOODS),
                                        'restart_after_improvement':True}})
        return packed
    if method.startswith('alns'):
        config=alns_config(method)
        operators=config['destroy']
        repairs=config['repair']
    else:
        config={'neighborhoods':[m.__name__ for m in MOVES]}
        operators=['random','worst','related']
        repairs=['greedy','regret2']
    weights={x:1.0 for x in operators+repairs}
    while steps<max_iterations and time.perf_counter()<deadline:
        steps+=1
        if method=='sa':
            op=MOVES[int(rng.integers(len(MOVES)))]
            cand=op(current,rng,obj.dem,obj.cap,obj.require_all)
            value=obj.score(cand)
            temp=max(1.,cur*.02)*(.01**(steps/max(max_iterations,1)))
            accept=value<cur or rng.random()<math.exp(min(0.,(cur-value)/temp))
            if accept and math.isfinite(value):
                current=as_list(cand);cur=value;accepted+=1
        else:
            # ALNS, including explicit operator ablations.
            def pick(names):
                if not config['adaptive_weights']:return names[int(rng.integers(len(names)))]
                w=np.array([weights[x] for x in names],dtype=float)
                return names[int(rng.choice(len(names),p=w/w.sum()))]
            rem=pick(operators);rep=pick(repairs)
            q=max(2,int(rng.integers(max(3,obj.n//10),max(4,obj.n//3+1))))
            partial,removed=sample_destroy(current,rem,obj,rng,q)
            cand=repair(partial,removed,rep,obj,rng)
            stats[rem][0]+=1;stats[rep][0]+=1
            if cand is None:
                failure_repairs+=1;continue
            if config['local_search'] and steps%5==0:
                cand,_,_,_=first_improvement(cand,obj,deadline,max_evals=60)
            value=obj.score(cand)
            # nontrivial late uphill moves allowed for exploration
            progress=steps/max(max_iterations,1)
            temp=max(1.,cur*.01)*(.015**progress)
            old_cur=cur
            accept=value<old_cur or rng.random()<math.exp(min(0.,(old_cur-value)/temp))
            if accept and math.isfinite(value):
                current=as_list(cand);cur=value;accepted+=1
            reward=0
            if value<best_val-1e-8:reward=5
            elif value<old_cur-1e-8:reward=2
            elif accept:reward=1
            stats[rem][1]+=reward;stats[rep][1]+=reward
            if config['adaptive_weights'] and steps%50==0:
                for op in weights:
                    u,rew,_=stats[op]
                    if u>0: weights[op]=max(.05,.8*weights[op]+.2*(rew/u))
                    stats[op][0]=0;stats[op][1]=0
        if cur<best_val-1e-8:
            best=as_list(current);best_val=cur;improvements+=1
    search_wall=time.perf_counter()-start_search
    verify(task,best)
    result=pack(best,best_val,initial_score,meta,obj,method,seed,
                steps,accepted,improvements,failure_repairs,0,constructed_wall,search_wall,stats)
    result.update({'termination_reason':('wall_budget' if time.perf_counter()>=deadline else 'iteration_budget'),
                   'local_optimum_verified':False,
                   'budget_unit':'search_proposals',
                   'method_config':config,
                   'final_operator_weights':dict(weights) if method.startswith('alns') else {}})
    return result


def pack(routes,cost,init_cost,meta,obj,method,seed,iterations,accepted,improved,failed,unused,construct_sec,search_sec,stats):
    return {'method':method,'routes':routes,'loads':verify(obj.task,routes),
            'surrogate_travel_sec':float(cost),'initial_surrogate_travel_sec':float(init_cost),
            'iterations':int(iterations),'accepted_moves':int(accepted),
            'best_improvements':int(improved),'failed_repairs':int(failed),
            'construction_wall_sec':float(construct_sec),'search_wall_sec':float(search_sec),
            'score_calls':obj.score_calls,'route_score_calls':obj.unconstrained_route_calls,
            'seed':int(seed),'greedy_meta':meta,
            'operators_stats':{k:list(v) for k,v in stats.items()}}
