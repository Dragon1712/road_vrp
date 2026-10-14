#!/usr/bin/env python3
"""Generate ten or more independent synthetic-order TD-CVRP30 tasks on Chicago OSM.

Standalone within this solver suite: only road_eval.py/models.py and existing
`dynamic_costs` are needed. No external network or older tdcvrp package.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import networkx as nx
import numpy as np
from road_eval import Road
from models import verify


def corridor_candidates(road, allowed, radius):
    g=nx.Graph()
    for row in road.edges.itertuples():
        u,v=str(row.osm_u),str(row.osm_v)
        length=float(row.length_m)
        if g.has_edge(u,v):
            g[u][v]['length']=min(g[u][v]['length'],length)
        else:
            g.add_edge(u,v,length=length)
    mapped_edges=road.edges.loc[road.is_mapped]
    sources=set(mapped_edges.osm_u.astype(str))|set(mapped_edges.osm_v.astype(str))
    if not sources:raise ValueError('No mapped Chicago monitoring corridors')
    distance=nx.multi_source_dijkstra_path_length(g,list(sources),cutoff=radius,weight='length')
    return sorted(set(distance)&set(allowed))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cost-dir',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--num-tasks',type=int,default=10)
    ap.add_argument('--customers',type=int,default=30)
    ap.add_argument('--vehicles',type=int,default=5)
    ap.add_argument('--capacity',type=int,default=45)
    ap.add_argument('--service-seconds',type=int,default=120)
    ap.add_argument('--seed',type=int,default=20261014)
    ap.add_argument('--strategy',choices=['uniform','mixed','corridor'],default='mixed')
    ap.add_argument('--radius-m',type=float,default=600)
    ap.add_argument('--dates',default='2025-06-05,2025-06-06,2025-06-03,2025-06-04,2025-06-07,2025-06-08')
    ap.add_argument('--departure-clock',default='08:00:00')
    args=ap.parse_args()
    if args.num_tasks<10:raise ValueError('This experiment requires at least 10 distinct tasks')
    if args.customers<args.vehicles or min(args.customers,args.vehicles,args.capacity)<1:
        raise ValueError('Invalid customers/vehicles/capacity')
    out=args.output_dir
    if out.exists() and any(out.iterdir()):
        raise SystemExit('Refusing to overwrite nonempty tasks directory. Use a NEW --output-dir')
    dates=[z.strip() for z in args.dates.split(',') if z.strip()]
    if len(dates)<6 or len(set(dates))!=len(dates):
        raise ValueError('At least six distinct scenario dates required')
    road=Road(args.cost_dir)
    scenarios=[]
    for date in dates:
        departure=date+' '+args.departure_clock
        sec=road.to_sec(departure)
        if sec+4*3600+3600>=road.end_sec:
            raise ValueError(f'{departure} has insufficient future traffic horizon')
        scenarios.append({'scenario_id':date+'_'+args.departure_clock[:5].replace(':',''),
                          'departure_local':departure})
    allowed=road.largest_scc_nodes()
    if len(allowed)<args.customers+1:
        raise ValueError('Largest SCC too small for selected task size')
    near=corridor_candidates(road,allowed,args.radius_m)
    if args.strategy=='corridor' and len(near)<args.customers+1:
        raise ValueError('Not enough mapped-corridor-nearby road nodes')
    rng=np.random.default_rng(args.seed)
    seen=set();rows=[]
    out.mkdir(parents=True,exist_ok=False) if not out.exists() else None
    for task_i in range(args.num_tasks):
        if args.strategy=='uniform':
            nodes=rng.choice(allowed,args.customers+1,replace=False).tolist()
        elif args.strategy=='corridor':
            nodes=rng.choice(near,args.customers+1,replace=False).tolist()
        else:
            n_near=min(len(near),int(round((args.customers+1)*.7)))
            near_nodes=rng.choice(near,n_near,replace=False).tolist() if n_near else []
            pool=sorted(set(allowed)-set(near_nodes))
            more=rng.choice(pool,args.customers+1-len(near_nodes),replace=False).tolist()
            nodes=near_nodes+more
            rng.shuffle(nodes)
        depot=str(nodes[0]); cust_nodes=list(map(str,nodes[1:]))
        n=args.customers
        customer_ids=np.arange(1,n+1)
        rng.shuffle(customer_ids)
        seed_routes=[list(map(int,seq)) for seq in np.array_split(customer_ids,args.vehicles)]
        for attempt in range(1000):
            demand=rng.integers(2,8,size=n).tolist()
            if all(sum(demand[x-1] for x in r)<=args.capacity for r in seed_routes):break
        else:
            raise ValueError(f'Could not generate feasible seed routes for capacity={args.capacity}')
        task={'task_id':f'task_{task_i:04d}',
              'kind':'SYNTHETIC delivery orders on actual Chicago OSM network',
              'depot_node':depot,
              'customers':[{'id':j+1,'node':cust_nodes[j],'demand':demand[j]} for j in range(n)],
              'vehicles':int(args.vehicles),'capacity':int(args.capacity),
              'service_seconds':int(args.service_seconds),'require_all_vehicles':True,
              'seed_routes':seed_routes,'scenarios':scenarios,
              'sampling_strategy':args.strategy,'matched_corridor_radius_m':args.radius_m,
              'seed':args.seed+task_i}
        verify(task,seed_routes)
        signature=hashlib.sha256(json.dumps({'depot':depot,'customers':task['customers']},sort_keys=True).encode()).hexdigest()
        if signature in seen:raise AssertionError(f'Duplicate task content: {task_i}')
        seen.add(signature)
        (out/(task['task_id']+'.json')).write_text(json.dumps(task,ensure_ascii=False,indent=2),encoding='utf-8')
        rows.append({'task_id':task['task_id'],'customer_count':n,
                     'scenario_count':len(scenarios),'demand_sum':sum(demand),
                     'signature_sha256':signature})
    manifest={'num_distinct_tasks':len(seen),'min_scenarios_per_task':len(scenarios),
              'all_seed_routes_capacity_feasible':True,'tasks':rows,
              'sampling_strategy':args.strategy,'root_seed':args.seed,
              'warning':'Customer orders/demand are synthetic, not real Chicago logistics records.'}
    (out/'task_validation.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'manifest.json').write_text(json.dumps({'tasks':len(rows),'customers_each':args.customers,
        'vehicles_each':args.vehicles,'scenarios_per_task':len(scenarios),
        'task_files':[r['task_id']+'.json' for r in rows]},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'num_distinct_tasks':len(seen),'scenarios_each':len(scenarios),
        'valid_seed_routes':True,'output_dir':str(out)},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
