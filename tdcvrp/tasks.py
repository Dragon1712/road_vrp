"""Generate repeatable real-road / synthetic-demand, capacity-feasible TD-CVRP30 tasks."""
from __future__ import annotations
import json
from pathlib import Path
import networkx as nx
import numpy as np
from .road import RoadCosts


def monitoring_proximity(road, radius_m):
    """Undirected graph-network distance to a Chicago-mapped road endpoint.

    This is for choosing interesting synthetic locations, not real customer geography.
    """
    U=nx.Graph()
    for e in road.edges.itertuples(index=False):
        u,v=str(e.osm_u),str(e.osm_v)
        length=float(e.length_m)
        if U.has_edge(u,v):
            U[u][v]['distance_m']=min(U[u][v]['distance_m'],length)
        else:
            U.add_edge(u,v,distance_m=length)
    mask=road.mapped
    sources=set(road.edges.loc[mask,'osm_u'])|set(road.edges.loc[mask,'osm_v'])
    if not sources:
        raise ValueError('No mapped roads: corridor-aware customer sampling unavailable')
    dist=nx.multi_source_dijkstra_path_length(U,list(sources),cutoff=radius_m,weight='distance_m')
    return {str(k):float(v) for k,v in dist.items()}


def generate(road, output_dir, *, num_tasks=10, customers=30, vehicles=5,
             capacity=45, service_seconds=120, seed=2026, radius_m=600,
             strategy='mixed', dates=None, departure_clock='08:00:00'):
    output_dir=Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    if not dates: dates=['2025-06-05','2025-06-06','2025-06-03','2025-06-04','2025-06-07','2025-06-08']
    if customers<vehicles or vehicles<1 or capacity<=0: raise ValueError('Invalid customers/vehicles/capacity')
    # Largest SCC guarantees depot->all customers->depot connectivity in directed graph.
    allowed=set(road.largest_scc)
    near=monitoring_proximity(road,radius_m)
    corridor=sorted(allowed & set(near))
    all_nodes=sorted(allowed)
    if len(corridor)<customers+1 and strategy=='corridor':
        raise ValueError(f'Only {len(corridor)} nearby nodes; increase --radius-m')
    if len(all_nodes)<customers+1: raise ValueError('Not enough SCC nodes')
    rng=np.random.default_rng(seed)
    tasks=[]
    for i in range(num_tasks):
        if strategy=='mixed':
            ratio=0.7
            n_near=min(int(round((customers+1)*ratio)),len(corridor))
            nearpick=rng.choice(corridor,n_near,replace=False).tolist() if n_near else []
            rest=sorted(set(all_nodes)-set(nearpick))
            farpick=rng.choice(rest,customers+1-len(nearpick),replace=False).tolist()
            nodes=nearpick+farpick
            rng.shuffle(nodes)
        else:
            pool=corridor if strategy=='corridor' else all_nodes
            nodes=rng.choice(pool,customers+1,replace=False).tolist()
        depot=str(nodes[0]); cust_nodes=list(map(str,nodes[1:]))
        demands=rng.integers(2,8,size=customers).tolist()
        indices=np.arange(1,customers+1)
        rng.shuffle(indices)
        # Partition into EXACTLY K nonempty routes, and redraw demands if a bin overloads.
        seed_routes=[list(map(int,part)) for part in np.array_split(indices,vehicles)]
        for _ in range(1000):
            if all(sum(demands[c-1] for c in r)<=capacity for r in seed_routes):
                break
            demands=rng.integers(2,8,size=customers).tolist()
        else:
            raise ValueError('Could not create a capacity-feasible seed: increase vehicle capacity')
        scenarios=[{'scenario_id':day+'_'+departure_clock[:5].replace(':',''),
                    'departure_local':day+' '+departure_clock} for day in dates]
        for sc in scenarios: road.to_seconds(sc['departure_local'])
        task={'task_id':f'task_{i:04d}',
              'kind':'SYNTHETIC orders on observed Chicago roads',
              'depot_node':depot,
              'customers':[{'id':j+1,'node':cust_nodes[j],'demand':int(demands[j])} for j in range(customers)],
              'vehicles':int(vehicles), 'capacity':int(capacity),
              'service_seconds':int(service_seconds),
              'seed_routes':seed_routes,'require_all_vehicles':True,
              'scenarios':scenarios, 'seed':int(seed)+i,
              'sampling_strategy':strategy,
              'matched_corridor_radius_m':float(radius_m)}
        path=output_dir/(task['task_id']+'.json')
        path.write_text(json.dumps(task,ensure_ascii=False,indent=2),encoding='utf-8')
        tasks.append(task)
    manifest={'tasks':len(tasks),'customers_each':customers,'vehicles_each':vehicles,
              'scenarios_per_task':len(dates),'departure_clock':departure_clock,
              'seed':seed,'strategy':strategy,'radius_m':radius_m,
              'num_largest_scc_nodes':len(all_nodes),'num_corridor_candidate_nodes':len(corridor),
              'task_files':[t['task_id']+'.json' for t in tasks],
              'limits':'Orders are synthetic. Dynamic road costs are ex-post Chicago monitoring corridor observations + static estimates.'}
    (output_dir/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    return manifest
