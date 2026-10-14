#!/usr/bin/env python3
"""Offline benchmark of multiple TD-CVRP heuristics and ALNS ablations.

Use only one task and scenario for smoke test. Historical traffic oracle is hindsight.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
import time
from pathlib import Path
import pandas as pd
from models import Objective,verify
from road_eval import Road,ODCost
from solvers import METHODS,run_solver


def dump(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')
    os.replace(temp,path)


def checkpoint(out,rows,solutions):
    # A partial crash between CSV and JSON is caught on resume via key comparison.
    tmp=out/'metrics.tmp.csv'
    pd.DataFrame(rows).to_csv(tmp,index=False)
    os.replace(tmp,out/'metrics.csv')
    dump(out/'solutions.json',solutions)


def result_key(record):
    return (str(record['task_id']),str(record['scenario_id']),str(record['solver']),
            str(record['planner']),int(record['seed']))



def load_od(args,road,task,sc):
    """Deterministic per-task/scenario OD cache, invalidated when files change."""
    import numpy as np
    node_ids=[str(task['depot_node'])]+[str(x['node']) for x in task['customers']]
    inputs=[]
    for name in ('edge_costs.npz','edge_index.csv','time_index.csv'):
        f=Path(args.cost_dir)/name
        st=f.stat()
        inputs.append((str(f.resolve()),st.st_size,st.st_mtime_ns))
    key_payload={'nodes':node_ids,'scenario':sc['departure_local'],
                 'hours':args.od_hours,'step':args.od_step_min,'input_files':inputs}
    key=hashlib.sha256(json.dumps(key_payload,sort_keys=True).encode()).hexdigest()[:20]
    cache_dir=Path(args.od_cache_dir) if args.od_cache_dir else Path(args.output_dir)/'od_cache'
    cache_dir.mkdir(parents=True,exist_ok=True)
    path=cache_dir/(key+'.npz')
    if path.is_file() and not args.rebuild_od_cache:
        with np.load(path,allow_pickle=False) as z:
            od=object.__new__(ODCost)
            od.nodes=node_ids
            od.index={n:i for i,n in enumerate(node_ids)}
            od.start=road.to_sec(sc['departure_local'])
            od.step=args.od_step_min*60.
            od.knots=z['knots']
            od.static=z['static']
            od.dynamic=z['dynamic']
            od.preprocessing_wall_sec=0.
        if od.dynamic.shape!=(len(od.knots),len(node_ids),len(node_ids)):
            raise ValueError('Corrupted cached OD shape')
        return od,'cache hit'
    od=ODCost(road,task,sc,step_minutes=args.od_step_min,hours=args.od_hours,quiet=True)
    np.savez_compressed(path,knots=od.knots,static=od.static,dynamic=od.dynamic)
    return od,'computed+cached'


def run(args):
    road=Road(args.cost_dir)
    files=sorted(Path(args.task_dir).glob('task_*.json'))
    if not files:raise ValueError('Cannot find task_*.json files')
    files=files[:args.max_tasks] if args.max_tasks else files
    methods=[x.strip() for x in args.methods.split(',') if x.strip()]
    assert methods and set(methods).issubset(METHODS),f'Available: {METHODS}'
    modes=[x.strip() for x in args.planners.split(',') if x.strip()]
    assert modes and set(modes).issubset(('static','oracle'))
    seeds=[int(x) for x in args.seeds.split(',')]
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True)
    # Configuration is locked for resume; changing solver/budget/inputs requires a fresh directory.
    inputs=[]
    for name in ('edge_costs.npz','edge_index.csv','time_index.csv'):
        f=Path(args.cost_dir)/name
        st=f.stat();inputs.append((str(f.resolve()),st.st_size,st.st_mtime_ns))
    for f in files:
        st=f.stat();inputs.append((str(f.resolve()),st.st_size,st.st_mtime_ns))
    config={
        'methods':methods,'planners':modes,'seeds':seeds,
        'task_files':[str(f.resolve()) for f in files],
        'max_scenarios':args.max_scenarios,'vehicle_rule':args.vehicle_rule,
        'od_step_min':args.od_step_min,'od_hours':args.od_hours,
        'max_iterations':args.max_iterations,'wall_sec':args.wall_sec,
        'detailed_legs':args.detailed_legs,'input_files':inputs,
    }
    # Normalize tuples to JSON arrays for stable on-disk resume comparison.
    config=json.loads(json.dumps(config,sort_keys=True))
    confpath=output/'run_config.json'
    metrics_path=output/'metrics.csv'
    solutions_path=output/'solutions.json'
    if args.resume:
        if not confpath.exists():
            raise ValueError('--resume requires existing run_config.json')
        if json.loads(confpath.read_text(encoding='utf-8'))!=config:
            raise ValueError('Resume configuration or input files have changed; use a new output directory')
        if metrics_path.exists()!=solutions_path.exists():
            raise ValueError('Partial output exists: metrics.csv and solutions.json must both be present')
        results=pd.read_csv(metrics_path).to_dict('records') if metrics_path.exists() else []
        all_solutions=json.loads(solutions_path.read_text(encoding='utf-8')) if solutions_path.exists() else []
        if len(results)!=len(all_solutions) or [result_key(x) for x in results]!=[result_key(x) for x in all_solutions]:
            raise ValueError('Resume metrics/solutions mismatch; reconcile or restart in new folder')
    else:
        if confpath.exists() or metrics_path.exists() or solutions_path.exists():
            raise ValueError('Results already exist; use --resume or a NEW output directory')
        dump(confpath,config)
        results=[];all_solutions=[]
    completed={result_key(x) for x in results}
    if len(completed)!=len(results):
        raise ValueError('Duplicate completed keys in resumed results')
    print(f'RUN {len(files)} tasks; {len(methods)} methods; {len(modes)} modes; '
          f'{len(seeds)} seeds; {len(completed)} records already complete',flush=True)
    pending=0
    for fp in files:
        task=json.loads(fp.read_text(encoding='utf-8'))
        task['require_all_vehicles']=(args.vehicle_rule=='exactly')
        for sc in task['scenarios'][:args.max_scenarios if args.max_scenarios else None]:
            scenario_keys=[(task['task_id'],sc['scenario_id'],method,mode,seed)
                           for seed in seeds for method in methods for mode in modes]
            if all(k in completed for k in scenario_keys):
                print(f'{task["task_id"]} / {sc["scenario_id"]}: all records already complete; SKIP',flush=True)
                continue
            print(f'\n{task["task_id"]} / {sc["scenario_id"]}: precomputing/loading OD ({args.od_step_min}min)',flush=True)
            od,cache_status=load_od(args,road,task,sc)
            print(f'  OD {cache_status}; per-run preprocessing {od.preprocessing_wall_sec:.2f}s',flush=True)
            for seed in seeds:
                for method in methods:
                    for mode in modes:
                        key=(task['task_id'],sc['scenario_id'],method,mode,seed)
                        if key in completed:continue
                        obj=Objective(task,od,mode)
                        start=time.perf_counter()
                        solved=run_solver(method,task,obj,seed=seed,max_iterations=args.max_iterations,wall_sec=args.wall_sec)
                        solve_wall=time.perf_counter()-start
                        verify(task,solved['routes'])
                        ex=road.exact(task,solved['routes'],sc,collect_legs=args.detailed_legs)
                        entry={'task_id':task['task_id'],'scenario_id':sc['scenario_id'],
                               'departure_local':sc['departure_local'],
                               'solver':method,'planner':mode,'seed':seed,
                               'vehicle_rule':args.vehicle_rule,
                               'surrogate_travel_sec':solved['surrogate_travel_sec'],
                               'initial_surrogate_travel_sec':solved['initial_surrogate_travel_sec'],
                               'ex_post_road_travel_sec':ex['ex_post_road_travel_sec'],
                               'ex_post_total_vehicle_elapsed_sec':ex['ex_post_total_vehicle_elapsed_sec'],
                               'observed_edge_share':ex['observed_edge_share'],
                               'num_vehicles_used':ex['num_vehicles_used'],
                               'vehicle_loads':','.join(map(str,ex['vehicle_loads'])),
                               'construct_wall_sec':solved['construction_wall_sec'],
                               'search_wall_sec':solved['search_wall_sec'],
                               'total_solver_wall_sec':solve_wall,
                               'od_precompute_wall_sec':od.preprocessing_wall_sec,
                               'iterations':solved['iterations'],
                               'score_calls':solved['score_calls'],
                               'route_score_calls':solved['route_score_calls'],
                               'accepted_moves':solved['accepted_moves'],
                               'improvements':solved['best_improvements'],
                               'failed_repairs':solved['failed_repairs'],
                               'termination_reason':solved.get('termination_reason',''),
                               'local_optimum_verified':solved.get('local_optimum_verified',False),
                               'budget_unit':solved.get('budget_unit','')}
                        results.append(entry)
                        all_solutions.append({'task_id':task['task_id'],
                                              'scenario_id':sc['scenario_id'],
                                              'departure_local':sc['departure_local'],
                                              'solver':method,'planner':mode,'seed':seed,
                                              'vehicle_rule':args.vehicle_rule,
                                              'routes':solved['routes'],
                                              'loads':ex['vehicle_loads'],
                                              'recorded_cost_sec':ex['ex_post_road_travel_sec'],
                                              'observed_edge_share':ex['observed_edge_share'],
                                              'solver_diagnostics':{k:v for k,v in solved.items() if k not in ('routes','loads')},
                                              'route_details':ex['route_details'] if args.detailed_legs else []})
                        print(f'  {method:20s} {mode:6s} seed={seed} '
                              f'approx={entry["surrogate_travel_sec"]:.1f} '
                              f'ex-post={entry["ex_post_road_travel_sec"]:.1f}s '
                              f'wall={entry["total_solver_wall_sec"]:.2f}s',flush=True)
                        completed.add(key);pending+=1
                        if pending>=args.checkpoint_every:
                            checkpoint(output,results,all_solutions)
                            pending=0
    if pending:
        checkpoint(output,results,all_solutions)
    df=pd.DataFrame(results)
    paired=df.pivot_table(index=['task_id','scenario_id','solver','seed'],columns='planner',
                          values='ex_post_road_travel_sec',aggfunc='first')
    if {'oracle','static'}.issubset(paired.columns):
        gain=paired.dropna(subset=['oracle','static']).copy()
        gain['gain_pct']=100*(gain['static']-gain['oracle'])/gain['static']
        gain.reset_index().to_csv(output/'paired_gains.csv',index=False)
    summary={'records':len(df),'tasks':len(files),'methods':methods,'planners':modes,'seeds':seeds,
             'vehicle_rule':args.vehicle_rule,
             'budget_rule':'Each method same wall clock search cap and proposal cap; actual work and OD preprocessing reported separately. Greedy stops after construction.',
             'objective':'Total fleet ROAD TRAVEL time (excluding service, not minmax/makespan)',
             'ex_post':'Every solution evaluated under its scenario with EXACT road-edge time-dependent Dijkstra; the static planner ALSO gets time-dependent per-leg road rerouting.',
             'hindsight':'Oracle planner sees future historical traffic costs and is NOT a deployable traffic forecast.',
             'limit':'Limited Chicago observations + OSM static fallback. Stronger teacher still not mathematically optimal.',
             'config':{k:str(v) for k,v in vars(args).items()}}
    dump(output/'benchmark_summary.json',summary)
    print('SAVED',output/'metrics.csv',output/'solutions.json',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cost-dir',type=Path,required=True)
    p.add_argument('--task-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--methods',default='greedy,sa,vnd,alns')
    p.add_argument('--planners',default='static,oracle')
    p.add_argument('--seeds',default='42,43')
    p.add_argument('--max-tasks',type=int,default=1)
    p.add_argument('--max-scenarios',type=int,default=2)
    p.add_argument('--vehicle-rule',choices=['exactly','at-most'],default='exactly')
    p.add_argument('--od-step-min',type=int,default=20)
    p.add_argument('--od-hours',type=float,default=4)
    p.add_argument('--od-cache-dir',default='',help='Optional shared OD cache folder')
    p.add_argument('--rebuild-od-cache',action='store_true')
    p.add_argument('--max-iterations',type=int,default=1000)
    p.add_argument('--wall-sec',type=float,default=4.0)
    p.add_argument('--detailed-legs',action='store_true')
    p.add_argument('--resume',action='store_true',help='Resume same run/config after interruption')
    p.add_argument('--checkpoint-every',type=int,default=5,help='Persist every N completed solves')
    args=p.parse_args()
    if args.max_iterations<=0:raise ValueError('Positive iterations required')
    if args.wall_sec<0:raise ValueError('Negative wall budget')
    if args.checkpoint_every<1:raise ValueError('Positive checkpoint interval required')
    run(args)

if __name__=='__main__':main()
