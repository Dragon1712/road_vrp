"""CLI for reproducible, fully offline Chicago TD-CVRP pilot."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from .road import RoadCosts
from .od import ODSurface, precompute
from .tasks import generate
from .solver import solve, evaluate_exact


def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=lambda x: x.item() if isinstance(x,np.generic) else str(x)),encoding='utf-8')


def task_files(task_dir):
    return sorted(Path(task_dir).glob('task_*.json'))


def get_surface(road, task, scenario, cache_dir, hours, step, rebuild=False):
    """Cached OD sample surfaces tied to road dataset version and task nodes."""
    node_ids=[str(task['depot_node'])]+[str(c['node']) for c in task['customers']]
    files=[road.folder/'edge_costs.npz',road.folder/'edge_index.csv',road.folder/'time_index.csv']
    fingerprint=[(str(f),f.stat().st_size,f.stat().st_mtime_ns) for f in files]
    metadata={'task_id':task['task_id'],'nodes':node_ids,
              'departure':scenario['departure_local'],'hours':hours,'step':step,
              'input_fingerprint':fingerprint}
    key=hashlib.sha256(json.dumps(metadata,sort_keys=True).encode()).hexdigest()[:20]
    cache_dir=Path(cache_dir);cache_dir.mkdir(parents=True,exist_ok=True)
    path=cache_dir/f'{task["task_id"]}_{scenario["scenario_id"]}_{key}.npz'
    if path.exists() and not rebuild:
        with np.load(path,allow_pickle=False) as z:
            return ODSurface(node_ids,z['knots'],z['dynamic'],z['static']),True
    surface=precompute(road,node_ids,scenario['departure_local'],hours,step)
    np.savez_compressed(path,knots=surface.knots,dynamic=surface.dynamic,static=surface.static)
    return surface,False


def do_generate(args):
    road=RoadCosts(args.cost_dir)
    dates=[x.strip() for x in args.dates.split(',') if x.strip()]
    manifest=generate(road,args.output_dir,num_tasks=args.num_tasks,customers=args.customers,
                      vehicles=args.vehicles,capacity=args.capacity,
                      service_seconds=args.service_seconds,seed=args.seed,
                      radius_m=args.radius_m,strategy=args.strategy,dates=dates,
                      departure_clock=args.departure_clock)
    print(json.dumps(manifest,ensure_ascii=False,indent=2))


def do_solve(args):
    road=RoadCosts(args.cost_dir)
    files=task_files(args.task_dir)
    if not files: raise ValueError('No task_*.json in '+str(args.task_dir))
    if args.max_tasks: files=files[:args.max_tasks]
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    records=[];brief=[]
    for fi,path in enumerate(files):
        task=json.loads(path.read_text(encoding='utf-8'))
        scenarios=task['scenarios'][:args.max_scenarios] if args.max_scenarios else task['scenarios']
        print(f'\nTASK {task["task_id"]}: {len(scenarios)} traffic scenarios',flush=True)
        for si,sc in enumerate(scenarios):
            surf,cached=get_surface(road,task,sc,args.cache_dir,args.od_hours,args.od_step_minutes,args.rebuild_cache)
            print('  scenario',sc['scenario_id'],'OD cache', 'hit' if cached else 'created',flush=True)
            depart=road.to_seconds(sc['departure_local'])
            rows=[]
            for mode in ('static','oracle'):
                # Static solution same seed, task and demand across scenarios; only evaluation conditions change.
                run=solve(task,surf,depart,mode,iterations=args.iterations,
                          restarts=args.restarts,seed=args.seed+fi*10007)
                exact=evaluate_exact(road,task,run['routes'],sc['departure_local'])
                metrics={k:v for k,v in exact.items() if k!='route_details'}
                rec={'task_id':task['task_id'],'scenario_id':sc['scenario_id'],
                     'departure_local':sc['departure_local'],
                     'planner':mode,'routes':run['routes'],'loads':run['loads'],
                     'iterations':run['iterations'],'search_seed':run['seed'],
                     'od_step_minutes':args.od_step_minutes,'od_hours':args.od_hours,
                     'search_objective_travel_sec':run['planned_travel_sec'],
                     'search_wall_sec':run['search_wall_sec'],
                     'search_history':run['history'],
                     'cache_info':run['objective_route_cache'],
                     'road_exact_ex_post':metrics,
                     'route_details':exact['route_details']}
                records.append(rec)
                row={'task_id':task['task_id'],'scenario_id':sc['scenario_id'],
                     'planner':mode,'departure_local':sc['departure_local'],
                     'search_objective_travel_sec':round(run['planned_travel_sec'],4),
                     'ex_post_road_travel_sec':round(exact['realized_travel_sec'],4),
                     'ex_post_total_vehicle_elapsed_sec':round(exact['realized_total_elapsed_vehicle_sec'],4),
                     'observed_edge_share':round(exact['observed_entry_share'],6),
                     'search_wall_sec':round(run['search_wall_sec'],4),
                     'iterations':args.iterations,
                     'num_vehicles_used':sum(bool(r) for r in run['routes']),
                     'vehicle_loads':','.join(map(str,run['loads']))}
                rows.append(row);brief.append(row)
                print(f'    {mode}: planned={run["planned_travel_sec"]:.1f}s; road-ex-post={exact["realized_travel_sec"]:.1f}s; observed traversals={exact["observed_entry_share"]:.1%}',flush=True)
            static=next(r for r in rows if r['planner']=='static')
            oracle=next(r for r in rows if r['planner']=='oracle')
            gain=static['ex_post_road_travel_sec']-oracle['ex_post_road_travel_sec']
            print(f'    same-scenario ex-post static-sequence minus oracle-sequence: {gain:.2f}s',flush=True)
            pd.DataFrame(brief).to_csv(out/'metrics.csv',index=False)
            write_json(out/'solutions.json',records)
    cfg={'cost_dir':str(args.cost_dir),'task_dir':str(args.task_dir),
         'evaluated_tasks':len(files),'records':len(records),
         'parameters':vars(args),
         'interpretation':'oracle sees historical future traffic; static sequence evaluated with EX-POST TD shortest road legs, so comparison isolates multi-vehicle assignment+visit order. Neither is a deployable prediction-based policy.',
         'limitations':'Search uses OD grid interpolation; final cost recomputed with exact road graph TDSP. No TW; all vehicles required and capacities enforced.'}
    write_json(out/'run_summary.json',cfg)
    print('\nSAVED',out/'metrics.csv',out/'solutions.json',flush=True)


def do_regress(args):
    road=RoadCosts(args.cost_dir)
    checks=[]
    for reference,depart in [(args.morning,'2025-06-05 08:00:00'),
                             (args.evening,'2025-06-05 17:00:00')]:
        df=pd.read_csv(reference,dtype={'origin':str,'destination':str})
        if len(df)!=100: raise ValueError(f'Expected 100 OD reference rows, got {len(df)} from {reference}')
        if not df[['origin','destination']].notna().all().all():raise ValueError('Missing OD nodes')
        at=road.to_seconds(depart)
        for r in df.itertuples(index=False):
            sp,static_path=road.shortest(r.origin,r.destination,at,'static',True)
            oracle,dyn_path=road.shortest(r.origin,r.destination,at,'dynamic',True)
            realized=road.evaluate_fixed_edge_path(static_path,at)
            measured=road.measure_path(dyn_path,at)
            vals={
                'static_planned_sec':sp,
                'static_route_realized_sec':realized,
                'oracle_td_route_sec':oracle,
                'oracle_improvement_sec':max(0.,realized-oracle),
                'oracle_td_edges':len(dyn_path),
                'oracle_td_observed_edges_at_entry_slot':measured}
            deltas={k:float(vals[k])-float(getattr(r,k)) for k in vals}
            secs={k:abs(deltas[k]) for k in list(vals)[:4]}
            count_keys=['oracle_td_edges','oracle_td_observed_edges_at_entry_slot']
            ok_time=max(secs.values())<=args.tolerance_sec
            ok_count=all(abs(deltas[k])<1e-6 for k in count_keys)
            checks.append({'reference':Path(reference).name,'departure':depart,
                           'origin':r.origin,'destination':r.destination,
                           'time_pass':bool(ok_time),'path_tie_counts_match':bool(ok_count),
                           'max_time_error_sec':max(secs.values()),
                           'edge_count_delta':deltas['oracle_td_edges'],
                           'measured_count_delta':deltas['oracle_td_observed_edges_at_entry_slot']})
    checks_df=pd.DataFrame(checks)
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    checks_df.to_csv(out/'tdsp_regression_details.csv',index=False)
    report={'pairs_checked':len(checks),'time_cost_passed':int(checks_df.time_pass.sum()),
            'path_traversal_count_consistent':int(checks_df.path_tie_counts_match.sum()),
            'max_absolute_time_error_sec':float(checks_df.max_time_error_sec.max()),
            'tolerance_sec':args.tolerance_sec,
            'time_cost_all_pass':bool(checks_df.time_pass.all()),
            'route_edge_counts_all_match':bool(checks_df.path_tie_counts_match.all()),
            'meaning':'100 morning +100 evening rows, same 100 OD pairs at each date/time. Time regression is authoritative; path tie counts are diagnostic.'}
    write_json(out/'tdsp_regression_report.json',report)
    print(json.dumps(report,indent=2))
    if not report['time_cost_all_pass']: raise SystemExit('FAILED TDSP time regression; check edge_index/cost files and interpolator')


def do_regret(args):
    """Task-conditioned cross traffic-scenario evaluation of approximate teacher solutions."""
    road=RoadCosts(args.cost_dir)
    sols=json.loads(Path(args.solutions).read_text(encoding='utf-8'))
    bytask={}
    for r in sols:
        if r['planner']=='oracle': bytask.setdefault(r['task_id'],[]).append(r)
    long=[];pairs=[]
    for tid,entries in bytask.items():
        task=json.loads((Path(args.task_dir)/(tid+'.json')).read_text(encoding='utf-8'))
        # Same wall-clock time is REQUIRED for a fair fixed-task regret comparison.
        clocks={r['departure_local'][11:] for r in entries}
        if len(clocks)!=1:raise ValueError('Mixed departure times in regret comparison, separate by clock')
        cost={}
        for i,sol in enumerate(entries):
            for j,sc in enumerate(entries):
                ev=evaluate_exact(road,task,sol['routes'],sc['departure_local'])
                cost[(i,j)]=float(ev['realized_travel_sec'])
                long.append({'task_id':tid,'solution_scenario':sol['scenario_id'],
                             'evaluation_scenario':sc['scenario_id'],
                             'ex_post_travel_sec':cost[(i,j)]})
        for i in range(len(entries)):
            for j in range(i+1,len(entries)):
                dij=cost[i,j]-cost[j,j]
                dji=cost[j,i]-cost[i,i]
                r_ij=max(0.,dij)/max(abs(cost[j,j]),1e-6)
                r_ji=max(0.,dji)/max(abs(cost[i,i]),1e-6)
                pairs.append({'task_id':tid,'scenario_a':entries[i]['scenario_id'],
                              'scenario_b':entries[j]['scenario_id'],
                              'raw_delta_a_route_in_b_sec':dij,
                              'raw_delta_b_route_in_a_sec':dji,
                              'regret_a_to_b':r_ij,'regret_b_to_a':r_ji,
                              'symmetric_regret':(r_ij+r_ji)/2,
                              'negative_teacher_difference':bool(dij< -1e-5 or dji< -1e-5)})
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(long).to_csv(out/'cross_costs.csv',index=False)
    pd.DataFrame(pairs).to_csv(out/'regret_pairs.csv',index=False)
    summary={'tasks':len(bytask),'total_cross_evaluations':len(long),
             'pairs':len(pairs),'nonzero_symmetric_regret_pairs':sum(p['symmetric_regret']>1e-6 for p in pairs),
             'negative_teacher_difference_pairs':sum(p['negative_teacher_difference'] for p in pairs),
             'warning':'References are approximate fixed-budget heuristics, not exact optima. Clipped negative differences reflect teacher noise; regrets are hindsight costs on mixed observed/fallback road model.'}
    write_json(out/'regret_summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))



def do_od_audit(args):
    """Compare sampled-OD arrival interpolation with exact road TDSP off-grid."""
    road=RoadCosts(args.cost_dir)
    files=task_files(args.task_dir)
    if not files: raise ValueError('Missing tasks')
    task=json.loads(files[0].read_text(encoding='utf-8'))
    sc=task['scenarios'][0]
    surf,cached=get_surface(road,task,sc,args.cache_dir,args.od_hours,args.od_step_minutes)
    rng=np.random.default_rng(args.seed)
    depart=road.to_seconds(sc['departure_local'])
    records=[]
    for _ in range(args.samples):
        a,b=map(int,rng.choice(len(surf.nodes),size=2,replace=False))
        # Sample interior region and deliberately off-grid instants.
        t=float(depart+rng.uniform(1.,min(args.audit_hours*3600.,surf.knots[-1]-depart-100.)))
        approx=surf.travel(a,b,t,'oracle')
        exact,_=road.shortest(surf.nodes[a],surf.nodes[b],t,mode='dynamic',path=False)
        records.append({'origin':surf.nodes[a],'target':surf.nodes[b],
                        'departure_rel_sec':t,'approx_sec':approx,'exact_sec':exact,
                        'absolute_error_sec':abs(approx-exact),
                        'relative_error':abs(approx-exact)/max(exact,1e-6)})
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(records).to_csv(out/'od_approx_samples.csv',index=False)
    df=pd.DataFrame(records)
    report={'sample_count':len(df),'cache_hit':cached,
            'mean_abs_error_sec':float(df.absolute_error_sec.mean()),
            'p95_abs_error_sec':float(df.absolute_error_sec.quantile(.95)),
            'max_abs_error_sec':float(df.absolute_error_sec.max()),
            'mean_relative_error':float(df.relative_error.mean()),
            'p95_relative_error':float(df.relative_error.quantile(.95)),
            'diagnostic':'Use smaller --od-step-minutes if errors are large; final metrics always use exact road-leg TDSP.'}
    write_json(out/'od_approx_summary.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))


def make_parser():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    g=sub.add_parser('generate',help='Create feasible 30-customer, five-vehicle tasks on SCC')
    g.add_argument('--cost-dir',type=Path,required=True)
    g.add_argument('--output-dir',type=Path,required=True)
    g.add_argument('--num-tasks',type=int,default=10)
    g.add_argument('--customers',type=int,default=30)
    g.add_argument('--vehicles',type=int,default=5)
    g.add_argument('--capacity',type=int,default=45)
    g.add_argument('--service-seconds',type=int,default=120)
    g.add_argument('--seed',type=int,default=2026)
    g.add_argument('--radius-m',type=float,default=600)
    g.add_argument('--strategy',choices=['mixed','corridor','uniform'],default='mixed')
    g.add_argument('--dates',default='2025-06-05,2025-06-06,2025-06-03,2025-06-04,2025-06-07,2025-06-08')
    g.add_argument('--departure-clock',default='08:00:00')
    g.set_defaults(action=do_generate)
    s=sub.add_parser('solve',help='Fixed-budget static vs hindsight TD-VRP30, detailed exact ex-post replay')
    s.add_argument('--cost-dir',type=Path,required=True)
    s.add_argument('--task-dir',type=Path,required=True)
    s.add_argument('--output-dir',type=Path,required=True)
    s.add_argument('--cache-dir',type=Path,required=True)
    s.add_argument('--max-tasks',type=int,default=1)
    s.add_argument('--max-scenarios',type=int,default=2)
    s.add_argument('--od-hours',type=float,default=4.)
    s.add_argument('--od-step-minutes',type=float,default=20.)
    s.add_argument('--iterations',type=int,default=1500)
    s.add_argument('--restarts',type=int,default=2)
    s.add_argument('--seed',type=int,default=42)
    s.add_argument('--rebuild-cache',action='store_true')
    s.set_defaults(action=do_solve)
    r=sub.add_parser('regress',help='Re-evaluate 100 old morning and 100 evening OD records (time regression)')
    r.add_argument('--cost-dir',type=Path,required=True)
    r.add_argument('--morning',type=Path,required=True)
    r.add_argument('--evening',type=Path,required=True)
    r.add_argument('--output-dir',type=Path,required=True)
    r.add_argument('--tolerance-sec',type=float,default=0.01)
    r.set_defaults(action=do_regress)
    a=sub.add_parser('od-audit',help='Check sampled OD arrival interpolation against exact TDSP')
    a.add_argument('--cost-dir',type=Path,required=True)
    a.add_argument('--task-dir',type=Path,required=True)
    a.add_argument('--cache-dir',type=Path,required=True)
    a.add_argument('--output-dir',type=Path,required=True)
    a.add_argument('--od-hours',type=float,default=4.)
    a.add_argument('--od-step-minutes',type=float,default=20.)
    a.add_argument('--audit-hours',type=float,default=2.)
    a.add_argument('--samples',type=int,default=100)
    a.add_argument('--seed',type=int,default=42)
    a.set_defaults(action=do_od_audit)
    q=sub.add_parser('regret',help='Cross-evaluate approx teacher visit sequences between historical traffic days')
    q.add_argument('--cost-dir',type=Path,required=True)
    q.add_argument('--task-dir',type=Path,required=True)
    q.add_argument('--solutions',type=Path,required=True)
    q.add_argument('--output-dir',type=Path,required=True)
    q.set_defaults(action=do_regret)
    return p


def main():
    args=make_parser().parse_args()
    args.action(args)

if __name__=='__main__':main()
