#!/usr/bin/env python3
"""Task-wise candidate pool with exact ex-post cross-scenario evaluation.

Pooled hindsight teacher is the best solution AMONG SEARCHED candidates, not an optimum.
Do NOT train/evaluate on same tasks/dates without proper splits.
"""
import argparse
import json
from pathlib import Path
from collections import defaultdict
import pandas as pd
from road_eval import Road
from models import verify


def canonical(routes):
    # identical vehicles; remove permutation of vehicle labels, not customer order
    return tuple(sorted(tuple(map(int,r)) for r in routes))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cost-dir',required=True)
    p.add_argument('--task-dir',required=True)
    p.add_argument('--solutions',required=True)
    p.add_argument('--output-dir',required=True)
    args=p.parse_args()
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    records=json.loads(Path(args.solutions).read_text(encoding='utf-8'))
    bytask=defaultdict(list)
    for r in records:bytask[r['task_id']].append(r)
    road=Road(args.cost_dir)
    teachers=[];costrows=[];regrets=[]
    for tid,items in sorted(bytask.items()):
        task=json.loads((Path(args.task_dir)/(tid+'.json')).read_text(encoding='utf-8'))
        for r in items:
            task['require_all_vehicles']=(r.get('vehicle_rule','exactly')=='exactly')
            verify(task,r['routes'])
        if len({r.get('vehicle_rule','exactly') for r in items})!=1:
            raise ValueError('Teacher candidates must share same vehicle constraints')
        candidates={canonical(r['routes']):r for r in items}
        scen={r['scenario_id']:r['departure_local'] for r in items}
        if len({x[11:] for x in scen.values()})!=1:
            raise ValueError('Cross-scenario comparison needs identical departure clock times')
        cached={}
        for sc,t in sorted(scen.items()):
            for i,(routes,record) in enumerate(candidates.items()):
                sec=road.exact(task,routes,{'departure_local':t})['ex_post_road_travel_sec']
                cached[(sc,routes)]=sec
                costrows.append({'task_id':tid,'scenario':sc,'candidate_id':i,'ex_post_sec':sec})
        ref={sc:min(candidates,key=lambda r:cached[(sc,r)]) for sc in scen}
        for sc,key in sorted(ref.items()):
            exemplar=candidates[key]
            teachers.append({'task_id':tid,'scenario_id':sc,'departure_local':scen[sc],
                             'routes':[list(r) for r in key], 'ex_post_sec':cached[(sc,key)],
                             'source_solver':exemplar['solver'],'source_seed':exemplar['seed'],
                             'label':'best_in_candidate_pool_hindsight'})
        scenarios=sorted(scen)
        for i,sc1 in enumerate(scenarios):
            for sc2 in scenarios[i+1:]:
                a,b=ref[sc1],ref[sc2]
                ab=cached[(sc2,a)]-cached[(sc2,b)]
                ba=cached[(sc1,b)]-cached[(sc1,a)]
                if min(ab,ba)<-1e-5:raise AssertionError('Pool teacher not best in common candidate pool')
                regret=(max(0,ab)/cached[(sc2,b)]+max(0,ba)/cached[(sc1,a)])/2
                regrets.append({'task_id':tid,'scenario_a':sc1,'scenario_b':sc2,
                                'same_decision':a==b,'regret_a_to_b':max(0,ab)/cached[(sc2,b)],
                                'regret_b_to_a':max(0,ba)/cached[(sc1,a)],
                                'symmetric_regret':regret})
        print(tid,'unique plans',len(candidates),'scenarios',len(scen),flush=True)
    pd.DataFrame(costrows).to_csv(out/'candidate_costs.csv',index=False)
    pd.DataFrame(regrets).to_csv(out/'regret_pairs.csv',index=False)
    (out/'teachers.json').write_text(json.dumps(teachers,indent=2,ensure_ascii=False),encoding='utf-8')
    print('SAVED',out,'teachers',len(teachers),'regret pairs',len(regrets))

if __name__=='__main__':main()
