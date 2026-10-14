#!/usr/bin/env python3
"""Synthetic-only integration tests: no Chicago traffic or deployment claims."""
from pathlib import Path
import json
import tempfile
import numpy as np
import pandas as pd
from road_eval import Road,ODCost
from models import Objective,verify
from solvers import METHODS,run_solver


def make_fixture(base):
    base=Path(base);d=base/'cost';d.mkdir(parents=True,exist_ok=True)
    n=6
    edges=[]
    for x in range(n):
        for y in range(n):
            for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                xx,yy=x+dx,y+dy
                if 0<=xx<n and 0<=yy<n: edges.append((f'{x}-{y}',f'{xx}-{yy}','0'))
    edges.sort()
    e=pd.DataFrame(edges,columns=['osm_u','osm_v','osm_key'])
    e.insert(0,'edge_idx',np.arange(len(e),dtype=int))
    e['static_travel_sec']=25.
    e['length_m']=100.
    e.to_csv(d/'edge_index.csv',index=False)
    times=pd.date_range('2025-06-05 06:00:00',periods=200,freq='10min')
    pd.DataFrame({'time_local':times.strftime('%Y-%m-%d %H:%M:%S')}).to_csv(d/'time_index.csv',index=False)
    cost=np.full((len(times),len(e)),25.,dtype=np.float32)
    observed=np.zeros(cost.shape,dtype=bool)
    mapped=np.zeros(len(e),dtype=bool)
    for i in range(len(e)):
        if i%3==0:
            mapped[i]=True
            for t in range(12,160):
                observed[t,i]=True
                cost[t,i]=45. if (t//7)%2 else 33.
    # This construction is FIFO at 10-minute knots.
    arr=np.maximum.accumulate(np.arange(len(times))[:,None]*600.+cost,axis=0)
    fifo=(arr-np.arange(len(times))[:,None]*600.).astype(np.float32)
    np.savez_compressed(d/'edge_costs.npz',travel_sec_fifo=fifo,travel_sec_raw=cost,
                        chicago_observed_mask=observed,is_mapped_edge=mapped,slot_seconds=600.,
                        slot_local_naive_ns=times.asi8)
    nodes=sorted(set(e.osm_u)|set(e.osm_v))
    selected=nodes[:13]
    task={'task_id':'task_0000','depot_node':selected[0],
          'customers':[{'id':i,'node':selected[i],'demand':3} for i in range(1,13)],
          'vehicles':3,'capacity':15,'service_seconds':30,
          'require_all_vehicles':True,
          'seed_routes':[[1,2,3,4],[5,6,7,8],[9,10,11,12]],
          'scenarios':[{'scenario_id':'2025-06-05_0800','departure_local':'2025-06-05 08:00:00'},
                       {'scenario_id':'2025-06-06_0800','departure_local':'2025-06-06 08:00:00'}]}
    td=base/'tasks';td.mkdir(parents=True,exist_ok=True)
    (td/'task_0000.json').write_text(json.dumps(task,indent=2))
    return d,td,task


def unit():
    with tempfile.TemporaryDirectory() as tmp:
        d,td,task=make_fixture(tmp)
        road=Road(d)
        for sc in task['scenarios']:
            od=ODCost(road,task,sc,step_minutes=20,hours=3)
            for method in METHODS:
                for mode in ('static','oracle'):
                    obj=Objective(task,od,mode)
                    result=run_solver(method,task,obj,max_iterations=50,wall_sec=1,seed=3)
                    verify(task,result['routes'])
                    ex=road.exact(task,result['routes'],sc)
                    assert ex['num_vehicles_used']==3
                    assert ex['ex_post_road_travel_sec']>0
                    assert sum(ex['vehicle_loads'])==36
                    assert result['surrogate_travel_sec'] <= result['initial_surrogate_travel_sec'] + 1e-5
            print('Synthetic methods passed:',sc['scenario_id'],len(METHODS),'methods * 2 planners')
        t=dict(task,require_all_vehicles=False)
        for method in ('greedy','sa','alns'):
            obj=Objective(t,od,'oracle')
            res=run_solver(method,t,obj,max_iterations=120,wall_sec=1,seed=7)
            verify(t,res['routes'])
        print('AT-MOST-K feasibility passed')

if __name__=='__main__':unit()
