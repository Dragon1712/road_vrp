"""End-to-end synthetic offline test. NEVER represents Chicago observations."""
from __future__ import annotations
from pathlib import Path
import json
import tempfile
import numpy as np
import pandas as pd
from tdcvrp.road import RoadCosts
from tdcvrp.od import precompute
from tdcvrp.tasks import generate
from tdcvrp.solver import solve, evaluate_exact, validate


def make_synthetic(folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    N=8
    e=[]
    for y in range(N):
        for x in range(N):
            for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                xx,yy=x+dx,y+dy
                if 0<=xx<N and 0<=yy<N:
                    e.append((f'{x}-{y}',f'{xx}-{yy}',str(0)))
    e.sort()
    edges=pd.DataFrame(e,columns=['osm_u','osm_v','osm_key'])
    edges.insert(0,'edge_idx',np.arange(len(edges)))
    edges['length_m']=100
    edges['static_travel_sec']=18.0
    edges['mapped_segment_id']=['mock-'+str(i) if i%3==0 else '' for i in range(len(edges))]
    edges.to_csv(folder/'edge_index.csv',index=False)
    times=pd.date_range('2025-06-05 06:00:00',periods=240,freq='10min')
    pd.DataFrame({'time_local':times.astype(str)}).to_csv(folder/'time_index.csv',index=False)
    arr=np.full((len(times),len(edges)),18.0,dtype=np.float32)
    mask=np.zeros(arr.shape,dtype=bool)
    mapped=np.zeros(len(edges),dtype=bool)
    for j in range(len(edges)):
        if j%3==0:
            mapped[j]=True
            for t in range(len(times)):
                mask[t,j]=True
                arr[t,j]=32 if 12<=t<=23 else (25 if 36<=t<=48 else 18)
    # FIFO sampled arrival must increase; congestion drop to 18 could violate FIFO at a boundary.
    a=np.maximum.accumulate(arr.astype(float)+np.arange(len(times))[:,None]*600,axis=0)
    fifo=(a-np.arange(len(times))[:,None]*600).astype(np.float32)
    np.savez_compressed(folder/'edge_costs.npz',travel_sec_raw=arr,travel_sec_fifo=fifo,
                        chicago_observed_mask=mask,is_mapped_edge=mapped,
                        slot_local_naive_ns=times.asi8,slot_seconds=600.)
    return folder


def run():
    with tempfile.TemporaryDirectory() as tmp:
        folder=make_synthetic(Path(tmp)/'road')
        road=RoadCosts(folder)
        assert len(road.largest_scc)==64
        # FIFO sanity check at off-grid instants, each OSM edge.
        for j in range(len(road.edges)):
            t1=road.to_seconds('2025-06-05 08:02:00')
            t2=t1+100
            assert t1+road.cost(j,t1)<=t2+road.cost(j,t2)+1e-6
        tasks_dir=Path(tmp)/'tasks'
        report=generate(road,tasks_dir,num_tasks=1,customers=30,vehicles=5,
                        capacity=45,seed=3,strategy='corridor',radius_m=350,
                        dates=['2025-06-05'],departure_clock='08:00:00')
        task=json.loads((tasks_dir/'task_0000.json').read_text())
        validate(task,task['seed_routes'])
        ids=[task['depot_node']]+[c['node'] for c in task['customers']]
        surface=precompute(road,ids,'2025-06-05 08:00:00',horizon_hours=2,step_minutes=20,quiet=True)
        departure=road.to_seconds('2025-06-05 08:00:00')
        results=[]
        for mode in ['static','oracle']:
            res=solve(task,surface,departure,mode,iterations=200,seed=13,restarts=2)
            e=evaluate_exact(road,task,res['routes'],'2025-06-05 08:00:00')
            validate(task,res['routes'])
            assert e['realized_travel_sec']>0 and e['edge_traversals']>0
            assert e['observed_edge_entries']<=e['edge_traversals']
            assert all(x['load']<=45 for x in e['route_details'])
            results.append((mode,res['planned_travel_sec'],e['realized_travel_sec']))
        assert results[0][0]=='static' and results[1][0]=='oracle'
        print('SYNTHETIC TEST PASS',report['num_largest_scc_nodes'],results)

if __name__=='__main__':run()
