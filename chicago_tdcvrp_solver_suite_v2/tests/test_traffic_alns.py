"""Synthetic-only unit and benchmark IO tests for search & traffic operators.

If official ALNS is absent, a declared TEST DOUBLE is used. This only tests the
adapter protocol and feasibility, NOT the official library's performance.
"""
from __future__ import annotations
import copy
import json
import sys
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

from tests.test_synthetic import make_fixture
from tests.test_upstream_bridge import install_test_double
from road_eval import Road,ODCost
from models import Objective,verify
from upstream_alns_adapter import VRPState, CombinedStop, canonical
from traffic_alns_adapter import (TRAFFIC_METHODS, TrafficFeatures, EnhancedOperators,
                                  get_config,run_traffic_solver)


def main():
    for name in TRAFFIC_METHODS:
        cfg=get_config(name,'oracle')
        assert cfg['destroy'] and cfg['repair']
    assert get_config('upstream_plus','oracle')['signal']=='static'
    assert get_config('upstream_traffic','oracle')['signal']=='traffic'
    assert not get_config('upstream_traffic_no_t_related','oracle')['traffic_related_enabled']
    assert not get_config('upstream_traffic_no_t_worst','oracle')['traffic_worst_enabled']
    assert 'regret3' not in get_config('upstream_traffic_no_regret3','oracle')['repair']
    assert not get_config('upstream_traffic_no_local','oracle')['local_search']
    assert not get_config('upstream_traffic_uniform','oracle')['adaptive_weights']
    assert get_config('upstream_traffic','static')['signal']=='static'
    rng=np.random.default_rng(13)
    with tempfile.TemporaryDirectory() as td:
        d,taskdir,task=make_fixture(td)
        road=Road(d);sc=task['scenarios'][0]
        od=ODCost(road,task,sc,step_minutes=20,hours=3)
        dynamic_obj=Objective(task,od,'oracle')
        static_obj=Objective(task,od,'static')
        original=canonical(task['seed_routes'])
        full=VRPState(original,dynamic_obj)
        traffic=TrafficFeatures(dynamic_obj,True)
        static=TrafficFeatures(dynamic_obj,False)
        assert traffic.dynamic_enabled and not static.dynamic_enabled
        assert np.isfinite(traffic.node_delay).all()
        assert np.isfinite(traffic.node_variability).all()
        assert np.max(np.abs(traffic.dynamic_symmetric-static.static_symmetric))>1e-3, 'Synthetic traffic signal must differ from static'
        assert traffic.static_symmetric.shape==(13,13)
        # Static mode MUST be invariant to changes in the historical future.
        base=TrafficFeatures(static_obj,True)
        old=od.dynamic.copy();od.dynamic[:]=od.dynamic*1.30
        changed=TrafficFeatures(static_obj,True)
        assert np.array_equal(base.dynamic_symmetric,changed.dynamic_symmetric)
        assert np.array_equal(base.node_delay,changed.node_delay)
        od.dynamic[:]=old
        # Validate each destroy/repair pair, every time respecting exactly-K.
        for dname in get_config('upstream_traffic','oracle')['destroy']:
            for repair in ['greedy','regret2','regret3']:
                ops=EnhancedOperators(dynamic_obj,CombinedStop(100,0),get_config('upstream_traffic','oracle'))
                partial=ops.destroy(dname,full,rng)
                assert full.routes==original
                all_customers=[c for route in partial.routes for c in route]+list(partial.removed)
                assert sorted(all_customers)==list(range(1,13)),(dname,all_customers)
                outcome=ops.repair(repair,partial,rng)
                verify(task,outcome.routes)
                assert np.isfinite(outcome.objective())
        try:
            import alns
        except ImportError:
            install_test_double()
            print('NOTE: using declared ALNS API TEST DOUBLE; official runtime untested')
        else:
            print('Official ALNS module:',getattr(alns,'__version__','unknown'))
        for mode in ['oracle','static']:
            for name in TRAFFIC_METHODS:
                obj=Objective(task,od,mode)
                sol=run_traffic_solver(name,task,obj,seed=42,max_iterations=15,wall_sec=3.)
                verify(task,sol['routes'])
                assert sol['surrogate_travel_sec'] <= sol['initial_surrogate_travel_sec']+1e-5
                assert sol['traffic_diagnostics']['dynamic_signal_enabled']==(mode=='oracle' and get_config(name,mode)['signal']=='traffic')
                ex=road.exact(task,sol['routes'],sc)
                assert np.isfinite(ex['ex_post_road_travel_sec'])
        # End-to-end benchmark compatible with earlier v2 results.
        import benchmark_traffic_alns as patched
        out=Path(td)/'benchmark'
        args=['benchmark_traffic_alns.py', '--cost-dir',str(d),'--task-dir',str(taskdir),
              '--output-dir',str(out),'--od-cache-dir',str(Path(td)/'cache'),
              '--methods','upstream_alns,upstream_plus,upstream_traffic,upstream_traffic_no_t_worst',
              '--planners','static,oracle','--seeds','42','--max-tasks','1','--max-scenarios','1',
              '--od-hours','3','--max-iterations','12','--wall-sec','3']
        orig=sys.argv;sys.argv=args
        try:
            patched.base.main()
        finally:sys.argv=orig
        df=pd.read_csv(out/'metrics.csv')
        assert len(df)==8 and df.ex_post_road_travel_sec.notna().all()
        assert df.duplicated(['task_id','scenario_id','seed','solver','planner']).sum()==0
        solutions=json.loads((out/'solutions.json').read_text())
        assert len(solutions)==8 and all(r['route_details']==[] for r in solutions)
        import summarize_traffic_operators as diagnostics
        orig=sys.argv
        sys.argv=['summarize_traffic_operators.py','--solutions',str(out/'solutions.json'),
                  '--output-dir',str(out/'diagnostics')]
        try:diagnostics.main()
        finally:sys.argv=orig
        assert (out/'diagnostics'/'operator_run_summary.csv').exists()
        print('All enhanced/traffic operator and v2-benchmark compatibility tests PASSED')

if __name__=='__main__':main()
