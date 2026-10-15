"""Contract/unit tests: synthetic Chicago fixture; no claim of real-ALNS installation.

When official `alns` is absent, test with a tiny API-compatible stand-in to
exercise our custom operators + benchmark IO; this IS NOT upstream algorithm QA.
"""
from __future__ import annotations
import json
import sys
import types
import tempfile
from pathlib import Path
import time
import numpy as np
from tests.test_synthetic import make_fixture
from road_eval import Road,ODCost
from models import Objective,verify
from upstream_alns_adapter import (ChicagoOperators, VRPState, CombinedStop,
                                   config_for,run_upstream_solver,UPSTREAM_METHODS,canonical)


def install_test_double():
    # Only for testing our original adapter logic when offline environment has no PyPI.
    root=types.ModuleType('alns');root.__version__='TEST_DOUBLE_NOT_OFFICIAL'
    select=types.ModuleType('alns.select');accept=types.ModuleType('alns.accept')
    class RouletteWheel:
        def __init__(self,scores,decay,num_destroy,num_repair):
            assert len(scores)==4 and 0<=decay<=1
    class RandomSelect:
        def __init__(self,num_destroy,num_repair):pass
    class SimulatedAnnealing:
        @classmethod
        def autofit(cls,init_obj,worse,accept_prob,num_iters):
            assert init_obj>0 and 0<worse<1 and 0<accept_prob<1 and num_iters>0
            return cls()
    class ALNS:
        def __init__(self,rng):self.rng=rng;self.d=[];self.r=[]
        def add_destroy_operator(self,f,name=None):self.d.append((name,f))
        def add_repair_operator(self,f,name=None):self.r.append((name,f))
        def iterate(self,initial_solution,op_select,accept,stop):
            best=initial_solution;current=initial_solution;count=0
            while not stop(self.rng,best,current):
                partial=self.d[count%len(self.d)][1](current,self.rng)
                candidate=self.r[count%len(self.r)][1](partial,self.rng)
                assert len(candidate.removed)==0
                if candidate.objective()<best.objective():best=candidate
                current=candidate
                count+=1
            return types.SimpleNamespace(best_state=best)
    root.ALNS=ALNS;select.RouletteWheel=RouletteWheel
    select.RandomSelect=RandomSelect;accept.SimulatedAnnealing=SimulatedAnnealing
    sys.modules.update({'alns':root,'alns.select':select,'alns.accept':accept})


def run():
    for m in UPSTREAM_METHODS:
        cfg=config_for(m)
        assert len(cfg['destroy'])>=2 and cfg['repair']
    assert 'related' not in config_for('upstream_no_related')['destroy']
    assert 'worst' not in config_for('upstream_no_worst')['destroy']
    assert config_for('upstream_greedy_repair')['repair']==['greedy']
    assert not config_for('upstream_uniform')['adaptive_weights']
    assert not config_for('upstream_no_local')['local_search']
    stop=CombinedStop(7,0)
    for i in range(7):assert not stop(None,None,None)
    assert stop(None,None,None) and stop.calls==7 and stop.reason=='iteration_budget'
    with tempfile.TemporaryDirectory() as td:
        costdir,taskdir,task=make_fixture(td)
        road=Road(costdir)
        od=ODCost(road,task,task['scenarios'][0],step_minutes=20,hours=3)
        obj=Objective(task,od,'oracle')
        # Immutability and feasibility of each destroy/repair operator pair.
        initial=canonical(task['seed_routes'])
        base=VRPState(initial,obj)
        ops=ChicagoOperators(obj,CombinedStop(50,0),False)
        rng=np.random.default_rng(44)
        for dname in ('random','worst','related'):
            for rname in ('greedy','regret2'):
                destroyed=ops.destroy(dname,base,rng)
                assert base.routes==initial and base.removed==()
                built=ops.repair(rname,destroyed,rng)
                verify(task,built.routes)
                assert sorted([c for r in built.routes for c in r])==list(range(1,13))
                assert np.isfinite(built.objective())
        try:
            import alns
        except ImportError:
            install_test_double()
            print('OFFICIAL alns unavailable: using API-compatible TEST DOUBLE; not a real library validation')
        else:
            print('Official alns module found:',getattr(alns,'__version__','unknown'))
        for mode in ['static','oracle']:
            for name in UPSTREAM_METHODS:
                objective=Objective(task,od,mode)
                result=run_upstream_solver(name,task,objective,seed=1,
                                           max_iterations=14,wall_sec=2.)
                verify(task,result['routes'])
                assert result['surrogate_travel_sec']<=result['initial_surrogate_travel_sec']+1e-6
                assert result['iterations']==14
                exact=road.exact(task,result['routes'],task['scenarios'][0])
                assert np.isfinite(exact['ex_post_road_travel_sec'])
                print(name,mode,round(exact['ex_post_road_travel_sec'],2))
    print('Adapter contract tests PASSED (synthetic data only)')


def benchmark_io_test():
    # End-to-end integration with ORIGINAL benchmark_suite; uses synthetic costs.
    import pandas as pd
    import benchmark_official_alns as patched
    with tempfile.TemporaryDirectory() as td:
        costdir,taskdir,task=make_fixture(td)
        out=Path(td)/'benchmark'
        old=list(sys.argv)
        sys.argv=['benchmark_official_alns.py',
                  '--cost-dir',str(costdir),'--task-dir',str(taskdir),
                  '--output-dir',str(out),'--od-cache-dir',str(Path(td)/'cache'),
                  '--methods','alns,upstream_alns,upstream_no_related',
                  '--planners','static,oracle','--seeds','42',
                  '--max-tasks','1','--max-scenarios','1',
                  '--od-hours','3','--max-iterations','12','--wall-sec','2']
        try:
            patched.base.main()
        finally:
            sys.argv=old
        df=pd.read_csv(out/'metrics.csv')
        assert len(df)==6 and set(df.solver)=={'alns','upstream_alns','upstream_no_related'}
        assert set(df.planner)=={'static','oracle'}
        assert df.ex_post_road_travel_sec.notna().all()
        solutions=json.loads((out/'solutions.json').read_text())
        assert len(solutions)==len(df)
        print('End-to-end benchmark format PASSED:',len(df),'synthetic records')


if __name__=='__main__':
    run()
    benchmark_io_test()


