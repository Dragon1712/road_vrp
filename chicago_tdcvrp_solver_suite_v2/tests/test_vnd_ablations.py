"""Test genuine VND local optimum and five mechanically distinct ALNS ablations."""
import math
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
from tests.test_synthetic import make_fixture
from road_eval import Road, ODCost
from models import Objective,verify
from solvers import VND_NEIGHBORHOODS,alns_config,iter_neighborhood,vnd_descent,run_solver


def main():
    with tempfile.TemporaryDirectory() as tmp:
        costs,tasks,original=make_fixture(tmp)
        sc=original['scenarios'][0]
        # Small 6-customer instance makes truly exhaustive local-neighbor checks possible.
        task=dict(original)
        task['customers']=[dict(x) for x in original['customers'][:6]]
        task['seed_routes']=[[1,2],[3,4],[5,6]]
        task['vehicles']=3
        task['capacity']=15
        road=Road(costs)
        od=ODCost(road,task,sc,step_minutes=20,hours=3)
        obj=Objective(task,od,'oracle')
        start=task['seed_routes']
        partial=vnd_descent(start,obj,max_evaluations=1)
        assert partial['termination_reason']=='evaluation_budget'
        assert not partial['local_optimum_verified']
        result=vnd_descent(start,obj,max_evaluations=100000)
        assert result['local_optimum_verified'],result
        best=result['cost']; sol=result['routes']
        assert best<=obj.score(start)+1e-7
        verify(task,sol)
        tested=0
        for neigh in VND_NEIGHBORHOODS:
            for cand in iter_neighborhood(sol,neigh,obj):
                assert obj.score(cand)>=best-1e-7, (neigh,sol,cand)
                tested+=1
        assert tested>40
        print('VND exhaustive local optimum verified: %d neighbors' % tested)

        configs={m:alns_config(m) for m in ['alns','alns_no_related','alns_no_worst',
                   'alns_greedy_repair','alns_uniform','alns_no_local']}
        assert 'related' in configs['alns']['destroy']
        assert 'related' not in configs['alns_no_related']['destroy']
        assert 'worst' not in configs['alns_no_worst']['destroy']
        assert configs['alns_greedy_repair']['repair']==['greedy']
        assert not configs['alns_uniform']['adaptive_weights']
        assert not configs['alns_no_local']['local_search']
        for method,conf in configs.items():
            res=run_solver(method,task,Objective(task,od,'oracle'),seed=42,
                           max_iterations=12,wall_sec=4.)
            verify(task,res['routes'])
            assert res['method_config']==conf
            assert set(res['operators_stats']).issubset(set(conf['destroy']+conf['repair']))
            if method=='alns_uniform':
                assert all(w==1.0 for w in res['final_operator_weights'].values())
        print('ALNS 5 ablation switch checks PASSED')


if __name__=='__main__':main()
