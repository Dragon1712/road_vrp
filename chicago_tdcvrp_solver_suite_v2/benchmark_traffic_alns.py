#!/usr/bin/env python3
"""Run N-Wouda/ALNS traffic ablations through unchanged v2 benchmark.

This is a new entrypoint, so old v2 scripts/results are not overwritten.
"""
import benchmark_official_alns as patched
from traffic_alns_adapter import TRAFFIC_METHODS,run_traffic_solver

base=patched.base
_previous=base.run_solver
base.METHODS=list(dict.fromkeys(list(base.METHODS)+list(TRAFFIC_METHODS)))


def dispatch(method,task,obj,seed=42,max_iterations=1000,wall_sec=3.):
    if method in TRAFFIC_METHODS:
        return run_traffic_solver(method,task,obj,seed=seed,
                                  max_iterations=max_iterations,wall_sec=wall_sec)
    return _previous(method,task,obj,seed=seed,
                     max_iterations=max_iterations,wall_sec=wall_sec)

base.run_solver=dispatch

if __name__=='__main__':base.main()
