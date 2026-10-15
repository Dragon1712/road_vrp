#!/usr/bin/env python3
"""Invoke EXISTING benchmark_suite unchanged, adding N-Wouda/ALNS as a method.

Example:
 python chicago_tdcvrp_solver_suite_v2/benchmark_official_alns.py \
  --cost-dir ... --task-dir ... --methods alns,upstream_alns \
  --planners oracle --seeds 42,43,44 --output-dir ...
"""
from __future__ import annotations
import benchmark_suite as base
from upstream_alns_adapter import UPSTREAM_METHODS, run_upstream_solver

_old_solver=base.run_solver
base.METHODS=list(base.METHODS)+list(UPSTREAM_METHODS)


def _dispatch(method,task,obj,seed=42,max_iterations=1000,wall_sec=3.):
    if method in UPSTREAM_METHODS:
        return run_upstream_solver(method,task,obj,seed=seed,
                                   max_iterations=max_iterations,wall_sec=wall_sec)
    return _old_solver(method,task,obj,seed=seed,max_iterations=max_iterations,wall_sec=wall_sec)

base.run_solver=_dispatch

if __name__=='__main__':
    base.main()
