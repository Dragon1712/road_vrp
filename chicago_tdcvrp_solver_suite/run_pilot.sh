#!/usr/bin/env bash
# Run from ~/code/WZL/road_vrp
set -euo pipefail
python chicago_tdcvrp_solver_suite/benchmark_suite.py \
 --cost-dir datasets/processed/chicago/dynamic_costs \
 --task-dir datasets/processed/chicago/tdcvrp30/tasks \
 --output-dir runs/chicago_solver_suite/base \
 --od-cache-dir runs/chicago_solver_suite/od_cache \
 --methods greedy,sa,vnd,alns --planners static,oracle \
 --seeds 42,43,44,45,46 --max-tasks 1 --max-scenarios 2 \
 --vehicle-rule exactly --od-step-min 20 --od-hours 4 \
 --max-iterations 5000 --wall-sec 5
python chicago_tdcvrp_solver_suite/summarize_results.py \
 --metrics runs/chicago_solver_suite/base/metrics.csv \
 --output-dir runs/chicago_solver_suite/base/summary
