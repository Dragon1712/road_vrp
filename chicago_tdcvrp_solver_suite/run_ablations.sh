#!/usr/bin/env bash
# Run from ~/code/WZL/road_vrp ; same tasks, OD cache and seeds as baseline
set -euo pipefail
python chicago_tdcvrp_solver_suite/benchmark_suite.py \
 --cost-dir datasets/processed/chicago/dynamic_costs \
 --task-dir datasets/processed/chicago/tdcvrp30/tasks \
 --output-dir runs/chicago_solver_suite/ablations \
 --od-cache-dir runs/chicago_solver_suite/od_cache \
 --methods alns,alns_no_related,alns_no_worst,alns_greedy_repair,alns_uniform,alns_no_local \
 --planners oracle --seeds 42,43,44,45,46 \
 --max-tasks 1 --max-scenarios 2 \
 --vehicle-rule exactly --od-step-min 20 --od-hours 4 \
 --max-iterations 5000 --wall-sec 5
python chicago_tdcvrp_solver_suite/summarize_results.py \
 --metrics runs/chicago_solver_suite/ablations/metrics.csv \
 --output-dir runs/chicago_solver_suite/ablations/summary
