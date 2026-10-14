#!/usr/bin/env bash
# Run from ~/code/WZL/road_vrp, AFTER installing the original tdcvrp/ and this suite.
# STAGE=smoke|base|ablations|all; each stage writes to its own output directory.
set -euo pipefail
STAGE="${1:-smoke}"
RESUME_ARGS=()
if [ "${RESUME:-0}" = "1" ]; then RESUME_ARGS=(--resume); fi
ROOT=chicago_tdcvrp_solver_suite_v2
COST=datasets/processed/chicago/dynamic_costs
TASK=datasets/processed/chicago/tdcvrp30/tasks_10
CACHE=runs/chicago_solver_suite_v2/od_cache
OUT=runs/chicago_solver_suite_v2

if [ "$STAGE" = "generate" ]; then
  python "$ROOT/prepare_10_tasks.py" --cost-dir "$COST" --output-dir "$TASK" \
    --num-tasks 10 --customers 30 --vehicles 5 --capacity 45 \
    --strategy mixed --radius-m 600 --seed 20261014
  exit 0
fi
if [ ! -f "$TASK/task_validation.json" ]; then
  echo "Missing validated tasks: run 'bash $ROOT/run_10tasks.sh generate' first" >&2
  exit 2
fi

runbase() {
  local out="$1"; local tasks="$2"; local sc="$3"; local seeds="$4"; local budget="$5"; local wall="$6";
  python "$ROOT/benchmark_suite.py" \
    --cost-dir "$COST" --task-dir "$TASK" --output-dir "$out" --od-cache-dir "$CACHE" \
    --methods greedy,sa,vnd,alns --planners static,oracle --seeds "$seeds" \
    --max-tasks "$tasks" --max-scenarios "$sc" --vehicle-rule exactly \
    --od-step-min 20 --od-hours 4 --max-iterations "$budget" --wall-sec "$wall" \
    --checkpoint-every 5 "${RESUME_ARGS[@]}"
  python "$ROOT/summarize_results.py" --metrics "$out/metrics.csv" --output-dir "$out/summary"
}
runablation() {
  local out="$1"; local tasks="$2"; local sc="$3"; local seeds="$4"; local budget="$5"; local wall="$6";
  python "$ROOT/benchmark_suite.py" \
    --cost-dir "$COST" --task-dir "$TASK" --output-dir "$out" --od-cache-dir "$CACHE" \
    --methods alns,alns_no_related,alns_no_worst,alns_greedy_repair,alns_uniform,alns_no_local \
    --planners oracle --seeds "$seeds" \
    --max-tasks "$tasks" --max-scenarios "$sc" --vehicle-rule exactly \
    --od-step-min 20 --od-hours 4 --max-iterations "$budget" --wall-sec "$wall" \
    --checkpoint-every 5 "${RESUME_ARGS[@]}"
  python "$ROOT/summarize_results.py" --metrics "$out/metrics.csv" --output-dir "$out/summary"
  python "$ROOT/summarize_ablations.py" --metrics "$out/metrics.csv" --output-dir "$out/summary"
}
case "$STAGE" in
  smoke)
    runbase "$OUT/smoke_base" 1 2 42,43 500 2
    runablation "$OUT/smoke_ablations" 1 2 42,43 500 2 ;;
  base)
    runbase "$OUT/base_10tasks" 10 6 42,43,44 5000 5 ;;
  ablations)
    runablation "$OUT/ablations_10tasks" 10 6 42,43,44 5000 5 ;;
  all)
    runbase "$OUT/base_10tasks" 10 6 42,43,44 5000 5
    runablation "$OUT/ablations_10tasks" 10 6 42,43,44 5000 5 ;;
  *) echo "Usage: bash $ROOT/run_10tasks.sh generate|smoke|base|ablations|all" >&2; exit 2 ;;
esac
