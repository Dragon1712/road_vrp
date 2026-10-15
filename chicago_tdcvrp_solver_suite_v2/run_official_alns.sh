#!/usr/bin/env bash
# Execute from road_vrp repository root, with the v2 code and v2 task data intact.
set -euo pipefail
STAGE="${1:-smoke}"
ROOT="chicago_tdcvrp_solver_suite_v2"
COST="datasets/processed/chicago/dynamic_costs"
TASK="datasets/processed/chicago/tdcvrp30/tasks_10"
CACHE="runs/chicago_solver_suite_v2/od_cache"
OUT="runs/chicago_solver_suite_v2/upstream_library"
RESUME_ARGS=()
if [ "${RESUME:-0}" = "1" ]; then RESUME_ARGS=(--resume); fi

if [ ! -f "$TASK/task_0000.json" ]; then
    echo "Missing $TASK/task_0000.json: sync original task data first." >&2
    exit 2
fi
python -c 'from alns import ALNS; from alns.select import RouletteWheel; from alns.accept import SimulatedAnnealing; print("Official alns API available")'

case "$STAGE" in
  smoke) TASKS=1; SC=2; SEEDS='42'; METHODS='alns,upstream_alns'; BUDGET=50; WALL=2; RUN='smoke' ;;
  compare) TASKS=10; SC=6; SEEDS='42,43,44'; METHODS='alns,upstream_alns'; BUDGET=5000; WALL=5; RUN='compare' ;;
  ablations) TASKS=10; SC=6; SEEDS='42,43,44'; METHODS='upstream_alns,upstream_no_related,upstream_no_worst,upstream_greedy_repair,upstream_uniform,upstream_no_local'; BUDGET=5000; WALL=5; RUN='ablations' ;;
  *) echo "Usage: bash $ROOT/run_official_alns.sh smoke|compare|ablations" >&2; exit 2 ;;
esac
python "$ROOT/benchmark_official_alns.py" \
  --cost-dir "$COST" --task-dir "$TASK" --od-cache-dir "$CACHE" \
  --output-dir "$OUT/$RUN" --max-tasks "$TASKS" --max-scenarios "$SC" \
  --methods "$METHODS" --planners oracle --seeds "$SEEDS" \
  --vehicle-rule exactly --od-step-min 20 --od-hours 4 \
  --max-iterations "$BUDGET" --wall-sec "$WALL" \
  --checkpoint-every 5 "${RESUME_ARGS[@]}"
python "$ROOT/summarize_results.py" --metrics "$OUT/$RUN/metrics.csv" \
  --output-dir "$OUT/$RUN/summary"
if [ "$STAGE" = "compare" ]; then
    python "$ROOT/compare_official_alns.py" \
      --metrics "$OUT/$RUN/metrics.csv" --reference alns \
      --output-dir "$OUT/$RUN/comparison"
elif [ "$STAGE" = "ablations" ]; then
    python "$ROOT/compare_official_alns.py" \
      --metrics "$OUT/$RUN/metrics.csv" --reference upstream_alns \
      --output-dir "$OUT/$RUN/comparison"
fi
echo "Results: $OUT/$RUN/metrics.csv"
