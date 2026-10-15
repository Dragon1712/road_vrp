#!/usr/bin/env bash
# Run from ~/code/WZL/road_vrp. Versioned output: no old results overwritten.
set -euo pipefail
STAGE="${1:-smoke}"
ROOT=chicago_tdcvrp_solver_suite_v2
COST=datasets/processed/chicago/dynamic_costs
TASK=datasets/processed/chicago/tdcvrp30/tasks_10
CACHE=runs/chicago_solver_suite_v2/od_cache
OUT=runs/chicago_solver_suite_v2/traffic_alns
SEEDS="${SEEDS:-42,43,44}"
BUDGET="${BUDGET:-5000}"
WALL="${WALL:-5}"
TASKS="${TASKS:-10}"
SCENARIOS="${SCENARIOS:-6}"
RESUME_ARGS=()
if [ "${RESUME:-0}" = "1" ];then RESUME_ARGS+=(--resume);fi
if [ ! -f "$TASK/task_0000.json" ];then echo "Missing task files: $TASK" >&2;exit 2;fi
python - <<'PY'
from alns import ALNS
from alns.accept import SimulatedAnnealing
from alns.select import RouletteWheel,RandomSelect
print('Official ALNS installed; import PASS')
PY
run_one() {
  local subdir="$1" methods="$2" tasks="$3" sc="$4" seeds="$5" budget="$6" wall="$7"
  local dir="$OUT/$subdir"
  python "$ROOT/benchmark_traffic_alns.py" \
    --cost-dir "$COST" --task-dir "$TASK" --od-cache-dir "$CACHE" \
    --output-dir "$dir" --max-tasks "$tasks" --max-scenarios "$sc" \
    --methods "$methods" --planners oracle --seeds "$seeds" \
    --vehicle-rule exactly --od-step-min 20 --od-hours 4 \
    --max-iterations "$budget" --wall-sec "$wall" \
    --checkpoint-every 5 "${RESUME_ARGS[@]}"
  python "$ROOT/summarize_results.py" --metrics "$dir/metrics.csv" --output-dir "$dir/summary"
  python "$ROOT/compare_official_alns.py" --metrics "$dir/metrics.csv" \
    --reference upstream_plus --output-dir "$dir/vs_search_only"
  python "$ROOT/compare_official_alns.py" --metrics "$dir/metrics.csv" \
    --reference upstream_traffic --output-dir "$dir/vs_traffic_full"
  echo "Saved $dir"
}
case "$STAGE" in
 smoke)
   run_one smoke "upstream_alns,upstream_plus,upstream_traffic,upstream_traffic_no_t_related,upstream_traffic_no_t_worst,upstream_traffic_no_regret3,upstream_traffic_no_local,upstream_traffic_uniform" 1 1 42 30 2 ;;
 compare)
   run_one compare "upstream_alns,upstream_plus,upstream_traffic" "$TASKS" "$SCENARIOS" "$SEEDS" "$BUDGET" "$WALL" ;;
 ablations)
   run_one ablations "upstream_plus,upstream_traffic,upstream_traffic_no_t_related,upstream_traffic_no_t_worst,upstream_traffic_no_regret3,upstream_traffic_no_local,upstream_traffic_uniform" "$TASKS" "$SCENARIOS" "$SEEDS" "$BUDGET" "$WALL" ;;
 *) echo 'Usage: bash chicago_tdcvrp_solver_suite_v2/run_traffic_alns.sh smoke|compare|ablations' >&2;exit 2 ;;
esac
