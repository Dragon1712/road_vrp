#!/usr/bin/env bash
# Run from your road_vrp project root after unzipping chicago_tdcvrp30_offline.zip.
set -euo pipefail
COST=datasets/processed/chicago/dynamic_costs
TASK=datasets/processed/chicago/tdcvrp30/tasks
RUN=runs/chicago_tdcvrp30

python tdcvrp30.py regress \
  --cost-dir "$COST" \
  --morning fixtures/tdsp_morning.csv \
  --evening fixtures/tdsp_evening.csv \
  --output-dir "$RUN/regression"

python tdcvrp30.py generate \
  --cost-dir "$COST" --output-dir "$TASK" \
  --num-tasks 10 --customers 30 --vehicles 5 \
  --capacity 45 --service-seconds 120 \
  --strategy mixed --radius-m 600 --seed 2026

python tdcvrp30.py solve \
  --cost-dir "$COST" --task-dir "$TASK" \
  --cache-dir "$RUN/od_cache" --output-dir "$RUN/pilot" \
  --max-tasks 1 --max-scenarios 2 \
  --od-hours 4 --od-step-minutes 20 \
  --iterations 1500 --restarts 2 --seed 42

python tdcvrp30.py od-audit \
  --cost-dir "$COST" --task-dir "$TASK" \
  --cache-dir "$RUN/od_cache" --output-dir "$RUN/od_audit" \
  --od-hours 4 --od-step-minutes 20 --samples 100

echo "DONE. Read $RUN/regression/tdsp_regression_report.json and $RUN/pilot/metrics.csv"
