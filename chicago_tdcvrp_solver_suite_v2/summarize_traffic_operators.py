#!/usr/bin/env python3
"""Aggregate traffic-aware operator statistics from v2 solutions.json.

Reports exact recorded algorithm diagnostic counters, not inferred gains.
"""
from pathlib import Path
import argparse
import json
import pandas as pd


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--solutions',required=True,type=Path)
    p.add_argument('--output-dir',required=True,type=Path)
    a=p.parse_args()
    records=json.loads(a.solutions.read_text(encoding='utf-8'))
    items=[]; op_rows=[]
    for r in records:
        d=r.get('solver_diagnostics') or {}
        c=d.get('method_config') or {}
        traffic=d.get('traffic_diagnostics') or {}
        entry={
            'task_id':r['task_id'],'scenario_id':r['scenario_id'],
            'solver':r['solver'],'planner':r['planner'],'seed':r['seed'],
            'recorded_cost_sec':float(r['recorded_cost_sec']),
            'traffic_signal_enabled':bool(traffic.get('dynamic_signal_enabled',False)),
            'traffic_related_enabled':bool(c.get('traffic_related_enabled',False)),
            'traffic_worst_enabled':bool(c.get('traffic_worst_enabled',False)),
            'mean_node_delay':float(traffic.get('node_delay_mean',0.)),
            'mean_temporal_variability':float(traffic.get('node_temporal_variability_mean',0.)),
            'changed_proposals':int(traffic.get('changed_proposals',0)),
            'failed_repairs':int(d.get('failed_repairs',0)),
            'local_improvements':int(traffic.get('local_improvements',0)),
            'local_score_calls':int(traffic.get('local_score_calls',0)),
            'iterations':int(d.get('iterations',0)),
            'score_calls':int(d.get('score_calls',0)),
        }
        items.append(entry)
        for kind,ops in (d.get('operator_usage') or {}).items():
            for name,count in ops.items():
                op_rows.append({**{k:entry[k] for k in ['task_id','scenario_id','solver','planner','seed']},
                               'operator_kind':kind,'operator':name,'use_count':int(count)})
    out=a.output_dir;out.mkdir(parents=True,exist_ok=True)
    x=pd.DataFrame(items)
    x.to_csv(out/'operator_diagnostics_by_run.csv',index=False)
    if op_rows:
        z=pd.DataFrame(op_rows)
        z.to_csv(out/'operator_usage_by_run.csv',index=False)
        (z.groupby(['solver','planner','operator_kind','operator'],as_index=False)
          .agg(total_uses=('use_count','sum'),runs_used=('use_count','size'))
          .to_csv(out/'operator_usage_summary.csv',index=False))
    summary=(x.groupby(['solver','planner'],as_index=False)
        .agg(n=('recorded_cost_sec','size'),mean_cost_sec=('recorded_cost_sec','mean'),
             mean_changed_proposals=('changed_proposals','mean'),
             mean_failed_repairs=('failed_repairs','mean'),
             mean_local_improvements=('local_improvements','mean'),
             mean_score_calls=('score_calls','mean'),
             mean_node_delay=('mean_node_delay','mean'),
             mean_temporal_variability=('mean_temporal_variability','mean'),
             traffic_runs=('traffic_signal_enabled','sum')))
    summary.to_csv(out/'operator_run_summary.csv',index=False)
    print(summary.to_string(index=False,float_format=lambda x:f'{x:.3f}'))

if __name__=='__main__':main()
