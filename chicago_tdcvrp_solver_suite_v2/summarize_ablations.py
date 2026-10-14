#!/usr/bin/env python3
"""Matched-seed ALNS ablation: compare each variant with Full ALNS on identical task/date/seed.

For interpretable analyses, first aggregate within tasks, not across seeds as if
independent traffic/VRP cases. More positive delta -> FULL ALNS better.
"""
from pathlib import Path
import argparse
import pandas as pd
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metrics',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    x=pd.read_csv(args.metrics)
    for k in ('task_id','scenario_id','seed','solver','planner','ex_post_road_travel_sec','total_solver_wall_sec'):
        if k not in x.columns:raise ValueError(f'Missing {k}')
    x=x[x.planner=='oracle']
    if x.empty or 'alns' not in set(x.solver):raise ValueError('Oracle Full ALNS reference missing')
    if x.duplicated(['task_id','scenario_id','seed','solver']).any():raise ValueError('Duplicate matched keys')
    keys=['task_id','scenario_id','seed']
    ref=x[x.solver=='alns'][keys+['ex_post_road_travel_sec','total_solver_wall_sec']].rename(columns={
        'ex_post_road_travel_sec':'full_alns_cost_sec','total_solver_wall_sec':'full_alns_solver_sec'})
    y=x[x.solver!='alns'].merge(ref,on=keys,how='inner',validate='many_to_one')
    expected=len(ref)
    variants=set(x.solver)-{'alns'}
    for v in variants:
        if len(y[y.solver==v])!=expected:
            raise ValueError(f'{v} has unmatched runs; expected {expected}, observed {len(y[y.solver==v])}. Complete the experiment first.')
    y['full_alns_advantage_sec']=y.ex_post_road_travel_sec-y.full_alns_cost_sec
    y['full_alns_advantage_pct']=100*y.full_alns_advantage_sec/y.ex_post_road_travel_sec
    y['full_alns_win']=y.full_alns_advantage_sec>1e-6
    args.output_dir.mkdir(parents=True,exist_ok=True)
    y.to_csv(args.output_dir/'paired_ablation_by_seed.csv',index=False)
    within=(y.groupby(['solver','task_id','scenario_id'],as_index=False)
        .agg(mean_full_alns_advantage_pct=('full_alns_advantage_pct','mean'),
             sd_over_seeds_pct=('full_alns_advantage_pct','std'),
             full_alns_win_fraction=('full_alns_win','mean'),
             mean_full_alns_advantage_sec=('full_alns_advantage_sec','mean'),
             num_seeds=('seed','nunique')))
    within.to_csv(args.output_dir/'paired_ablation_by_instance.csv',index=False)
    summary=(within.groupby('solver',as_index=False)
        .agg(mean_advantage_pct=('mean_full_alns_advantage_pct','mean'),
             std_across_instances_pct=('mean_full_alns_advantage_pct','std'),
             mean_full_alns_win_fraction=('full_alns_win_fraction','mean'),
             mean_advantage_sec=('mean_full_alns_advantage_sec','mean'),
             independent_tasks=('task_id','nunique'),
             task_scenarios=('scenario_id','size')))
    summary.to_csv(args.output_dir/'alns_ablation_summary.csv',index=False)
    print(summary.to_string(index=False,float_format=lambda q:f'{q:.3f}'))
    print('Note: +advantage means removing module is worse. Results are relative to Full ALNS under matched seed.')
    print('One task over several dates is NOT several independent VRP tasks.')

if __name__=='__main__':main()
