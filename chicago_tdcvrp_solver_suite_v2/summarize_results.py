#!/usr/bin/env python3
"""Aggregate solver benchmarks without confusing multiple seeds with independent tasks."""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--metrics',required=True)
    p.add_argument('--output-dir',required=True)
    args=p.parse_args()
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(args.metrics)
    must=['task_id','scenario_id','solver','planner','seed','ex_post_road_travel_sec','total_solver_wall_sec']
    missing=[x for x in must if x not in df.columns]
    if missing:raise ValueError('Missing metrics fields: '+str(missing))
    if df.duplicated(['task_id','scenario_id','solver','planner','seed']).any():
        raise ValueError('Duplicate seed-task-scenario-solver-planner keys')
    keys=['task_id','scenario_id']
    df['pair_best_cost_sec']=df.groupby(keys).ex_post_road_travel_sec.transform('min')
    df['pool_gap_pct']=100*(df.ex_post_road_travel_sec/df.pair_best_cost_sec-1)
    df.to_csv(out/'metrics_with_pool_gap.csv',index=False)
    summary=(df.groupby(['solver','planner'],as_index=False)
               .agg(records=('ex_post_road_travel_sec','size'),
                    mean_cost_sec=('ex_post_road_travel_sec','mean'),
                    std_cost_sec=('ex_post_road_travel_sec','std'),
                    mean_pool_gap_pct=('pool_gap_pct','mean'),
                    median_pool_gap_pct=('pool_gap_pct','median'),
                    mean_solver_sec=('total_solver_wall_sec','mean'),
                    mean_od_precompute_sec=('od_precompute_wall_sec','mean'),
                    mean_score_calls=('score_calls','mean'),
                    mean_observed_edge_share=('observed_edge_share','mean')))
    summary.to_csv(out/'solver_summary.csv',index=False)
    pairing=(df.pivot(index=['task_id','scenario_id','solver','seed'],columns='planner',
                      values='ex_post_road_travel_sec'))
    if {'static','oracle'}.issubset(pairing.columns):
        paired=pairing.dropna(subset=['static','oracle']).reset_index()
        paired['oracle_gain_pct']=100*(paired['static']-paired['oracle'])/paired['static']
        paired.to_csv(out/'paired_gain_by_seed.csv',index=False)
        # Aggregate over random seeds within each *instance+traffic scenario* first.
        inst=(paired.groupby(['task_id','scenario_id','solver'],as_index=False)
              .agg(mean_gain_over_seeds_pct=('oracle_gain_pct','mean'),
                   std_gain_over_seeds_pct=('oracle_gain_pct','std'),
                   win_rate_over_seeds=('oracle_gain_pct',lambda z: float((z>0).mean())),
                   num_seeds=('oracle_gain_pct','size')))
        inst.to_csv(out/'per_instance_stability.csv',index=False)
        rep=(inst.groupby('solver',as_index=False)
            .agg(mean_instance_gain_pct=('mean_gain_over_seeds_pct','mean'),
                 std_instance_gain_pct=('mean_gain_over_seeds_pct','std'),
                 mean_win_rate=('win_rate_over_seeds','mean'),
                 n_instances=('mean_gain_over_seeds_pct','size')))
        rep.to_csv(out/'paired_summary_by_solver.csv',index=False)
    print('Saved aggregated CSVs to',out)
    print(summary.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    print('NOTE: Best-in-evaluated-pool gap is NOT a bound to the true optimum; seeds share the same underlying tasks.')

if __name__=='__main__':main()
