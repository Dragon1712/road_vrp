#!/usr/bin/env python3
"""Paired, task-blocked comparisons of external ALNS against a reference solver.

Positive gain_sec := reference cost - compared method cost (positive favors
compared method). Bootstrap is task-blocked, NOT an independent-run CI.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--metrics',required=True,type=Path)
    ap.add_argument('--reference',default='alns')
    ap.add_argument('--output-dir',required=True,type=Path)
    ap.add_argument('--bootstrap',default=2000,type=int)
    args=ap.parse_args()
    x=pd.read_csv(args.metrics)
    x=x[x.planner=='oracle'].copy()
    key=['task_id','scenario_id','seed']
    required=key+['solver','ex_post_road_travel_sec','total_solver_wall_sec']
    for c in required:
        if c not in x.columns:raise ValueError('Missing '+c)
    if x.duplicated(key+['solver']).any():raise ValueError('Duplicate task/scenario/seed/solver key')
    if args.reference not in set(x.solver):raise ValueError('Reference not in metrics')
    baseline=x[x.solver==args.reference][key+['ex_post_road_travel_sec','total_solver_wall_sec']]
    baseline=baseline.rename(columns={'ex_post_road_travel_sec':'reference_sec',
                                      'total_solver_wall_sec':'reference_solver_wall_sec'})
    y=x[x.solver!=args.reference].merge(baseline,on=key,how='inner',validate='many_to_one')
    for solver in sorted(set(x.solver)-{args.reference}):
        expect=len(baseline)
        got=len(y[y.solver==solver])
        if expect!=got:raise ValueError(f'Unmatched {solver}: {got} vs {expect}')
    y['gain_sec']=y.reference_sec-y.ex_post_road_travel_sec
    y['gain_pct']=100*y.gain_sec/y.reference_sec
    y['win']=y.gain_sec>1e-6
    tasks=(y.groupby(['solver','task_id'],as_index=False)
              .agg(mean_gain_sec=('gain_sec','mean'),
                   mean_gain_pct=('gain_pct','mean'),
                   win_rate=('win','mean'),
                   n_runs=('gain_pct','size')))
    rng=np.random.default_rng(20261015)
    summary=[]
    for solver,g in tasks.groupby('solver'):
        gains=g.mean_gain_pct.to_numpy(float)
        secs=g.mean_gain_sec.to_numpy(float)
        n=len(g)
        draws=rng.integers(0,n,size=(max(1,args.bootstrap),n))
        boot=np.mean(gains[draws],axis=1)
        summary.append({'solver':solver,'reference':args.reference,
                        'num_tasks':n,'paired_runs':int(len(y[y.solver==solver])),
                        'mean_gain_pct':float(gains.mean()),
                        'mean_gain_sec':float(secs.mean()),
                        'task_boot_ci_lower_pct':float(np.quantile(boot,.025)),
                        'task_boot_ci_upper_pct':float(np.quantile(boot,.975)),
                        'tasks_with_mean_positive':int((gains>0).sum()),
                        'overall_paired_win_rate':float(y[y.solver==solver].win.mean())})
    args.output_dir.mkdir(parents=True,exist_ok=True)
    y.to_csv(args.output_dir/'paired_results.csv',index=False)
    tasks.to_csv(args.output_dir/'task_means.csv',index=False)
    df=pd.DataFrame(summary)
    df.to_csv(args.output_dir/'solver_comparison.csv',index=False)
    print(df.to_string(index=False,float_format=lambda v:f'{v:.3f}'))
    print('Task-level bootstrap is exploratory; traffic days shared between tasks.')

if __name__=='__main__':main()
