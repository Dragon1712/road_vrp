#!/usr/bin/env python3
"""Offline cleaning of Chicago Traffic Tracker historical segment speed CSV.

Dependencies: pandas, numpy (optional pyarrow for parquet).
No Internet access needed. Does NOT perform OSM matching or VRP routing.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = [
    'time', 'segment_id', 'speed', 'street', 'direction', 'length',
    'start_latitude', 'start_longitude', 'end_latitude', 'end_longitude'
]
STATIC = [
    'segment_id', 'street', 'direction', 'from_street', 'to_street',
    'length', 'street_heading', 'start_latitude', 'start_longitude',
    'end_latitude', 'end_longitude'
]


def save_table(df, path, fmt):
    if fmt == 'parquet':
        try:
            df.to_parquet(path.with_suffix('.parquet'), index=False)
        except ImportError as e:
            raise SystemExit('Parquet needs pyarrow: python -m pip install pyarrow') from e
    else:
        df.to_csv(path.with_suffix('.csv'), index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input-dir', required=True, type=Path, help='Directory containing traffic_*.csv')
    ap.add_argument('--pattern', default='traffic_*.csv')
    ap.add_argument('--output-dir', required=True, type=Path)
    ap.add_argument('--format', choices=['csv', 'parquet'], default='csv')
    ap.add_argument('--step-min', type=int, default=10)
    args = ap.parse_args()
    if args.step_min < 1 or 1440 % args.step_min != 0:
        ap.error('--step-min must divide 1440')

    files = sorted(args.input_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f'No files match {args.input_dir / args.pattern}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    original_rows = 0
    for f in files:
        x = pd.read_csv(f, low_memory=False)
        x.columns = x.columns.str.strip().str.lower()
        missing = [c for c in REQUIRED if c not in x]
        if missing:
            raise SystemExit(f'{f}: missing fields: {missing}')
        for col in STATIC:
            if col not in x:
                x[col] = np.nan
        for col in ['bus_count', 'message_count']:
            if col not in x:
                x[col] = np.nan
        use = list(dict.fromkeys(REQUIRED + STATIC + ['bus_count', 'message_count']))
        x = x[use].copy()
        original_rows += len(x)
        frames.append(x)
        print(f'Loaded {f.name}: {len(x):,} rows')

    df = pd.concat(frames, ignore_index=True)
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    numeric = ['segment_id', 'speed', 'length', 'start_latitude', 'start_longitude',
               'end_latitude', 'end_longitude', 'bus_count', 'message_count']
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['time', 'segment_id']).copy()
    df['segment_id'] = df['segment_id'].astype('int64')
    df['slot'] = df['time'].dt.floor(f'{args.step_min}min')
    df.sort_values(['slot', 'segment_id', 'time'], inplace=True, kind='mergesort')
    duplicates = int(df.duplicated(['slot', 'segment_id']).sum())
    df = df.drop_duplicates(['slot', 'segment_id'], keep='last').copy()
    invalid_speed = (~df['speed'].gt(0)) | (~np.isfinite(df['speed']))
    df['observed'] = ~invalid_speed
    df['speed_mph'] = df['speed'].where(df['observed']).astype(float)
    df['speed_kph'] = df['speed_mph'] * 1.609344
    df['length_m'] = df['length'] * 1609.344
    df['segment_travel_sec'] = 3600.0 * df['length'] / df['speed_mph']
    df.loc[df['length'].le(0) | df['length'].isna(), 'segment_travel_sec'] = np.nan
    for c in ['street', 'direction', 'from_street', 'to_street', 'street_heading']:
        df[c] = df[c].astype('string').str.strip()

    # Spatial attributes belong to segment IDs; mode/first per segment should stay stable.
    static_cols = [c for c in STATIC if c != 'segment_id']
    seg = df.groupby('segment_id', sort=True)[static_cols].first().reset_index()
    seg['length_m'] = pd.to_numeric(seg['length'], errors='coerce') * 1609.344
    seg = seg.sort_values('segment_id').reset_index(drop=True)

    cols = ['slot', 'time', 'segment_id', 'speed_mph', 'speed_kph',
            'segment_travel_sec', 'observed', 'bus_count', 'message_count']
    obs = df[cols].sort_values(['slot', 'segment_id']).reset_index(drop=True)
    save_table(seg, args.output_dir/'segments_static', args.format)
    save_table(obs, args.output_dir/'traffic_long', args.format)

    # Dense arrays for neural network input; retain NaNs and an explicit mask.
    start = df['slot'].min().normalize()
    stop = df['slot'].max().normalize() + pd.Timedelta(days=1)
    clock = pd.date_range(start, stop, freq=f'{args.step_min}min', inclusive='left')
    columns = pd.Index(seg['segment_id'], name='segment_id')
    pivot = obs.pivot(index='slot', columns='segment_id', values='speed_kph')
    pivot = pivot.reindex(index=clock, columns=columns)
    arr = pivot.to_numpy(dtype=np.float32, copy=True)
    mask = np.isfinite(arr)
    np.savez_compressed(args.output_dir/'traffic_tensor.npz',
                        speed_kph=arr, observed_mask=mask, segment_ids=columns.to_numpy(dtype=np.int64),
                        # epoch nanos from naive local timestamps; treat as wall-clock, not UTC.
                        time_local_naive_ns=clock.asi8)
    pd.DataFrame({'time_local': clock.strftime('%Y-%m-%d %H:%M:%S')}).to_csv(args.output_dir/'time_index.csv', index=False)
    pd.DataFrame({'column_index': np.arange(len(columns)), 'segment_id': columns}).to_csv(args.output_dir/'segment_index.csv', index=False)

    by_day = obs.assign(date=obs['slot'].dt.date.astype(str)).groupby('date').agg(
        rows=('segment_id', 'size'),
        valid=('observed', 'sum'),
        segments=('segment_id', 'nunique'))
    by_day['valid_ratio'] = by_day['valid'] / by_day['rows']
    by_day.reset_index().to_csv(args.output_dir/'daily_quality.csv', index=False)
    issues = {}
    for c in ['direction','street','length','start_latitude','start_longitude','end_latitude','end_longitude']:
        mismatch = df.groupby('segment_id')[c].nunique(dropna=True)
        issues[c] = int((mismatch > 1).sum())
    report = {
        'files': [f.name for f in files], 'original_rows': original_rows,
        'kept_after_cleanup': int(len(obs)), 'duplicate_segment_slots_removed': duplicates,
        'segments': int(len(seg)), 'time_slots': int(len(clock)),
        'traffic_tensor_shape': list(arr.shape),
        'valid_observations': int(mask.sum()),
        'coverage_of_full_segment_time_grid': round(float(mask.mean()), 5),
        'invalid_speed_rows': int(invalid_speed.sum()),
        'length_unit_input': 'mile', 'speed_unit_input': 'mph',
        'derived_speed_unit': 'km/h', 'derived_travel_time_unit': 'seconds',
        'timezone_handling': 'source local-like naive TIME retained; verify timezone with provider; no UTC conversion applied',
        'static_field_changes_by_segment': issues,
        'notes': [
            'Missing (including -1 mph) speeds are NaN; never replace with zero.',
            'The 10-minute grid is a preprocessing assumption based on source nominal frequency.',
            'Road segments here are monitored corridors, NOT a complete connected OSM graph.',
            'No training/test split, imputation, OSM match, or VRP solving has occurred.',
            'Record segment id order by segment_index.csv for any later tensors.'
        ]
    }
    (args.output_dir/'quality_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print('DONE:', args.output_dir)

if __name__=='__main__':
    main()
