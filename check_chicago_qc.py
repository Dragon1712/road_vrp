#!/usr/bin/env python3
"""Offline QC for output of prepare_chicago.py; requires numpy, pandas.
Usage: python check_chicago_qc.py --data /path/to/processed/chicago
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, required=True,
                    help='Output directory containing traffic_tensor.npz, traffic_long.csv, and indexes')
    ap.add_argument('--out', type=Path, default=None, help='Where to save diagnostics (default: data/qc)')
    args = ap.parse_args()
    root = args.data
    out = args.out or root / 'qc'
    out.mkdir(parents=True, exist_ok=True)
    required = ['traffic_tensor.npz', 'time_index.csv', 'segment_index.csv', 'traffic_long.csv', 'segments_static.csv']
    missing = [str(root / name) for name in required if not (root / name).is_file()]
    if missing:
        raise SystemExit('Missing files:\n' + '\n'.join(missing))

    pack = np.load(root / 'traffic_tensor.npz', allow_pickle=False)
    mask = pack['observed_mask'].astype(bool)
    speeds = pack['speed_kph']
    assert mask.shape == speeds.shape
    times = pd.to_datetime(pd.read_csv(root / 'time_index.csv')['time_local'])
    seg_index = pd.read_csv(root / 'segment_index.csv')
    assert mask.shape == (len(times), len(seg_index)), f'Axis shape mismatch: {mask.shape}'
    observed_per_slot = mask.sum(axis=1)
    observed_per_segment = mask.sum(axis=0)

    # Distinguish an entire missing raw snapshot from a snapshot with invalid speed rows.
    seen_slots = set()
    present_rows = 0
    for c in pd.read_csv(root / 'traffic_long.csv', usecols=['slot'], chunksize=200_000):
        present_rows += len(c)
        seen_slots.update(c['slot'].astype(str).str.replace(r'^(\d{4}-\d{2}-\d{2})$', r'\1 00:00:00', regex=True).str.slice(0, 19))
    str_times = times.dt.strftime('%Y-%m-%d %H:%M:%S')
    had_records = str_times.isin(seen_slots).to_numpy(dtype=bool)
    completely_absent = ~had_records
    no_valid_speed = observed_per_slot == 0

    days = times.dt.strftime('%Y-%m-%d')
    slot_detail = pd.DataFrame({
        'slot_local_naive': str_times,
        'day': days,
        'present_in_raw': had_records,
        'n_valid_speed': observed_per_slot,
        'fraction_valid_speed': observed_per_slot / mask.shape[1],
    })
    slot_detail.to_csv(out / 'slot_quality.csv', index=False)

    summary = slot_detail.groupby('day', sort=True).agg(
        time_slots=('day', 'size'),
        wholly_absent_slots=('present_in_raw', lambda x: int((~x).sum())),
        zero_valid_speed_slots=('n_valid_speed', lambda x: int((x == 0).sum())),
        valid_speed_observations=('n_valid_speed', 'sum'),
        valid_speed_ratio=('fraction_valid_speed', 'mean'),
    )
    summary.to_csv(out / 'day_coverage.csv')

    seg_coverage = seg_index.copy()
    seg_coverage['valid_speed_count'] = observed_per_segment
    seg_coverage['valid_speed_ratio'] = observed_per_segment / mask.shape[0]
    static = pd.read_csv(root / 'segments_static.csv')
    seg_coverage = seg_coverage.merge(static, on='segment_id', how='left', validate='one_to_one')
    seg_coverage.sort_values('valid_speed_ratio', ascending=False).to_csv(out / 'segment_coverage.csv', index=False)

    n_slots, n_seg = mask.shape
    print('\n=== OVERALL ===')
    print(f'Time slots: {n_slots}; segments: {n_seg}; full cells: {n_slots*n_seg:,}')
    print(f'Observed valid speeds: {int(mask.sum()):,} ({100*mask.mean():.2f}% of grid)')
    print(f'Raw long-table rows: {present_rows:,}')
    print(f'Entirely absent raw time slots: {int(completely_absent.sum())}')
    print(f'Time slots with zero valid speeds: {int(no_valid_speed.sum())}')
    print('Segments with >= 50% / 70% / 80% valid:', *(int((seg_coverage.valid_speed_ratio >= v).sum()) for v in (0.5, 0.7, 0.8)))
    print('\n=== DAILY ===')
    print(summary.to_string(float_format=lambda x: f'{x:.4f}'))
    print('\n=== MISSING RAW SLOTS (first 30) ===')
    print(str_times[completely_absent].head(30).to_string(index=False))
    print('\n=== BEST COVERAGE SEGMENTS (first 10) ===')
    print(seg_coverage[['segment_id', 'valid_speed_ratio', 'street', 'direction']].head(10).to_string(index=False))
    print('\n=== SPATIAL EXTENT ===')
    for key in ('start_latitude', 'start_longitude', 'end_latitude', 'end_longitude'):
        if key in static:
            v = pd.to_numeric(static[key], errors='coerce')
            print(f'{key}: {v.min():.6f} to {v.max():.6f}')
    print(f'\nSaved QC files to: {out}')
    print('NOTE: Treat source times as local naive until timezone semantics are verified.')


if __name__ == '__main__':
    main()
