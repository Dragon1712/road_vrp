#!/usr/bin/env python3
"""Produce a CONSERVATIVE, non-conflicting static Chicago-monitor -> OSM-edge mapping.

All sources are local; no internet required. Input expected from earlier Chicago
matching/audit scripts. This does NOT assert geospatial ground truth and does
NOT generate traffic speed/travel-time predictions.
"""
import argparse
import json
from pathlib import Path
import pandas as pd

EDGE = ['osm_u', 'osm_v', 'osm_key']

def norm_id(x):
    s = str(x).strip()
    try:
        if s and float(s).is_integer():
            return str(int(float(s)))
    except ValueError:
        pass
    return s

def main():
    p = argparse.ArgumentParser(description='Create high-confidence conflict-free preliminary map for Chicago TD-VRP pilot')
    p.add_argument('--match-dir', required=True, help='Folder with accepted_osm_edge_mapping.csv and audit/strict_candidate_matches.csv')
    p.add_argument('--output-dir', default=None)
    p.add_argument('--min-coverage', type=float, default=0.50)
    p.add_argument('--allow-global-conflict-segments', action='store_true', help='Expert-only: do not conservatively exclude all segments appearing in full accepted mapping conflicts')
    args = p.parse_args()
    base = Path(args.match_dir)
    out = Path(args.output_dir) if args.output_dir else base / 'safe_mapping'
    out.mkdir(parents=True,exist_ok=True)

    strict = pd.read_csv(base/'audit'/'strict_candidate_matches.csv',dtype={'segment_id':str})
    mapping = pd.read_csv(base/'accepted_osm_edge_mapping.csv',dtype='string')
    confpath = base/'audit'/'edge_conflicts.csv'
    conflicts = pd.read_csv(confpath,dtype='string') if confpath.exists() else pd.DataFrame()
    required_strict = {'segment_id', 'valid_speed_ratio'}
    required_mapping = set(EDGE + ['segment_id', 'osm_length_m'])
    if not required_strict.issubset(strict.columns):
        raise ValueError('Missing strict fields: '+str(required_strict-set(strict.columns)))
    if not required_mapping.issubset(mapping.columns):
        raise ValueError('Missing mapping fields: '+str(required_mapping-set(mapping.columns)))

    strict['segment_id'] = strict['segment_id'].map(norm_id)
    strict['valid_speed_ratio'] = pd.to_numeric(strict.valid_speed_ratio,errors='coerce')
    mapping['segment_id'] = mapping['segment_id'].map(norm_id)
    for field in EDGE:
        mapping[field] = mapping[field].astype(str).str.strip()
    mapping['osm_length_m'] = pd.to_numeric(mapping['osm_length_m'],errors='coerce')

    strict = strict.drop_duplicates('segment_id').copy()
    eligible = strict[strict.valid_speed_ratio.ge(args.min_coverage)].copy()
    eligible_ids = set(eligible.segment_id)
    mm = mapping[mapping.segment_id.isin(eligible_ids)].copy()
    bad_lengths = set(mm.loc[~(mm.osm_length_m.gt(0) & mm.osm_length_m.notna()), 'segment_id'])
    mm = mm[mm.osm_length_m.gt(0) & mm.osm_length_m.notna()].copy()
    # If any edge of a segment is malformed, drop the entire monitored segment.
    excluded = {sid:'invalid_osm_edge_length' for sid in bad_lengths}

    global_conflict_ids = set()
    if not conflicts.empty and 'segment_ids' in conflicts.columns:
        for s in conflicts['segment_ids'].dropna():
            global_conflict_ids.update(norm_id(z) for z in str(s).split(',') if z.strip())
    if not args.allow_global_conflict_segments:
        for sid in eligible_ids & global_conflict_ids:
            excluded[sid] = 'appears_in_full_mapping_conflict'

    for sid in eligible_ids - set(mm.segment_id):
        excluded.setdefault(sid,'no_valid_mapped_osm_edge')

    # Recalculate after conservative pruning and eliminate ALL remaining internal collisions,
    # dropping entire affected monitored segments, never arbitrarily picking a winner.
    while True:
        mm = mm[~mm.segment_id.isin(excluded)].copy()
        duplicated = mm.groupby(EDGE,dropna=False).segment_id.nunique()
        bad_edges = duplicated[duplicated.gt(1)].index
        if len(bad_edges)==0:break
        affected = set(mm.set_index(EDGE).loc[bad_edges].segment_id)
        for sid in affected:excluded[sid] = 'internal_osm_edge_conflict'
    mm = mm.drop_duplicates(EDGE+['segment_id']).copy()

    safe_ids=set(mm.segment_id)
    eligible=eligible[eligible.segment_id.isin(safe_ids)].copy()
    totals=(mm.groupby('segment_id').osm_length_m.sum().rename('mapped_path_length_m').reset_index())
    mm=mm.merge(totals,on='segment_id',validate='many_to_one')
    mm=mm.merge(eligible[['segment_id','valid_speed_ratio']],on='segment_id',validate='many_to_one')
    mm['path_fraction']=mm.osm_length_m/mm.mapped_path_length_m
    assert mm[EDGE].duplicated().sum()==0, 'Unexpected OSM edge conflicts remain'
    assert mm['path_fraction'].between(0,1.00001).all(), 'Invalid path fractions'
    mm=mm.sort_values(['segment_id']+EDGE)

    eligible.to_csv(out/'selected_monitor_segments.csv',index=False)
    mm.to_csv(out/'safe_osm_edge_mapping.csv',index=False)
    pd.DataFrame([{'segment_id':sid,'reason':reason} for sid,reason in sorted(excluded.items())],
                 columns=['segment_id','reason']).to_csv(out/'excluded_monitor_segments.csv',index=False)
    report={
        'total_strict_candidates':int(len(strict)),
        'strict_at_or_above_min_coverage':int(len(eligible_ids)),
        'minimum_valid_observation_ratio':float(args.min_coverage),
        'segments_conflicting_in_full_accepted_mapping':int(len(global_conflict_ids)),
        'eligible_segments_excluded_due_to_any_conflict_or_quality':int(len(excluded)),
        'selected_monitor_segments':int(len(safe_ids)),
        'selected_unique_osm_directed_edges':int(len(mm)),
        'remaining_osm_edge_collisions':int(mm[EDGE].duplicated().sum()),
        'path_fraction_sum_min':float(mm.groupby('segment_id').path_fraction.sum().min()) if len(mm) else None,
        'path_fraction_sum_max':float(mm.groupby('segment_id').path_fraction.sum().max()) if len(mm) else None,
        'WARNING':'Geometrically strict is not independently verified. Inspect map alignment and road direction; do not treat this as ground truth.'
    }
    (out/'safe_mapping_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    print('Saved:',out)

if __name__=='__main__':main()
