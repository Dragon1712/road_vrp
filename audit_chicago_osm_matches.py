#!/usr/bin/env python3
"""Offline audit of outputs from match_segments_offline.py.

Requires only pandas. No network access. This does not validate geographic correctness;
it highlights provisional matches requiring human review before edge-speed transfer.
"""
import argparse
import json
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description='Audit preliminary Chicago Traffic Tracker to OSM matching')
    ap.add_argument('--match-dir', required=True, help='Directory containing segment_matches.csv and accepted_osm_edge_mapping.csv')
    ap.add_argument('--output-dir', default=None)
    ap.add_argument('--high-coverage', type=float, default=.5)
    ap.add_argument('--strict-street', type=float, default=.8)
    ap.add_argument('--strict-length-min', type=float, default=.8)
    ap.add_argument('--strict-length-max', type=float, default=1.25)
    ap.add_argument('--strict-snap-m', type=float, default=100.)
    args=ap.parse_args()

    md=Path(args.match_dir)
    out=Path(args.output_dir) if args.output_dir else md / 'audit'
    out.mkdir(parents=True,exist_ok=True)
    m=pd.read_csv(md/'segment_matches.csv')
    e=pd.read_csv(md/'accepted_osm_edge_mapping.csv')
    required=['segment_id','match_status','valid_speed_ratio','snap_start_m','snap_end_m','length_ratio','street_fraction']
    missing=[x for x in required if x not in m.columns]
    if missing: raise ValueError('Missing segment match fields: '+str(missing))

    m['valid_speed_ratio']=pd.to_numeric(m.valid_speed_ratio,errors='coerce')
    accepted=m.match_status.eq('accepted')
    considered=m.match_status.isin(['accepted','poor_match_review','same_endpoint','no_directed_path'])
    high=m.valid_speed_ratio.ge(args.high_coverage)
    strict=(accepted & m.street_fraction.ge(args.strict_street)
            & m.length_ratio.between(args.strict_length_min,args.strict_length_max)
            & m.snap_start_m.le(args.strict_snap_m) & m.snap_end_m.le(args.strict_snap_m))

    # For visibility: an OSM edge mapped to multiple independent monitored segment IDs.
    conflicts=pd.DataFrame(columns=['osm_u','osm_v','osm_key','n_segments','segment_ids'])
    if not e.empty:
        grp=(e.groupby(['osm_u','osm_v','osm_key'],dropna=False)['segment_id']
             .agg(n_segments='nunique',segment_ids=lambda ss: ','.join(map(str,sorted(set(ss))))).reset_index())
        conflicts=grp[grp.n_segments.gt(1)].sort_values('n_segments',ascending=False)
    conflicts.to_csv(out/'edge_conflicts.csv',index=False)

    review=m.copy()
    review['review_reason']=''
    review.loc[review.match_status.eq('poor_match_review'),'review_reason'] += 'poor_match;'
    review.loc[accepted & ~strict,'review_reason'] += 'accepted_but_not_strict;'
    review.loc[accepted & high,'review_reason'] += 'high_coverage_priority;'
    review=review[review['review_reason'].ne('')].copy()
    review['review_score']=(review.match_status.eq('poor_match_review').astype(int)*5
                            + high.reindex(review.index).astype(int)*3
                            + review.street_fraction.lt(args.strict_street).astype(int)*2
                            + (~review.length_ratio.between(args.strict_length_min,args.strict_length_max)).astype(int)*2)
    review=review.sort_values(['review_score','valid_speed_ratio'],ascending=False)
    review.to_csv(out/'review_priority.csv',index=False)
    m.loc[strict].to_csv(out/'strict_candidate_matches.csv',index=False)

    counts={str(k):int(v) for k,v in m.match_status.value_counts().to_dict().items()}
    summary={
        'all_rows':int(len(m)),
        'status_counts':counts,
        'within_candidate_path_scope':int(considered.sum()),
        'accepted':int(accepted.sum()),
        'accepted_rate_within_candidate_path_scope':round(float(accepted.sum()/considered.sum()),4) if considered.sum() else None,
        'accepted_high_coverage':int((accepted&high).sum()),
        'within_candidate_scope_high_coverage':int((considered&high).sum()),
        'accepted_rate_within_scope_high_coverage':round(float((accepted&high).sum()/(considered&high).sum()),4) if (considered&high).sum() else None,
        'strict_candidate_matches':int(strict.sum()),
        'strict_high_coverage_matches':int((strict&high).sum()),
        'mapping_rows':int(len(e)),
        'unique_directed_osm_edges':int(len(e[['osm_u','osm_v','osm_key']].drop_duplicates())) if not e.empty else 0,
        'conflicting_osm_edges':int(len(conflicts)),
        'notes':[
            'Accepted is preliminary, not geospatial ground truth.',
            'no_close_endpoint typically means out-of-area or endpoint snap failure; do not use it in local-area denominator without a geographic scope definition.',
            'Strict matches are review candidates based on stronger thresholds, not verified correct matches.',
            'A monitored segment can span multiple OSM edges; collision edges require explicit speed assignment rules.'
        ]
    }
    (out/'audit_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(summary,indent=2,ensure_ascii=False))
    print('\nSaved audit files to:',out)

if __name__ == '__main__': main()
