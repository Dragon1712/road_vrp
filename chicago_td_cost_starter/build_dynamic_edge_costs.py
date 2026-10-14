#!/usr/bin/env python3
"""Chicago 2025 measured corridor -> OSM directed-edge, time-dependent costs.

Offline, reproducible pilot. Outputs observed/fallback masks, raw sampled costs,
and an optional FIFO-adjusted sampled-arrival envelope suitable for linear
interpolation in continuous-time TD shortest paths.

Only observed Chicago segment_travel_sec records produce observed edge labels.
All missing/unmatched edges use static *estimates*, never relabeled as measured.
"""
import argparse
import json
import math
from pathlib import Path
import re

import networkx as nx
import numpy as np
import pandas as pd

EDGE = ["osm_u", "osm_v", "osm_key"]
KM_PER_MILE = 1.609344


def normalize_id(x):
    s = str(x).strip()
    if re.fullmatch(r"[+-]?\d+\.0+", s):
        return s.split(".")[0]
    return s


def parse_highway(value):
    """OSMnx GraphML often encodes OSM lists as text representations."""
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""
    val = str(value or "").lower()
    for highway in ("motorway", "trunk", "primary", "secondary", "tertiary", "residential", "living_street", "service", "unclassified"):
        if highway in val:
            return highway
    return "other"


DEFAULT_SPEED_KPH = {
    "motorway": 80, "trunk": 65, "primary": 50, "secondary": 45,
    "tertiary": 40, "residential": 30, "living_street": 15,
    "service": 20, "unclassified": 35, "other": 35,
}


def static_speed(edge):
    """Static routing estimate, NOT a measured Chicago travel speed."""
    speed_kph = edge.get("speed_kph", None)
    try:
        val = float(speed_kph)
        if np.isfinite(val) and val > 0:
            return val, "osm_preexisting_speed_kph"
    except (TypeError, ValueError):
        pass
    raw = edge.get("maxspeed", "")
    if isinstance(raw, (tuple, list)):
        raw = raw[0] if raw else ""
    match = re.search(r"\d+(?:\.\d+)?", str(raw))
    if match:
        speed = float(match.group())
        suffix = str(raw).lower()
        # This is a Chicago-only project. Bare OSM maxspeed uses US road convention.
        value = speed if "km" in suffix or "kph" in suffix else speed * KM_PER_MILE
        if 5 <= value <= 140:
            return value, "osm_maxspeed_assumed_us_mph_if_unspecified"
    category = parse_highway(edge.get("highway", ""))
    return DEFAULT_SPEED_KPH[category], "highway_class_heuristic"


def read_graph_edges(graphml):
    G = nx.read_graphml(graphml)
    if not G.is_directed():
        raise ValueError("GraphML must be directed; use OSMnx drive GraphML")
    edges = []
    if G.is_multigraph():
        iterator = ((u, v, k, d) for u, v, k, d in G.edges(keys=True, data=True))
    else:
        iterator = ((u, v, 0, d) for u, v, d in G.edges(data=True))
    for u, v, key, attr in iterator:
        try:
            length_m = float(attr.get("length", float("nan")))
        except (TypeError, ValueError):
            length_m = float("nan")
        if not np.isfinite(length_m) or length_m <= 0:
            raise ValueError(f"Missing/invalid OSM edge length: {(u,v,key)}")
        speed, source = static_speed(attr)
        edges.append({
            "osm_u": normalize_id(u), "osm_v": normalize_id(v), "osm_key": normalize_id(key),
            "length_m": length_m,
            "static_speed_kph": speed,
            "static_speed_source": source,
            "static_travel_sec": 3.6 * length_m / speed,
            "highway": parse_highway(attr.get("highway", "")),
        })
    if not edges:
        raise ValueError("Empty OSM graph")
    edge_df = pd.DataFrame(edges)
    if edge_df.duplicated(EDGE).any():
        raise ValueError("Duplicate OSM directed edge identifiers while loading GraphML")
    edge_df = edge_df.sort_values(EDGE).reset_index(drop=True)
    edge_df.insert(0, "edge_idx", np.arange(len(edge_df), dtype=np.int64))
    return G, edge_df


def build(args):
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    G, edges = read_graph_edges(args.graphml)
    mapping = pd.read_csv(args.mapping, dtype={k: "string" for k in EDGE + ["segment_id"]})
    needed = set(EDGE + ["segment_id", "path_fraction", "osm_length_m"])
    if not needed.issubset(mapping.columns):
        raise ValueError(f"Mapping missing columns: {sorted(needed-set(mapping.columns))}")
    for k in EDGE + ["segment_id"]:
        mapping[k] = mapping[k].map(normalize_id)
    mapping["path_fraction"] = pd.to_numeric(mapping["path_fraction"], errors="coerce")
    mapping["osm_length_m"] = pd.to_numeric(mapping["osm_length_m"], errors="coerce")
    if mapping.duplicated(EDGE).any():
        raise ValueError("Mapping contains repeated OSM directed edges: resolve collisions first")
    if mapping["path_fraction"].isna().any() or (mapping["path_fraction"] <= 0).any():
        raise ValueError("Missing or invalid path_fraction")
    fraction_sums = mapping.groupby("segment_id").path_fraction.sum()
    if not np.allclose(fraction_sums.to_numpy(), 1.0, atol=1e-4):
        raise ValueError("One or more segment path_fraction sums != 1")
    if mapping.empty:
        raise ValueError("Mapping has zero rows; inspect safe_mapping_summary.json")

    merged = mapping.merge(edges[EDGE + ["edge_idx", "length_m"]], on=EDGE,
                           how="left", validate="one_to_one", indicator=True)
    not_found = merged["_merge"] != "both"
    if not_found.any():
        examples = merged.loc[not_found, EDGE].head(8).to_dict("records")
        raise ValueError(f"{not_found.sum()} mapped OSM edge keys not in GraphML. Examples: {examples}. "
                         "Use the EXACT GraphML from the original matching step.")
    if not np.allclose(merged["osm_length_m"], merged["length_m"], rtol=.02, atol=2.0):
        failures = (~np.isclose(merged["osm_length_m"], merged["length_m"], rtol=.02, atol=2.0)).sum()
        raise ValueError(f"{failures} mapping edge lengths do not agree with GraphML: check versions and IDs")
    mapping = merged.drop(columns=["_merge", "length_m"])
    edges["mapped_segment_id"] = ""
    edges["mapped_path_fraction"] = np.nan
    for row in mapping.itertuples(index=False):
        ix = int(row.edge_idx)
        edges.at[ix, "mapped_segment_id"] = row.segment_id
        edges.at[ix, "mapped_path_fraction"] = row.path_fraction

    tdf = pd.read_csv(args.time_index)
    if "time_local" not in tdf.columns:
        raise ValueError("time_index.csv missing time_local")
    times = pd.DatetimeIndex(pd.to_datetime(tdf["time_local"], errors="raise"))
    if len(times) < 2 or not times.is_monotonic_increasing or times.has_duplicates:
        raise ValueError("time_index.csv timestamps must be unique, sorted, and at least 2 slots")
    if len(set(np.diff(times.view("int64")))) > 1:
        raise ValueError("Time index is not uniform; FIFO interp requires uniform steps")
    dt_sec = float((times[1]-times[0]).total_seconds())
    if dt_sec <= 0:
        raise ValueError("Invalid grid step")

    # Only read columns already standardized by prepare_chicago.py.
    obs = pd.read_csv(args.traffic_long,
                      usecols=["slot", "segment_id", "segment_travel_sec", "observed"],
                      dtype={"segment_id": "string"})
    obs["segment_id"] = obs["segment_id"].map(normalize_id)
    valid_bool = obs["observed"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
    obs["segment_travel_sec"] = pd.to_numeric(obs["segment_travel_sec"], errors="coerce")
    obs["slot"] = pd.to_datetime(obs["slot"], errors="coerce")
    use_segments = pd.Index(mapping["segment_id"].unique()).sort_values()
    obs = obs[obs["segment_id"].isin(set(use_segments)) & valid_bool &
              np.isfinite(obs["segment_travel_sec"]) & (obs["segment_travel_sec"] > 0)].copy()
    dup = obs.duplicated(["slot", "segment_id"])
    if dup.any():
        raise ValueError(f"Found {dup.sum()} duplicate valid segment/time records")
    pivot = obs.pivot(index="slot", columns="segment_id", values="segment_travel_sec")
    pivot = pivot.reindex(index=times, columns=use_segments)
    mon_sec = pivot.to_numpy(dtype=np.float32)

    num_t, num_e = len(times), len(edges)
    # The full graph is kept, including roads with no Chicago measurements.
    tt = np.broadcast_to(edges["static_travel_sec"].to_numpy(np.float32),
                         (num_t, num_e)).copy()
    observed_mask = np.zeros((num_t, num_e), dtype=bool)
    edge_to_seg_col = {k: i for i, k in enumerate(use_segments)}
    for row in mapping.itertuples(index=False):
        j = int(row.edge_idx)
        x = mon_sec[:, edge_to_seg_col[row.segment_id]]
        good = np.isfinite(x) & (x > 0)
        # Monitoring corridor travel time, not speed, is redistributed by length.
        tt[good, j] = x[good] * float(row.path_fraction)
        observed_mask[good, j] = True
    mapped_edge = edges["mapped_segment_id"].ne("").to_numpy(bool)
    if observed_mask[:, ~mapped_edge].any():
        raise AssertionError("Observed traffic erroneously assigned to unmonitored edges")
    if not np.isfinite(tt).all() or (tt <= 0).any():
        raise ValueError("Invalid raw travel times generated")

    # Sampled-grid FIFO correction: A_i = t_i + c_i must be nondecreasing.
    # TD shortest path must LINEARLY INTERPOLATE the arrival times between knots,
    # not use piecewise-constant costs or linearly interpolate raw speed.
    t_rel = np.arange(num_t, dtype=np.float64) * dt_sec
    raw_arrivals = tt.astype(np.float64) + t_rel[:, None]
    fifo_arrivals = np.maximum.accumulate(raw_arrivals, axis=0)
    tt_fifo = (fifo_arrivals - t_rel[:, None]).astype(np.float32)
    positive_adjustment = tt_fifo.astype(np.float64)-tt.astype(np.float64)
    non_fifo_count = int((positive_adjustment > 1e-3).sum())
    non_fifo_max_add = float(np.maximum(positive_adjustment, 0).max())

    edges.to_csv(out / "edge_index.csv", index=False)
    pd.DataFrame({"time_local": times.strftime("%Y-%m-%d %H:%M:%S")}).to_csv(
        out / "time_index.csv", index=False)
    np.savez_compressed(out / "edge_costs.npz", travel_sec_raw=tt,
                        travel_sec_fifo=tt_fifo,
                        chicago_observed_mask=observed_mask,
                        is_mapped_edge=mapped_edge,
                        slot_local_naive_ns=times.asi8,
                        slot_seconds=dt_sec)

    observed_per_slot = observed_mask.sum(axis=1)
    by_slot = pd.DataFrame({"time_local": times.strftime("%Y-%m-%d %H:%M:%S"),
                            "mapped_observed_edges": observed_per_slot,
                            "mapped_observation_ratio": observed_per_slot / max(int(mapped_edge.sum()), 1),
                            "all_graph_measured_edge_ratio": observed_per_slot / num_e})
    by_slot.to_csv(out / "observed_edge_coverage_by_slot.csv", index=False)
    daily = by_slot.assign(date=times.date.astype(str)).groupby("date").agg(
        mean_mapped_observation_ratio=("mapped_observation_ratio", "mean"),
        mean_graph_measured_edge_ratio=("all_graph_measured_edge_ratio", "mean"),
        slots=("time_local", "count"))
    daily.to_csv(out / "observed_edge_coverage_by_day.csv")
    report = {
        "osm_nodes": G.number_of_nodes(), "osm_directed_edges": num_e,
        "unique_monitored_corridors": len(use_segments),
        "candidate_mapped_directed_edges": int(mapped_edge.sum()),
        "fraction_graph_edges_candidate_mapped": float(mapped_edge.mean()),
        "time_slots": num_t, "step_minutes": dt_sec / 60,
        "observed_mapped_edge_cells": int(observed_mask.sum()),
        "fraction_observed_within_mapped_edge_grid": float(observed_mask[:, mapped_edge].mean()),
        "fraction_observed_within_entire_graph_grid": float(observed_mask.mean()),
        "time_slots_no_mapped_observations": int((observed_per_slot == 0).sum()),
        "fifo_grid_values_adjusted": non_fifo_count,
        "fifo_max_added_seconds": non_fifo_max_add,
        "cost_sources": {
            "chicago_observed_mask_true": "Chicago measured monitoring corridor travel time, allocated to candidate OSM path",
            "mapped_but_mask_false": "OSM static estimated fallback; NOT observed",
            "unmapped_edges": "OSM static estimated fallback; NOT observed"
        },
        "limits": [
            "Geometric matching is a candidate; validate direction and street shape against a map.",
            "Both dynamic tensors contain EX-POST historical labels; not usable as known future traffic in deployment.",
            "raw sampled travel time may violate FIFO; use fifo tensor with linear arrival interpolation for TDSP.",
            "OSM static speeds are assumptions/reference attributes, not verified traffic speeds.",
            "Source time kept local-like naive; verify clock interpretation.",
        ]
    }
    (out / "dynamic_cost_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("Saved:", out)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graphml", type=Path, required=True)
    p.add_argument("--mapping", type=Path, required=True)
    p.add_argument("--traffic-long", type=Path, required=True)
    p.add_argument("--time-index", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    build(p.parse_args())


if __name__ == "__main__":
    main()
