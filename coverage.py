"""
analysis/coverage.py

Voronoi-based coverage gap analysis for power play / penalty kill.

Key insight: on a power play, the PP unit tries to create large undefended
regions (Voronoi cells with no PK player nearby). The size of these gaps
directly relates to passing lane quality and scoring chance generation.

Metrics:
  - pk_voronoi_areas: area each PK player is responsible to cover (ft²)
  - undefended_area: OZ ice with no PK player within coverage radius
  - max_pk_cell: size of the biggest PK coverage responsibility
  - coverage_balance: std dev of PK cell sizes (high = unbalanced / exploitable)
  - pp_space_generated: area of OZ "owned" by PP players
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from geometry import (
    undefended_area, voronoi_areas, coverage_grid,
    centroid, hull_area, formation_features,
)
from config import OZ_X_START, OZ_X_END, RINK_WIDTH

logger = logging.getLogger(__name__)

OZ_BOUNDS = (OZ_X_START, OZ_X_END, 0.0, RINK_WIDTH)
OZ_AREA   = (OZ_X_END - OZ_X_START) * RINK_WIDTH   # 75 × 85 = 6375 ft²


# ── Per-Frame Coverage Metrics ─────────────────────────────────────────────────

def frame_coverage(pp_positions: np.ndarray,
                    pk_positions: np.ndarray,
                    coverage_radius: float = 10.0) -> dict:
    """
    Compute coverage metrics for a single tracking frame.

    pp_positions: (n_pp, 2) — PP player positions
    pk_positions: (n_pk, 2) — PK player positions
    coverage_radius: distance within which a PK player 'covers' a point (ft)

    Returns dict of scalar metrics.
    """
    # Voronoi areas for PK players in OZ
    pk_oz = pk_positions[pk_positions[:, 0] >= OZ_X_START]
    if len(pk_oz) == 0:
        return _empty_coverage()

    pk_areas = voronoi_areas(pk_oz, OZ_BOUNDS, n_samples=2000)
    pk_area_vals = list(pk_areas.values())

    # Undefended space
    undef = undefended_area(pp_positions, pk_positions, OZ_BOUNDS, n_samples=3000)

    # PP Voronoi areas
    pp_oz = pp_positions[pp_positions[:, 0] >= OZ_X_START]
    pp_areas = voronoi_areas(pp_oz, OZ_BOUNDS, n_samples=2000) if len(pp_oz) > 0 else {}
    pp_area_sum = sum(pp_areas.values())

    return {
        "undefended_area_ft2": undef,
        "undefended_pct":      undef / OZ_AREA,
        "max_pk_cell_ft2":     max(pk_area_vals) if pk_area_vals else 0.0,
        "mean_pk_cell_ft2":    float(np.mean(pk_area_vals)) if pk_area_vals else 0.0,
        "pk_coverage_balance": float(np.std(pk_area_vals)) if len(pk_area_vals) > 1 else 0.0,
        "pp_space_ft2":        pp_area_sum,
        "n_pk_in_oz":          len(pk_oz),
    }


def _empty_coverage() -> dict:
    return {
        "undefended_area_ft2": OZ_AREA,
        "undefended_pct":      1.0,
        "max_pk_cell_ft2":     0.0,
        "mean_pk_cell_ft2":    0.0,
        "pk_coverage_balance": 0.0,
        "pp_space_ft2":        OZ_AREA,
        "n_pk_in_oz":          0,
    }


# ── Sequence-Level Coverage Summary ───────────────────────────────────────────

def coverage_over_time(tracking: pd.DataFrame,
                        pp_team: str = "PP",
                        pk_team: str = "PK",
                        sample_every_n: int = 3) -> pd.DataFrame:
    """
    Compute frame-by-frame coverage metrics over a PP possession.
    tracking: DataFrame with frame, team, x, y columns.
    Returns DataFrame with one row per sampled frame + all coverage metrics.
    """
    frame_ids = sorted(tracking["frame"].unique())
    rows = []

    for i, fid in enumerate(frame_ids):
        if i % sample_every_n != 0:
            continue
        frame_data = tracking[tracking["frame"] == fid]
        pp_pos = frame_data[frame_data["team"] == pp_team][["x", "y"]].dropna().to_numpy()
        pk_pos = frame_data[frame_data["team"] == pk_team][["x", "y"]].dropna().to_numpy()

        if len(pp_pos) < 2 or len(pk_pos) < 1:
            continue

        metrics = frame_coverage(pp_pos, pk_pos)
        metrics["frame"] = fid
        metrics["t_seconds"] = fid / 10.0
        rows.append(metrics)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def pp_coverage_summary(tracking: pd.DataFrame,
                         pp_team: str = "PP",
                         pk_team: str = "PK") -> dict:
    """
    Summary statistics of coverage over an entire PP sequence.
    Returns dict of mean/max/min metrics + interpretation flags.
    """
    cov = coverage_over_time(tracking, pp_team, pk_team)
    if cov.empty:
        return {}

    summary = {
        "mean_undefended_ft2":     cov["undefended_area_ft2"].mean(),
        "max_undefended_ft2":      cov["undefended_area_ft2"].max(),
        "mean_undefended_pct":     cov["undefended_pct"].mean(),
        "mean_pk_balance":         cov["pk_coverage_balance"].mean(),
        "max_pk_cell_ft2":         cov["max_pk_cell_ft2"].max(),
        "mean_pp_space_ft2":       cov["pp_space_ft2"].mean(),
        # Flags for coaching staff
        "pk_overloaded_one_side":  bool(cov["pk_coverage_balance"].mean() > 200),
        "large_undefended_slot":   bool(cov["undefended_area_ft2"].mean() > 2000),
    }
    return summary


# ── Passing Lane Analysis ──────────────────────────────────────────────────────

def passing_lane_quality(from_pos: np.ndarray,
                          to_pos: np.ndarray,
                          pk_positions: np.ndarray,
                          lane_width: float = 5.0) -> dict:
    """
    Evaluate quality of a potential passing lane from one PP player to another.

    Computes:
      - lane_distance: length of pass (ft)
      - min_defender_proximity: closest PK player to the lane centerline (ft)
      - lane_open: True if no PK player within lane_width feet of center line
      - shot_angle_improvement: change in angle to net if pass is completed

    from_pos, to_pos: (2,) arrays
    pk_positions: (n_pk, 2) array
    """
    vec = to_pos - from_pos
    lane_len = float(np.linalg.norm(vec))
    if lane_len < 0.1:
        return {"lane_distance": 0, "min_defender_proximity": 99, "lane_open": True}

    unit = vec / lane_len

    # Project each PK player onto the lane, measure perpendicular distance
    proximities = []
    for pk in pk_positions:
        to_pk = pk - from_pos
        proj = np.dot(to_pk, unit)
        if 0 <= proj <= lane_len:
            perp = np.linalg.norm(to_pk - proj * unit)
            proximities.append(perp)

    min_prox = min(proximities) if proximities else 99.0

    # Shot angle: angle to goal from each position
    goal = np.array([189.0, 42.5])
    angle_from = float(np.degrees(np.arctan2(
        abs(from_pos[1] - 42.5), abs(goal[0] - from_pos[0])
    )))
    angle_to = float(np.degrees(np.arctan2(
        abs(to_pos[1] - 42.5), abs(goal[0] - to_pos[0])
    )))

    return {
        "lane_distance":            round(lane_len, 1),
        "min_defender_proximity":   round(min_prox, 1),
        "lane_open":                min_prox > lane_width,
        "from_angle_to_net":        round(angle_from, 1),
        "to_angle_to_net":          round(angle_to, 1),
        "shot_angle_improvement":   round(angle_from - angle_to, 1),  # positive = better
    }


def all_pp_passing_lanes(pp_positions: np.ndarray,
                           pk_positions: np.ndarray) -> pd.DataFrame:
    """
    Evaluate all possible PP-to-PP passing lanes at once.
    Returns DataFrame sorted by lane quality (open lanes with best shot angle improvement).
    """
    rows = []
    n = len(pp_positions)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            result = passing_lane_quality(pp_positions[i], pp_positions[j], pk_positions)
            result["from_player"] = i
            result["to_player"] = j
            rows.append(result)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df.sort_values("shot_angle_improvement", ascending=False).reset_index(drop=True)
