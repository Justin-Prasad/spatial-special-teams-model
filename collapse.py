"""
analysis/collapse.py

PK defensive collapse speed analysis.

After a PP zone entry, how quickly does the PK unit contract its shape?
This is a pure physics / kinematics problem:

  - centroid velocity: how fast the PK unit's center of mass moves toward net (ft/s)
  - hull shrink rate: how quickly the defensive coverage area contracts (ft²/s)
  - angular compression: how quickly the unit closes its angular spread
  - individual player velocity: which PK players are slowest to respond

Good PK units collapse quickly and predictably.
Slow collapse = PP has time to move the puck to the slot unopposed.
"""

import logging

import numpy as np
import pandas as pd

from analysis.geometry import (
    hull_area, centroid, angular_spread,
    compute_collapse_metrics, mean_nearest_neighbor_distance,
)
from config import COLLAPSE_FRAMES, TRACKING_HZ

logger = logging.getLogger(__name__)


# ── Zone Entry Detection ───────────────────────────────────────────────────────

def detect_zone_entries(events: pd.DataFrame,
                         team_col: str = "team",
                         event_type_col: str = "event",
                         zone_entry_label: str = "Zone Entry") -> pd.DataFrame:
    """
    Find all zone entry events in an event DataFrame.
    Returns subset of events that are zone entries, with frame index if available.
    """
    if event_type_col not in events.columns:
        return pd.DataFrame()
    entries = events[events[event_type_col].str.contains(zone_entry_label, na=False, case=False)]
    logger.info(f"Found {len(entries)} zone entries")
    return entries.copy()


# ── Collapse Window Extraction ─────────────────────────────────────────────────

def extract_collapse_window(tracking: pd.DataFrame,
                              entry_frame: int,
                              pk_team: str = "PK",
                              window_frames: int = COLLAPSE_FRAMES) -> list[np.ndarray]:
    """
    Extract PK player position arrays for `window_frames` frames following a zone entry.

    Returns list of (n_pk, 2) arrays, one per frame.
    Empty list if insufficient data.
    """
    end_frame = entry_frame + window_frames
    pk_frames = tracking[
        (tracking["team"] == pk_team) &
        (tracking["frame"] >= entry_frame) &
        (tracking["frame"] <= end_frame)
    ]

    frame_ids = sorted(pk_frames["frame"].unique())
    arrays = []
    for fid in frame_ids:
        pos = pk_frames[pk_frames["frame"] == fid][["x", "y"]].dropna().to_numpy()
        if len(pos) >= 2:
            arrays.append(pos)

    return arrays


# ── Collapse Metrics ───────────────────────────────────────────────────────────

def collapse_metrics_from_tracking(tracking: pd.DataFrame,
                                    entry_frame: int,
                                    pk_team: str = "PK") -> dict:
    """
    Full collapse analysis starting from a specific zone entry frame.
    Combines geometry.compute_collapse_metrics with hockey-specific interpretation.
    """
    frames = extract_collapse_window(tracking, entry_frame, pk_team)
    if len(frames) < 3:
        return {}

    dt = 1.0 / TRACKING_HZ
    base = compute_collapse_metrics(frames, dt=dt)

    # Additional hockey-specific metrics
    initial_area   = base["hull_areas"][0] if base["hull_areas"] else 0
    final_area     = base["hull_areas"][-1] if base["hull_areas"] else 0
    initial_cent_x = base["centroid_xs"][0] if base["centroid_xs"] else 0
    final_cent_x   = base["centroid_xs"][-1] if base["centroid_xs"] else 0

    # Initial/final angular spread
    init_ang = angular_spread(frames[0])  if frames else 0
    final_ang = angular_spread(frames[-1]) if frames else 0

    window_s = len(frames) * dt
    base.update({
        "initial_hull_area_ft2":      round(initial_area, 1),
        "final_hull_area_ft2":        round(final_area, 1),
        "hull_reduction_pct":         round((initial_area - final_area) / max(initial_area, 1) * 100, 1),
        "centroid_x_advance_ft":      round(initial_cent_x - final_cent_x, 2),   # negative = moved toward net
        "initial_angular_spread":     round(init_ang, 3),
        "final_angular_spread":       round(final_ang, 3),
        "angular_compression":        round(init_ang - final_ang, 3),             # positive = compressed
        "window_seconds":             round(window_s, 1),
        "collapse_grade":             _grade_collapse(base["mean_collapse_speed"], initial_area - final_area),
    })
    return base


def _grade_collapse(mean_speed: float, area_reduction: float) -> str:
    """
    Simple qualitative grading of PK collapse quality.
    For coaching staff reports.
    """
    if mean_speed > 6 and area_reduction > 200:
        return "A — Excellent collapse"
    if mean_speed > 4 and area_reduction > 100:
        return "B — Good collapse"
    if mean_speed > 2:
        return "C — Adequate"
    return "D — Slow collapse"


# ── Individual Player Velocity ─────────────────────────────────────────────────

def individual_player_velocities(tracking: pd.DataFrame,
                                   entry_frame: int,
                                   pk_team: str = "PK",
                                   window_frames: int = 20) -> pd.DataFrame:
    """
    Compute per-player average speed (ft/s) in the collapse window after a zone entry.
    Identifies the slowest responders on the PK unit.

    Returns DataFrame: player_id, mean_speed_ft_s, max_speed_ft_s, dist_traveled_ft
    """
    end_frame = entry_frame + window_frames
    dt = 1.0 / TRACKING_HZ
    pk = tracking[
        (tracking["team"] == pk_team) &
        (tracking["frame"] >= entry_frame) &
        (tracking["frame"] <= end_frame)
    ].sort_values(["player_id", "frame"])

    results = []
    for pid, player_df in pk.groupby("player_id"):
        if len(player_df) < 2:
            continue
        xs = player_df["x"].to_numpy()
        ys = player_df["y"].to_numpy()
        dists = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
        speeds = dists / dt
        results.append({
            "player_id":       pid,
            "mean_speed_ft_s": round(float(speeds.mean()), 2),
            "max_speed_ft_s":  round(float(speeds.max()), 2),
            "dist_traveled_ft":round(float(dists.sum()), 1),
        })

    return pd.DataFrame(results).sort_values("mean_speed_ft_s")


# ── Multi-Sequence Collapse Summary ───────────────────────────────────────────

def summarize_pk_collapses(tracking: pd.DataFrame,
                             entry_frames: list[int],
                             pk_team: str = "PK") -> pd.DataFrame:
    """
    Run collapse analysis for multiple zone entries and aggregate.
    entry_frames: list of frame indices where zone entries occurred.
    Returns DataFrame with one row per zone entry + metrics.
    """
    rows = []
    for i, ef in enumerate(entry_frames):
        m = collapse_metrics_from_tracking(tracking, ef, pk_team)
        if m:
            m["entry_idx"] = i
            m["entry_frame"] = ef
            # Flatten list columns to scalars for summary
            m["final_hull_shrink_rate"] = m["hull_shrink_rates"][-1] if m.get("hull_shrink_rates") else 0
            for k in ["hull_areas", "centroid_xs", "centroid_velocities", "hull_shrink_rates"]:
                m.pop(k, None)
            rows.append(m)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ── Simulated Collapse (for demo / unit tests) ─────────────────────────────────

def simulate_pk_collapse(formation: str = "box",
                          collapse_speed: str = "fast",
                          n_frames: int = 50,
                          seed: int = 0) -> list[np.ndarray]:
    """
    Generate synthetic PK position sequences simulating a collapse.
    Used for dashboard demos and unit testing without real tracking data.

    formation: "box" | "diamond"
    collapse_speed: "fast" | "slow"
    """
    rng = np.random.default_rng(seed)

    formations = {
        "box": np.array([[165,25],[165,60],[155,32],[155,53]], dtype=float),
        "diamond": np.array([[175,42],[160,28],[160,57],[148,42]], dtype=float),
    }
    base = formations.get(formation, formations["box"])
    target_x = 180.0   # collapse toward net
    speed_factor = 0.6 if collapse_speed == "fast" else 0.15

    frames = []
    positions = base.copy()
    for f in range(n_frames):
        noise = rng.normal(0, 0.2, size=positions.shape)
        # Move toward net (higher x)
        dx = (target_x - positions[:, 0]) * speed_factor * (1 / n_frames)
        # Compress y toward center (42.5)
        dy = (42.5 - positions[:, 1]) * speed_factor * 0.5 * (1 / n_frames)
        positions = positions + np.column_stack([dx, dy]) + noise
        frames.append(positions.copy())

    return frames
