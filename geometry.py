"""
analysis/geometry.py

Core spatial geometry for hockey special teams analysis.

All functions operate on numpy arrays of player positions (n_players, 2).
Units: feet. Coordinate origin: BDC standard (0,0 = bottom-left defensive zone).

Key geometric concepts:
  - Convex hull: the smallest convex polygon containing all players.
    Area = team's spatial footprint. Shrinking hull on PK = good collapse.
  - Voronoi: partitions ice into regions owned by nearest player.
    Measures how much undefended space a PP creates.
  - Centroid velocity: physics metric for how fast a unit moves collectively.
  - Angular spread: how spread-out the unit is rotationally.
"""

import numpy as np
from scipy.spatial import ConvexHull, Voronoi, voronoi_plot_2d
from scipy.spatial.distance import cdist
from typing import Optional


# ── Convex Hull Metrics ────────────────────────────────────────────────────────

def hull_area(positions: np.ndarray) -> float:
    """
    Convex hull area of player positions (ft²).
    Larger = more spread-out formation.
    Returns 0.0 if fewer than 3 players or degenerate hull.
    """
    if len(positions) < 3:
        return 0.0
    try:
        return float(ConvexHull(positions).volume)  # 'volume' = area in 2D
    except Exception:
        return 0.0


def hull_perimeter(positions: np.ndarray) -> float:
    """Perimeter of convex hull (ft)."""
    if len(positions) < 3:
        return 0.0
    try:
        hull = ConvexHull(positions)
        verts = positions[hull.vertices]
        shifted = np.roll(verts, -1, axis=0)
        return float(np.sum(np.linalg.norm(shifted - verts, axis=1)))
    except Exception:
        return 0.0


def hull_vertices(positions: np.ndarray) -> np.ndarray:
    """Return hull vertices in order. Falls back to all positions."""
    if len(positions) < 3:
        return positions
    try:
        hull = ConvexHull(positions)
        return positions[hull.vertices]
    except Exception:
        return positions


# ── Centroid & Spread ──────────────────────────────────────────────────────────

def centroid(positions: np.ndarray) -> np.ndarray:
    """Mean (x, y) of all player positions."""
    return positions.mean(axis=0)


def spread_x(positions: np.ndarray) -> float:
    """Range of x coordinates (depth of formation)."""
    return float(positions[:, 0].max() - positions[:, 0].min())


def spread_y(positions: np.ndarray) -> float:
    """Range of y coordinates (width of formation)."""
    return float(positions[:, 1].max() - positions[:, 1].min())


def angular_spread(positions: np.ndarray) -> float:
    """
    Std dev of angles from centroid to each player (radians).
    High value = players spread around centroid (open formation).
    Low value = players clustered in one direction (tight/directional formation).
    """
    c = centroid(positions)
    angles = np.arctan2(positions[:, 1] - c[1], positions[:, 0] - c[0])
    return float(np.std(angles))


# ── Inter-Player Distance ──────────────────────────────────────────────────────

def pairwise_distances(positions: np.ndarray) -> np.ndarray:
    """(n, n) symmetric matrix of Euclidean distances (ft)."""
    return cdist(positions, positions)


def mean_pairwise_distance(positions: np.ndarray) -> float:
    """Mean distance between all player pairs (ft)."""
    D = pairwise_distances(positions)
    n = len(positions)
    if n < 2:
        return 0.0
    upper = D[np.triu_indices(n, k=1)]
    return float(upper.mean())


def mean_nearest_neighbor_distance(positions: np.ndarray) -> float:
    """Mean distance from each player to their nearest teammate (ft)."""
    D = pairwise_distances(positions)
    np.fill_diagonal(D, np.inf)
    return float(D.min(axis=1).mean())


def max_gap(positions: np.ndarray) -> float:
    """
    Largest gap between any two adjacent players (in the sorted-y ordering).
    Proxy for the biggest 'hole' in a defensive line.
    """
    sorted_y = np.sort(positions[:, 1])
    if len(sorted_y) < 2:
        return 0.0
    return float(np.diff(sorted_y).max())


# ── Voronoi Coverage ──────────────────────────────────────────────────────────

def voronoi_areas(positions: np.ndarray,
                   bounds: tuple[float, float, float, float],
                   n_samples: int = 5000) -> dict[int, float]:
    """
    Estimate each player's Voronoi region area (ft²) within the given bounds.
    Uses Monte Carlo sampling — fast enough for per-frame use.

    bounds: (x_min, x_max, y_min, y_max)
    Returns: {player_index: area_ft2}
    """
    x_min, x_max, y_min, y_max = bounds
    rng = np.random.default_rng(0)
    pts = rng.uniform(size=(n_samples, 2))
    pts[:, 0] = pts[:, 0] * (x_max - x_min) + x_min
    pts[:, 1] = pts[:, 1] * (y_max - y_min) + y_min

    # Assign each sample to nearest player
    D = cdist(pts, positions)
    owners = D.argmin(axis=1)

    total_area = (x_max - x_min) * (y_max - y_min)
    areas = {}
    for i in range(len(positions)):
        areas[i] = float((owners == i).mean() * total_area)
    return areas


def undefended_area(pp_positions: np.ndarray,
                     pk_positions: np.ndarray,
                     oz_bounds: tuple = (125, 200, 0, 85),
                     n_samples: int = 8000) -> float:
    """
    Estimate ice surface area (ft²) in the OZ not covered by any PK player,
    where 'covered' means within 10 feet of a PK skater.

    This quantifies how much open space the PP has created.
    """
    x_min, x_max, y_min, y_max = oz_bounds
    rng = np.random.default_rng(1)
    pts = rng.uniform(size=(n_samples, 2))
    pts[:, 0] = pts[:, 0] * (x_max - x_min) + x_min
    pts[:, 1] = pts[:, 1] * (y_max - y_min) + y_min

    # Distance from each sample point to nearest PK player
    D = cdist(pts, pk_positions)
    min_pk_dist = D.min(axis=1)

    COVERAGE_RADIUS_FT = 10.0
    uncovered = (min_pk_dist > COVERAGE_RADIUS_FT).mean()
    total_oz = (x_max - x_min) * (y_max - y_min)
    return float(uncovered * total_oz)


def coverage_grid(positions: np.ndarray,
                  bounds: tuple = (125, 200, 0, 85),
                  resolution: float = 1.0) -> np.ndarray:
    """
    Build a 2D grid of which player owns each grid cell (index of nearest player).
    Returns: 2D integer array (x_cells × y_cells)
    """
    x_min, x_max, y_min, y_max = bounds
    xs = np.arange(x_min, x_max + resolution, resolution)
    ys = np.arange(y_min, y_max + resolution, resolution)
    XX, YY = np.meshgrid(xs, ys)
    grid_pts = np.column_stack([XX.ravel(), YY.ravel()])
    D = cdist(grid_pts, positions)
    owners = D.argmin(axis=1).reshape(XX.shape)
    return owners


# ── Physics Metrics ────────────────────────────────────────────────────────────

def centroid_velocity(positions_t1: np.ndarray,
                       positions_t2: np.ndarray,
                       dt: float = 0.1) -> float:
    """
    Speed of centroid movement between two frames (ft/s).
    dt: time between frames (seconds). Default 0.1 = 10Hz.
    """
    c1 = centroid(positions_t1)
    c2 = centroid(positions_t2)
    dist = np.linalg.norm(c2 - c1)
    return float(dist / dt)


def hull_shrink_rate(area_t1: float, area_t2: float, dt: float = 0.1) -> float:
    """
    Rate of convex hull area change (ft²/s). Negative = shrinking (collapsing).
    A fast-collapsing PK unit has a large negative shrink rate after a zone entry.
    """
    return (area_t2 - area_t1) / dt


def compute_collapse_metrics(tracking_frames: list[np.ndarray],
                              dt: float = 0.1) -> dict:
    """
    Given a time-ordered list of PK player position arrays (each shape n×2),
    compute collapse physics metrics over the sequence.

    Returns dict:
      hull_areas: list of hull areas per frame
      centroid_xs: list of centroid x positions per frame
      centroid_velocities: list of centroid speeds per frame (ft/s)
      hull_shrink_rates: list of area change rates (ft²/s)
      total_hull_change: overall change in hull area
      mean_collapse_speed: mean centroid velocity (ft/s)
    """
    if len(tracking_frames) < 2:
        return {}

    areas = [hull_area(f) for f in tracking_frames]
    centroids = [centroid(f) for f in tracking_frames]
    cent_x = [float(c[0]) for c in centroids]

    velocities = [
        centroid_velocity(tracking_frames[i], tracking_frames[i+1], dt)
        for i in range(len(tracking_frames) - 1)
    ]
    shrink_rates = [
        hull_shrink_rate(areas[i], areas[i+1], dt)
        for i in range(len(areas) - 1)
    ]

    return {
        "hull_areas": areas,
        "centroid_xs": cent_x,
        "centroid_velocities": velocities,
        "hull_shrink_rates": shrink_rates,
        "total_hull_change": areas[-1] - areas[0],
        "mean_collapse_speed": float(np.mean(velocities)) if velocities else 0.0,
        "max_collapse_speed": float(np.max(velocities)) if velocities else 0.0,
    }


# ── Formation Feature Vector ───────────────────────────────────────────────────

def formation_features(positions: np.ndarray) -> dict[str, float]:
    """
    Extract full geometric feature vector for clustering.
    positions: (n_players, 2) — either PP or PK positions at one snapshot.
    """
    if len(positions) == 0:
        return {k: 0.0 for k in [
            "hull_area", "centroid_x", "centroid_y", "centroid_y_offset",
            "max_spread_x", "max_spread_y", "mean_nn_dist", "angular_spread",
            "mean_pairwise_dist", "max_gap_y",
        ]}

    c = centroid(positions)
    return {
        "hull_area":          hull_area(positions),
        "centroid_x":         float(c[0]),
        "centroid_y":         float(c[1]),
        "centroid_y_offset":  float(abs(c[1] - 42.5)),
        "max_spread_x":       spread_x(positions),
        "max_spread_y":       spread_y(positions),
        "mean_nn_dist":       mean_nearest_neighbor_distance(positions),
        "angular_spread":     angular_spread(positions),
        "mean_pairwise_dist": mean_pairwise_distance(positions),
        "max_gap_y":          max_gap(positions),
    }
