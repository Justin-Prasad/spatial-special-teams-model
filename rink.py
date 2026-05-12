"""
utils/rink.py

Rink geometry: coordinate helpers, zone classification,
matplotlib rink drawing, and BDC coordinate normalization.

BDC coordinate system:
  (0, 0) = bottom-left corner of the defensive zone (from eventing team's POV)
  x increases toward the offensive zone (up ice)
  y increases from left board to right board
  Rink: x ∈ [0, 200], y ∈ [0, 85]
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Arc, Circle, FancyArrowPatch
from matplotlib.lines import Line2D
from config import (
    RINK_LENGTH, RINK_WIDTH, BLUE_LINE_X, GOAL_LINE_X,
    NET_X, NET_Y_TOP, NET_Y_BOT, CREASE_RADIUS,
)


# ── Coordinate Helpers ─────────────────────────────────────────────────────────

def flip_to_offensive(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalize coordinates so all events are in the offensive zone
    (x > 100). Flips events where x < 100.
    Returns (x_norm, y_norm).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = x < 100
    x[mask] = RINK_LENGTH - x[mask]
    y[mask] = RINK_WIDTH  - y[mask]
    return x, y


def distance(x1, y1, x2, y2) -> float:
    """Euclidean distance in feet."""
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def angle_to_goal(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Angle (degrees) from position (x, y) to goal center (189, 42.5).
    0 = straight on, 90 = parallel to goal line.
    """
    dx = NET_X - x
    dy = 42.5 - y
    return np.degrees(np.arctan2(np.abs(dy), np.abs(dx)))


def in_offensive_zone(x: np.ndarray) -> np.ndarray:
    """Boolean mask: True if x is in offensive zone (past blue line)."""
    return np.asarray(x) > BLUE_LINE_X


def zone_classify(x: float) -> str:
    """Classify x coordinate into zone string."""
    if x < 75:
        return "DZ"
    if x < 125:
        return "NZ"
    return "OZ"


# ── Inter-Player Geometry ──────────────────────────────────────────────────────

def pairwise_distances(positions: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Euclidean distances for a set of player positions.
    positions: (n_players, 2) array of (x, y)
    Returns: (n_players, n_players) symmetric distance matrix
    """
    n = len(positions)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = distance(positions[i, 0], positions[i, 1],
                         positions[j, 0], positions[j, 1])
            D[i, j] = D[j, i] = d
    return D


def nearest_neighbor_distances(positions: np.ndarray) -> np.ndarray:
    """
    For each player, distance to their nearest teammate.
    positions: (n_players, 2)
    Returns: (n_players,) array
    """
    D = pairwise_distances(positions)
    np.fill_diagonal(D, np.inf)
    return D.min(axis=1)


def centroid(positions: np.ndarray) -> np.ndarray:
    """Mean position of a set of players. Returns (x, y)."""
    return positions.mean(axis=0)


def angular_spread(positions: np.ndarray) -> float:
    """
    Standard deviation of angles from centroid to each player.
    Measures how 'spread' the unit is angularly — high = wide formation.
    """
    c = centroid(positions)
    angles = np.arctan2(positions[:, 1] - c[1], positions[:, 0] - c[0])
    return float(np.std(angles))


# ── Convex Hull ────────────────────────────────────────────────────────────────

def convex_hull_area(positions: np.ndarray) -> float:
    """
    Compute convex hull area (ft²) for a set of player positions.
    Uses Shoelace formula on the scipy hull.
    Returns 0 if fewer than 3 players.
    """
    if len(positions) < 3:
        return 0.0
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(positions)
        return float(hull.volume)   # 'volume' is area in 2D
    except Exception:
        return 0.0


def convex_hull_vertices(positions: np.ndarray) -> np.ndarray:
    """
    Return the vertices of the convex hull in order.
    Returns positions unchanged if hull fails.
    """
    if len(positions) < 3:
        return positions
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(positions)
        return positions[hull.vertices]
    except Exception:
        return positions


# ── Rink Drawing ───────────────────────────────────────────────────────────────

def draw_rink(ax: plt.Axes, half: str = "offensive", alpha: float = 0.6) -> None:
    """
    Draw a simplified rink on the given axes.
    half: "offensive" (x=100..200), "full" (x=0..200), or "defensive"
    """
    ax.set_facecolor("#e8f4f8")

    if half == "offensive":
        x0, x1 = 100, 200
    elif half == "defensive":
        x0, x1 = 0, 100
    else:
        x0, x1 = 0, 200

    # Boards
    boards = mpatches.FancyBboxPatch(
        (x0, 0), x1 - x0, RINK_WIDTH,
        boxstyle="round,pad=2",
        linewidth=1.5, edgecolor="#333", facecolor="#eaf4fb", alpha=alpha, zorder=0,
    )
    ax.add_patch(boards)

    # Center ice / blue lines
    if half == "full":
        ax.axvline(100, color="#cc0000", lw=1.5, alpha=0.5, zorder=1)   # center red line
    ax.axvline(BLUE_LINE_X, color="#2255cc", lw=2, alpha=0.6, zorder=1)  # offensive blue line

    # Goal line
    ax.axvline(GOAL_LINE_X, color="#cc0000", lw=1.2, alpha=0.5, zorder=1)

    # Crease
    crease = mpatches.Wedge(
        (GOAL_LINE_X, RINK_WIDTH / 2), CREASE_RADIUS,
        theta1=90, theta2=270,
        facecolor="#93c5fd", edgecolor="#2255cc", lw=1, alpha=0.4, zorder=1
    )
    ax.add_patch(crease)

    # Net
    net = mpatches.Rectangle(
        (GOAL_LINE_X, NET_Y_BOT), 2, NET_Y_TOP - NET_Y_BOT,
        facecolor="#666", edgecolor="#333", lw=1.5, alpha=0.8, zorder=2
    )
    ax.add_patch(net)

    # Faceoff circles (offensive zone)
    for fy in [20.5, 64.5]:
        circle = Circle((169, fy), 15, fill=False, edgecolor="#cc0000",
                         lw=1, alpha=0.3, zorder=1)
        ax.add_patch(circle)
        ax.plot(169, fy, "o", color="#cc0000", ms=3, alpha=0.4, zorder=1)

    ax.set_xlim(x0 - 3, x1 + 3)
    ax.set_ylim(-3, RINK_WIDTH + 3)
    ax.set_aspect("equal")
    ax.axis("off")
