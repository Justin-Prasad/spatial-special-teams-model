"""
config.py — Central configuration for Hockey Spatial Analysis.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent
DATA_RAW   = ROOT / "data" / "raw"
DATA_PROC  = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models" / "saved"
REPORT_DIR = ROOT / "reports" / "output"

for d in [DATA_RAW, DATA_PROC, MODELS_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Big Data Cup Data URLs (public GitHub releases) ───────────────────────────
BDC_URLS = {
    # 2021 edition: Erie Otters (OHL) + women's Olympic/NCAA data
    "bdc2021_events": (
        "https://raw.githubusercontent.com/bigdatacup/Big-Data-Cup-2021/"
        "main/hackathon_womens_data.csv"
    ),
    "bdc2021_otters": (
        "https://raw.githubusercontent.com/bigdatacup/Big-Data-Cup-2021/"
        "main/hackathon_nwhl.csv"
    ),
}

# ── Rink Dimensions (feet) ─────────────────────────────────────────────────────
RINK_LENGTH   = 200.0
RINK_WIDTH    = 85.0
RINK_HALF     = RINK_LENGTH / 2          # 100 ft — center ice x
GOAL_LINE_X   = 189.0                    # distance from own goal line
BLUE_LINE_X   = 125.0                    # offensive blue line from own end
NET_X         = 189.0
NET_Y_TOP     = 44.66
NET_Y_BOT     = 40.33
CREASE_RADIUS = 6.0                      # ft

# BDC coordinate system: (0,0) = bottom-left of defensive zone (from eventing team POV)
# x increases up-ice (toward offensive zone), y increases from left to right boards
BDC_LENGTH = 200.0
BDC_WIDTH  = 85.0

# ── Special Teams ──────────────────────────────────────────────────────────────
PP_SKATER_COUNTS  = (5, 4)   # (advantage_team, pk_team) skaters
PK_SKATER_COUNTS  = (4, 5)

# Minimum tracking frames for a valid special teams segment
MIN_PP_FRAMES = 30    # ~3 seconds at 10Hz

# ── Formation Clustering ───────────────────────────────────────────────────────
N_FORMATIONS      = 5       # K-Means k
DBSCAN_EPS        = 3.5     # feet — spatial epsilon for DBSCAN
DBSCAN_MIN_SAMPLES = 8

# Geometric feature set used for clustering
FORMATION_FEATURES = [
    "hull_area",           # convex hull area (ft²)
    "centroid_x",          # mean x position (ft)
    "centroid_y",          # mean y position (ft)
    "centroid_y_offset",   # |centroid_y - 42.5| — distance from center line
    "max_spread_x",        # max - min x across players
    "max_spread_y",        # max - min y across players
    "mean_nn_dist",        # mean nearest-neighbor distance
    "angular_spread",      # std of angles from centroid to players
]

# ── Coverage / Voronoi ─────────────────────────────────────────────────────────
VORONOI_GRID_RES  = 1.0     # ft — resolution for coverage grid
OZ_X_START        = 125.0   # offensive zone starts at blue line
OZ_X_END          = 200.0

# ── Collapse Physics ──────────────────────────────────────────────────────────
COLLAPSE_WINDOW_S  = 5.0    # seconds after zone entry to measure collapse
TRACKING_HZ        = 10     # frames per second in BDC tracking data
COLLAPSE_FRAMES    = int(COLLAPSE_WINDOW_S * TRACKING_HZ)

# ── Reporting ─────────────────────────────────────────────────────────────────
REPORT_TOP_N = 10
