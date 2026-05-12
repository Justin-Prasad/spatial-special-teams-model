"""
utils/data_loader.py

Downloads and loads Big Data Cup event + tracking data.
All data is publicly available on GitHub — no API keys, no license restrictions.

Big Data Cup GitHub: https://github.com/bigdatacup
"""

import io
import logging
import time
from pathlib import Path

import pandas as pd
import numpy as np
import requests

from config import BDC_URLS, DATA_RAW, DATA_PROC

logger = logging.getLogger(__name__)


# ── Download ───────────────────────────────────────────────────────────────────

def _download(url: str, dest: Path, retries: int = 3) -> Path:
    """Download a URL to dest with retry logic. Returns dest path."""
    if dest.exists():
        logger.info(f"Cached: {dest.name}")
        return dest
    for attempt in range(retries):
        try:
            logger.info(f"Downloading {url} ...")
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            logger.info(f"Saved {dest.name} ({len(resp.content) / 1024:.0f} KB)")
            return dest
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            logger.warning(f"  Attempt {attempt+1} failed: {e}. Retrying in {wait}s...")
            time.sleep(wait)


def download_all() -> dict[str, Path]:
    """Download all Big Data Cup datasets. Returns {name: path}."""
    paths = {}
    for name, url in BDC_URLS.items():
        dest = DATA_RAW / f"{name}.csv"
        paths[name] = _download(url, dest)
    return paths


# ── Loading & Cleaning ─────────────────────────────────────────────────────────

def load_events(path: Path | None = None) -> pd.DataFrame:
    """
    Load and clean BDC event data.
    Standardizes column names, parses coordinates, adds derived fields.
    """
    if path is None:
        # Try both editions
        for name in ["bdc2021_events", "bdc2021_otters"]:
            p = DATA_RAW / f"{name}.csv"
            if p.exists():
                path = p
                break
    if path is None or not path.exists():
        raise FileNotFoundError(
            "BDC event data not found. Run: python pipeline.py --stage download"
        )

    df = pd.read_csv(path, low_memory=False)

    # Normalize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Parse coordinate columns (BDC uses 'x_coordinate', 'y_coordinate')
    coord_map = {
        "x_coordinate": "x",
        "y_coordinate": "y",
        "x_coordinate_2": "x2",    # destination coordinate for passes etc.
        "y_coordinate_2": "y2",
    }
    df = df.rename(columns={k: v for k, v in coord_map.items() if k in df.columns})

    # Ensure numeric coordinates
    for col in ["x", "y", "x2", "y2"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Parse period clock → seconds elapsed in period
    if "clock" in df.columns:
        df["period_seconds"] = df["clock"].apply(_parse_clock)

    # Classify zone
    if "x" in df.columns:
        df["zone"] = df["x"].apply(lambda v: _zone(v) if pd.notna(v) else None)

    # Special teams flag from skater counts
    if "home_team_skaters" in df.columns and "away_team_skaters" in df.columns:
        df = _add_strength_state(df)

    logger.info(f"Loaded {len(df):,} events from {path.name}")
    return df


def load_tracking(path: Path | None = None) -> pd.DataFrame:
    """
    Load BDC player tracking data (x/y at ~10Hz).
    Returns one row per player per frame.
    """
    if path is None:
        for name in ["bdc2021_tracking", "tracking"]:
            p = DATA_RAW / f"{name}.csv"
            if p.exists():
                path = p
                break
    if path is None or not path.exists():
        logger.warning("Tracking data not found — spatial analyses will use simulated data.")
        return pd.DataFrame()

    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    for col in ["x", "y"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    logger.info(f"Loaded {len(df):,} tracking rows")
    return df


# ── Special Teams Extraction ───────────────────────────────────────────────────

def extract_special_teams(events: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Split event data into PP and PK subsets.
    Returns {"pp": DataFrame, "pk": DataFrame, "ev": DataFrame}
    """
    if "strength_state" not in events.columns:
        logger.warning("No strength_state column — returning all events as EV")
        return {"ev": events, "pp": pd.DataFrame(), "pk": pd.DataFrame()}

    pp = events[events["strength_state"] == "PP"].copy()
    pk = events[events["strength_state"] == "PK"].copy()
    ev = events[events["strength_state"] == "5v5"].copy()

    logger.info(f"Events — 5v5: {len(ev):,} | PP: {len(pp):,} | PK: {len(pk):,}")
    return {"pp": pp, "pk": pk, "ev": ev}


def extract_pp_sequences(events: pd.DataFrame,
                          min_events: int = 5) -> list[pd.DataFrame]:
    """
    Extract individual power play sequences as a list of DataFrames.
    A new PP starts when strength state changes to PP.
    min_events: skip very short sequences (likely errors).
    """
    if "strength_state" not in events.columns:
        return []

    sequences = []
    current = []
    prev_state = None

    for _, row in events.iterrows():
        state = row.get("strength_state")
        if state == "PP":
            current.append(row)
        else:
            if current and len(current) >= min_events:
                sequences.append(pd.DataFrame(current))
            current = []
        prev_state = state

    if current and len(current) >= min_events:
        sequences.append(pd.DataFrame(current))

    logger.info(f"Extracted {len(sequences)} PP sequences")
    return sequences


# ── Synthetic Data (when tracking not available) ───────────────────────────────

def simulate_pp_tracking(n_frames: int = 300,
                          formation: str = "umbrella",
                          seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic player tracking data for a power play.
    Used for development and dashboard demos when BDC tracking is unavailable.

    Formations: "umbrella", "overload", "1-3-1"
    Returns DataFrame with columns: frame, player_id, team, x, y, role
    """
    rng = np.random.default_rng(seed)

    # Base positions for each formation (5 PP players, OZ perspective)
    formations = {
        "umbrella": {
            "positions": np.array([
                [170, 42.5],   # high slot
                [155, 25.0],   # left half-wall
                [155, 60.0],   # right half-wall
                [140, 15.0],   # left point
                [140, 70.0],   # right point
            ]),
            "roles": ["high_slot", "left_wing", "right_wing", "left_point", "right_point"],
        },
        "overload": {
            "positions": np.array([
                [175, 30.0],   # net-front
                [165, 20.0],   # low left
                [165, 40.0],   # mid left
                [148, 20.0],   # left point
                [145, 55.0],   # right point
            ]),
            "roles": ["net_front", "low_left", "mid_left", "left_point", "right_point"],
        },
        "1-3-1": {
            "positions": np.array([
                [185, 42.5],   # net-front
                [165, 20.0],   # left wing
                [165, 42.5],   # center
                [165, 65.0],   # right wing
                [140, 42.5],   # point
            ]),
            "roles": ["net_front", "left_wing", "center", "right_wing", "point"],
        },
    }

    # PK positions (4 players, box formation)
    pk_positions_base = np.array([
        [175, 30.0],   # low left
        [175, 55.0],   # low right
        [158, 30.0],   # high left
        [158, 55.0],   # high right
    ])
    pk_roles = ["low_left", "low_right", "high_left", "high_right"]

    fm = formations.get(formation, formations["umbrella"])
    pp_base = fm["positions"]
    pp_roles = fm["roles"]

    rows = []
    for frame in range(n_frames):
        t = frame / 10.0  # seconds

        # PP players: drift with small random walk + sinusoidal pattern movement
        for i, (base, role) in enumerate(zip(pp_base, pp_roles)):
            drift_x = rng.normal(0, 0.3)
            drift_y = rng.normal(0, 0.3)
            wobble_y = 1.5 * np.sin(t * 0.8 + i * 1.2)
            x = float(np.clip(base[0] + drift_x, 125, 198))
            y = float(np.clip(base[1] + drift_y + wobble_y, 3, 82))
            rows.append({
                "frame": frame, "player_id": f"PP{i+1}", "team": "PP",
                "x": x, "y": y, "role": role,
            })

        # PK players: slightly tighter random walk, collapse toward net on odd frames
        collapse = 0.05 if frame % 20 < 5 else 0.0
        for i, (base, role) in enumerate(zip(pk_positions_base, pk_roles)):
            x = float(np.clip(base[0] + rng.normal(0, 0.4) + collapse * (base[0] - 180), 125, 198))
            y = float(np.clip(base[1] + rng.normal(0, 0.4) + collapse * (base[1] - 42.5) * 0.3, 3, 82))
            rows.append({
                "frame": frame, "player_id": f"PK{i+1}", "team": "PK",
                "x": x, "y": y, "role": role,
            })

    return pd.DataFrame(rows)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_clock(clock_str: str) -> float:
    """Convert 'MM:SS' string to seconds elapsed (from 20:00)."""
    try:
        parts = str(clock_str).split(":")
        minutes, seconds = int(parts[0]), int(parts[1])
        return (20 * 60) - (minutes * 60 + seconds)
    except Exception:
        return float("nan")


def _zone(x: float) -> str:
    if x < 75:
        return "DZ"
    if x < 125:
        return "NZ"
    return "OZ"


def _add_strength_state(df: pd.DataFrame) -> pd.DataFrame:
    """Infer strength state from home/away skater counts."""
    home = pd.to_numeric(df["home_team_skaters"], errors="coerce").fillna(5)
    away = pd.to_numeric(df["away_team_skaters"], errors="coerce").fillna(5)

    def state(h, a):
        if h == a:
            return "5v5" if h >= 5 else "4v4"
        if h > a:
            return "PP"   # home team on PP
        return "PK"       # home team on PK

    df["strength_state"] = [state(h, a) for h, a in zip(home, away)]
    return df
