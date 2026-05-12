"""
pipeline.py — End-to-end orchestration for Hockey Spatial Analysis.

Usage:
    python pipeline.py --stage download
    python pipeline.py --stage features
    python pipeline.py --stage analyze
    python pipeline.py --stage report
    python pipeline.py --all
"""

import argparse
import logging
import sys

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("pipeline.log")],
)
logger = logging.getLogger("pipeline")

from config import DATA_RAW, DATA_PROC, MODELS_DIR


# ── Stage: Download ────────────────────────────────────────────────────────────

def stage_download() -> None:
    from utils.data_loader import download_all
    paths = download_all()
    logger.info(f"Downloaded {len(paths)} datasets")
    for name, path in paths.items():
        logger.info(f"  {name}: {path}")


# ── Stage: Features ────────────────────────────────────────────────────────────

def stage_features() -> None:
    from utils.data_loader import load_events, extract_special_teams
    from analysis.geometry import formation_features
    from utils.rink import flip_to_offensive

    try:
        events = load_events()
    except FileNotFoundError:
        logger.warning("No event data found. Run --stage download first.")
        logger.info("Using synthetic data for feature demonstration.")
        _demo_features()
        return

    # Normalize coordinates to offensive zone perspective
    if "x" in events.columns and "y" in events.columns:
        x, y = flip_to_offensive(events["x"].to_numpy(), events["y"].to_numpy())
        events["x_norm"] = x
        events["y_norm"] = y

    st = extract_special_teams(events)

    # Save processed subsets
    for name, df in st.items():
        if not df.empty:
            out = DATA_PROC / f"events_{name}.parquet"
            df.to_parquet(out, index=False)
            logger.info(f"Saved {name}: {len(df):,} events → {out}")

    logger.info("Feature stage complete")


def _demo_features() -> None:
    """Run feature engineering on synthetic data for demo purposes."""
    from utils.data_loader import simulate_pp_tracking
    from analysis.formations import extract_formation_snapshots

    for formation in ["umbrella", "overload", "1-3-1"]:
        tracking = simulate_pp_tracking(n_frames=200, formation=formation, seed=42)
        snapshots = extract_formation_snapshots(tracking, sample_every_n=2)
        out = DATA_PROC / f"snapshots_{formation}.parquet"
        snapshots.to_parquet(out, index=False)
        logger.info(f"Demo snapshots ({formation}): {len(snapshots)} rows → {out}")


# ── Stage: Analyze ─────────────────────────────────────────────────────────────

def stage_analyze() -> None:
    from utils.data_loader import simulate_pp_tracking
    from analysis.formations import FormationDetector, extract_formation_snapshots
    from analysis.coverage import coverage_over_time, pp_coverage_summary
    from analysis.collapse import simulate_pk_collapse, compute_collapse_metrics

    logger.info("=== Formation Detection ===")
    # Combine synthetic formations for clustering demo
    all_snapshots = []
    for formation, seed in [("umbrella", 1), ("overload", 2), ("1-3-1", 3)]:
        t = simulate_pp_tracking(500, formation=formation, seed=seed)
        snaps = extract_formation_snapshots(t, sample_every_n=3)
        snaps["true_formation"] = formation
        all_snapshots.append(snaps)

    combined = pd.concat(all_snapshots, ignore_index=True)
    combined.to_parquet(DATA_PROC / "all_snapshots.parquet", index=False)

    detector = FormationDetector(n_clusters=5)
    labels = detector.fit_predict(combined)
    combined["cluster"] = labels
    combined.to_parquet(DATA_PROC / "snapshots_clustered.parquet", index=False)

    summary = detector.cluster_summary()
    summary.to_csv(DATA_PROC / "formation_summary.csv", index=False)
    logger.info(f"\n{summary.to_string()}")

    detector.save(MODELS_DIR / "formation_detector.pkl")

    logger.info("=== Coverage Analysis ===")
    for formation in ["umbrella", "overload", "1-3-1"]:
        tracking = simulate_pp_tracking(300, formation=formation, seed=42)
        cov = coverage_over_time(tracking)
        cov.to_parquet(DATA_PROC / f"coverage_{formation}.parquet", index=False)
        summary_cov = pp_coverage_summary(tracking)
        logger.info(f"  {formation}: {summary_cov}")

    logger.info("=== Collapse Physics ===")
    for speed, seed in [("fast", 1), ("slow", 2)]:
        frames = simulate_pk_collapse(collapse_speed=speed, n_frames=50, seed=seed)
        metrics = compute_collapse_metrics(frames, dt=0.1)
        logger.info(
            f"  {speed}: mean_speed={metrics.get('mean_collapse_speed',0):.2f} ft/s | "
            f"hull_change={metrics.get('total_hull_change',0):.1f} ft²"
        )

    logger.info("Analysis stage complete")


# ── Stage: Report ──────────────────────────────────────────────────────────────

def stage_report() -> None:
    from reports.generate_report import generate_report
    generate_report()


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Hockey Spatial Analysis Pipeline")
    parser.add_argument("--stage", choices=["download", "features", "analyze", "report", "all"])
    parser.add_argument("--all", dest="run_all", action="store_true")
    args = parser.parse_args()

    run_all = args.run_all or args.stage == "all"

    if run_all or args.stage == "download":
        logger.info("=== STAGE: DOWNLOAD ===")
        stage_download()

    if run_all or args.stage == "features":
        logger.info("=== STAGE: FEATURES ===")
        stage_features()

    if run_all or args.stage == "analyze":
        logger.info("=== STAGE: ANALYZE ===")
        stage_analyze()

    if run_all or args.stage == "report":
        logger.info("=== STAGE: REPORT ===")
        stage_report()


if __name__ == "__main__":
    main()
