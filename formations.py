"""
analysis/formations.py

Formation detection: identify recurring PP/PK structural configurations
using geometric features and unsupervised clustering.

Pipeline:
  1. Sample player positions from tracking data at regular intervals
  2. Compute geometric feature vector per snapshot (hull area, spread, etc.)
  3. K-Means clustering to find N dominant formations
  4. DBSCAN for outlier detection (formations that don't fit any cluster)
  5. Label clusters with hockey-meaningful names based on feature ranges

Each cluster = a formation archetype the team uses repeatedly.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import joblib

from geometry import formation_features, centroid, hull_area
from config import (
    FORMATION_FEATURES, N_FORMATIONS,
    DBSCAN_EPS, DBSCAN_MIN_SAMPLES, MODELS_DIR,
)

logger = logging.getLogger(__name__)


# ── Feature Extraction ─────────────────────────────────────────────────────────

def extract_formation_snapshots(tracking: pd.DataFrame,
                                 team_col: str = "team",
                                 team_label: str = "PP",
                                 sample_every_n: int = 5) -> pd.DataFrame:
    """
    From tracking data, sample every N frames and compute geometric
    feature vector for the specified team.

    Returns DataFrame with one row per snapshot, columns = FORMATION_FEATURES.
    """
    frames = tracking[tracking[team_col] == team_label].copy()
    frame_ids = sorted(frames["frame"].unique())

    rows = []
    for i, fid in enumerate(frame_ids):
        if i % sample_every_n != 0:
            continue
        fdata = frames[frames["frame"] == fid]
        positions = fdata[["x", "y"]].dropna().to_numpy()
        if len(positions) < 3:
            continue
        feats = formation_features(positions)
        feats["frame"] = fid
        rows.append(feats)

    if not rows:
        logger.warning(f"No valid snapshots for team={team_label}")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    logger.info(f"Extracted {len(df):,} formation snapshots for {team_label}")
    return df


# ── K-Means Clustering ─────────────────────────────────────────────────────────

class FormationDetector:
    """
    Detects recurring PP or PK formations via K-Means + DBSCAN.

    Usage:
        fd = FormationDetector(n_clusters=5)
        fd.fit(snapshot_df)
        labels = fd.predict(snapshot_df)
        fd.save("models/saved/formation_kmeans.pkl")
    """

    def __init__(self, n_clusters: int = N_FORMATIONS):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        self.feature_cols = FORMATION_FEATURES
        self.cluster_profiles_: Optional[pd.DataFrame] = None
        self.silhouette_: float = 0.0
        self.fitted_ = False

    def fit(self, snapshots: pd.DataFrame) -> "FormationDetector":
        """
        Fit scaler + K-Means on formation snapshots.
        Also runs DBSCAN to identify outlier formations.
        """
        X = self._prepare(snapshots)
        X_scaled = self.scaler.fit_transform(X)
        self.kmeans.fit(X_scaled)

        labels = self.kmeans.labels_
        if len(set(labels)) > 1:
            self.silhouette_ = silhouette_score(X_scaled, labels)
        logger.info(
            f"K-Means fit: {self.n_clusters} clusters, "
            f"silhouette={self.silhouette_:.3f}"
        )

        # Cluster profiles
        df = snapshots[self.feature_cols].copy()
        df["cluster"] = labels
        self.cluster_profiles_ = df.groupby("cluster")[self.feature_cols].mean()
        self.cluster_names_ = self._name_clusters(self.cluster_profiles_)

        self.fitted_ = True
        return self

    def predict(self, snapshots: pd.DataFrame) -> np.ndarray:
        """Return cluster labels for snapshot rows."""
        self._check_fitted()
        X = self._prepare(snapshots)
        X_scaled = self.scaler.transform(X)
        return self.kmeans.predict(X_scaled)

    def fit_predict(self, snapshots: pd.DataFrame) -> np.ndarray:
        self.fit(snapshots)
        return self.predict(snapshots)

    def dbscan_outliers(self, snapshots: pd.DataFrame) -> np.ndarray:
        """
        Run DBSCAN and return boolean mask of outlier snapshots (label == -1).
        Outliers are formation snapshots that don't belong to any dominant cluster.
        """
        X = self._prepare(snapshots)
        X_scaled = self.scaler.transform(X)
        labels = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(X_scaled)
        return labels == -1

    def cluster_summary(self) -> pd.DataFrame:
        """Return human-readable summary of each cluster."""
        self._check_fitted()
        rows = []
        for cluster_id, name in self.cluster_names_.items():
            profile = self.cluster_profiles_.loc[cluster_id]
            rows.append({
                "cluster": cluster_id,
                "name": name,
                "hull_area_ft2": round(profile["hull_area"], 1),
                "centroid_x": round(profile["centroid_x"], 1),
                "width_ft": round(profile["max_spread_y"], 1),
                "depth_ft": round(profile["max_spread_x"], 1),
                "mean_spacing_ft": round(profile["mean_nn_dist"], 1),
            })
        return pd.DataFrame(rows)

    def save(self, path) -> None:
        joblib.dump(self, path)
        logger.info(f"Saved FormationDetector to {path}")

    @classmethod
    def load(cls, path) -> "FormationDetector":
        return joblib.load(path)

    # ── Internals ────────────────────────────────────────────────────────────────

    def _prepare(self, snapshots: pd.DataFrame) -> np.ndarray:
        """Select and fill feature columns."""
        cols = [c for c in self.feature_cols if c in snapshots.columns]
        return snapshots[cols].fillna(0).to_numpy(dtype=np.float64)

    def _name_clusters(self, profiles: pd.DataFrame) -> dict[int, str]:
        """
        Heuristically name clusters based on geometric profiles.
        Naming conventions:
          - High hull area + wide spread → "Overload"
          - Centered + balanced angular spread → "Umbrella"
          - Low centroid_x (deep) + high spread_y → "1-3-1"
          - Small hull area + close spacing → "Box PP" / "Tight PK"
          - High centroid_x (close to net) → "Net-front"
        """
        names = {}
        for cid, row in profiles.iterrows():
            area = row.get("hull_area", 0)
            cx   = row.get("centroid_x", 150)
            sy   = row.get("max_spread_y", 30)
            sx   = row.get("max_spread_x", 30)
            ang  = row.get("angular_spread", 1.0)
            nn   = row.get("mean_nn_dist", 15)

            if cx > 170 and area < 400:
                name = "Net-front cluster"
            elif area > 900 and sy > 55:
                name = "Overload wide"
            elif sy > 45 and sx < 25 and ang > 1.2:
                name = "Umbrella"
            elif sx > 40 and sy > 40:
                name = "1-3-1 spread"
            elif nn < 12 and area < 500:
                name = "Compact box"
            elif cx < 150:
                name = "Deep possession"
            else:
                name = f"Formation {cid+1}"
            names[cid] = name

        return names

    def _check_fitted(self):
        if not self.fitted_:
            raise RuntimeError("Call .fit() first.")


# ── Optimal K Selection ────────────────────────────────────────────────────────

def find_optimal_k(snapshots: pd.DataFrame,
                    k_range: range = range(2, 10)) -> tuple[int, pd.DataFrame]:
    """
    Run K-Means for a range of k values, return the k with best silhouette score.
    Also returns a DataFrame of scores for elbow-curve plotting.
    """
    from sklearn.preprocessing import StandardScaler
    X = snapshots[FORMATION_FEATURES].fillna(0).to_numpy()
    X_scaled = StandardScaler().fit_transform(X)

    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertia = km.inertia_
        sil = silhouette_score(X_scaled, labels) if k > 1 else 0.0
        results.append({"k": k, "inertia": inertia, "silhouette": sil})
        logger.info(f"  k={k}: inertia={inertia:.1f}, silhouette={sil:.3f}")

    df = pd.DataFrame(results)
    best_k = int(df.loc[df["silhouette"].idxmax(), "k"])
    logger.info(f"Optimal k = {best_k}")
    return best_k, df
