"""
tests/test_geometry.py

Unit tests for analysis/geometry.py — the core spatial math engine.
Run with: pytest tests/ -v
"""

import numpy as np
import pytest
from analysis.geometry import (
    hull_area, hull_perimeter, centroid, spread_x, spread_y,
    angular_spread, pairwise_distances, mean_pairwise_distance,
    mean_nearest_neighbor_distance, max_gap, voronoi_areas,
    undefended_area, centroid_velocity, hull_shrink_rate,
    compute_collapse_metrics, formation_features,
)


# ── Convex Hull ────────────────────────────────────────────────────────────────

class TestHullArea:
    def test_square(self):
        # 10×10 square → area 100
        pts = np.array([[0,0],[10,0],[10,10],[0,10]], dtype=float)
        assert abs(hull_area(pts) - 100.0) < 1.0

    def test_triangle(self):
        # right triangle base=10, height=10 → area=50
        pts = np.array([[0,0],[10,0],[0,10]], dtype=float)
        assert abs(hull_area(pts) - 50.0) < 1.0

    def test_collinear_returns_zero(self):
        # Collinear points → degenerate hull
        pts = np.array([[0,0],[5,0],[10,0]], dtype=float)
        assert hull_area(pts) == 0.0

    def test_fewer_than_three_returns_zero(self):
        assert hull_area(np.array([[1,2],[3,4]])) == 0.0

    def test_hockey_formation(self):
        # Realistic umbrella PP positions — should be a reasonable area
        pts = np.array([
            [170,42.5],[155,25],[155,60],[140,15],[140,70]
        ], dtype=float)
        area = hull_area(pts)
        assert 500 < area < 3000


class TestHullPerimeter:
    def test_square(self):
        pts = np.array([[0,0],[10,0],[10,10],[0,10]], dtype=float)
        assert abs(hull_perimeter(pts) - 40.0) < 1.0

    def test_fewer_than_three(self):
        assert hull_perimeter(np.array([[0,0],[1,1]])) == 0.0


# ── Centroid & Spread ──────────────────────────────────────────────────────────

class TestCentroid:
    def test_square(self):
        pts = np.array([[0,0],[4,0],[4,4],[0,4]], dtype=float)
        c = centroid(pts)
        assert abs(c[0] - 2.0) < 0.01
        assert abs(c[1] - 2.0) < 0.01

    def test_single_point(self):
        pts = np.array([[7.0, 3.0]])
        c = centroid(pts)
        assert c[0] == 7.0 and c[1] == 3.0


class TestSpread:
    def test_spread_x(self):
        pts = np.array([[130,10],[170,70]], dtype=float)
        assert abs(spread_x(pts) - 40.0) < 0.01

    def test_spread_y(self):
        pts = np.array([[150,10],[150,75]], dtype=float)
        assert abs(spread_y(pts) - 65.0) < 0.01


class TestAngularSpread:
    def test_symmetric(self):
        # Points evenly distributed around a center → high angular spread
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        pts = np.column_stack([10*np.cos(angles) + 160, 10*np.sin(angles) + 42])
        spread = angular_spread(pts)
        assert spread > 1.5    # high spread

    def test_clustered(self):
        # All points in same direction → low angular spread
        pts = np.array([[160,40],[161,40],[162,40],[163,40]], dtype=float)
        spread = angular_spread(pts)
        assert spread < 0.5


# ── Distances ──────────────────────────────────────────────────────────────────

class TestDistances:
    def setup_method(self):
        self.pts = np.array([[0,0],[3,4],[6,0]], dtype=float)

    def test_pairwise_symmetric(self):
        D = pairwise_distances(self.pts)
        assert D.shape == (3, 3)
        assert np.allclose(D, D.T)
        assert np.all(np.diag(D) == 0)

    def test_known_distance(self):
        D = pairwise_distances(self.pts)
        # (0,0) to (3,4) → distance 5
        assert abs(D[0, 1] - 5.0) < 0.01

    def test_mean_pairwise(self):
        result = mean_pairwise_distance(self.pts)
        assert result > 0

    def test_nn_distance(self):
        nn = mean_nearest_neighbor_distance(self.pts)
        assert nn > 0

    def test_max_gap(self):
        pts = np.array([[150,10],[150,25],[150,60],[150,80]], dtype=float)
        gap = max_gap(pts)
        assert abs(gap - 35.0) < 0.1   # gap between y=25 and y=60


# ── Voronoi / Coverage ─────────────────────────────────────────────────────────

class TestVoronoiAreas:
    def test_areas_positive(self):
        pts = np.array([[150,20],[150,65],[175,42]], dtype=float)
        areas = voronoi_areas(pts, bounds=(125, 200, 0, 85), n_samples=1000)
        for v in areas.values():
            assert v > 0

    def test_areas_sum_approx_total(self):
        pts = np.array([[150,20],[150,65],[175,42]], dtype=float)
        areas = voronoi_areas(pts, bounds=(125, 200, 0, 85), n_samples=2000)
        total = sum(areas.values())
        oz_area = 75 * 85
        # Should be within 5% of total OZ area
        assert abs(total - oz_area) / oz_area < 0.05


class TestUndefendedArea:
    def test_no_pk_players_full_undefended(self):
        pp = np.array([[160,42]], dtype=float)
        pk = np.array([[200,42]], dtype=float)   # outside OZ
        undef = undefended_area(pp, pk, oz_bounds=(125,200,0,85), n_samples=1000)
        oz = 75 * 85
        assert undef > oz * 0.9   # almost everything undefended

    def test_pk_covering_most_oz(self):
        # 4 PK players covering key spots
        pp = np.array([[160,42]], dtype=float)
        pk = np.array([[165,25],[165,60],[155,30],[155,55]], dtype=float)
        undef = undefended_area(pp, pk, oz_bounds=(125,200,0,85), n_samples=1000)
        # Some area still undefended but less than full OZ
        assert undef < 75 * 85 * 0.8


# ── Physics Metrics ────────────────────────────────────────────────────────────

class TestPhysicsMetrics:
    def test_centroid_velocity_stationary(self):
        pos = np.array([[160,40],[165,45],[170,50]], dtype=float)
        v = centroid_velocity(pos, pos, dt=0.1)
        assert v == pytest.approx(0.0, abs=1e-6)

    def test_centroid_velocity_moving(self):
        pos1 = np.array([[160,40],[165,45]], dtype=float)
        pos2 = np.array([[161,40],[166,45]], dtype=float)  # shifted x by 1
        v = centroid_velocity(pos1, pos2, dt=0.1)
        assert abs(v - 10.0) < 0.5   # 1 ft / 0.1s = 10 ft/s

    def test_hull_shrink_rate_negative(self):
        # Area decreasing → negative rate
        rate = hull_shrink_rate(500.0, 400.0, dt=0.1)
        assert rate == pytest.approx(-1000.0, abs=1.0)

    def test_hull_shrink_rate_growing(self):
        rate = hull_shrink_rate(400.0, 500.0, dt=0.1)
        assert rate > 0


class TestCollapseMetrics:
    def test_returns_all_keys(self):
        # Two frames, two players each
        frames = [
            np.array([[160,30],[160,55],[150,30],[150,55]], dtype=float),
            np.array([[163,33],[163,52],[153,33],[153,52]], dtype=float),
        ]
        result = compute_collapse_metrics(frames, dt=0.1)
        assert "hull_areas" in result
        assert "centroid_velocities" in result
        assert "hull_shrink_rates" in result
        assert "mean_collapse_speed" in result

    def test_hull_areas_list_length(self):
        frames = [
            np.array([[160,30],[160,55],[150,30],[150,55]], dtype=float),
            np.array([[162,32],[162,53],[152,32],[152,53]], dtype=float),
            np.array([[164,34],[164,51],[154,34],[154,51]], dtype=float),
        ]
        result = compute_collapse_metrics(frames, dt=0.1)
        assert len(result["hull_areas"]) == 3
        assert len(result["centroid_velocities"]) == 2


# ── Formation Features ─────────────────────────────────────────────────────────

class TestFormationFeatures:
    def test_all_keys_present(self):
        pts = np.array([[170,42],[155,25],[155,60],[140,15],[140,70]], dtype=float)
        feats = formation_features(pts)
        expected = [
            "hull_area","centroid_x","centroid_y","centroid_y_offset",
            "max_spread_x","max_spread_y","mean_nn_dist","angular_spread",
            "mean_pairwise_dist","max_gap_y",
        ]
        for k in expected:
            assert k in feats, f"Missing key: {k}"

    def test_empty_positions(self):
        feats = formation_features(np.array([]).reshape(0, 2))
        assert all(v == 0.0 for v in feats.values())

    def test_hull_area_positive_for_valid(self):
        pts = np.array([[170,42],[155,25],[155,60],[140,15],[140,70]], dtype=float)
        feats = formation_features(pts)
        assert feats["hull_area"] > 0

    def test_centroid_y_offset(self):
        # Symmetric formation → centroid near y=42.5 → small offset
        pts = np.array([[160,35],[160,50],[150,30],[150,55],[145,42]], dtype=float)
        feats = formation_features(pts)
        assert feats["centroid_y_offset"] < 10
