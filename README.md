# Hockey Special Teams Spatial Analysis

Geometric and physics-based analysis of power play and penalty kill structure
using the Big Data Cup tracking dataset (free, public — Stathletes / GitHub).

Quantifies *how* teams organize spatially on special teams — formation shape,
coverage gaps, defensive collapse speed, and PP unit efficiency — using real
x/y player coordinate data.

---

## What This Project Does

### 1. Formation Detection (Clustering)
Groups PP and PK snapshots into recurring formations using geometric features:
convex hull area, centroid position, inter-player distance matrix, angular spread.
K-Means + DBSCAN identify the dominant structural configurations each team uses.

### 2. Voronoi Coverage Analysis
Decomposes the offensive zone into Voronoi regions owned by each player at each
tracking frame. Measures the size of *undefended* space created by a PP unit —
and how it correlates with shot generation.

### 3. Defensive Collapse Geometry (Physics)
After a PP zone entry, how fast does the PK unit contract its defensive shape?
Tracks centroid velocity (ft/s), convex hull shrink rate (ft²/s), and angular
compression over time. Surfaces which PK units are structurally slowest to respond.

### 4. Pass Network Spatial Analysis
Maps passing lanes geometrically: distance, angle, defender proximity.
Identifies which PP passing sequences open the largest coverage gaps.

---

## Data Source

**Big Data Cup** — Stathletes / University of Toronto  
Public GitHub: `github.com/bigdatacup`  
License: Publicly released for research; no restrictions on use.

Data includes:
- Event data: shots, passes, zone entries, faceoffs (x/y coordinates)
- Tracking data: player x/y positions at ~10Hz, player IDs, puck location

Download: `python pipeline.py --stage download`

---

## Project Structure

```
hockey-spatial/
├── data/
│   ├── raw/              # Downloaded Big Data Cup CSVs
│   └── processed/        # Cleaned, feature-engineered parquets
├── analysis/
│   ├── geometry.py       # Core spatial math (convex hull, Voronoi, distances)
│   ├── formations.py     # Formation detection (clustering)
│   ├── coverage.py       # Voronoi coverage gap analysis
│   ├── collapse.py       # PK collapse speed / physics metrics
│   └── passing.py        # Pass network spatial analysis
├── models/
│   └── formation_classifier.py  # Formation labeling model
├── utils/
│   ├── data_loader.py    # Big Data Cup ingestion
│   ├── rink.py           # Rink geometry constants + drawing helpers
│   └── viz.py            # Matplotlib / Plotly visualization helpers
├── reports/
│   └── generate_report.py
├── tests/
│   └── test_geometry.py
├── pipeline.py
├── config.py
└── requirements.txt
```

---

## Quickstart

```bash
pip install -r requirements.txt

# Download Big Data Cup data
python pipeline.py --stage download

# Feature engineering
python pipeline.py --stage features

# Run all analyses
python pipeline.py --stage analyze

# Generate stakeholder report
python pipeline.py --stage report
```

---

## Resume Bullet

**Hockey Special Teams Spatial Analysis | Python, scikit-learn, SciPy, SQL, Git**
- Analyzed power play and penalty kill structure using Big Data Cup player tracking data,
  applying Voronoi decomposition and convex hull geometry to quantify defensive coverage gaps
- Built formation detection system using K-Means and DBSCAN clustering on geometric features
  (inter-player distances, centroid spread, angular configuration) to identify recurring PP/PK structures
- Modeled PK defensive collapse speed using physics-based metrics (centroid velocity, hull shrink rate)
  to evaluate how quickly units respond to zone entries
- Produced automated stakeholder reports communicating spatial findings to non-technical hockey ops staff
