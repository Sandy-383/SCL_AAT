"""
GTFS (General Transit Feed Specification) Loader.

Reads standard GTFS CSV files and exposes clean DataFrames.
If no GTFS feed is available, a synthetic city network is generated
so that the optimizer can run immediately without any external data.
"""

import os
import math
import random
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

try:
    from sklearn.cluster import KMeans
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


# ─── GTFS File Loader ────────────────────────────────────────────────────────

class GTFSLoader:
    """Load and validate a GTFS feed from a directory of CSV files."""

    REQUIRED_FILES = ["stops.txt", "routes.txt", "trips.txt", "stop_times.txt"]
    OPTIONAL_FILES = ["shapes.txt", "calendar.txt", "agency.txt"]

    def __init__(self, gtfs_dir: str):
        self.gtfs_dir = gtfs_dir
        self.stops       : Optional[pd.DataFrame] = None
        self.routes      : Optional[pd.DataFrame] = None
        self.trips       : Optional[pd.DataFrame] = None
        self.stop_times  : Optional[pd.DataFrame] = None
        self.shapes      : Optional[pd.DataFrame] = None

    def load(self) -> bool:
        """Load all GTFS files. Returns True if successful."""
        for fname in self.REQUIRED_FILES:
            path = os.path.join(self.gtfs_dir, fname)
            if not os.path.exists(path):
                logger.warning("Required GTFS file missing: %s", path)
                return False

        self.stops      = pd.read_csv(os.path.join(self.gtfs_dir, "stops.txt"))
        self.routes     = pd.read_csv(os.path.join(self.gtfs_dir, "routes.txt"))
        self.trips      = pd.read_csv(os.path.join(self.gtfs_dir, "trips.txt"))
        self.stop_times = pd.read_csv(os.path.join(self.gtfs_dir, "stop_times.txt"))

        shapes_path = os.path.join(self.gtfs_dir, "shapes.txt")
        if os.path.exists(shapes_path):
            self.shapes = pd.read_csv(shapes_path)

        self._validate()
        logger.info("GTFS loaded: %d stops, %d routes, %d trips",
                    len(self.stops), len(self.routes), len(self.trips))
        return True

    def _validate(self):
        """Basic schema validation."""
        assert "stop_id"   in self.stops.columns,       "stops.txt missing stop_id"
        assert "stop_lat"  in self.stops.columns,       "stops.txt missing stop_lat"
        assert "stop_lon"  in self.stops.columns,       "stops.txt missing stop_lon"
        assert "route_id"  in self.routes.columns,      "routes.txt missing route_id"
        assert "trip_id"   in self.trips.columns,       "trips.txt missing trip_id"
        assert "stop_id"   in self.stop_times.columns,  "stop_times.txt missing stop_id"
        assert "trip_id"   in self.stop_times.columns,  "stop_times.txt missing trip_id"

    def get_stop_sequence(self, trip_id: str) -> pd.DataFrame:
        """Return ordered stop sequence for a given trip."""
        seq = self.stop_times[self.stop_times["trip_id"] == trip_id].copy()
        seq.sort_values("stop_sequence", inplace=True)
        return seq.merge(self.stops[["stop_id", "stop_lat", "stop_lon"]], on="stop_id")

    def summary(self) -> Dict:
        return {
            "num_stops" : len(self.stops)  if self.stops  is not None else 0,
            "num_routes": len(self.routes) if self.routes is not None else 0,
            "num_trips" : len(self.trips)  if self.trips  is not None else 0,
        }


# ─── Synthetic City Generator ────────────────────────────────────────────────

class SyntheticCityGenerator:
    """
    Generates a realistic synthetic city transit network when no GTFS data
    is available. Useful for benchmarking and development.

    The city is modelled as:
    - A central business district (CBD) with dense stops
    - Several residential zones radiating outward
    - Stops connected by realistic road-like edges
    """

    def __init__(
        self,
        num_stops  : int   = 200,
        num_routes : int   = 10,
        city_name  : str   = "SynthCity",
        center_lat : float = 12.9716,   # Bengaluru coordinates as default
        center_lon : float = 77.5946,
        city_radius_km: float = 15.0,
        seed       : int   = 42,
    ):
        self.num_stops      = num_stops
        self.num_routes     = num_routes
        self.city_name      = city_name
        self.center_lat     = center_lat
        self.center_lon     = center_lon
        self.city_radius_km = city_radius_km
        self.seed           = seed
        random.seed(seed)
        np.random.seed(seed)

    def generate(self) -> Dict:
        """
        Returns a dict with:
          stops     : pd.DataFrame  (stop_id, stop_name, stop_lat, stop_lon, zone, demand)
          routes    : list of dicts  (route_id, stop_sequence, headway_min)
          adj_matrix: np.ndarray    (num_stops × num_stops travel-time matrix in minutes)
          demand    : np.ndarray    (num_stops × num_stops OD demand matrix)
        """
        stops      = self._generate_stops()
        adj_matrix = self._build_adjacency(stops)
        demand     = self._generate_demand(stops)

        # Pre-compute cluster labels so they can be stored and visualized
        coords        = stops[["stop_lat", "stop_lon"]].values
        cluster_labels = self._kmeans_labels(coords, self.num_routes)

        routes = self._generate_routes(stops, adj_matrix, cluster_labels)

        logger.info("[Synthetic] City '%s': %d stops, %d routes generated.",
                    self.city_name, len(stops), len(routes))

        return {
            "stops"         : stops,
            "routes"        : routes,
            "adj_matrix"    : adj_matrix,
            "demand"        : demand,
            "city_name"     : self.city_name,
            "cluster_labels": cluster_labels,   # (n,) int array — one label per stop
        }

    # ── Private helpers ──────────────────────────────────────────────────────

    def _generate_stops(self) -> pd.DataFrame:
        """Place stops using a CBD-core + radial-zone distribution."""
        DEG_PER_KM_LAT = 1.0 / 110.574
        DEG_PER_KM_LON = 1.0 / (111.320 * math.cos(math.radians(self.center_lat)))

        stop_rows = []

        # CBD: 20% of stops in the inner 20% radius
        cbd_count = int(self.num_stops * 0.20)
        cbd_radius = self.city_radius_km * 0.20
        for i in range(cbd_count):
            r     = random.uniform(0, cbd_radius)
            theta = random.uniform(0, 2 * math.pi)
            lat   = self.center_lat + r * math.cos(theta) * DEG_PER_KM_LAT
            lon   = self.center_lon + r * math.sin(theta) * DEG_PER_KM_LON
            stop_rows.append({
                "stop_id"  : f"S{i:04d}",
                "stop_name": f"CBD_Stop_{i}",
                "stop_lat" : round(lat, 6),
                "stop_lon" : round(lon, 6),
                "zone"     : "CBD",
                "demand"   : random.randint(500, 2000),
            })

        # Residential zones: 80% of stops across 4 radial zones
        zone_names = ["North", "South", "East", "West"]
        per_zone   = (self.num_stops - cbd_count) // 4
        for z, zone_name in enumerate(zone_names):
            base_angle = z * (math.pi / 2)
            for j in range(per_zone):
                r     = random.uniform(cbd_radius, self.city_radius_km)
                theta = base_angle + random.uniform(-math.pi / 4, math.pi / 4)
                lat   = self.center_lat + r * math.cos(theta) * DEG_PER_KM_LAT
                lon   = self.center_lon + r * math.sin(theta) * DEG_PER_KM_LON
                idx   = cbd_count + z * per_zone + j
                stop_rows.append({
                    "stop_id"  : f"S{idx:04d}",
                    "stop_name": f"{zone_name}_Stop_{j}",
                    "stop_lat" : round(lat, 6),
                    "stop_lon" : round(lon, 6),
                    "zone"     : zone_name,
                    "demand"   : random.randint(50, 800),
                })

        return pd.DataFrame(stop_rows)

    def _build_adjacency(self, stops: pd.DataFrame) -> np.ndarray:
        """
        Build a travel-time adjacency matrix (minutes) using Haversine distances
        and an assumed average speed of 25 km/h.
        Only connects stops within a realistic neighborhood radius.
        """
        n      = len(stops)
        matrix = np.full((n, n), np.inf)
        np.fill_diagonal(matrix, 0.0)

        lats = stops["stop_lat"].values
        lons = stops["stop_lon"].values

        AVG_SPEED_KMH    = 25.0
        MAX_DIRECT_KM    = self.city_radius_km * 0.30   # Max direct edge length
        TRAFFIC_FACTOR   = 1.3                           # Simulate congestion

        for i in range(n):
            for j in range(i + 1, n):
                dist_km = _haversine(lats[i], lons[i], lats[j], lons[j])
                if dist_km <= MAX_DIRECT_KM:
                    time_min = (dist_km / AVG_SPEED_KMH) * 60 * TRAFFIC_FACTOR
                    matrix[i][j] = round(time_min, 2)
                    matrix[j][i] = round(time_min, 2)

        # Ensure connectivity using Floyd-Warshall
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if matrix[i][k] + matrix[k][j] < matrix[i][j]:
                        matrix[i][j] = matrix[i][k] + matrix[k][j]

        return matrix

    def _generate_demand(self, stops: pd.DataFrame) -> np.ndarray:
        """
        OD demand matrix: demand[i][j] = expected passengers/hour from stop i to j.
        CBD stops attract more inbound demand during peak hours.
        """
        n      = len(stops)
        demand = np.zeros((n, n))
        demands = stops["demand"].values

        for i in range(n):
            for j in range(n):
                if i != j:
                    # Gravity model: proportional to demand at both stops
                    demand[i][j] = (demands[i] * demands[j]) ** 0.5 / (n * 10)

        return demand.astype(np.float32)

    def _generate_routes(
        self,
        stops: pd.DataFrame,
        adj_matrix: np.ndarray,
        labels: np.ndarray,
    ) -> list:
        """
        Generate bus routes using pre-computed K-means cluster labels.

        Each cluster becomes one route: stops are ordered greedily by proximity
        within the cluster, then the nearest CBD stop is appended as terminal.
        """
        n        = len(stops)
        cbd_mask = stops["zone"] == "CBD"
        cbd_ids  = stops.index[cbd_mask].tolist()

        logger.info(
            "[KMeans] Clustered %d stops into %d geographic route zones.",
            n, self.num_routes
        )

        routes: list = []
        for r in range(self.num_routes):
            cluster_idx = np.where(labels == r)[0].tolist()

            # Merge tiny clusters (< 3 stops) with their nearest neighbour cluster
            if len(cluster_idx) < 3:
                cluster_idx = self._expand_cluster(cluster_idx, labels, r, n)

            # Order stops within cluster via greedy nearest-neighbour
            ordered = self._greedy_order(cluster_idx, adj_matrix)

            # Append nearest CBD stop as terminal (if not already in route)
            cbd_terminal = self._nearest_cbd(ordered[-1], cbd_ids, adj_matrix)
            if cbd_terminal not in ordered:
                ordered.append(cbd_terminal)

            routes.append({
                "route_id"      : f"R{r:03d}",
                "stop_sequence" : ordered,
                "headway_min"   : random.randint(10, 30),
                "num_vehicles"  : random.randint(3, 10),
            })

        return routes

    # ── K-means helpers ───────────────────────────────────────────────────────

    def _kmeans_labels(self, coords: np.ndarray, k: int) -> np.ndarray:
        """Return cluster label (0..k-1) for each stop."""
        if _SKLEARN_AVAILABLE:
            km = KMeans(n_clusters=k, random_state=self.seed, n_init=10)
            return km.fit_predict(coords)
        else:
            return self._numpy_kmeans(coords, k)

    def _numpy_kmeans(self, coords: np.ndarray, k: int, max_iter: int = 100) -> np.ndarray:
        """Lightweight K-means using NumPy only (fallback)."""
        rng = np.random.default_rng(self.seed)
        # Random initialisation from data points
        center_idx = rng.choice(len(coords), size=k, replace=False)
        centers    = coords[center_idx].copy()

        labels = np.zeros(len(coords), dtype=int)
        for _ in range(max_iter):
            # Assignment step
            dists  = np.linalg.norm(coords[:, None, :] - centers[None, :, :], axis=2)
            new_labels = dists.argmin(axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            # Update step
            for c in range(k):
                mask = labels == c
                if mask.any():
                    centers[c] = coords[mask].mean(axis=0)

        return labels

    def _expand_cluster(
        self,
        cluster_idx: List[int],
        labels: np.ndarray,
        target_label: int,
        n: int,
        min_size: int = 3,
    ) -> List[int]:
        """Borrow stops from the most populated other cluster to reach min_size."""
        pool = [i for i in range(n) if labels[i] != target_label]
        extra = random.sample(pool, min(min_size - len(cluster_idx), len(pool)))
        return cluster_idx + extra

    def _greedy_order(self, idx: List[int], adj_matrix: np.ndarray) -> List[int]:
        """Order stops in idx by greedy nearest-neighbour starting from a random stop."""
        if len(idx) <= 1:
            return list(idx)

        ordered  = [random.choice(idx)]
        remaining = set(idx) - {ordered[0]}

        while remaining:
            current = ordered[-1]
            dists   = [(adj_matrix[current][j], j) for j in remaining]
            dists.sort()
            ordered.append(dists[0][1])
            remaining.remove(dists[0][1])

        return ordered

    def _nearest_cbd(self, stop: int, cbd_ids: List[int], adj_matrix: np.ndarray) -> int:
        """Return the CBD stop closest (by travel time) to `stop`."""
        best_t, best_s = np.inf, cbd_ids[0]
        for s in cbd_ids:
            t = adj_matrix[stop][s]
            if t < best_t:
                best_t, best_s = t, s
        return best_s


# ─── Utility ──────────────────────────────────────────────────────────────────

def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in km."""
    R    = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a    = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def load_or_generate(gtfs_dir: str, **synth_kwargs) -> Dict:
    """
    Attempt to load a real GTFS feed from gtfs_dir.
    Falls back to synthetic generation if files are missing.
    """
    loader = GTFSLoader(gtfs_dir)
    if loader.load():
        # Convert loaded GTFS into the internal dict format
        stops = loader.stops[["stop_id", "stop_name", "stop_lat", "stop_lon"]].copy()
        stops["zone"]   = "unknown"
        stops["demand"] = 100  # placeholder
        stops = stops.reset_index(drop=True)

        n          = len(stops)
        adj_matrix = np.full((n, n), np.inf)
        np.fill_diagonal(adj_matrix, 0.0)

        lats = stops["stop_lat"].values
        lons = stops["stop_lon"].values
        for i in range(n):
            for j in range(i + 1, n):
                d = _haversine(lats[i], lons[i], lats[j], lons[j])
                t = (d / 25.0) * 60 * 1.3
                if d < 3.0:
                    adj_matrix[i][j] = adj_matrix[j][i] = t

        demand = np.ones((n, n), dtype=np.float32)

        # Build route list from trip stop sequences (first num_routes trips)
        routes     = []
        trip_ids   = loader.trips["trip_id"].tolist()
        stop_id_to_idx = {s: i for i, s in enumerate(stops["stop_id"])}

        for idx, tid in enumerate(trip_ids[:synth_kwargs.get("num_routes", 10)]):
            seq    = loader.get_stop_sequence(tid)
            idxs   = [stop_id_to_idx[s] for s in seq["stop_id"] if s in stop_id_to_idx]
            if len(idxs) >= 2:
                routes.append({
                    "route_id"      : f"R{idx:03d}",
                    "stop_sequence" : idxs,
                    "headway_min"   : 15,
                    "num_vehicles"  : 5,
                })

        return {"stops": stops, "routes": routes, "adj_matrix": adj_matrix,
                "demand": demand, "city_name": "GTFS_City"}
    else:
        logger.info("GTFS not found — generating synthetic city network.")
        gen = SyntheticCityGenerator(**synth_kwargs)
        return gen.generate()
