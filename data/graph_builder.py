"""
Transit Graph Builder.

Converts the raw city data dict (from gtfs_loader) into a NetworkX directed
graph used for shortest-path calculations, demand routing, and fitness evaluation.

Node attributes : stop_id, stop_name, lat, lon, zone, demand
Edge attributes : travel_time (min), distance_km, route_ids
"""

import math
import logging
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class TransitGraph:
    """
    Encapsulates the city transit network as a directed graph.

    Key responsibilities:
    - Build the graph from city data
    - Provide shortest travel-time paths (with transfer penalties)
    - Compute coverage and connectivity metrics
    - Support incremental route updates during optimization
    """

    TRANSFER_PENALTY_MIN = 5.0  # minutes added for each transfer

    def __init__(self, city_data: Dict):
        self.stops      : pd.DataFrame  = city_data["stops"].reset_index(drop=True)
        self.routes     : List[Dict]    = city_data["routes"]
        self.adj_matrix : np.ndarray    = city_data["adj_matrix"]   # [n×n] minutes
        self.demand     : np.ndarray    = city_data["demand"]        # [n×n] passengers/hr
        self.city_name  : str           = city_data.get("city_name", "City")

        self.n = len(self.stops)
        self.G : nx.DiGraph = nx.DiGraph()

        self._stop_idx_map : Dict[str, int] = {
            sid: i for i, sid in enumerate(self.stops["stop_id"])
        }

        self._build_base_graph()
        self._add_routes_to_graph()
        self._compute_metrics()

    # ── Graph Construction ────────────────────────────────────────────────────

    def _build_base_graph(self):
        """Add all stops as nodes."""
        for idx, row in self.stops.iterrows():
            self.G.add_node(
                idx,
                stop_id   = row["stop_id"],
                stop_name = row.get("stop_name", f"Stop_{idx}"),
                lat       = row["stop_lat"],
                lon       = row["stop_lon"],
                zone      = row.get("zone", "unknown"),
                demand    = row.get("demand", 0),
            )

    def _add_routes_to_graph(self):
        """
        Add directed edges for each route's consecutive stop pairs.
        Multiple routes between the same stops accumulate in edge attributes.
        """
        for route in self.routes:
            seq = route["stop_sequence"]
            for k in range(len(seq) - 1):
                u, v = seq[k], seq[k + 1]
                if u >= self.n or v >= self.n:
                    continue
                t = self.adj_matrix[u][v]
                if t == np.inf:
                    # Estimate travel time from Haversine distance
                    t = self._estimate_travel_time(u, v)

                if self.G.has_edge(u, v):
                    # Keep the fastest travel time; accumulate route list
                    self.G[u][v]["travel_time"] = min(self.G[u][v]["travel_time"], t)
                    self.G[u][v]["route_ids"].append(route["route_id"])
                else:
                    dist = self._haversine_idx(u, v)
                    self.G.add_edge(
                        u, v,
                        travel_time  = t,
                        distance_km  = dist,
                        route_ids    = [route["route_id"]],
                    )

    def _estimate_travel_time(self, u: int, v: int, speed_kmh: float = 25.0) -> float:
        dist = self._haversine_idx(u, v)
        return (dist / speed_kmh) * 60 * 1.3

    # ── Metrics ───────────────────────────────────────────────────────────────

    def _compute_metrics(self):
        """Pre-compute graph-level statistics used in fitness evaluation."""
        self.num_connected_components = nx.number_weakly_connected_components(self.G)
        reachable = sum(1 for n in self.G.nodes() if self.G.degree(n) > 0)
        self.coverage_ratio = reachable / self.n if self.n > 0 else 0.0

        logger.info(
            "[TransitGraph] %s | Nodes: %d | Edges: %d | Coverage: %.1f%% | Components: %d",
            self.city_name, self.G.number_of_nodes(), self.G.number_of_edges(),
            self.coverage_ratio * 100, self.num_connected_components
        )

    # ── Shortest Path ─────────────────────────────────────────────────────────

    def shortest_travel_time(self, source: int, target: int) -> float:
        """Return shortest travel time (minutes) between two stop indices."""
        if source == target:
            return 0.0
        try:
            path   = nx.dijkstra_path(self.G, source, target, weight="travel_time")
            routes_used = set()
            total_time  = 0.0
            transfers   = 0

            for i in range(len(path) - 1):
                u, v       = path[i], path[i + 1]
                edge_data  = self.G[u][v]
                total_time += edge_data["travel_time"]
                edge_routes = set(edge_data["route_ids"])

                if i == 0:
                    routes_used = edge_routes
                else:
                    if not routes_used.intersection(edge_routes):
                        transfers  += 1
                        total_time += self.TRANSFER_PENALTY_MIN
                    routes_used = edge_routes

            return total_time
        except nx.NetworkXNoPath:
            return np.inf

    def all_pairs_travel_time(self) -> np.ndarray:
        """
        Compute full n×n travel time matrix using Dijkstra.
        Used once during fitness evaluation setup.
        """
        logger.info("Computing all-pairs travel time (%d×%d)…", self.n, self.n)
        matrix = np.full((self.n, self.n), np.inf)
        np.fill_diagonal(matrix, 0.0)

        lengths = dict(nx.all_pairs_dijkstra_path_length(self.G, weight="travel_time"))
        for u, dists in lengths.items():
            for v, d in dists.items():
                matrix[u][v] = d

        return matrix

    # ── Route Update API ──────────────────────────────────────────────────────

    def apply_solution(self, routes: List[Dict]) -> "TransitGraph":
        """
        Return a new TransitGraph with the provided routes applied.
        Used during fitness evaluation to test a wolf's position.
        """
        from copy import deepcopy
        new_data = {
            "stops"      : self.stops,
            "routes"     : routes,
            "adj_matrix" : self.adj_matrix,
            "demand"     : self.demand,
            "city_name"  : self.city_name,
        }
        return TransitGraph(new_data)

    # ── Coverage Analysis ────────────────────────────────────────────────────

    def compute_demand_coverage(self, routes: Optional[List[Dict]] = None) -> float:
        """
        Fraction of total OD demand covered by existing (or given) routes.
        An OD pair is 'covered' if both origin and destination are on the same route
        or reachable with ≤1 transfer.
        """
        if routes is None:
            routes = self.routes

        served_stops = set()
        for r in routes:
            for s in r["stop_sequence"]:
                served_stops.add(s)

        total_demand  = self.demand.sum()
        covered_demand = 0.0
        for i in range(self.n):
            for j in range(self.n):
                if i != j and i in served_stops and j in served_stops:
                    covered_demand += self.demand[i][j]

        return covered_demand / total_demand if total_demand > 0 else 0.0

    def compute_route_overlap(self, routes: Optional[List[Dict]] = None) -> float:
        """Fraction of edges covered by more than one route (redundancy metric)."""
        if routes is None:
            routes = self.routes
        edge_counts: Dict[Tuple[int, int], int] = {}
        for r in routes:
            seq = r["stop_sequence"]
            for k in range(len(seq) - 1):
                e = (seq[k], seq[k + 1])
                edge_counts[e] = edge_counts.get(e, 0) + 1
        if not edge_counts:
            return 0.0
        return sum(1 for c in edge_counts.values() if c > 1) / len(edge_counts)

    # ── Utility ───────────────────────────────────────────────────────────────

    def _haversine_idx(self, u: int, v: int) -> float:
        lu, lau = self.stops.at[u, "stop_lat"], self.stops.at[u, "stop_lon"]
        lv, lav = self.stops.at[v, "stop_lat"], self.stops.at[v, "stop_lon"]
        return _haversine(lu, lau, lv, lav)

    def stop_coords(self, idx: int) -> Tuple[float, float]:
        return (self.stops.at[idx, "stop_lat"], self.stops.at[idx, "stop_lon"])

    def get_route_length_km(self, route: Dict) -> float:
        seq = route["stop_sequence"]
        return sum(self._haversine_idx(seq[i], seq[i+1]) for i in range(len(seq)-1))

    def summary(self) -> Dict:
        return {
            "city"          : self.city_name,
            "num_stops"     : self.n,
            "num_routes"    : len(self.routes),
            "num_edges"     : self.G.number_of_edges(),
            "coverage_ratio": self.coverage_ratio,
            "components"    : self.num_connected_components,
        }


# ─── Utility ─────────────────────────────────────────────────────────────────

def _haversine(lat1, lon1, lat2, lon2) -> float:
    R    = 6371.0
    phi1 = math.radians(lat1);  phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a    = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
