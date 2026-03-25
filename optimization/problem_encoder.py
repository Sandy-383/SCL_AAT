"""
Problem Encoder: Route Solution ↔ Continuous Position Vector

The Grey Wolf Optimizer works in a continuous R^d space.
This module translates between:
  - GWO position vector  x ∈ [0, 1]^d
  - Transit solution     List[Dict]  (route_id, stop_sequence, headway_min, num_vehicles)

Encoding scheme (per route r, R routes total):
──────────────────────────────────────────────
Each route is allocated a fixed-length 'slot' of dimension D_r:

  [stop_0/n, stop_1/n, ..., stop_{K-1}/n,   <- stop selection (K dims)
   headway_norm,                             <- headway in [0,1]
   vehicles_norm]                            <- vehicle count in [0,1]

where K = max_stops_per_route (padded with -1 sentinel for shorter routes).

Total dimension d = R × (K + 2)

Decoding:
  - stop_i = round(x_i × (n-1))  clamped to valid stop indices
  - headway = min_hw + x × (max_hw - min_hw)
  - vehicles = 1 + round(x × (max_vehicles - 1))

Duplicate stops within a route are resolved by bumping forward.
"""

import numpy as np
from typing import List, Dict, Tuple
from config import PROBLEM_CONFIG


class RouteEncoder:
    """
    Encodes/decodes GWO position vectors to transit route solutions.

    Parameters
    ----------
    num_stops      : total number of stops in the city network
    num_routes     : number of routes to optimize
    max_stops      : maximum stops per route (determines dimension per route)
    min_headway    : minimum headway in minutes
    max_headway    : maximum headway in minutes
    max_vehicles   : maximum vehicles per route
    """

    def __init__(
        self,
        num_stops    : int,
        num_routes   : int   = PROBLEM_CONFIG["num_routes"],
        max_stops    : int   = PROBLEM_CONFIG["max_stops_per_route"],
        min_stops    : int   = PROBLEM_CONFIG["min_stops_per_route"],
        min_headway  : float = PROBLEM_CONFIG["min_headway_min"],
        max_headway  : float = PROBLEM_CONFIG["max_headway_min"],
        max_vehicles : int   = 15,
    ):
        self.num_stops    = num_stops
        self.num_routes   = num_routes
        self.max_stops    = max_stops
        self.min_stops    = min_stops
        self.min_headway  = min_headway
        self.max_headway  = max_headway
        self.max_vehicles = max_vehicles

        # Dimension breakdown per route
        self.dims_per_route = max_stops + 2   # stops + headway + vehicles
        self.dimension      = num_routes * self.dims_per_route

    # ── Encoding ─────────────────────────────────────────────────────────────

    def encode(self, routes: List[Dict]) -> np.ndarray:
        """
        Convert a list of route dicts → position vector x ∈ [0,1]^d.
        Used to initialise wolves near known good solutions.
        """
        x = np.zeros(self.dimension, dtype=np.float64)

        for r_idx, route in enumerate(routes[:self.num_routes]):
            base          = r_idx * self.dims_per_route
            seq           = route["stop_sequence"]

            # Stop indices (normalised)
            for k in range(self.max_stops):
                if k < len(seq):
                    x[base + k] = seq[k] / (self.num_stops - 1)
                else:
                    x[base + k] = -1.0 / (self.num_stops - 1)  # sentinel

            # Headway (normalised)
            hw = route.get("headway_min", 15)
            x[base + self.max_stops] = (hw - self.min_headway) / (
                self.max_headway - self.min_headway
            )

            # Vehicles (normalised)
            nv = route.get("num_vehicles", 5)
            x[base + self.max_stops + 1] = (nv - 1) / (self.max_vehicles - 1)

        return np.clip(x, 0.0, 1.0)

    # ── Decoding ─────────────────────────────────────────────────────────────

    def decode(self, x: np.ndarray) -> List[Dict]:
        """
        Convert position vector x ∈ [0,1]^d → list of route dicts.
        This is called once per wolf per iteration — must be fast.
        """
        x = np.clip(x, 0.0, 1.0)
        routes = []

        for r_idx in range(self.num_routes):
            base = r_idx * self.dims_per_route

            # ── Decode stop sequence ──────────────────────────────────────
            raw_stops = x[base : base + self.max_stops]
            # Map [0,1] → [0, n-1] integer indices
            stop_indices = np.round(raw_stops * (self.num_stops - 1)).astype(int)
            stop_indices = np.clip(stop_indices, 0, self.num_stops - 1)

            # Remove duplicates while preserving order
            seen      = set()
            seq       = []
            for s in stop_indices:
                if s not in seen:
                    seq.append(int(s))
                    seen.add(s)

            # Enforce minimum route length
            if len(seq) < self.min_stops:
                # Pad with neighbouring stops
                for extra in range(self.num_stops):
                    if extra not in seen:
                        seq.append(extra)
                        seen.add(extra)
                    if len(seq) >= self.min_stops:
                        break

            # ── Decode headway ────────────────────────────────────────────
            hw_raw  = float(x[base + self.max_stops])
            headway = self.min_headway + hw_raw * (self.max_headway - self.min_headway)
            headway = round(headway)

            # ── Decode vehicles ───────────────────────────────────────────
            nv_raw   = float(x[base + self.max_stops + 1])
            vehicles = 1 + round(nv_raw * (self.max_vehicles - 1))
            vehicles = int(np.clip(vehicles, 1, self.max_vehicles))

            routes.append({
                "route_id"      : f"R{r_idx:03d}",
                "stop_sequence" : seq,
                "headway_min"   : headway,
                "num_vehicles"  : vehicles,
            })

        return routes

    # ── Random Initialisation ─────────────────────────────────────────────────

    def random_population(self, num_wolves: int, rng: np.random.Generator) -> np.ndarray:
        """
        Generate a random initial population matrix of shape (num_wolves, dim).
        Positions are uniform in [0, 1]^d.
        """
        return rng.random((num_wolves, self.dimension))

    def perturb(self, x: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
        """Add Gaussian noise to a position (for mutation during reinit)."""
        noise = rng.normal(0, sigma, x.shape)
        return np.clip(x + noise, 0.0, 1.0)

    # ── Dimension Info ────────────────────────────────────────────────────────

    def info(self) -> Dict:
        return {
            "num_stops"      : self.num_stops,
            "num_routes"     : self.num_routes,
            "max_stops"      : self.max_stops,
            "dims_per_route" : self.dims_per_route,
            "total_dimension": self.dimension,
        }
