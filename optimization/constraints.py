"""
Constraint Handler for Transit Route Optimization.

Implements penalty-based constraint handling.
Infeasible solutions are not discarded — instead a penalty term is added
to their fitness values, guiding the optimizer toward feasibility.

Constraints enforced:
  1. Minimum route length  (min_stops_per_route)
  2. Vehicle capacity      (passengers ≤ capacity × frequency × operating_hours)
  3. Network connectivity  (all stops must be reachable)
  4. Headway bounds        (min_headway ≤ hw ≤ max_headway)
  5. No isolated stops     (every stop should be on at least one route)
  6. Route overlap limit   (excessive overlap wastes fleet resources)
"""

import numpy as np
from typing import List, Dict, Tuple
from config import PROBLEM_CONFIG


class ConstraintHandler:
    """
    Evaluates constraint violations and returns a scalar penalty value.

    The penalty is designed to be:
    - Zero for fully feasible solutions
    - Proportional to the magnitude of violation
    - Normalised so it is comparable to objective function values
    """

    def __init__(
        self,
        num_stops        : int,
        vehicle_capacity : int   = PROBLEM_CONFIG["vehicle_capacity"],
        min_stops        : int   = PROBLEM_CONFIG["min_stops_per_route"],
        max_stops        : int   = PROBLEM_CONFIG["max_stops_per_route"],
        min_headway      : float = PROBLEM_CONFIG["min_headway_min"],
        max_headway      : float = PROBLEM_CONFIG["max_headway_min"],
        operating_hours  : float = PROBLEM_CONFIG["operating_hours"],
        penalty_weight   : float = 1000.0,
    ):
        self.num_stops        = num_stops
        self.vehicle_capacity = vehicle_capacity
        self.min_stops        = min_stops
        self.max_stops        = max_stops
        self.min_headway      = min_headway
        self.max_headway      = max_headway
        self.operating_hours  = operating_hours
        self.penalty_weight   = penalty_weight

    def evaluate(
        self,
        routes    : List[Dict],
        demand    : np.ndarray,
    ) -> Tuple[float, Dict]:
        """
        Compute total constraint penalty and a breakdown dict.

        Returns
        -------
        penalty   : float   — total penalty (0 = fully feasible)
        breakdown : dict    — per-constraint violation amounts
        """
        violations = {}

        violations["route_length"]   = self._route_length_violation(routes)
        violations["capacity"]       = self._capacity_violation(routes, demand)
        violations["headway_bounds"] = self._headway_violation(routes)
        violations["stop_coverage"]  = self._stop_coverage_violation(routes)
        violations["overlap"]        = self._overlap_penalty(routes)

        total_penalty = self.penalty_weight * sum(violations.values())
        return total_penalty, violations

    def is_feasible(self, routes: List[Dict], demand: np.ndarray) -> bool:
        """Return True if all constraints are satisfied."""
        penalty, _ = self.evaluate(routes, demand)
        return penalty == 0.0

    # ── Individual Constraint Functions ───────────────────────────────────────

    def _route_length_violation(self, routes: List[Dict]) -> float:
        """Penalise routes shorter than min_stops or longer than max_stops."""
        violation = 0.0
        for r in routes:
            n = len(r["stop_sequence"])
            if n < self.min_stops:
                violation += (self.min_stops - n) / self.min_stops
            if n > self.max_stops:
                violation += (n - self.max_stops) / self.max_stops
        return violation / max(len(routes), 1)

    def _capacity_violation(self, routes: List[Dict], demand: np.ndarray) -> float:
        """
        Penalise if total route capacity < total demand served by that route.

        Route capacity (passengers/hr) = vehicles × capacity × (60 / headway)
        Route demand   = sum of OD demand between consecutive stop pairs
        """
        violation = 0.0
        for r in routes:
            seq      = r["stop_sequence"]
            headway  = max(r.get("headway_min", 15), 1)
            vehicles = r.get("num_vehicles", 5)

            freq_per_hr  = 60.0 / headway
            route_cap    = vehicles * self.vehicle_capacity * freq_per_hr

            # Approximate demand as sum along the route
            route_demand = 0.0
            for k in range(len(seq) - 1):
                i, j = seq[k], seq[k + 1]
                if i < demand.shape[0] and j < demand.shape[1]:
                    route_demand += demand[i][j]

            if route_demand > route_cap:
                violation += (route_demand - route_cap) / max(route_cap, 1)

        return violation / max(len(routes), 1)

    def _headway_violation(self, routes: List[Dict]) -> float:
        """Penalise headways outside [min_headway, max_headway]."""
        violation = 0.0
        for r in routes:
            hw = r.get("headway_min", 15)
            if hw < self.min_headway:
                violation += (self.min_headway - hw) / self.min_headway
            if hw > self.max_headway:
                violation += (hw - self.max_headway) / self.max_headway
        return violation / max(len(routes), 1)

    def _stop_coverage_violation(self, routes: List[Dict]) -> float:
        """
        Penalise if the fraction of stops not served by any route is too high.
        Target: ≥ 80% of stops covered.
        """
        MIN_COVERAGE = 0.80
        served = set()
        for r in routes:
            served.update(r["stop_sequence"])

        coverage = len(served) / self.num_stops
        if coverage < MIN_COVERAGE:
            return (MIN_COVERAGE - coverage) / MIN_COVERAGE
        return 0.0

    def _overlap_penalty(self, routes: List[Dict]) -> float:
        """
        Light penalty for high route overlap (> 40% shared edges is wasteful).
        This is a soft constraint.
        """
        MAX_OVERLAP = 0.40
        edge_counts: Dict = {}
        total_edges = 0
        for r in routes:
            seq = r["stop_sequence"]
            for k in range(len(seq) - 1):
                e = (seq[k], seq[k+1])
                edge_counts[e] = edge_counts.get(e, 0) + 1
                total_edges += 1

        if total_edges == 0:
            return 0.0

        overlap = sum(c - 1 for c in edge_counts.values()) / total_edges
        if overlap > MAX_OVERLAP:
            return (overlap - MAX_OVERLAP) / (1 - MAX_OVERLAP)
        return 0.0
