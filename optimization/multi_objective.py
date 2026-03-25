"""
Multi-Objective Fitness Evaluator and Pareto Front Handler.

Objectives (all minimised internally — coverage is negated):
──────────────────────────────────────────────────────────────
  f1 : Average passenger travel time (minutes)
  f2 : Average number of transfers per trip
  f3 : Operational cost (fleet cost per hour)
  f4 : 1 - demand coverage ratio  (so lower = more coverage)

Pareto Front:
  - Non-dominated sorting (NSGA-II style)
  - Crowding distance for diversity preservation
  - Hypervolume indicator for solution quality measurement
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


# ─── Objective Evaluator ─────────────────────────────────────────────────────

class ObjectiveEvaluator:
    """
    Evaluates all four objectives for a candidate transit solution.

    Parameters
    ----------
    graph        : TransitGraph — city network with adjacency data
    demand       : np.ndarray  — OD demand matrix [n×n]
    constraints  : ConstraintHandler
    config       : PROBLEM_CONFIG dict
    """

    CO2_PER_KM    = 0.089     # kg CO₂ per km per vehicle
    FUEL_COST_KM  = 8.0       # ₹ per km
    DRIVER_COST_H = 150.0     # ₹ per driver per hour

    def __init__(self, graph, demand: np.ndarray, constraints, config: Dict):
        self.graph       = graph
        self.demand      = demand
        self.constraints = constraints
        self.config      = config
        self.n           = graph.n

        # Pre-compute travel time matrix for base network
        logger.info("Pre-computing base travel-time matrix…")
        self._base_tt = graph.all_pairs_travel_time()

    def evaluate(self, routes: List[Dict]) -> np.ndarray:
        """
        Evaluate objectives for a single solution (list of routes).

        Returns
        -------
        objectives : np.ndarray shape (4,)  — [f1, f2, f3, f4]
        """
        # Build travel time matrix for this specific solution
        g_sol = self.graph.apply_solution(routes)
        tt    = g_sol.all_pairs_travel_time()

        f1 = self._avg_travel_time(tt)
        f2 = self._avg_transfers(routes, tt)
        f3 = self._operational_cost(routes, g_sol)
        f4 = 1.0 - g_sol.compute_demand_coverage(routes)

        penalty, _ = self.constraints.evaluate(routes, self.demand)

        objectives = np.array([f1, f2, f3, f4], dtype=np.float64)
        objectives += penalty   # penalty shifts all objectives equally

        return objectives

    def evaluate_batch(self, solutions: List[List[Dict]]) -> np.ndarray:
        """
        Evaluate a batch of solutions.
        Returns matrix of shape (num_solutions, 4).
        """
        results = np.zeros((len(solutions), 4), dtype=np.float64)
        for i, routes in enumerate(solutions):
            results[i] = self.evaluate(routes)
        return results

    # ── Objective Functions ──────────────────────────────────────────────────

    def _avg_travel_time(self, tt: np.ndarray) -> float:
        """
        Demand-weighted average travel time.
        Only OD pairs with finite travel time contribute.
        """
        mask     = np.isfinite(tt) & (tt > 0)
        weighted = (tt * self.demand)[mask]
        denom    = self.demand[mask].sum()
        if denom == 0:
            return 999.0
        return float(weighted.sum() / denom)

    def _avg_transfers(self, routes: List[Dict], tt: np.ndarray) -> float:
        """
        Estimate average transfers per trip.
        Approximated by counting stops served by each route and comparing
        OD connectivity: if same route serves both stops → 0 transfers,
        otherwise 1+ transfer.
        """
        # Map stop → set of routes serving it
        stop_routes: Dict[int, set] = {}
        for r in routes:
            for s in r["stop_sequence"]:
                stop_routes.setdefault(s, set()).add(r["route_id"])

        total_transfers = 0.0
        total_pairs     = 0

        for i in range(self.n):
            for j in range(self.n):
                if i == j or not np.isfinite(tt[i][j]):
                    continue
                ri = stop_routes.get(i, set())
                rj = stop_routes.get(j, set())
                if ri.isdisjoint(rj):
                    total_transfers += 1   # at least one transfer needed
                total_pairs += 1

        return total_transfers / max(total_pairs, 1)

    def _operational_cost(self, routes: List[Dict], g_sol) -> float:
        """
        Total fleet operational cost per hour (₹).
          = Σ_r  vehicles_r × (fuel_cost × route_length_km / speed + driver_cost)
        """
        op_hours = self.config.get("operating_hours", 18)
        speed    = self.config.get("avg_speed_kmh", 25.0)
        total    = 0.0

        for r in routes:
            length_km = g_sol.get_route_length_km(r)
            vehicles  = r.get("num_vehicles", 5)
            trip_h    = length_km / speed
            fuel      = self.FUEL_COST_KM * length_km * vehicles
            driver    = self.DRIVER_COST_H * trip_h * vehicles
            total    += (fuel + driver) * op_hours

        # Normalise to thousands (₹ thousands per day)
        return total / 1000.0


# ─── Pareto Front ─────────────────────────────────────────────────────────────

class ParetoFront:
    """
    NSGA-II style Pareto front computation with crowding distance.

    Usage:
        pf = ParetoFront()
        ranks, crowding = pf.compute(objectives)   # objectives: (N, M) array
        front0_idx = np.where(ranks == 0)[0]
    """

    def compute(
        self,
        objectives: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Non-dominated sorting + crowding distance.

        Parameters
        ----------
        objectives : (N, M) array — N solutions, M objectives (all minimised)

        Returns
        -------
        ranks     : (N,) int  — Pareto rank (0 = non-dominated front)
        crowding  : (N,) float — crowding distance (higher = more diverse)
        """
        N = len(objectives)
        ranks    = self._fast_non_dominated_sort(objectives)
        crowding = self._crowding_distance(objectives, ranks)
        return ranks, crowding

    def _fast_non_dominated_sort(self, obj: np.ndarray) -> np.ndarray:
        N     = len(obj)
        ranks = np.full(N, -1, dtype=int)

        domination_count  = np.zeros(N, dtype=int)
        dominated_by_me   = [[] for _ in range(N)]

        for i in range(N):
            for j in range(i + 1, N):
                dom_ij = self._dominates(obj[i], obj[j])
                dom_ji = self._dominates(obj[j], obj[i])
                if dom_ij:
                    dominated_by_me[i].append(j)
                    domination_count[j] += 1
                elif dom_ji:
                    dominated_by_me[j].append(i)
                    domination_count[i] += 1

        # Front 0: solutions not dominated by anyone
        current_front = [i for i in range(N) if domination_count[i] == 0]
        rank = 0
        while current_front:
            for i in current_front:
                ranks[i] = rank
            next_front = []
            for i in current_front:
                for j in dominated_by_me[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            current_front = next_front
            rank += 1

        return ranks

    def _dominates(self, a: np.ndarray, b: np.ndarray) -> bool:
        """Return True if a dominates b (a ≤ b in all, a < b in at least one)."""
        return bool(np.all(a <= b) and np.any(a < b))

    def _crowding_distance(self, obj: np.ndarray, ranks: np.ndarray) -> np.ndarray:
        N, M   = obj.shape
        crowd  = np.zeros(N)
        max_rank = ranks.max()

        for r in range(max_rank + 1):
            front = np.where(ranks == r)[0]
            if len(front) <= 2:
                crowd[front] = np.inf
                continue

            for m in range(M):
                sorted_idx = front[np.argsort(obj[front, m])]
                crowd[sorted_idx[0]]  = np.inf
                crowd[sorted_idx[-1]] = np.inf
                f_range = obj[sorted_idx[-1], m] - obj[sorted_idx[0], m]
                if f_range == 0:
                    continue
                for k in range(1, len(sorted_idx) - 1):
                    crowd[sorted_idx[k]] += (
                        obj[sorted_idx[k+1], m] - obj[sorted_idx[k-1], m]
                    ) / f_range

        return crowd

    # ── Quality Indicators ───────────────────────────────────────────────────

    def hypervolume(
        self,
        objectives   : np.ndarray,
        reference_pt : Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute hypervolume indicator (WFG algorithm — 2D/3D exact, ND approx).
        The reference point is the nadir point + 10% margin if not provided.
        """
        # Take only the Pareto front (rank 0) solutions
        ranks, _ = self.compute(objectives)
        front    = objectives[ranks == 0]

        if len(front) == 0:
            return 0.0

        if reference_pt is None:
            reference_pt = front.max(axis=0) * 1.1

        M = front.shape[1]
        if M == 2:
            return self._hv_2d(front, reference_pt)
        else:
            return self._hv_approx(front, reference_pt)

    def _hv_2d(self, front: np.ndarray, ref: np.ndarray) -> float:
        """Exact 2D hypervolume."""
        sorted_f = front[np.argsort(front[:, 0])]
        hv       = 0.0
        prev_x   = ref[0]

        for i in range(len(sorted_f) - 1, -1, -1):
            p  = sorted_f[i]
            hv += (prev_x - p[0]) * (ref[1] - p[1])
            prev_x = p[0]
        return hv

    def _hv_approx(self, front: np.ndarray, ref: np.ndarray) -> float:
        """
        Monte Carlo approximation of hypervolume for M > 2 dimensions.
        Samples 50,000 random points in the bounding box.
        """
        N_SAMPLES  = 50_000
        ideal      = front.min(axis=0)
        rng        = np.random.default_rng(0)
        samples    = rng.uniform(ideal, ref, size=(N_SAMPLES, front.shape[1]))

        dominated  = 0
        for p in samples:
            if np.any(np.all(front <= p, axis=1)):
                dominated += 1

        box_volume = np.prod(ref - ideal)
        return (dominated / N_SAMPLES) * box_volume

    def generational_distance(
        self,
        approx_front   : np.ndarray,
        true_front     : np.ndarray,
    ) -> float:
        """
        GD: average Euclidean distance from each approx solution
        to the nearest true Pareto front solution.
        """
        dist = 0.0
        for p in approx_front:
            dist += np.min(np.linalg.norm(true_front - p, axis=1))
        return dist / len(approx_front)

    def inverted_generational_distance(
        self,
        approx_front : np.ndarray,
        true_front   : np.ndarray,
    ) -> float:
        """
        IGD: average distance from each true solution to nearest approx solution.
        Lower is better.
        """
        dist = 0.0
        for p in true_front:
            dist += np.min(np.linalg.norm(approx_front - p, axis=1))
        return dist / len(true_front)
