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
        # Fast path: update pre-computed base matrix with transit edges
        # (avoids rebuilding the full graph + running all-pairs Dijkstra)
        tt = self._apply_routes_fast(routes)

        f1 = self._avg_travel_time(tt)
        f2 = self._avg_transfers_fast(routes)
        f3 = self._operational_cost_fast(routes)
        f4 = 1.0 - self._demand_coverage_fast(routes)

        penalty, _ = self.constraints.evaluate(routes, self.demand)

        objectives = np.array([f1, f2, f3, f4], dtype=np.float64)
        objectives += penalty

        return objectives

    def _apply_routes_fast(self, routes: List[Dict]) -> np.ndarray:
        """
        Update the pre-computed base travel-time matrix with transit route edges.

        Uses vectorised Floyd-Warshall relaxation (one pass per edge) instead of
        rebuilding the full networkx graph and re-running Dijkstra.
        """
        tt = self._base_tt.copy()

        for r in routes:
            seq = r["stop_sequence"]
            for k in range(len(seq) - 1):
                u, v = int(seq[k]), int(seq[k + 1])
                if not (0 <= u < self.n and 0 <= v < self.n):
                    continue
                t = float(self.graph.adj_matrix[u][v])
                if not np.isfinite(t) or t <= 0:
                    t = self.graph._estimate_travel_time(u, v)
                if t < tt[u][v]:
                    tt[u][v] = t
                    # Vectorised relaxation: for all (i,j), candidate = tt[i,u] + t + tt[v,j]
                    col_u = tt[:, u]          # shape (n,)
                    row_v = tt[v, :]          # shape (n,)
                    candidates = col_u[:, None] + t + row_v[None, :]  # shape (n,n)
                    np.minimum(tt, candidates, out=tt)

        return tt

    def _avg_transfers_fast(self, routes: List[Dict]) -> float:
        """
        Estimate fraction of OD pairs requiring a transfer.
        A stop pair needs a transfer if no single route covers both stops.
        """
        # Map stop_id -> set of route indices
        stop_routes: Dict[int, set] = {}
        for idx, r in enumerate(routes):
            for s in r["stop_sequence"]:
                stop_routes.setdefault(int(s), set()).add(idx)

        # Build served-stop mask
        served = np.zeros(self.n, dtype=bool)
        for s in stop_routes:
            if 0 <= s < self.n:
                served[s] = True

        if not served.any():
            return 1.0

        served_idx = np.where(served)[0]

        # For each stop pair: does at least one route cover both?
        transfers = 0
        total = 0
        for i in served_idx:
            for j in served_idx:
                if i == j:
                    continue
                ri = stop_routes.get(int(i), set())
                rj = stop_routes.get(int(j), set())
                if ri.isdisjoint(rj):
                    transfers += 1
                total += 1

        return transfers / max(total, 1)

    def _operational_cost_fast(self, routes: List[Dict]) -> float:
        """
        Fleet operational cost (Rs thousands/day) using base graph geometry.
        Does not require a rebuilt solution graph.
        """
        op_hours = self.config.get("operating_hours", 18)
        speed    = self.config.get("avg_speed_kmh", 25.0)
        total    = 0.0
        for r in routes:
            length_km = self.graph.get_route_length_km(r)
            vehicles  = r.get("num_vehicles", 5)
            trip_h    = length_km / speed
            fuel      = self.FUEL_COST_KM  * length_km * vehicles
            driver    = self.DRIVER_COST_H * trip_h    * vehicles
            total    += (fuel + driver) * op_hours
        return total / 1000.0

    def _demand_coverage_fast(self, routes: List[Dict]) -> float:
        """
        Fraction of total OD demand covered by routes (both stops served).
        """
        served: set = set()
        for r in routes:
            for s in r["stop_sequence"]:
                if 0 <= int(s) < self.n:
                    served.add(int(s))

        if not served:
            return 0.0

        total_demand   = self.demand.sum()
        if total_demand == 0:
            return 0.0

        served_arr = np.array(sorted(served), dtype=int)
        covered    = self.demand[np.ix_(served_arr, served_arr)].sum()
        # Subtract diagonal (i==j trivially covered but not meaningful)
        covered   -= self.demand[served_arr, served_arr].sum()
        return float(covered / total_demand)

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

        # Guarantee ref dominates every point in this front
        reference_pt = np.maximum(reference_pt, front.max(axis=0) * 1.01 + 1e-6)

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
