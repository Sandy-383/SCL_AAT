"""
Ant Colony Optimization (ACO) — CPU Baseline.

Multi-objective ACO (MOACO) for route optimization.
Pheromone matrix guides ants to prefer high-demand, low-travel-time edges.
"""

import time
import logging
import numpy as np
from typing import List, Dict, Tuple, Callable

from optimization.problem_encoder import RouteEncoder
from optimization.multi_objective import ParetoFront

logger = logging.getLogger(__name__)


class AntColonyOptimization:
    def __init__(
        self,
        encoder        : RouteEncoder,
        fitness_fn     : Callable,
        num_ants       : int   = 100,
        max_iterations : int   = 500,
        alpha          : float = 1.0,   # pheromone weight
        beta           : float = 2.0,   # heuristic weight
        rho            : float = 0.1,   # evaporation rate
        seed           : int   = 42,
        verbose        : bool  = True,
    ):
        self.encoder        = encoder
        self.fitness_fn     = fitness_fn
        self.num_ants       = num_ants
        self.max_iterations = max_iterations
        self.alpha          = alpha
        self.beta           = beta
        self.rho            = rho
        self.rng            = np.random.default_rng(seed)
        self.verbose        = verbose
        self.dim            = encoder.dimension
        self.pareto         = ParetoFront()

        # Pheromone matrix over continuous position space (discretized)
        self.N_BINS = 20   # discretise each dimension into 20 bins
        self.pheromone = np.ones((self.dim, self.N_BINS), dtype=np.float64)

        self.history        = {"hypervolume": [], "best_f1": []}
        self.total_time     = 0.0
        self.fitness_evals  = 0

    def optimize(self) -> Tuple[List[Dict], np.ndarray]:
        arch_pos = np.empty((0, self.dim))
        arch_obj = np.empty((0, 4))
        start    = time.perf_counter()

        for iteration in range(self.max_iterations):
            # Construct ant solutions using pheromone-guided sampling
            positions = self._construct_solutions()
            obj       = self._eval_pop(positions)

            # Update archive
            arch_pos, arch_obj = self._update_archive(arch_pos, arch_obj, positions, obj)

            # Update pheromones
            self._update_pheromones(arch_pos, arch_obj)

            hv = self.pareto.hypervolume(arch_obj)
            self.history["hypervolume"].append(hv)
            self.history["best_f1"].append(float(arch_obj[:, 0].min()) if len(arch_obj) else 999.0)

        self.total_time = time.perf_counter() - start
        best_idx = int(np.argmin(arch_obj[:, 0]))
        return self.encoder.decode(arch_pos[best_idx]), arch_obj[best_idx]

    def _construct_solutions(self):
        """Sample positions using pheromone-weighted probabilities per dimension."""
        positions = np.zeros((self.num_ants, self.dim))

        for d in range(self.dim):
            # Probability of choosing each bin = pheromone^alpha
            probs = self.pheromone[d] ** self.alpha
            probs /= probs.sum()
            # Sample bins for all ants
            bins  = self.rng.choice(self.N_BINS, size=self.num_ants, p=probs)
            # Convert bin to continuous value (uniform within bin)
            positions[:, d] = (bins + self.rng.random(self.num_ants)) / self.N_BINS

        return np.clip(positions, 0.0, 1.0)

    def _update_pheromones(self, arch_pos, arch_obj):
        """Deposit pheromones for archive solutions, then evaporate."""
        # Evaporate
        self.pheromone *= (1 - self.rho)
        self.pheromone = np.clip(self.pheromone, 0.01, 10.0)

        if len(arch_pos) == 0:
            return

        # Deposit: each archive solution deposits proportional to 1/(f1+1)
        quality = 1.0 / (arch_obj[:, 0] + 1.0)
        quality /= quality.sum()

        for sol_idx, pos in enumerate(arch_pos):
            for d in range(self.dim):
                bin_idx = min(int(pos[d] * self.N_BINS), self.N_BINS - 1)
                self.pheromone[d, bin_idx] += quality[sol_idx]

    def _eval_pop(self, positions):
        results = np.zeros((len(positions), 4))
        for i, x in enumerate(positions):
            results[i] = self.fitness_fn(self.encoder.decode(x))
            self.fitness_evals += 1
        return results

    def _update_archive(self, ap, ao, new_p, new_o):
        if len(ap) == 0:
            cp, co = new_p, new_o
        else:
            cp = np.vstack([ap, new_p])
            co = np.vstack([ao, new_o])

        ranks, crowd = self.pareto.compute(co)
        front0 = np.where(ranks == 0)[0]
        ap, ao = cp[front0], co[front0]

        MAX_ARCH = 100
        if len(ap) > MAX_ARCH:
            _, c = self.pareto.compute(ao)
            keep = np.argsort(-c)[:MAX_ARCH]
            ap, ao = ap[keep], ao[keep]

        return ap, ao

    def summary(self):
        return {
            "algorithm"     : "ACO",
            "total_time_s"  : round(self.total_time, 3),
            "fitness_evals" : self.fitness_evals,
            "final_hv"      : self.history["hypervolume"][-1] if self.history["hypervolume"] else 0.0,
        }
