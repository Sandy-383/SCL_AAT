"""
Genetic Algorithm (GA) — CPU Baseline.

NSGA-II style multi-objective GA for route optimization comparison.
"""

import time
import logging
import numpy as np
from typing import List, Dict, Tuple, Callable

from optimization.problem_encoder import RouteEncoder
from optimization.multi_objective import ParetoFront

logger = logging.getLogger(__name__)


class GeneticAlgorithm:
    def __init__(
        self,
        encoder         : RouteEncoder,
        fitness_fn      : Callable,
        population      : int   = 100,
        max_generations : int   = 500,
        crossover_rate  : float = 0.8,
        mutation_rate   : float = 0.02,
        seed            : int   = 42,
        verbose         : bool  = True,
    ):
        self.encoder        = encoder
        self.fitness_fn     = fitness_fn
        self.pop_size       = population
        self.max_gen        = max_generations
        self.cx_rate        = crossover_rate
        self.mut_rate       = mutation_rate
        self.rng            = np.random.default_rng(seed)
        self.verbose        = verbose
        self.dim            = encoder.dimension
        self.pareto         = ParetoFront()

        self.history        = {"hypervolume": [], "best_f1": []}
        self.total_time     = 0.0
        self.fitness_evals  = 0

    def optimize(self) -> Tuple[List[Dict], np.ndarray]:
        pop = self.encoder.random_population(self.pop_size, self.rng)
        obj = self._eval_pop(pop)

        start = time.perf_counter()

        for gen in range(self.max_gen):
            # Tournament selection
            parents = self._tournament_select(pop, obj)

            # SBX crossover
            offspring = self._crossover(parents)

            # Polynomial mutation
            offspring = self._mutate(offspring)
            offspring = np.clip(offspring, 0.0, 1.0)

            off_obj = self._eval_pop(offspring)

            # NSGA-II environmental selection
            combined     = np.vstack([pop, offspring])
            combined_obj = np.vstack([obj, off_obj])
            pop, obj     = self._nsga2_select(combined, combined_obj)

            hv = self.pareto.hypervolume(obj)
            self.history["hypervolume"].append(hv)
            self.history["best_f1"].append(float(obj[:, 0].min()))

        self.total_time = time.perf_counter() - start
        best_idx = int(np.argmin(obj[:, 0]))
        return self.encoder.decode(pop[best_idx]), obj[best_idx]

    def _eval_pop(self, pop):
        results = np.zeros((len(pop), 4))
        for i, x in enumerate(pop):
            results[i] = self.fitness_fn(self.encoder.decode(x))
            self.fitness_evals += 1
        return results

    def _tournament_select(self, pop, obj, k=2):
        N = len(pop)
        selected = np.zeros_like(pop)
        ranks, crowding = self.pareto.compute(obj)

        for i in range(N):
            candidates = self.rng.choice(N, k, replace=False)
            best = candidates[0]
            for c in candidates[1:]:
                if (ranks[c] < ranks[best]
                        or (ranks[c] == ranks[best] and crowding[c] > crowding[best])):
                    best = c
            selected[i] = pop[best]
        return selected

    def _crossover(self, parents):
        N        = len(parents)
        eta      = 15.0  # SBX distribution index
        offspring = parents.copy()

        for i in range(0, N - 1, 2):
            if self.rng.random() < self.cx_rate:
                u  = self.rng.random(self.dim)
                beta = np.where(u <= 0.5,
                                (2 * u) ** (1 / (eta + 1)),
                                (1 / (2 * (1 - u))) ** (1 / (eta + 1)))
                offspring[i]     = 0.5 * ((1 + beta) * parents[i] + (1 - beta) * parents[i+1])
                offspring[i + 1] = 0.5 * ((1 - beta) * parents[i] + (1 + beta) * parents[i+1])

        return np.clip(offspring, 0.0, 1.0)

    def _mutate(self, pop):
        eta = 20.0  # polynomial mutation index
        for i in range(len(pop)):
            mask = self.rng.random(self.dim) < self.mut_rate
            if mask.any():
                u     = self.rng.random(self.dim)
                delta = np.where(u < 0.5,
                                 (2 * u) ** (1 / (eta + 1)) - 1,
                                 1 - (2 * (1 - u)) ** (1 / (eta + 1)))
                pop[i] += mask * delta
        return pop

    def _nsga2_select(self, pop, obj):
        ranks, crowding = self.pareto.compute(obj)
        # Sort by (rank ASC, crowding DESC) and pick top pop_size
        sort_key = ranks - crowding * 1e-9   # tie-break by crowding
        idx      = np.argsort(sort_key)[:self.pop_size]
        return pop[idx], obj[idx]

    def summary(self):
        return {
            "algorithm"     : "GA (NSGA-II)",
            "total_time_s"  : round(self.total_time, 3),
            "fitness_evals" : self.fitness_evals,
            "final_hv"      : self.history["hypervolume"][-1] if self.history["hypervolume"] else 0.0,
        }
