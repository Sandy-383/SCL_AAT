"""
Particle Swarm Optimization (PSO) — CPU Baseline.

Classic multi-objective PSO (MOPSO) with Pareto archive for comparison.
"""

import time
import logging
import numpy as np
from typing import List, Dict, Tuple, Callable

from optimization.problem_encoder import RouteEncoder
from optimization.multi_objective import ParetoFront

logger = logging.getLogger(__name__)


class PSO:
    def __init__(
        self,
        encoder        : RouteEncoder,
        fitness_fn     : Callable,
        num_particles  : int   = 100,
        max_iterations : int   = 500,
        w              : float = 0.7,    # inertia
        c1             : float = 1.5,    # cognitive coefficient
        c2             : float = 1.5,    # social coefficient
        seed           : int   = 42,
        archive_size   : int   = 100,
        verbose        : bool  = True,
    ):
        self.encoder        = encoder
        self.fitness_fn     = fitness_fn
        self.num_particles  = num_particles
        self.max_iterations = max_iterations
        self.w              = w
        self.c1             = c1
        self.c2             = c2
        self.seed           = seed
        self.archive_size   = archive_size
        self.verbose        = verbose
        self.dim            = encoder.dimension

        self.rng            = np.random.default_rng(seed)
        self.pareto         = ParetoFront()

        self.history        = {"hypervolume": [], "best_f1": []}
        self.total_time     = 0.0
        self.fitness_evals  = 0

    def optimize(self) -> Tuple[List[Dict], np.ndarray]:
        pos      = self.encoder.random_population(self.num_particles, self.rng)
        vel      = self.rng.uniform(-0.1, 0.1, (self.num_particles, self.dim))
        pbest    = pos.copy()
        pbest_obj = self._eval_pop(pos)

        arch_pos = pbest.copy()
        arch_obj = pbest_obj.copy()

        start = time.perf_counter()

        for t in range(self.max_iterations):
            # Select global best from archive (least crowded)
            gbest = self._select_gbest(arch_pos, arch_obj)

            r1 = self.rng.random((self.num_particles, self.dim))
            r2 = self.rng.random((self.num_particles, self.dim))

            vel = (self.w * vel
                   + self.c1 * r1 * (pbest - pos)
                   + self.c2 * r2 * (gbest - pos))
            pos = np.clip(pos + vel, 0.0, 1.0)

            obj = self._eval_pop(pos)

            # Update personal bests
            for i in range(self.num_particles):
                if self._dominates(obj[i], pbest_obj[i]):
                    pbest[i]     = pos[i].copy()
                    pbest_obj[i] = obj[i].copy()

            # Update archive
            arch_pos, arch_obj = self._update_archive(arch_pos, arch_obj, pos, obj)

            hv = self.pareto.hypervolume(arch_obj)
            self.history["hypervolume"].append(hv)
            self.history["best_f1"].append(float(arch_obj[:, 0].min()))

        self.total_time = time.perf_counter() - start
        best_idx = int(np.argmin(arch_obj[:, 0]))
        return self.encoder.decode(arch_pos[best_idx]), arch_obj[best_idx]

    def _eval_pop(self, positions):
        results = np.zeros((len(positions), 4))
        for i, x in enumerate(positions):
            results[i] = self.fitness_fn(self.encoder.decode(x))
            self.fitness_evals += 1
        return results

    def _dominates(self, a, b):
        return bool(np.all(a <= b) and np.any(a < b))

    def _select_gbest(self, arch_pos, arch_obj):
        _, crowd = self.pareto.compute(arch_obj)
        inv_c    = 1.0 / (crowd + 1e-9)
        probs    = inv_c / inv_c.sum()
        idx      = self.rng.choice(len(arch_pos), p=probs)
        return arch_pos[idx]

    def _update_archive(self, ap, ao, np_, no):
        cp = np.vstack([ap, np_])
        co = np.vstack([ao, no])
        ranks, crowd = self.pareto.compute(co)
        front0 = np.where(ranks == 0)[0]
        ap, ao = cp[front0], co[front0]
        if len(ap) > self.archive_size:
            _, c = self.pareto.compute(ao)
            keep = np.argsort(-c)[:self.archive_size]
            ap, ao = ap[keep], ao[keep]
        return ap, ao

    def summary(self):
        return {
            "algorithm"     : "PSO",
            "total_time_s"  : round(self.total_time, 3),
            "fitness_evals" : self.fitness_evals,
            "final_hv"      : self.history["hypervolume"][-1] if self.history["hypervolume"] else 0.0,
        }
