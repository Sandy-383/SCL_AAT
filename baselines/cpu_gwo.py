"""
CPU Grey Wolf Optimizer (Sequential Reference Implementation).

This is the standard GWO algorithm running on CPU with no parallelism.
It serves as:
  1. A correctness reference for the CUDA implementation
  2. A baseline for speedup benchmarking
  3. A fallback when no NVIDIA GPU is available

Algorithm Reference:
  Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014).
  Grey wolf optimizer. Advances in Engineering Software, 69, 46-61.

Multi-objective extension:
  Archive-based approach — maintains a Pareto archive of non-dominated solutions
  found during the search. Leader wolves (α, β, δ) are selected from the archive
  using a roulette-wheel mechanism biased toward less-crowded regions.
"""

import time
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from tqdm import tqdm

from optimization.problem_encoder import RouteEncoder
from optimization.multi_objective import ParetoFront
from config import GWO_CONFIG

logger = logging.getLogger(__name__)


class CPUGWO:
    """
    Multi-Objective Grey Wolf Optimizer — Sequential CPU Implementation.

    Parameters
    ----------
    encoder       : RouteEncoder   — encodes/decodes position vectors
    fitness_fn    : Callable       — f(routes) → np.ndarray shape (4,)
    num_wolves    : int
    max_iterations: int
    seed          : int
    archive_size  : int            — max Pareto archive size
    """

    def __init__(
        self,
        encoder        : RouteEncoder,
        fitness_fn     : Callable,
        num_wolves     : int = GWO_CONFIG["num_wolves"],
        max_iterations : int = GWO_CONFIG["max_iterations"],
        seed           : int = GWO_CONFIG["seed"],
        archive_size   : int = 100,
        verbose        : bool = True,
    ):
        self.encoder        = encoder
        self.fitness_fn     = fitness_fn
        self.num_wolves     = num_wolves
        self.max_iterations = max_iterations
        self.seed           = seed
        self.archive_size   = archive_size
        self.verbose        = verbose
        self.dim            = encoder.dimension

        self.rng            = np.random.default_rng(seed)
        self.pareto         = ParetoFront()

        # History for convergence plots
        self.history = {
            "best_f1"       : [],
            "best_f2"       : [],
            "best_f3"       : [],
            "best_f4"       : [],
            "hypervolume"   : [],
            "iteration_time": [],
        }

        # Statistics
        self.total_time   = 0.0
        self.fitness_evals = 0

    # ── Main Optimization Loop ────────────────────────────────────────────────

    def optimize(self) -> Tuple[List[Dict], np.ndarray]:
        """
        Run GWO and return:
          best_routes   : List[Dict]  — decoded best solution
          best_obj      : np.ndarray  — objective values [f1, f2, f3, f4]
        """
        logger.info("CPU-GWO | wolves=%d | iterations=%d | dim=%d",
                    self.num_wolves, self.max_iterations, self.dim)

        # ── Initialisation ──────────────────────────────────────────────────
        positions   = self.encoder.random_population(self.num_wolves, self.rng)
        objectives  = self._evaluate_population(positions)

        archive_pos = positions.copy()
        archive_obj = objectives.copy()
        archive_pos, archive_obj = self._update_archive(archive_pos, archive_obj,
                                                         positions, objectives)

        start_time = time.perf_counter()

        iterator = tqdm(range(self.max_iterations), desc="CPU-GWO",
                        disable=not self.verbose, ncols=90)

        for iteration in iterator:
            iter_start = time.perf_counter()

            # Linear decrease of 'a' from 2 → 0
            a = 2.0 * (1.0 - iteration / self.max_iterations)

            # Select α, β, δ from archive (least crowded leaders)
            alpha, beta, delta = self._select_leaders(archive_pos, archive_obj)

            # ── Position Update ─────────────────────────────────────────────
            # For each wolf w:
            #   A1 = 2a·r1 - a,   C1 = 2·r2
            #   D_alpha = |C1·alpha - X_w|
            #   X1 = alpha - A1·D_alpha      (and similarly for beta, delta)
            #   X_new = (X1 + X2 + X3) / 3

            r1 = self.rng.random((self.num_wolves, self.dim))
            r2 = self.rng.random((self.num_wolves, self.dim))
            r3 = self.rng.random((self.num_wolves, self.dim))
            r4 = self.rng.random((self.num_wolves, self.dim))
            r5 = self.rng.random((self.num_wolves, self.dim))
            r6 = self.rng.random((self.num_wolves, self.dim))

            A1 = 2 * a * r1 - a;  C1 = 2 * r2
            A2 = 2 * a * r3 - a;  C2 = 2 * r4
            A3 = 2 * a * r5 - a;  C3 = 2 * r6

            D_alpha = np.abs(C1 * alpha - positions)
            D_beta  = np.abs(C2 * beta  - positions)
            D_delta = np.abs(C3 * delta - positions)

            X1 = alpha - A1 * D_alpha
            X2 = beta  - A2 * D_beta
            X3 = delta - A3 * D_delta

            positions = (X1 + X2 + X3) / 3.0
            positions = np.clip(positions, 0.0, 1.0)

            # ── Fitness Evaluation ──────────────────────────────────────────
            objectives = self._evaluate_population(positions)

            # ── Archive Update ──────────────────────────────────────────────
            archive_pos, archive_obj = self._update_archive(
                archive_pos, archive_obj, positions, objectives
            )

            # ── Logging ─────────────────────────────────────────────────────
            iter_time = time.perf_counter() - iter_start
            self._log_iteration(iteration, archive_obj, iter_time)

            if self.verbose and iteration % 50 == 0:
                best_idx = np.argmin(archive_obj[:, 0])
                iterator.set_postfix({
                    "f1": f"{archive_obj[best_idx, 0]:.2f}",
                    "HV": f"{self.history['hypervolume'][-1]:.4f}",
                })

        self.total_time = time.perf_counter() - start_time
        logger.info("CPU-GWO done | %.2fs | %d evals", self.total_time, self.fitness_evals)

        # Return the least-crowded Pareto-front solution
        best_pos, best_obj = self._best_solution(archive_pos, archive_obj)
        best_routes = self.encoder.decode(best_pos)
        return best_routes, best_obj

    # ── Leader Selection ──────────────────────────────────────────────────────

    def _select_leaders(
        self,
        archive_pos : np.ndarray,
        archive_obj : np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Select α, β, δ from the Pareto archive.
        Uses roulette-wheel selection inversely proportional to crowding distance
        so that under-explored regions are preferred.
        """
        n = len(archive_pos)
        if n <= 3:
            # Not enough archive members — pick random
            idx = self.rng.choice(n, size=3, replace=(n < 3))
            return archive_pos[idx[0]], archive_pos[idx[1]], archive_pos[idx[2]]

        _, crowding = self.pareto.compute(archive_obj)

        # Finite crowding distances only
        finite_mask = np.isfinite(crowding)
        if finite_mask.sum() < 3:
            idx = self.rng.choice(n, size=3, replace=False)
        else:
            # Inverse crowding → probability (less crowded = higher chance)
            inv_crowd = 1.0 / (crowding + 1e-9)
            probs     = inv_crowd / inv_crowd.sum()
            idx       = self.rng.choice(n, size=3, replace=False, p=probs)

        return archive_pos[idx[0]], archive_pos[idx[1]], archive_pos[idx[2]]

    # ── Archive Management ────────────────────────────────────────────────────

    def _update_archive(
        self,
        arch_pos  : np.ndarray,
        arch_obj  : np.ndarray,
        new_pos   : np.ndarray,
        new_obj   : np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Merge current archive with new solutions, keep non-dominated only,
        prune to archive_size using crowding distance if needed.
        """
        combined_pos = np.vstack([arch_pos, new_pos])
        combined_obj = np.vstack([arch_obj, new_obj])

        # Keep only non-dominated (rank 0)
        ranks, crowding = self.pareto.compute(combined_obj)
        front0          = np.where(ranks == 0)[0]

        arch_pos = combined_pos[front0]
        arch_obj = combined_obj[front0]

        # Prune to archive_size using crowding distance
        if len(arch_pos) > self.archive_size:
            _, crowd = self.pareto.compute(arch_obj)
            keep     = np.argsort(-crowd)[:self.archive_size]  # keep most diverse
            arch_pos = arch_pos[keep]
            arch_obj = arch_obj[keep]

        return arch_pos, arch_obj

    # ── Best Solution ─────────────────────────────────────────────────────────

    def _best_solution(
        self,
        arch_pos : np.ndarray,
        arch_obj : np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return the solution with minimum f1 (travel time) from the archive."""
        best_idx = int(np.argmin(arch_obj[:, 0]))
        return arch_pos[best_idx], arch_obj[best_idx]

    # ── Fitness Evaluation ────────────────────────────────────────────────────

    def _evaluate_population(self, positions: np.ndarray) -> np.ndarray:
        """Evaluate all wolves. Pure Python loop — this is what CUDA replaces."""
        n        = len(positions)
        results  = np.zeros((n, 4), dtype=np.float64)
        for i in range(n):
            routes      = self.encoder.decode(positions[i])
            results[i]  = self.fitness_fn(routes)
            self.fitness_evals += 1
        return results

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log_iteration(
        self,
        iteration  : int,
        archive_obj: np.ndarray,
        iter_time  : float,
    ):
        hv = self.pareto.hypervolume(archive_obj)
        best_idx = np.argmin(archive_obj[:, 0])
        self.history["best_f1"].append(float(archive_obj[best_idx, 0]))
        self.history["best_f2"].append(float(archive_obj[best_idx, 1]))
        self.history["best_f3"].append(float(archive_obj[best_idx, 2]))
        self.history["best_f4"].append(float(archive_obj[best_idx, 3]))
        self.history["hypervolume"].append(hv)
        self.history["iteration_time"].append(iter_time)

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> Dict:
        return {
            "algorithm"       : "CPU-GWO",
            "num_wolves"      : self.num_wolves,
            "max_iterations"  : self.max_iterations,
            "dimension"       : self.dim,
            "total_time_s"    : round(self.total_time, 3),
            "fitness_evals"   : self.fitness_evals,
            "evals_per_second": round(self.fitness_evals / max(self.total_time, 1e-9), 1),
            "final_hv"        : self.history["hypervolume"][-1] if self.history["hypervolume"] else 0.0,
        }
