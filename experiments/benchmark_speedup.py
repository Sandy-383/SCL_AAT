"""
Speedup Benchmark.

Measures wall-clock execution time for all algorithms across
different problem sizes and records speedup of CUDA-GWO vs baselines.

Experiment design (publishable standard):
  - 5 independent runs per (algorithm × problem_size)
  - Report mean ± std
  - Wilcoxon signed-rank test for statistical significance
  - Problem sizes: 100, 200, 500, 1000 stops
"""

import time
import json
import logging
import numpy as np
from typing import Dict, List
from scipy import stats

logger = logging.getLogger(__name__)


class SpeedupBenchmark:
    """
    Runs timing experiments across algorithms and problem sizes.

    Parameters
    ----------
    output_dir : directory to save JSON results
    n_runs     : number of independent runs per configuration
    """

    PROBLEM_SIZES = [100, 200, 500, 1000]
    N_RUNS        = 5

    def __init__(self, output_dir: str = "results", n_runs: int = 5):
        self.output_dir = output_dir
        self.n_runs     = n_runs

    def run(
        self,
        encoder_factory,    # callable(num_stops) -> RouteEncoder
        fitness_factory,    # callable(city_data) -> fitness_fn
        city_generator,     # callable(num_stops) -> city_data dict
        algorithms          : Dict,  # {"cuda_gwo": CUDAGWOBinding, "cpu_gwo": CPUGWO, ...}
        max_iterations      : int = 100,   # Reduced for benchmarking
        verbose             : bool = True,
    ) -> Dict:
        """
        Run all benchmarks and return a results dict.

        Returns
        -------
        results : {
            "timings": {
                problem_size: {
                    algo: {"mean": float, "std": float, "runs": [float,...]}
                }
            },
            "speedup": {
                problem_size: {algo: float}  — speedup vs CUDA-GWO
            },
            "statistical_tests": {
                problem_size: {
                    "cuda_vs_cpu": {"p_value": float, "significant": bool}
                }
            }
        }
        """
        results = {"timings": {}, "speedup": {}, "statistical_tests": {}}

        for n_stops in self.PROBLEM_SIZES:
            logger.info("=== Benchmark: %d stops ===", n_stops)
            size_results = {}

            city_data = city_generator(n_stops)
            encoder   = encoder_factory(n_stops)
            fitness   = fitness_factory(city_data)

            for algo_name, algo_instance in algorithms.items():
                times = []
                for run in range(self.n_runs):
                    logger.info("  %s run %d/%d", algo_name, run+1, self.n_runs)

                    t0 = time.perf_counter()
                    try:
                        if algo_name == "cuda_gwo":
                            algo_instance.optimize(
                                encoder, fitness,
                                city_data["adj_matrix"],
                                city_data["demand"],
                                max_iterations=max_iterations,
                                verbose=False,
                            )
                        else:
                            # CPU baselines: reinitialise and run
                            algo = algo_instance(
                                encoder=encoder,
                                fitness_fn=fitness,
                                max_iterations=max_iterations,
                                verbose=False,
                            )
                            algo.optimize()
                    except Exception as e:
                        logger.error("  %s failed: %s", algo_name, e)
                        times.append(None)
                        continue

                    elapsed = time.perf_counter() - t0
                    times.append(elapsed)

                valid_times = [t for t in times if t is not None]
                size_results[algo_name] = {
                    "mean"  : float(np.mean(valid_times)) if valid_times else None,
                    "std"   : float(np.std(valid_times))  if valid_times else None,
                    "runs"  : valid_times,
                }

            results["timings"][n_stops] = size_results

            # Compute speedups relative to CUDA-GWO
            cuda_mean = size_results.get("cuda_gwo", {}).get("mean", None)
            if cuda_mean and cuda_mean > 0:
                results["speedup"][n_stops] = {
                    algo: size_results[algo]["mean"] / cuda_mean
                    for algo in size_results
                    if size_results[algo]["mean"] is not None
                }

            # Wilcoxon test: CUDA-GWO vs CPU-GWO
            cuda_runs = size_results.get("cuda_gwo", {}).get("runs", [])
            cpu_runs  = size_results.get("cpu_gwo",  {}).get("runs", [])
            if len(cuda_runs) >= 3 and len(cpu_runs) >= 3:
                try:
                    stat, p = stats.wilcoxon(cuda_runs[:len(cpu_runs)], cpu_runs[:len(cuda_runs)])
                    results["statistical_tests"][n_stops] = {
                        "cuda_vs_cpu": {
                            "statistic" : float(stat),
                            "p_value"   : float(p),
                            "significant": bool(p < 0.05),
                        }
                    }
                except Exception:
                    pass

        # Save results
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        out_path = os.path.join(self.output_dir, "speedup_benchmark.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Speedup benchmark results saved: %s", out_path)

        self._print_table(results)
        return results

    def _print_table(self, results: Dict):
        print("\n" + "="*70)
        print("SPEEDUP BENCHMARK RESULTS")
        print("="*70)
        for size in self.PROBLEM_SIZES:
            if size not in results["timings"]:
                continue
            print(f"\nProblem size: {size} stops")
            print(f"  {'Algorithm':<20} {'Mean (s)':<12} {'Std (s)':<10} {'Speedup':<10}")
            print("  " + "-"*52)
            for algo, vals in results["timings"][size].items():
                mean    = f"{vals['mean']:.3f}" if vals["mean"] else "N/A"
                std     = f"{vals['std']:.3f}"  if vals["std"]  else "N/A"
                speedup = results.get("speedup", {}).get(size, {}).get(algo, 1.0)
                su_str  = f"{speedup:.1f}×"
                print(f"  {algo:<20} {mean:<12} {std:<10} {su_str:<10}")
        print("="*70 + "\n")
