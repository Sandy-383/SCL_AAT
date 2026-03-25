"""
CUDA Grey Wolf Optimizer — Public Transport Route Optimization
==============================================================
Main entry point.

Usage:
    python main.py                          # full run: all algorithms
    python main.py --algo cuda_gwo          # single algorithm
    python main.py --algo cpu_gwo --quick   # quick test (100 iterations)
    python main.py --benchmark              # speedup benchmark only
    python main.py --gtfs path/to/gtfs      # use real GTFS feed
    python main.py --stops 500              # synthetic city with 500 stops

Workflow:
  1. Load / generate city network
  2. Build transit graph
  3. Set up encoder, fitness function, constraints
  4. Run selected algorithm(s)
  5. Evaluate solution quality
  6. Generate all visualizations
  7. Print/save results summary
"""

import os
import sys
import time
import logging
import argparse
import numpy as np

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

# ── Imports ───────────────────────────────────────────────────────────────────
from config import (
    GWO_CONFIG, PROBLEM_CONFIG, BASELINE_CONFIGS, DATA_DIR, OUTPUT_DIR, RESULTS_DIR
)
from data.gtfs_loader       import load_or_generate
from data.graph_builder     import TransitGraph
from optimization.problem_encoder  import RouteEncoder
from optimization.constraints      import ConstraintHandler
from optimization.multi_objective  import ObjectiveEvaluator, ParetoFront
from baselines.cpu_gwo      import CPUGWO
from baselines.pso          import PSO
from baselines.genetic_algorithm import GeneticAlgorithm
from baselines.aco          import AntColonyOptimization
from cuda_gwo.cuda_gwo_binding import CUDAGWOBinding
from visualization.route_map     import RouteMapVisualizer
from visualization.convergence_plot import ConvergencePlotter
from experiments.solution_quality   import SolutionQualityEvaluator


# ─── Argument Parser ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="CUDA-GWO Transit Route Optimizer"
    )
    p.add_argument("--algo",       default="all",
                   choices=["all", "cuda_gwo", "cpu_gwo", "pso", "ga", "aco"],
                   help="Algorithm to run (default: all)")
    p.add_argument("--stops",      type=int, default=200,
                   help="Number of stops in synthetic city (default: 200)")
    p.add_argument("--routes",     type=int, default=10,
                   help="Number of routes to optimize (default: 10)")
    p.add_argument("--wolves",     type=int, default=GWO_CONFIG["num_wolves"],
                   help="Population size (default: 100)")
    p.add_argument("--iters",      type=int, default=GWO_CONFIG["max_iterations"],
                   help="Max iterations (default: 500)")
    p.add_argument("--quick",      action="store_true",
                   help="Quick test: 50 wolves, 50 iterations")
    p.add_argument("--gtfs",       default=None,
                   help="Path to GTFS directory (uses synthetic if not provided)")
    p.add_argument("--benchmark",  action="store_true",
                   help="Run speedup benchmark across problem sizes")
    p.add_argument("--no-vis",     action="store_true",
                   help="Skip visualization generation")
    p.add_argument("--seed",       type=int, default=GWO_CONFIG["seed"])
    return p.parse_args()


# ─── City & Problem Setup ────────────────────────────────────────────────────

def setup_problem(args):
    """Load city data and initialise all problem components."""
    logger.info("Setting up problem: %d stops, %d routes", args.stops, args.routes)

    # Load / generate city network
    city_data = load_or_generate(
        gtfs_dir   = args.gtfs or DATA_DIR,
        num_stops  = args.stops,
        num_routes = args.routes,
        seed       = args.seed,
    )

    # Build transit graph
    graph = TransitGraph(city_data)
    logger.info("Graph: %s", graph.summary())

    # Problem encoder
    encoder = RouteEncoder(
        num_stops    = graph.n,
        num_routes   = args.routes,
        max_stops    = PROBLEM_CONFIG["max_stops_per_route"],
        min_stops    = PROBLEM_CONFIG["min_stops_per_route"],
        min_headway  = PROBLEM_CONFIG["min_headway_min"],
        max_headway  = PROBLEM_CONFIG["max_headway_min"],
    )
    logger.info("Encoder: %s", encoder.info())

    # Constraint handler
    constraints = ConstraintHandler(num_stops=graph.n)

    # Objective evaluator
    evaluator = ObjectiveEvaluator(
        graph       = graph,
        demand      = city_data["demand"],
        constraints = constraints,
        config      = PROBLEM_CONFIG,
    )

    # Fitness function (single callable for all algorithms)
    def fitness_fn(routes):
        return evaluator.evaluate(routes)

    return city_data, graph, encoder, fitness_fn


# ─── Algorithm Runners ───────────────────────────────────────────────────────

def run_cuda_gwo(encoder, fitness_fn, city_data, args) -> dict:
    binding = CUDAGWOBinding()
    if binding.available:
        logger.info("Running CUDA-GWO on GPU…")
    else:
        logger.info("CUDA not available — running CUDA-GWO in Python simulation mode")

    t0 = time.perf_counter()
    best_routes, best_obj, history = binding.optimize(
        encoder        = encoder,
        fitness_fn     = fitness_fn,
        adj_matrix     = city_data["adj_matrix"],
        demand         = city_data["demand"],
        num_wolves     = args.wolves,
        max_iterations = args.iters,
        seed           = args.seed,
        verbose        = True,
    )
    runtime = time.perf_counter() - t0

    return {
        "algo"        : "cuda_gwo",
        "best_routes" : best_routes,
        "best_obj"    : best_obj,
        "history"     : history,
        "runtime_s"   : runtime,
        "binding"     : binding,
    }


def run_cpu_gwo(encoder, fitness_fn, args) -> dict:
    logger.info("Running CPU-GWO…")
    t0   = time.perf_counter()
    algo = CPUGWO(
        encoder        = encoder,
        fitness_fn     = fitness_fn,
        num_wolves     = args.wolves,
        max_iterations = args.iters,
        seed           = args.seed,
        verbose        = True,
    )
    best_routes, best_obj = algo.optimize()
    runtime = time.perf_counter() - t0
    return {
        "algo"        : "cpu_gwo",
        "best_routes" : best_routes,
        "best_obj"    : best_obj,
        "history"     : algo.history,
        "runtime_s"   : runtime,
        "summary"     : algo.summary(),
    }


def run_pso(encoder, fitness_fn, args) -> dict:
    logger.info("Running PSO…")
    t0   = time.perf_counter()
    algo = PSO(
        encoder        = encoder,
        fitness_fn     = fitness_fn,
        num_particles  = args.wolves,
        max_iterations = args.iters,
        seed           = args.seed,
        verbose        = True,
    )
    best_routes, best_obj = algo.optimize()
    runtime = time.perf_counter() - t0
    return {
        "algo"        : "pso",
        "best_routes" : best_routes,
        "best_obj"    : best_obj,
        "history"     : algo.history,
        "runtime_s"   : runtime,
    }


def run_ga(encoder, fitness_fn, args) -> dict:
    logger.info("Running GA (NSGA-II)…")
    t0   = time.perf_counter()
    algo = GeneticAlgorithm(
        encoder         = encoder,
        fitness_fn      = fitness_fn,
        population      = args.wolves,
        max_generations = args.iters,
        seed            = args.seed,
        verbose         = True,
    )
    best_routes, best_obj = algo.optimize()
    runtime = time.perf_counter() - t0
    return {
        "algo"        : "ga",
        "best_routes" : best_routes,
        "best_obj"    : best_obj,
        "history"     : algo.history,
        "runtime_s"   : runtime,
    }


def run_aco(encoder, fitness_fn, args) -> dict:
    logger.info("Running ACO…")
    t0   = time.perf_counter()
    algo = AntColonyOptimization(
        encoder        = encoder,
        fitness_fn     = fitness_fn,
        num_ants       = args.wolves,
        max_iterations = args.iters,
        seed           = args.seed,
        verbose        = True,
    )
    best_routes, best_obj = algo.optimize()
    runtime = time.perf_counter() - t0
    return {
        "algo"        : "aco",
        "best_routes" : best_routes,
        "best_obj"    : best_obj,
        "history"     : algo.history,
        "runtime_s"   : runtime,
    }


# ─── Results & Visualization ─────────────────────────────────────────────────

def generate_outputs(run_results: list, city_data, graph, args):
    """Generate all output files: maps, plots, quality report."""
    pareto = ParetoFront()
    vis    = RouteMapVisualizer(graph, output_dir=OUTPUT_DIR)
    plotter = ConvergencePlotter(output_dir=OUTPUT_DIR)
    quality = SolutionQualityEvaluator(output_dir=RESULTS_DIR)

    histories         : dict = {}
    pareto_objectives : dict = {}
    timings           : dict = {}

    for res in run_results:
        algo = res["algo"]
        histories[algo]         = res["history"]
        timings[algo]           = res["runtime_s"]

        # Pareto front of this run (just best_obj for now — can be expanded)
        obj = np.array(res["best_obj"]).reshape(1, -1)
        pareto_objectives[algo] = obj

        quality.add_result(
            algo_name         = algo,
            pareto_objectives = obj,
            history           = res["history"],
            runtime_s         = res["runtime_s"],
        )

    # ── Route Maps ────────────────────────────────────────────────────────
    if not args.no_vis:
        # Find CUDA-GWO or first result for map
        map_res = next(
            (r for r in run_results if r["algo"] == "cuda_gwo"),
            run_results[0]
        )
        vis.save_interactive_map(map_res["best_routes"], filename="route_map.html")
        vis.save_static_map(map_res["best_routes"], filename="route_map_static.png")

        if len(run_results) > 1:
            vis.save_comparison_map(
                run_results[-1]["best_routes"],  # last baseline
                map_res["best_routes"],           # CUDA-GWO
            )

        # ── Convergence Plots ─────────────────────────────────────────────
        if len(histories) > 0:
            plotter.plot_hypervolume(histories)
            plotter.plot_objective_convergence(histories)
            plotter.plot_speedup(timings)
            plotter.plot_pareto_front(pareto_objectives)
            plotter.plot_dashboard(histories, timings, pareto_objectives)

    # ── Quality Report ────────────────────────────────────────────────────
    report = quality.evaluate()
    quality.save_latex_table(report)

    return report


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.quick:
        args.wolves = 50
        args.iters  = 50
        logger.info("Quick mode: 50 wolves, 50 iterations")

    # ── Setup ────────────────────────────────────────────────────────────
    city_data, graph, encoder, fitness_fn = setup_problem(args)

    # ── Select algorithms to run ────────────────────────────────────────
    algo_map = {
        "cuda_gwo" : lambda: run_cuda_gwo(encoder, fitness_fn, city_data, args),
        "cpu_gwo"  : lambda: run_cpu_gwo(encoder, fitness_fn, args),
        "pso"      : lambda: run_pso(encoder, fitness_fn, args),
        "ga"       : lambda: run_ga(encoder, fitness_fn, args),
        "aco"      : lambda: run_aco(encoder, fitness_fn, args),
    }

    to_run = list(algo_map.keys()) if args.algo == "all" else [args.algo]
    logger.info("Algorithms to run: %s", to_run)

    # ── Run algorithms ───────────────────────────────────────────────────
    run_results = []
    total_start = time.perf_counter()

    for algo_name in to_run:
        logger.info("\n" + "="*60)
        logger.info("ALGORITHM: %s", algo_name.upper())
        logger.info("="*60)
        try:
            res = algo_map[algo_name]()
            run_results.append(res)
            logger.info(
                "%s done | runtime: %.2fs | best f1: %.3f",
                algo_name, res["runtime_s"], res["best_obj"][0]
            )
        except Exception as e:
            logger.error("Failed to run %s: %s", algo_name, e, exc_info=True)

    total_time = time.perf_counter() - total_start
    logger.info("\nTotal runtime: %.2fs", total_time)

    if not run_results:
        logger.error("No algorithms completed successfully.")
        sys.exit(1)

    # ── Generate outputs ────────────────────────────────────────────────
    report = generate_outputs(run_results, city_data, graph, args)

    # ── Final summary ────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    for res in run_results:
        obj = res["best_obj"]
        print(
            f"  {res['algo']:<18} "
            f"f1={obj[0]:.2f} min  "
            f"f2={obj[1]:.3f} tx  "
            f"f3={obj[2]:.1f} k₹  "
            f"f4={obj[3]:.3f}  "
            f"time={res['runtime_s']:.2f}s"
        )
    print("="*60)
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
