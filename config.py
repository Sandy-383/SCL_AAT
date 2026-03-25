"""
Global configuration for CUDA-GWO Public Transport Route Optimization.
All hyperparameters, paths, and problem settings are centralized here.
"""

import os

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data", "gtfs")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(DATA_DIR,    exist_ok=True)
os.makedirs(OUTPUT_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── GWO Hyperparameters ─────────────────────────────────────────────────────
GWO_CONFIG = {
    "num_wolves"    : 100,       # Population size (wolves in the pack)
    "max_iterations": 500,       # Maximum iterations
    "dim"           : None,      # Set dynamically based on problem size
    "a_start"       : 2.0,       # Initial value of 'a' (linearly decreases to 0)
    "a_end"         : 0.0,
    "seed"          : 42,
}

# ─── CUDA Configuration ───────────────────────────────────────────────────────
CUDA_CONFIG = {
    "block_size"        : 256,   # Threads per block
    "shared_mem_bytes"  : 48 * 1024,  # 48 KB shared memory per block
    "device_id"         : 0,
}

# ─── Problem Parameters ───────────────────────────────────────────────────────
PROBLEM_CONFIG = {
    "num_routes"          : 10,     # Number of routes to optimize
    "min_stops_per_route" : 3,
    "max_stops_per_route" : 20,
    "vehicle_capacity"    : 60,     # Passengers per vehicle
    "min_headway_min"     : 5,      # Minimum headway (minutes)
    "max_headway_min"     : 60,     # Maximum headway (minutes)
    "operating_hours"     : 18,     # Daily operating hours
    "avg_speed_kmh"       : 25.0,   # Average bus speed
    "transfer_penalty"    : 5.0,    # Minutes penalty per transfer
    "co2_per_km"          : 0.089,  # kg CO2 per km per vehicle
}

# ─── Multi-Objective Weights (used in weighted-sum scalarization for display) ─
OBJECTIVE_WEIGHTS = {
    "travel_time" : 0.35,
    "transfers"   : 0.25,
    "cost"        : 0.25,
    "coverage"    : 0.15,   # Note: this is maximized, so inverted internally
}

# ─── Benchmark Baselines ─────────────────────────────────────────────────────
BASELINE_CONFIGS = {
    "cpu_gwo": {**GWO_CONFIG},
    "pso"    : {"num_particles": 100, "max_iterations": 500, "w": 0.7, "c1": 1.5, "c2": 1.5},
    "ga"     : {"population": 100, "max_generations": 500, "crossover_rate": 0.8, "mutation_rate": 0.02},
    "aco"    : {"num_ants": 100, "max_iterations": 500, "alpha": 1.0, "beta": 2.0, "rho": 0.1},
}

# ─── Visualization ────────────────────────────────────────────────────────────
VIZ_CONFIG = {
    "map_tiles"     : "OpenStreetMap",
    "route_colors"  : [
        "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
        "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    ],
    "stop_radius"   : 6,
    "route_weight"  : 3,
}
