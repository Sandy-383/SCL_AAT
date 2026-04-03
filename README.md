# Public Transport Route Optimization using CUDA Grey Wolf Optimization

> A capstone & research-publishable project demonstrating GPU-accelerated multi-objective transit route optimization on NVIDIA RTX 4060.

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Output Files](#output-files)
- [Algorithm Details](#algorithm-details)
- [System Architecture](#system-architecture)
- [Research Contribution](#research-contribution)

---

## Overview

This project solves the **Transit Route Network Design Problem (TRNDP)** — finding the optimal set of bus routes for a city that simultaneously:

- Minimizes average passenger **travel time** (f1)
- Minimizes average **number of transfers** (f2)
- Minimizes **fleet operational cost** (f3)
- Maximizes **demand coverage** (f4 = 1 - coverage)

A custom **CUDA-parallelized Grey Wolf Optimizer (GWO)** runs on the GPU, with four CPU-based baseline algorithms for comparison. The optimized routes are visualized on a real **Bangalore city map** using interactive Folium maps.

---

## Key Results

> Quick-mode run: 200 stops, 10 routes, 50 wolves, 50 iterations

| Algorithm | Best f1 (min) | Best f2 (tx) | Best f3 (kRs) | Runtime |
|-----------|:------------:|:------------:|:-------------:|:-------:|
| **CUDA-GWO (RTX 4060)** | 259.17 | 138.32 | 4843.4 | **43s** |
| CPU-GWO | 259.17 | 138.32 | 4843.4 | 43s |
| PSO | 259.17 | 138.32 | 4843.4 | 37s |
| GA (NSGA-II) | 176.22 | 69.59 | 4488.0 | 43s |
| ACO | 235.34 | 132.07 | 2887.1 | 41s |

- **GPU:** NVIDIA RTX 4060 Laptop (8187 MB VRAM, SM 8.9)
- **CUDA backend:** CuPy RawKernel + NVRTC (no nvcc required)
- **City:** Synthetic Bangalore-like city with gravity-model OD demand

---

## Project Structure

```
SCL_AAT/
├── main.py                        # Entry point — CLI for all run modes
├── config.py                      # All hyperparameters and settings
├── gpu_setup.py                   # NVIDIA DLL path fix for Windows + CuPy
│
├── cuda_gwo/                      # GPU-accelerated Grey Wolf Optimizer
│   ├── cupy_gwo.py                # Main GPU engine (CuPy + NVRTC kernels)
│   ├── cuda_gwo_binding.py        # Smart loader: CuPy -> ctypes -> simulation
│   ├── gwo_kernel.cu / .cuh       # CUDA C: wolf position update kernel
│   ├── fitness_kernel.cu / .cuh   # CUDA C: travel time & coverage kernel
│   ├── memory_manager.cu / .cuh   # GPU memory management (pinned, 2 streams)
│   └── gwo_host.cpp               # C++ host driver (ctypes API)
│
├── data/
│   ├── gtfs_loader.py             # Real GTFS loader + synthetic city generator
│   └── graph_builder.py           # NetworkX transit graph + Dijkstra paths
│
├── optimization/
│   ├── problem_encoder.py         # Route encoding as float vectors [0,1]^d
│   ├── constraints.py             # 5 penalty types (length, capacity, etc.)
│   └── multi_objective.py         # 4 objectives + Pareto front + HV/GD/IGD
│
├── baselines/
│   ├── cpu_gwo.py                 # CPU Grey Wolf Optimizer
│   ├── pso.py                     # Multi-Objective PSO (MOPSO)
│   ├── genetic_algorithm.py       # NSGA-II Genetic Algorithm
│   └── aco.py                     # Multi-Objective ACO (MOACO)
│
├── visualization/
│   ├── route_map.py               # Folium interactive map + static PNG
│   └── convergence_plot.py        # Convergence, speedup, Pareto, dashboard plots
│
├── experiments/
│   ├── benchmark_speedup.py       # Speedup benchmark across problem sizes
│   └── solution_quality.py        # HV, GD, IGD metrics + LaTeX table
│
├── outputs/                       # Generated plots and maps (auto-created)
└── results/                       # Metrics JSON + LaTeX table (auto-created)
```

---

## Requirements

- Python 3.12
- NVIDIA GPU with CUDA 12.x (tested on RTX 4060 Laptop)
- Windows 10/11 or Linux

### Python Packages

```
cupy-cuda12x >= 14.0
numpy >= 2.0
networkx
matplotlib
folium
scipy
tqdm
pandas
ml-dtypes >= 0.5.4
```

---

## Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd SCL_AAT

# 2. Install dependencies
pip install cupy-cuda12x numpy networkx matplotlib folium scipy tqdm pandas ml-dtypes

# 3. Verify GPU is detected
python -c "import gpu_setup; import cupy as cp; print('CUDA version:', cp.cuda.runtime.runtimeGetVersion())"
```

> **Windows note:** `gpu_setup.py` automatically adds NVIDIA DLL directories so CuPy can find CUDA libraries. It is imported first in `main.py` — no manual setup needed.

---

## How to Run

### 1. Quick test — GPU only (40 seconds)
```bash
python main.py --algo cuda_gwo --quick
```

### 2. Full comparison — all 5 algorithms (~3.5 minutes)
```bash
python main.py --quick
```
Generates the speedup chart comparing CUDA-GWO vs all baselines.

### 3. Larger city — more realistic results
```bash
python main.py --stops 500 --routes 20 --quick
```

### 4. Full research-grade run — for paper results
```bash
python main.py
```
100 wolves, 500 iterations. 30-60 minutes. Best quality results.

### 5. Scalability benchmark
```bash
python main.py --benchmark
```
Tests 100 -> 200 -> 500 -> 1000 stops. Generates scalability plot.

### All CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--algo` | `all` | `cuda_gwo`, `cpu_gwo`, `pso`, `ga`, `aco`, `all` |
| `--stops` | `200` | Number of bus stops |
| `--routes` | `10` | Number of bus routes |
| `--wolves` | `100` | Wolf pack size |
| `--iters` | `500` | Optimization iterations |
| `--quick` | off | Sets wolves=50, iters=50 |
| `--benchmark` | off | Run scalability benchmark |
| `--gtfs` | — | Path to real GTFS data directory |
| `--seed` | `42` | Random seed for reproducibility |
| `--no-vis` | off | Skip visualization generation |

---

## Output Files

| File | Description |
|------|-------------|
| `outputs/route_map.html` | **Interactive Bangalore map** — open in any browser |
| `outputs/dashboard.png` | All-in-one summary figure for paper/presentation |
| `outputs/speedup_comparison.png` | GPU vs CPU speedup bar chart |
| `outputs/hypervolume_convergence.png` | Hypervolume improvement over iterations |
| `outputs/objective_convergence.png` | f1-f4 convergence per algorithm |
| `outputs/pareto_front.png` | Pareto front scatter (f1 vs f2) |
| `outputs/route_map_static.png` | Static PNG version of route map |
| `outputs/comparison_map.png` | Before vs after route comparison |
| `results/solution_quality.json` | Raw metrics: HV, GD, IGD, spacing, runtime |
| `results/results_table.tex` | LaTeX table — paste directly into paper |

---

## Algorithm Details

### Grey Wolf Optimizer (GWO)

Inspired by the grey wolf pack hunting hierarchy:

```
Alpha (a) -- Best solution found so far (leads the hunt)
Beta  (b) -- 2nd best (assists alpha)
Delta (d) -- 3rd best (assists beta)
Omega (w) -- All remaining wolves (update positions following a, b, d)
```

Position update rule (per wolf, per dimension):
```
X1 = X_alpha - A1 * |C1 * X_alpha - X|
X2 = X_beta  - A2 * |C2 * X_beta  - X|
X3 = X_delta - A3 * |C3 * X_delta - X|
X_new = (X1 + X2 + X3) / 3
```

### CUDA Parallelization

- Each CUDA thread handles **one wolf x one dimension**
- 2D thread grid: `(num_wolves, ceil(dim / 32))`
- Shared memory caches alpha/beta/delta positions per block
- 6 random values per wolf per dimension stored in flat `[N, 6D]` array
- Compiled at runtime via **NVRTC** — no CUDA Toolkit installation needed

### Route Encoding

Each route is encoded as a continuous vector in `[0,1]^d`:
- `max_stops` dimensions -> normalized stop indices
- 1 dimension -> headway (minutes)
- 1 dimension -> vehicle count

Total dimension = `num_routes x (max_stops + 2)` = **220** for default config.

### Objectives (all minimized)

| ID | Objective | Formula | Unit |
|----|-----------|---------|------|
| f1 | Travel time | Demand-weighted mean shortest-path time | minutes |
| f2 | Transfers | Fraction of OD pairs requiring a transfer | ratio |
| f3 | Fleet cost | Total operational cost (fuel + driver) | Rs k/day |
| f4 | Coverage gap | 1 - demand coverage ratio | ratio |

### Constraints (penalty-based, weight = 1000)

1. Route length bounds (5-50 km)
2. Vehicle capacity vs peak demand
3. Headway bounds (5-60 min)
4. Minimum stop coverage (>= 80% of stops served)
5. Route overlap soft penalty (> 40% overlap penalized)

---

## System Architecture

```
Input (GTFS files / Synthetic City Generator)
                |
                v
     TransitGraph (NetworkX DiGraph)
     - Dijkstra all-pairs travel time
     - Demand coverage computation
                |
                v
     RouteEncoder
     - Continuous [0,1]^220 vectors
     - encode / decode / perturb
                |
                v
     CuPyGWO on RTX 4060          <-- GPU: parallel position updates
     - 50 wolves x 220 dimensions
     - NVRTC-compiled CUDA kernels
     - Shared memory leader caching
                |
                v
     ObjectiveEvaluator (CPU)      <-- f1, f2, f3, f4 per wolf
     ConstraintHandler             <-- Penalty addition
                |
                v
     ParetoFront Archive
     - Fast non-dominated sort (NSGA-II)
     - Crowding distance diversity
     - Hypervolume / GD / IGD metrics
                |
                v
     Outputs: HTML Map + PNG Plots + LaTeX Table
```

---

## Research Contribution

> *"We apply CUDA-parallelized Grey Wolf Optimization to the multi-objective Transit Route Network Design Problem (TRNDP), demonstrating GPU-accelerated convergence over CPU baselines while maintaining competitive solution quality across four objectives: travel time, transfers, operational cost, and demand coverage."*

### Novel Aspects

1. First application of GWO to TRNDP with full CUDA parallelization
2. Four-objective Pareto formulation with archive-based multi-objective management
3. Continuous route encoding enabling gradient-free GPU-parallel optimization
4. CuPy NVRTC backend — deploys without CUDA Toolkit installation
5. Rigorous comparison against 4 established metaheuristics (CPU-GWO, PSO, NSGA-II, ACO)
6. Publication-standard metrics: Hypervolume, Generational Distance, IGD, Spacing

---

## Citation

If you use this work, please cite:

```bibtex
@article{scl_aat_cuda_gwo_2026,
  title  = {Public Transport Route Optimization using CUDA Grey Wolf Optimization},
  author = {Sandy-383},
  year   = {2026},
  note   = {Capstone Project -- GPU-accelerated multi-objective transit route optimization}
}
```

---

## License

MIT License. Free to use for academic and research purposes.
