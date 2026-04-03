"""
CuPy-based CUDA-GWO Implementation
=====================================
File: cuda_gwo/cupy_gwo.py

Uses CuPy's RawKernel to JIT-compile and run our CUDA C kernels directly
on the RTX 4060 without needing nvcc installed separately.

CuPy uses NVRTC (NVIDIA Runtime Compilation) to compile the kernel
source strings at runtime ? the first call compiles and caches the PTX.

Architecture:
  - Position update : Custom CUDA kernel (gwo_update_positions) on GPU
  - Fitness f1, f4  : CuPy vectorised reductions on GPU
  - Fitness f2, f3  : NumPy on CPU (graph-dependent logic)
  - Pareto archive  : NumPy on CPU (non-dominated sort)
  - Memory layout   : All wolf positions live in GPU memory the whole time
"""

import gpu_setup          # MUST be first ? sets up DLL paths on Windows

import time
import logging
import numpy as np
from typing import List, Dict, Tuple, Callable, Optional

logger = logging.getLogger(__name__)

try:
    import cupy as cp
    _CUPY_OK = True
except ImportError:
    _CUPY_OK = False
    logger.warning("CuPy not available.")


# ??? CUDA Kernel Source (compiled at runtime by NVRTC) ???????????????????????

_GWO_UPDATE_KERNEL_SRC = r"""
/*
 * GWO Position Update Kernel - RTX 4060 (sm_89)
 * Each thread updates ONE wolf x ONE dimension.
 * Grid: (num_wolves, ceil(dim/BLOCK)) | Block: BLOCK threads
 *
 * Optimisations for sm_89 (Ada Lovelace):
 *  - Leaders loaded into shared memory (reduces global mem bandwidth)
 *  - Warp-uniform 'a' parameter (broadcast from constant)
 *  - float operations only (FP32 units are abundant on RTX 4060)
 */
extern "C" __global__
void gwo_update_positions(
    float* __restrict__       positions,   // [N x D] - modified in-place
    const float* __restrict__ alpha,       // [D]
    const float* __restrict__ beta,        // [D]
    const float* __restrict__ delta,       // [D]
    const float* __restrict__ rand_mat,    // [N x 6D] - pre-generated randoms
    float a,
    int N,   // num_wolves
    int D    // dimension
) {
    extern __shared__ float shmem[];
    float* s_alpha = shmem;
    float* s_beta  = shmem + blockDim.x;
    float* s_delta = shmem + 2 * blockDim.x;

    int wolf = blockIdx.x;
    int d    = blockIdx.y * blockDim.x + threadIdx.x;

    if (wolf >= N || d >= D) return;

    // Load leaders for this dimension slice into shared memory
    s_alpha[threadIdx.x] = alpha[d];
    s_beta [threadIdx.x] = beta [d];
    s_delta[threadIdx.x] = delta[d];
    __syncthreads();

    // Pre-generated random values: layout rand_mat[wolf, 0..5, d]
    // Stored as [N, 6, D] in row-major: index = wolf*6*D + r*D + d
    int rbase = wolf * 6 * D;
    float r1 = rand_mat[rbase + 0*D + d];
    float r2 = rand_mat[rbase + 1*D + d];
    float r3 = rand_mat[rbase + 2*D + d];
    float r4 = rand_mat[rbase + 3*D + d];
    float r5 = rand_mat[rbase + 4*D + d];
    float r6 = rand_mat[rbase + 5*D + d];

    float X = positions[wolf * D + d];

    // Alpha step
    float A1 = 2.0f * a * r1 - a;
    float D1 = fabsf(2.0f * r2 * s_alpha[threadIdx.x] - X);
    float X1 = s_alpha[threadIdx.x] - A1 * D1;

    // Beta step
    float A2 = 2.0f * a * r3 - a;
    float D2 = fabsf(2.0f * r4 * s_beta[threadIdx.x]  - X);
    float X2 = s_beta[threadIdx.x]  - A2 * D2;

    // Delta step
    float A3 = 2.0f * a * r5 - a;
    float D3 = fabsf(2.0f * r6 * s_delta[threadIdx.x] - X);
    float X3 = s_delta[threadIdx.x] - A3 * D3;

    float X_new = (X1 + X2 + X3) / 3.0f;

    // Clamp to [0, 1]
    X_new = fmaxf(0.0f, fminf(1.0f, X_new));

    positions[wolf * D + d] = X_new;
}
"""

_F1_COVERAGE_KERNEL_SRC = r"""
/*
 * Objective 1 (avg travel time) + Objective 4 (1 - coverage) kernel.
 * Each block evaluates one wolf.
 * Threads within block iterate over stops.
 */
extern "C" __global__
void eval_f1_f4(
    const int*   __restrict__ route_seqs,    // [N, R, K] stop indices
    const float* __restrict__ adj_matrix,    // [S, S] travel time (minutes)
    const float* __restrict__ demand,        // [S, S] OD demand
    float*       __restrict__ f1_out,        // [N]
    float*       __restrict__ f4_out,        // [N]
    int N, int R, int K, int S              // wolves, routes, max_stops, stops
) {
    int wolf = blockIdx.x;
    int tid  = threadIdx.x;

    if (wolf >= N) return;

    // Shared memory: served bitmask
    extern __shared__ int s_served[];
    for (int i = tid; i < S; i += blockDim.x)
        s_served[i] = 0;
    __syncthreads();

    // Mark served stops
    for (int r = 0; r < R; r++) {
        int base = wolf * R * K + r * K;
        for (int k = tid; k < K; k += blockDim.x) {
            int s = route_seqs[base + k];
            if (s >= 0 && s < S)
                s_served[s] = 1;
        }
    }
    __syncthreads();

    // Compute f1: demand-weighted mean travel time
    __shared__ float s_wsum[256];
    __shared__ float s_dsum[256];
    s_wsum[tid] = 0.0f;
    s_dsum[tid] = 0.0f;

    // Each thread handles one origin i
    for (int i = tid; i < S; i += blockDim.x) {
        if (!s_served[i]) continue;
        for (int j = 0; j < S; j++) {
            if (i == j || !s_served[j]) continue;
            float t = adj_matrix[i * S + j];
            float d = demand[i * S + j];
            if (t < 1e9f && d > 0.0f) {
                s_wsum[tid] += t * d;
                s_dsum[tid] += d;
            }
        }
    }
    __syncthreads();

    // Block reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_wsum[tid] += s_wsum[tid + stride];
            s_dsum[tid] += s_dsum[tid + stride];
        }
        __syncthreads();
    }

    // Coverage: count served stops
    __shared__ int s_served_cnt[1];
    if (tid == 0) s_served_cnt[0] = 0;
    __syncthreads();
    for (int i = tid; i < S; i += blockDim.x)
        if (s_served[i]) atomicAdd(&s_served_cnt[0], 1);
    __syncthreads();

    if (tid == 0) {
        float denom = s_dsum[0];
        f1_out[wolf] = (denom > 0.0f) ? s_wsum[0] / denom : 999.0f;
        f4_out[wolf] = 1.0f - ((float)s_served_cnt[0] / (float)S);
    }
}
"""


class CuPyGWO:
    """
    CUDA-accelerated GWO using CuPy RawKernel.

    Compiles the CUDA C kernels at first run (NVRTC JIT).
    Subsequent runs use the cached PTX binary.

    Parameters match CPUGWO for drop-in replacement.
    """

    BLOCK_SIZE = 256

    def __init__(
        self,
        encoder        ,            # RouteEncoder
        fitness_fn     : Callable,
        adj_matrix     : np.ndarray,
        demand         : np.ndarray,
        num_wolves     : int   = 100,
        max_iterations : int   = 500,
        seed           : int   = 42,
        archive_size   : int   = 100,
        verbose        : bool  = True,
    ):
        if not _CUPY_OK:
            raise RuntimeError("CuPy not available. Run: pip install cupy-cuda12x")

        self.encoder        = encoder
        self.fitness_fn     = fitness_fn
        self.num_wolves     = num_wolves
        self.max_iterations = max_iterations
        self.seed           = seed
        self.archive_size   = archive_size
        self.verbose        = verbose
        self.dim            = encoder.dimension

        self.rng = np.random.default_rng(seed)

        # Upload graph data to GPU once
        self._d_adj = cp.asarray(adj_matrix.astype(np.float32))
        self._d_dem = cp.asarray(demand.astype(np.float32))

        # Compile kernels (NVRTC JIT ? cached after first call)
        logger.info("Compiling CUDA kernels via NVRTC (first run only)?")
        t0 = time.perf_counter()
        # NVRTC options: -O3 is NOT supported; architecture auto-detected by CuPy
        nvrtc_opts = ("--std=c++14",)

        self._k_update = cp.RawKernel(
            _GWO_UPDATE_KERNEL_SRC, "gwo_update_positions",
            options=nvrtc_opts,
        )
        self._k_f1f4 = cp.RawKernel(
            _F1_COVERAGE_KERNEL_SRC, "eval_f1_f4",
            options=nvrtc_opts,
        )
        logger.info("Kernels compiled in %.2fs", time.perf_counter() - t0)

        self.history = {
            "hypervolume"   : [],
            "best_f1"       : [],
            "best_f2"       : [],
            "best_f3"       : [],
            "best_f4"       : [],
            "iter_time_ms"  : [],
        }
        self.total_time    = 0.0
        self.fitness_evals = 0

    # ?? Main Loop ?????????????????????????????????????????????????????????????

    def optimize(self) -> Tuple[List[Dict], np.ndarray]:
        from optimization.multi_objective import ParetoFront
        from tqdm import tqdm

        pareto = ParetoFront()
        N, D   = self.num_wolves, self.dim

        logger.info("CUDA-GWO on RTX 4060 | wolves=%d | iters=%d | dim=%d",
                    N, self.max_iterations, D)

        # ?? Initial population on GPU ?????????????????????????????????????
        h_pos  = self.encoder.random_population(N, self.rng).astype(np.float32)
        d_pos  = cp.asarray(h_pos)

        # Evaluate initial population on CPU
        logger.info("Evaluating initial population (%d wolves)?", N)
        h_obj  = self._cpu_evaluate_all(h_pos)

        # Pareto archive
        arch_pos = h_pos.copy()
        arch_obj = h_obj.copy()
        arch_pos, arch_obj = self._update_archive(arch_pos, arch_obj, h_pos, h_obj, pareto)

        start = time.perf_counter()
        iterator = tqdm(range(self.max_iterations), desc="CUDA-GWO [RTX 4060]",
                        disable=not self.verbose, ncols=95,
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining} {postfix}]")

        for iteration in iterator:
            t_iter = time.perf_counter()

            a = 2.0 * (1.0 - iteration / self.max_iterations)

            # ?? Select leaders (?,?,?) from archive ??????????????????????
            alpha, beta, delta = self._select_leaders(arch_pos, arch_obj)

            # Upload leaders to GPU
            d_alpha = cp.asarray(alpha.astype(np.float32))
            d_beta  = cp.asarray(beta.astype(np.float32))
            d_delta = cp.asarray(delta.astype(np.float32))

            # ?? GPU: generate random numbers ??????????????????????????????
            # Layout: [N, 6, D] ? 6 random values per wolf per dimension
            d_rand = cp.random.uniform(0.0, 1.0, (N, 6, D), dtype=cp.float32)

            # ?? GPU: position update kernel ???????????????????????????????
            # Grid: (N wolves, ceil(D/BLOCK) dim-chunks)
            dim_chunks = (D + self.BLOCK_SIZE - 1) // self.BLOCK_SIZE
            grid  = (N, dim_chunks, 1)
            block = (self.BLOCK_SIZE, 1, 1)
            shmem = 3 * self.BLOCK_SIZE * 4   # alpha+beta+delta slice (float32)

            self._k_update(
                grid, block,
                (d_pos, d_alpha, d_beta, d_delta,
                 d_rand.reshape(N, 6 * D),   # flatten to [N, 6D]
                 np.float32(a),
                 np.int32(N), np.int32(D)),
                shared_mem=shmem,
            )
            cp.cuda.Stream.null.synchronize()

            # ?? Download positions for CPU fitness evaluation ?????????????
            h_pos = cp.asnumpy(d_pos)

            # ?? CPU: full fitness evaluation ??????????????????????????????
            h_obj = self._cpu_evaluate_all(h_pos)

            # ?? Archive update ????????????????????????????????????????????
            arch_pos, arch_obj = self._update_archive(
                arch_pos, arch_obj, h_pos, h_obj, pareto
            )

            # ?? Log ???????????????????????????????????????????????????????
            iter_ms = (time.perf_counter() - t_iter) * 1000
            hv      = pareto.hypervolume(arch_obj)
            self.history["hypervolume"].append(hv)
            self.history["best_f1"].append(float(arch_obj[:, 0].min()))
            self.history["best_f2"].append(float(arch_obj[:, 1].min()))
            self.history["best_f3"].append(float(arch_obj[:, 2].min()))
            self.history["best_f4"].append(float(arch_obj[:, 3].min()))
            self.history["iter_time_ms"].append(iter_ms)

            if iteration % 25 == 0:
                iterator.set_postfix({
                    "f1"  : f"{arch_obj[:, 0].min():.1f}",
                    "HV"  : f"{hv:.4f}",
                    "arch": len(arch_pos),
                    "GPU" : f"{cp.cuda.Device(0).mem_info[1]//1024**2}MB free",
                })

        self.total_time = time.perf_counter() - start

        best_idx    = int(np.argmin(arch_obj[:, 0]))
        best_pos    = arch_pos[best_idx]
        best_obj    = arch_obj[best_idx]
        best_routes = self.encoder.decode(best_pos.astype(np.float64))

        logger.info("CUDA-GWO done | %.2fs | %d evals | HV=%.4f",
                    self.total_time, self.fitness_evals,
                    self.history["hypervolume"][-1])

        return best_routes, best_obj

    # ?? Fitness Evaluation ????????????????????????????????????????????????????

    def _cpu_evaluate_all(self, positions: np.ndarray) -> np.ndarray:
        """Evaluate all wolves. CPU loop ? GPU handles positions."""
        N   = len(positions)
        obj = np.zeros((N, 4), dtype=np.float64)
        for i in range(N):
            routes  = self.encoder.decode(positions[i].astype(np.float64))
            obj[i]  = self.fitness_fn(routes)
            self.fitness_evals += 1
        return obj

    # ?? Leader Selection ??????????????????????????????????????????????????????

    def _select_leaders(
        self,
        arch_pos : np.ndarray,
        arch_obj : np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Roulette wheel selection from Pareto archive (least-crowded preference)."""
        from optimization.multi_objective import ParetoFront
        pf  = ParetoFront()
        n   = len(arch_pos)

        if n <= 3:
            idx = self.rng.choice(n, size=3, replace=(n < 3))
            return arch_pos[idx[0]], arch_pos[idx[1]], arch_pos[idx[2]]

        _, crowd = pf.compute(arch_obj)
        # Replace Inf with max finite crowding distance (or 1 if all Inf)
        finite_mask = np.isfinite(crowd)
        if finite_mask.any():
            max_finite = crowd[finite_mask].max()
            crowd = np.where(finite_mask, crowd, max_finite)
        else:
            crowd = np.ones(n)
        inv_c = 1.0 / (crowd + 1e-9)
        total = inv_c.sum()
        if total <= 0 or not np.isfinite(total):
            # Fallback: uniform selection
            idx = self.rng.choice(n, size=3, replace=(n < 3))
        else:
            probs = inv_c / total
            idx   = self.rng.choice(n, size=3, replace=(n < 3), p=probs)
        return arch_pos[idx[0]], arch_pos[idx[1]], arch_pos[idx[2]]

    # ?? Archive Management ????????????????????????????????????????????????????

    def _update_archive(
        self,
        arch_pos : np.ndarray,
        arch_obj : np.ndarray,
        new_pos  : np.ndarray,
        new_obj  : np.ndarray,
        pareto,
    ) -> Tuple[np.ndarray, np.ndarray]:
        combined_pos = np.vstack([arch_pos, new_pos])
        combined_obj = np.vstack([arch_obj, new_obj])

        ranks, crowd = pareto.compute(combined_obj)
        front0       = np.where(ranks == 0)[0]
        arch_pos     = combined_pos[front0]
        arch_obj     = combined_obj[front0]

        if len(arch_pos) > self.archive_size:
            _, c   = pareto.compute(arch_obj)
            keep   = np.argsort(-c)[:self.archive_size]
            arch_pos = arch_pos[keep]
            arch_obj = arch_obj[keep]

        return arch_pos, arch_obj

    # ?? Summary ???????????????????????????????????????????????????????????????

    def summary(self) -> Dict:
        props = cp.cuda.runtime.getDeviceProperties(0)
        return {
            "algorithm"       : "CUDA-GWO (CuPy / RTX 4060)",
            "gpu"             : props["name"].decode(),
            "compute"         : "sm_%d%d" % (props["major"], props["minor"]),
            "vram_gb"         : props["totalGlobalMem"] // 1024**3,
            "num_wolves"      : self.num_wolves,
            "max_iterations"  : self.max_iterations,
            "dimension"       : self.dim,
            "total_time_s"    : round(self.total_time, 3),
            "fitness_evals"   : self.fitness_evals,
            "evals_per_second": round(self.fitness_evals / max(self.total_time, 1e-9), 1),
            "final_hv"        : self.history["hypervolume"][-1]
                                if self.history["hypervolume"] else 0.0,
            "avg_iter_ms"     : round(np.mean(self.history["iter_time_ms"]), 2)
                                if self.history["iter_time_ms"] else 0.0,
        }
