/*
 * CUDA Fitness Evaluation Kernel
 * ================================
 * File: cuda_gwo/fitness_kernel.cu
 *
 * GPU-accelerated evaluation of transit route fitness objectives.
 *
 * Architecture:
 * ─────────────
 * The fitness evaluation is the most compute-intensive part of GWO.
 * Each wolf's decoded routes must be evaluated for 4 objectives.
 *
 * Strategy (Hybrid GPU-CPU):
 * ──────────────────────────
 * Full end-to-end GPU fitness evaluation requires the entire transit graph
 * to reside on GPU and a GPU-resident shortest-path algorithm.
 * We implement a THREE-TIER approach:
 *
 *   Tier 1 [GPU] : Pre-computed travel-time matrix lookup
 *                  (adj_matrix is uploaded once; lookup is O(1) per wolf)
 *   Tier 2 [GPU] : Objective 1 (avg travel time) and Objective 4 (coverage)
 *                  computed entirely on GPU using the adj_matrix
 *   Tier 3 [CPU] : Objectives 2 (transfers) and 3 (cost) require
 *                  graph traversal — evaluated on CPU in parallel threads
 *
 * The adj_matrix (n×n float32) is stored in GPU global memory.
 * For n=200 stops: 200×200×4B = 160KB  (fits in L2 cache easily)
 * For n=1000 stops: 1000×1000×4B = 4MB (fits in GPU global memory)
 *
 * Kernel: evaluate_travel_time_kernel
 *   Grid:  (num_wolves) blocks
 *   Block: (n) threads where n = num_stops
 *   Each block evaluates f1 for one wolf using shared memory reduction.
 */

#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include "fitness_kernel.cuh"

#define MAX_STOPS_PER_ROUTE  20
#define MAX_ROUTES           10
#define MAX_ROUTE_STOPS      (MAX_ROUTES * MAX_STOPS_PER_ROUTE)

// ─── Objective 1: Average Travel Time ────────────────────────────────────────
/*
 * For each wolf, compute demand-weighted average travel time using
 * the pre-computed all-pairs travel time matrix.
 *
 * positions[wolf_id × dim + ...]  is decoded to route stop sequences
 * encoded as integer indices in the lower 16 bits of position×(n-1).
 *
 * This kernel assumes the DECODED stop sequences are already available
 * in route_seqs[wolf_id × max_route_stops].
 */
__global__ void evaluate_travel_time_kernel(
    const int*   __restrict__ route_seqs,      // [num_wolves × max_route_stops]
    const int*   __restrict__ route_lengths,   // [num_wolves × num_routes]
    const float* __restrict__ adj_matrix,      // [n × n] travel time in minutes
    const float* __restrict__ demand,          // [n × n] OD demand
    float*       __restrict__ f1_out,          // [num_wolves] — avg travel time
    float*       __restrict__ f4_out,          // [num_wolves] — 1 - coverage
    int num_wolves,
    int num_routes,
    int n_stops
) {
    int wolf_id = blockIdx.x;
    if (wolf_id >= num_wolves) return;

    // Build served-stop bitmask using shared memory
    // (works for n_stops ≤ 1024)
    extern __shared__ int s_served[];   // [n_stops] — 1 if stop is served
    int tid = threadIdx.x;

    // Initialise shared bitmask
    for (int i = tid; i < n_stops; i += blockDim.x)
        s_served[i] = 0;
    __syncthreads();

    // Mark served stops from all routes of this wolf
    const int* seqs   = route_seqs   + wolf_id * MAX_ROUTE_STOPS;
    const int* lengths = route_lengths + wolf_id * num_routes;

    int offset = 0;
    for (int r = 0; r < num_routes; r++) {
        int len = lengths[r];
        for (int k = tid; k < len; k += blockDim.x) {
            int s = seqs[offset + k];
            if (s >= 0 && s < n_stops)
                s_served[s] = 1;
        }
        offset += MAX_STOPS_PER_ROUTE;
    }
    __syncthreads();

    // ── Compute f1: demand-weighted travel time ───────────────────────────
    // Each thread handles one origin stop i, iterates over all j
    float thread_weighted_sum = 0.0f;
    float thread_demand_sum   = 0.0f;

    for (int i = tid; i < n_stops; i += blockDim.x) {
        if (!s_served[i]) continue;
        for (int j = 0; j < n_stops; j++) {
            if (i == j || !s_served[j]) continue;
            float t = adj_matrix[i * n_stops + j];
            float d = demand[i * n_stops + j];
            if (t < FLT_MAX && d > 0.0f) {
                thread_weighted_sum += t * d;
                thread_demand_sum   += d;
            }
        }
    }

    // Block reduction for f1
    __shared__ float s_wsum[256];
    __shared__ float s_dsum[256];
    s_wsum[tid] = thread_weighted_sum;
    s_dsum[tid] = thread_demand_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_wsum[tid] += s_wsum[tid + stride];
            s_dsum[tid] += s_dsum[tid + stride];
        }
        __syncthreads();
    }

    // ── Compute f4: 1 - coverage ──────────────────────────────────────────
    __shared__ int s_served_count[1];
    if (tid == 0) s_served_count[0] = 0;
    __syncthreads();

    for (int i = tid; i < n_stops; i += blockDim.x)
        if (s_served[i]) atomicAdd(&s_served_count[0], 1);
    __syncthreads();

    // Thread 0 writes results
    if (tid == 0) {
        float denom = s_dsum[0];
        f1_out[wolf_id] = (denom > 0.0f) ? s_wsum[0] / denom : 999.0f;
        f4_out[wolf_id] = 1.0f - ((float)s_served_count[0] / (float)n_stops);
    }
}

// ─── Position Decode Kernel ───────────────────────────────────────────────────
/*
 * Kernel: decode_positions_kernel
 *
 * Batch decode GWO position vectors into integer stop sequences.
 * This runs BEFORE fitness evaluation.
 *
 * Each thread decodes one (wolf, route, stop_slot) triple.
 * Output: route_seqs[wolf × max_route_stops + route × max_stops + k] = stop_idx
 */
__global__ void decode_positions_kernel(
    const float* __restrict__ positions,    // [num_wolves × dim]
    int*         __restrict__ route_seqs,   // [num_wolves × max_route_stops]
    int*         __restrict__ route_lengths,// [num_wolves × num_routes]
    int num_wolves,
    int num_routes,
    int max_stops_per_route,
    int n_stops,
    int dim_per_route   // = max_stops_per_route + 2
) {
    int wolf_id  = blockIdx.x;
    int route_id = blockIdx.y;
    int k        = threadIdx.x;   // stop slot index

    if (wolf_id >= num_wolves || route_id >= num_routes) return;
    if (k >= max_stops_per_route) return;

    int pos_base = wolf_id * (num_routes * dim_per_route)
                 + route_id * dim_per_route;
    int seq_base = wolf_id * MAX_ROUTE_STOPS
                 + route_id * max_stops_per_route;

    float raw   = positions[pos_base + k];
    int   stop  = (int)roundf(raw * (float)(n_stops - 1));
    stop = max(0, min(stop, n_stops - 1));

    route_seqs[seq_base + k] = stop;

    // Thread 0 of each route block records route length
    // (approximated as max_stops; duplicates handled on CPU side)
    if (k == 0) {
        route_lengths[wolf_id * num_routes + route_id] = max_stops_per_route;
    }
}

// ─── Penalty Addition Kernel ──────────────────────────────────────────────────
/*
 * Add constraint penalty (computed on CPU) to all objectives of infeasible wolves.
 * penalties[wolf_id] = scalar penalty value.
 */
__global__ void add_penalty_kernel(
    float*       __restrict__ objectives,  // [num_wolves × 4] — modified in-place
    const float* __restrict__ penalties,   // [num_wolves]
    int num_wolves
) {
    int wolf_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (wolf_id >= num_wolves) return;

    float p = penalties[wolf_id];
    int base = wolf_id * 4;
    objectives[base + 0] += p;
    objectives[base + 1] += p;
    objectives[base + 2] += p;
    objectives[base + 3] += p;
}
