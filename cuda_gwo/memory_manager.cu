/*
 * CUDA Memory Manager
 * ====================
 * File: cuda_gwo/memory_manager.cu
 *
 * Handles all GPU memory allocations, transfers, and lifecycle management
 * for the GWO optimizer.
 *
 * Design principles:
 *   - Allocate all GPU buffers ONCE at start of optimization
 *   - Reuse throughout all iterations (no per-iteration malloc/free)
 *   - Pinned host memory for fast async H↔D transfers
 *   - CUDA streams for overlap of computation and data transfer
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "memory_manager.cuh"

// ─── Error Checking Macro ─────────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

// ─── GWO Memory Pool ─────────────────────────────────────────────────────────

GWOMemory* gwo_memory_create(
    int num_wolves,
    int dim,
    int n_stops,
    int num_routes,
    int max_stops_per_route,
    int archive_size
) {
    GWOMemory* mem = (GWOMemory*)malloc(sizeof(GWOMemory));
    if (!mem) { fprintf(stderr, "Host malloc failed\n"); exit(1); }

    mem->num_wolves          = num_wolves;
    mem->dim                 = dim;
    mem->n_stops             = n_stops;
    mem->num_routes          = num_routes;
    mem->max_stops_per_route = max_stops_per_route;
    mem->archive_size        = archive_size;

    int max_route_stops = num_routes * max_stops_per_route;

    // ── Device allocations ───────────────────────────────────────────────

    // Population positions: [num_wolves × dim]
    CUDA_CHECK(cudaMalloc(&mem->d_positions,
        (size_t)num_wolves * dim * sizeof(float)));

    // Objectives: [num_wolves × 4]
    CUDA_CHECK(cudaMalloc(&mem->d_objectives,
        (size_t)num_wolves * 4 * sizeof(float)));

    // Leader positions: alpha, beta, delta — each [dim]
    CUDA_CHECK(cudaMalloc(&mem->d_alpha, (size_t)dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&mem->d_beta,  (size_t)dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&mem->d_delta, (size_t)dim * sizeof(float)));

    // Pareto archive: [archive_size × dim] positions + [archive_size × 4] objectives
    CUDA_CHECK(cudaMalloc(&mem->d_archive_positions,
        (size_t)archive_size * dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&mem->d_archive_objectives,
        (size_t)archive_size * 4 * sizeof(float)));

    // Decoded route sequences: [num_wolves × max_route_stops]
    CUDA_CHECK(cudaMalloc(&mem->d_route_seqs,
        (size_t)num_wolves * max_route_stops * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&mem->d_route_lengths,
        (size_t)num_wolves * num_routes * sizeof(int)));

    // Travel time and demand matrices: [n_stops × n_stops]
    CUDA_CHECK(cudaMalloc(&mem->d_adj_matrix,
        (size_t)n_stops * n_stops * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&mem->d_demand,
        (size_t)n_stops * n_stops * sizeof(float)));

    // Penalties: [num_wolves]
    CUDA_CHECK(cudaMalloc(&mem->d_penalties,
        (size_t)num_wolves * sizeof(float)));

    // cuRAND states: [num_wolves]
    CUDA_CHECK(cudaMalloc(&mem->d_rng_states,
        (size_t)num_wolves * sizeof(curandState)));

    // Leader selection output
    CUDA_CHECK(cudaMalloc(&mem->d_leader_indices, 3 * sizeof(int)));

    // Objective weights [4]
    CUDA_CHECK(cudaMalloc(&mem->d_obj_weights, 4 * sizeof(float)));

    // ── Pinned host allocations (fast DMA transfers) ─────────────────────
    CUDA_CHECK(cudaMallocHost(&mem->h_positions,
        (size_t)num_wolves * dim * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&mem->h_objectives,
        (size_t)num_wolves * 4 * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&mem->h_penalties,
        (size_t)num_wolves * sizeof(float)));

    // ── CUDA Streams ─────────────────────────────────────────────────────
    CUDA_CHECK(cudaStreamCreate(&mem->stream_compute));
    CUDA_CHECK(cudaStreamCreate(&mem->stream_transfer));

    fprintf(stdout,
        "[GWOMemory] Allocated: wolves=%d dim=%d n_stops=%d "
        "GPU_mem=%.1f MB\n",
        num_wolves, dim, n_stops,
        (float)gwo_memory_bytes(mem) / (1024.0f * 1024.0f));

    return mem;
}

size_t gwo_memory_bytes(const GWOMemory* mem) {
    int    N   = mem->num_wolves;
    int    D   = mem->dim;
    int    S   = mem->n_stops;
    int    R   = mem->num_routes;
    int    K   = mem->max_stops_per_route;
    int    A   = mem->archive_size;

    size_t bytes = 0;
    bytes += (size_t)N * D * sizeof(float);         // positions
    bytes += (size_t)N * 4 * sizeof(float);         // objectives
    bytes += (size_t)D * 3 * sizeof(float);         // alpha, beta, delta
    bytes += (size_t)A * D * sizeof(float);         // archive positions
    bytes += (size_t)A * 4 * sizeof(float);         // archive objectives
    bytes += (size_t)N * R * K * sizeof(int);       // route seqs
    bytes += (size_t)N * R * sizeof(int);           // route lengths
    bytes += (size_t)S * S * sizeof(float) * 2;     // adj_matrix + demand
    bytes += (size_t)N * sizeof(float);             // penalties
    bytes += (size_t)N * sizeof(curandState);       // rng states
    return bytes;
}

void gwo_memory_upload_graph(
    GWOMemory*   mem,
    const float* adj_matrix,    // host [n_stops × n_stops]
    const float* demand         // host [n_stops × n_stops]
) {
    size_t sz = (size_t)mem->n_stops * mem->n_stops * sizeof(float);
    CUDA_CHECK(cudaMemcpy(mem->d_adj_matrix, adj_matrix, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(mem->d_demand,     demand,     sz, cudaMemcpyHostToDevice));
}

void gwo_memory_upload_positions(GWOMemory* mem, const float* positions) {
    size_t sz = (size_t)mem->num_wolves * mem->dim * sizeof(float);
    CUDA_CHECK(cudaMemcpyAsync(mem->d_positions, positions, sz,
                               cudaMemcpyHostToDevice, mem->stream_transfer));
}

void gwo_memory_download_positions(GWOMemory* mem, float* positions) {
    size_t sz = (size_t)mem->num_wolves * mem->dim * sizeof(float);
    CUDA_CHECK(cudaMemcpyAsync(positions, mem->d_positions, sz,
                               cudaMemcpyDeviceToHost, mem->stream_transfer));
    CUDA_CHECK(cudaStreamSynchronize(mem->stream_transfer));
}

void gwo_memory_upload_objectives(GWOMemory* mem, const float* objectives) {
    size_t sz = (size_t)mem->num_wolves * 4 * sizeof(float);
    CUDA_CHECK(cudaMemcpyAsync(mem->d_objectives, objectives, sz,
                               cudaMemcpyHostToDevice, mem->stream_transfer));
}

void gwo_memory_download_objectives(GWOMemory* mem, float* objectives) {
    size_t sz = (size_t)mem->num_wolves * 4 * sizeof(float);
    CUDA_CHECK(cudaMemcpyAsync(objectives, mem->d_objectives, sz,
                               cudaMemcpyDeviceToHost, mem->stream_transfer));
    CUDA_CHECK(cudaStreamSynchronize(mem->stream_transfer));
}

void gwo_memory_set_leaders(
    GWOMemory*   mem,
    const float* alpha,
    const float* beta,
    const float* delta
) {
    size_t sz = (size_t)mem->dim * sizeof(float);
    CUDA_CHECK(cudaMemcpyAsync(mem->d_alpha, alpha, sz,
                               cudaMemcpyHostToDevice, mem->stream_transfer));
    CUDA_CHECK(cudaMemcpyAsync(mem->d_beta,  beta,  sz,
                               cudaMemcpyHostToDevice, mem->stream_transfer));
    CUDA_CHECK(cudaMemcpyAsync(mem->d_delta, delta, sz,
                               cudaMemcpyHostToDevice, mem->stream_transfer));
    CUDA_CHECK(cudaStreamSynchronize(mem->stream_transfer));
}

void gwo_memory_destroy(GWOMemory* mem) {
    if (!mem) return;

    cudaFree(mem->d_positions);
    cudaFree(mem->d_objectives);
    cudaFree(mem->d_alpha);
    cudaFree(mem->d_beta);
    cudaFree(mem->d_delta);
    cudaFree(mem->d_archive_positions);
    cudaFree(mem->d_archive_objectives);
    cudaFree(mem->d_route_seqs);
    cudaFree(mem->d_route_lengths);
    cudaFree(mem->d_adj_matrix);
    cudaFree(mem->d_demand);
    cudaFree(mem->d_penalties);
    cudaFree(mem->d_rng_states);
    cudaFree(mem->d_leader_indices);
    cudaFree(mem->d_obj_weights);

    cudaFreeHost(mem->h_positions);
    cudaFreeHost(mem->h_objectives);
    cudaFreeHost(mem->h_penalties);

    cudaStreamDestroy(mem->stream_compute);
    cudaStreamDestroy(mem->stream_transfer);

    free(mem);
}
