/*
 * memory_manager.cuh — GPU Memory Pool for GWO
 */

#pragma once
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    // Dimensions
    int num_wolves;
    int dim;
    int n_stops;
    int num_routes;
    int max_stops_per_route;
    int archive_size;

    // Device (GPU) buffers
    float*       d_positions;
    float*       d_objectives;
    float*       d_alpha;
    float*       d_beta;
    float*       d_delta;
    float*       d_archive_positions;
    float*       d_archive_objectives;
    int*         d_route_seqs;
    int*         d_route_lengths;
    float*       d_adj_matrix;
    float*       d_demand;
    float*       d_penalties;
    float*       d_obj_weights;
    int*         d_leader_indices;
    curandState* d_rng_states;

    // Pinned host buffers (for fast DMA)
    float*       h_positions;
    float*       h_objectives;
    float*       h_penalties;

    // CUDA streams
    cudaStream_t stream_compute;
    cudaStream_t stream_transfer;
} GWOMemory;

GWOMemory* gwo_memory_create(
    int num_wolves,
    int dim,
    int n_stops,
    int num_routes,
    int max_stops_per_route,
    int archive_size
);

size_t gwo_memory_bytes(const GWOMemory* mem);

void gwo_memory_upload_graph(
    GWOMemory*   mem,
    const float* adj_matrix,
    const float* demand
);

void gwo_memory_upload_positions(GWOMemory* mem, const float* positions);
void gwo_memory_download_positions(GWOMemory* mem, float* positions);
void gwo_memory_upload_objectives(GWOMemory* mem, const float* objectives);
void gwo_memory_download_objectives(GWOMemory* mem, float* objectives);

void gwo_memory_set_leaders(
    GWOMemory*   mem,
    const float* alpha,
    const float* beta,
    const float* delta
);

void gwo_memory_destroy(GWOMemory* mem);

#ifdef __cplusplus
}
#endif
