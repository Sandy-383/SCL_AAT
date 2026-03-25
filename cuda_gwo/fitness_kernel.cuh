/*
 * fitness_kernel.cuh — Public API for CUDA Fitness Kernels
 */

#pragma once
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

__global__ void evaluate_travel_time_kernel(
    const int*   route_seqs,
    const int*   route_lengths,
    const float* adj_matrix,
    const float* demand,
    float*       f1_out,
    float*       f4_out,
    int num_wolves,
    int num_routes,
    int n_stops
);

__global__ void decode_positions_kernel(
    const float* positions,
    int*         route_seqs,
    int*         route_lengths,
    int num_wolves,
    int num_routes,
    int max_stops_per_route,
    int n_stops,
    int dim_per_route
);

__global__ void add_penalty_kernel(
    float*       objectives,
    const float* penalties,
    int num_wolves
);

#ifdef __cplusplus
}
#endif
