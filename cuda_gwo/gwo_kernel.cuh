/*
 * gwo_kernel.cuh — Public API for CUDA GWO Kernels
 * =================================================
 * Include this header from gwo_host.cpp to access kernel launchers.
 */

#pragma once
#include <cuda_runtime.h>
#include <curand_kernel.h>

#ifdef __cplusplus
extern "C" {
#endif

// ─── Kernel Declarations ───────────────────────────────────────────────────

__global__ void init_curand_kernel(
    curandState* states,
    unsigned long long seed,
    int num_wolves
);

__global__ void gwo_update_positions(
    float*       positions,
    const float* alpha,
    const float* beta,
    const float* delta,
    curandState* rng_states,
    float a,
    int num_wolves,
    int dim
);

__global__ void select_leaders(
    const float* arch_positions,
    const float* arch_objectives,
    const float* weights,
    float*       alpha_out,
    float*       beta_out,
    float*       delta_out,
    int*         leader_indices,
    int arch_size,
    int dim
);

__global__ void scatter_objectives(
    float*       obj_matrix,
    const float* new_obj,
    int num_wolves
);

__global__ void clamp_positions(
    float* positions,
    int total_elements
);

#ifdef __cplusplus
}
#endif
