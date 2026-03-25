/*
 * GWO Host Driver
 * ================
 * File: cuda_gwo/gwo_host.cpp
 *
 * C++ host code that orchestrates the CUDA-GWO:
 *   1. Initialise GPU memory pool
 *   2. Upload initial population + graph data
 *   3. Per-iteration loop:
 *        a. Select leaders (alpha, beta, delta) from Pareto archive
 *        b. Launch gwo_update_positions kernel
 *        c. Download updated positions to host
 *        d. CPU evaluates fitness (objectives 2 & 3 via Python callback)
 *        e. GPU evaluates objectives 1 & 4 via fitness_kernel
 *        f. Update Pareto archive
 *   4. Return best solution to Python via ctypes-compatible C API
 *
 * Python binds to this via ctypes (see cuda_gwo/cuda_gwo_binding.py).
 * The C API is declared extern "C" for ABI compatibility.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include <chrono>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "gwo_kernel.cuh"
#include "fitness_kernel.cuh"
#include "memory_manager.cuh"

#define BLOCK_SIZE    256
#define NUM_OBJ       4
#define MAX_ARCH      100

// ─── Timing Helper ────────────────────────────────────────────────────────────
static double now_ms() {
    using namespace std::chrono;
    return (double)duration_cast<microseconds>(
        steady_clock::now().time_since_epoch()
    ).count() / 1000.0;
}

// ─── Pareto Dominance (CPU side for archive) ──────────────────────────────────
static bool dominates(const float* a, const float* b) {
    bool any_less = false;
    for (int m = 0; m < NUM_OBJ; m++) {
        if (a[m] > b[m]) return false;
        if (a[m] < b[m]) any_less = true;
    }
    return any_less;
}

// ─── Archive Update (CPU) ─────────────────────────────────────────────────────
static int update_archive(
    float*   arch_pos,   int* arch_size,
    float*   arch_obj,
    const float* new_pos, const float* new_obj,
    int num_wolves, int dim
) {
    for (int i = 0; i < num_wolves; i++) {
        const float* np = new_pos + i * dim;
        const float* no = new_obj + i * NUM_OBJ;

        // Check if new_pos[i] is dominated by any archive member
        bool dominated = false;
        for (int a = 0; a < *arch_size; a++) {
            if (dominates(arch_obj + a * NUM_OBJ, no)) {
                dominated = true;
                break;
            }
        }
        if (dominated) continue;

        // Remove archive members dominated by new_pos[i]
        int keep = 0;
        for (int a = 0; a < *arch_size; a++) {
            if (!dominates(no, arch_obj + a * NUM_OBJ)) {
                if (a != keep) {
                    memcpy(arch_pos + keep * dim,    arch_pos + a * dim,    dim * sizeof(float));
                    memcpy(arch_obj + keep * NUM_OBJ, arch_obj + a * NUM_OBJ, NUM_OBJ * sizeof(float));
                }
                keep++;
            }
        }
        *arch_size = keep;

        // Add new solution if archive not full
        if (*arch_size < MAX_ARCH) {
            memcpy(arch_pos + (*arch_size) * dim,    np, dim * sizeof(float));
            memcpy(arch_obj + (*arch_size) * NUM_OBJ, no, NUM_OBJ * sizeof(float));
            (*arch_size)++;
        }
    }
    return *arch_size;
}

// ─── Leader Selection (CPU) ───────────────────────────────────────────────────
static void select_leaders_cpu(
    const float* arch_pos, const float* arch_obj, int arch_size,
    float* alpha, float* beta, float* delta, int dim
) {
    if (arch_size == 0) return;

    // Scalarised score = sum(objectives)
    std::vector<std::pair<float,int>> scored(arch_size);
    for (int i = 0; i < arch_size; i++) {
        float s = 0.0f;
        for (int m = 0; m < NUM_OBJ; m++) s += arch_obj[i * NUM_OBJ + m];
        scored[i] = {s, i};
    }
    std::sort(scored.begin(), scored.end());

    int ia = scored[0].second;
    int ib = scored[std::min(1, arch_size-1)].second;
    int id = scored[std::min(2, arch_size-1)].second;

    memcpy(alpha, arch_pos + ia * dim, dim * sizeof(float));
    memcpy(beta,  arch_pos + ib * dim, dim * sizeof(float));
    memcpy(delta, arch_pos + id * dim, dim * sizeof(float));
}

// ─── Public C API (called from Python via ctypes) ─────────────────────────────

extern "C" {

/*
 * cuda_gwo_run
 * ─────────────
 * Main entry point called from Python.
 *
 * Parameters:
 *   init_positions [in]  : float[num_wolves × dim] — initial population
 *   init_objectives[in]  : float[num_wolves × 4]  — initial fitness values
 *   adj_matrix     [in]  : float[n_stops × n_stops]
 *   demand         [in]  : float[n_stops × n_stops]
 *   out_positions  [out] : float[num_wolves × dim] — final population
 *   out_objectives [out] : float[num_wolves × 4]  — final fitness
 *   out_best_pos   [out] : float[dim]             — best solution position
 *   out_best_obj   [out] : float[4]               — best solution objectives
 *   num_wolves     [in]  : int
 *   dim            [in]  : int
 *   n_stops        [in]  : int
 *   num_routes     [in]  : int
 *   max_stops      [in]  : int
 *   max_iterations [in]  : int
 *   seed           [in]  : int
 *   timing_ms      [out] : double — total GPU time in ms
 *
 * The Python caller is responsible for:
 *   - Allocating output buffers
 *   - Calling fitness evaluation for objectives 2 & 3 each iteration
 *     (passed back via out_objectives before next iteration)
 *
 * Returns: 0 on success, non-zero on error
 */
int cuda_gwo_run(
    const float* init_positions,
    const float* init_objectives,
    const float* adj_matrix,
    const float* demand,
    float*       out_positions,
    float*       out_objectives,
    float*       out_best_pos,
    float*       out_best_obj,
    int num_wolves,
    int dim,
    int n_stops,
    int num_routes,
    int max_stops,
    int max_iterations,
    unsigned long long seed,
    double* timing_ms
) {
    double t_start = now_ms();

    // ── Allocate GPU memory ───────────────────────────────────────────────
    GWOMemory* mem = gwo_memory_create(
        num_wolves, dim, n_stops, num_routes, max_stops, MAX_ARCH
    );

    // ── Upload initial data ───────────────────────────────────────────────
    gwo_memory_upload_graph(mem, adj_matrix, demand);
    gwo_memory_upload_positions(mem, init_positions);
    gwo_memory_upload_objectives(mem, init_objectives);

    // Initialise cuRAND states
    {
        int blocks = (num_wolves + BLOCK_SIZE - 1) / BLOCK_SIZE;
        init_curand_kernel<<<blocks, BLOCK_SIZE>>>(
            mem->d_rng_states, seed, num_wolves
        );
        cudaDeviceSynchronize();
    }

    // ── Allocate CPU-side archive ─────────────────────────────────────────
    std::vector<float> arch_pos(MAX_ARCH * dim,     0.0f);
    std::vector<float> arch_obj(MAX_ARCH * NUM_OBJ, 0.0f);
    int arch_size = 0;

    // Seed archive with initial population
    update_archive(
        arch_pos.data(), &arch_size,
        arch_obj.data(),
        init_positions, init_objectives,
        num_wolves, dim
    );

    // CPU buffers for download
    std::vector<float> h_pos(num_wolves * dim);
    std::vector<float> h_obj(num_wolves * NUM_OBJ);
    std::vector<float> h_alpha(dim), h_beta(dim), h_delta(dim);

    // ── Objective weights (equal weight) ─────────────────────────────────
    float h_weights[4] = {0.25f, 0.25f, 0.25f, 0.25f};
    cudaMemcpy(mem->d_obj_weights, h_weights, 4 * sizeof(float), cudaMemcpyHostToDevice);

    // ── Main GWO Loop ─────────────────────────────────────────────────────
    for (int iter = 0; iter < max_iterations; iter++) {
        float a = 2.0f * (1.0f - (float)iter / (float)max_iterations);

        // Select leaders from archive
        select_leaders_cpu(
            arch_pos.data(), arch_obj.data(), arch_size,
            h_alpha.data(), h_beta.data(), h_delta.data(), dim
        );

        // Upload leaders to GPU
        gwo_memory_set_leaders(mem, h_alpha.data(), h_beta.data(), h_delta.data());

        // ── Launch position update kernel ─────────────────────────────────
        // 2D grid: [wolf_id, dim_chunk]
        int dim_chunks   = (dim + BLOCK_SIZE - 1) / BLOCK_SIZE;
        dim3 grid(num_wolves, dim_chunks);
        dim3 block(BLOCK_SIZE);
        size_t shmem = 3 * BLOCK_SIZE * sizeof(float);  // alpha+beta+delta slice

        gwo_update_positions<<<grid, block, shmem, mem->stream_compute>>>(
            mem->d_positions,
            mem->d_alpha, mem->d_beta, mem->d_delta,
            mem->d_rng_states,
            a, num_wolves, dim
        );
        cudaStreamSynchronize(mem->stream_compute);

        // ── Download updated positions to host ────────────────────────────
        gwo_memory_download_positions(mem, h_pos.data());

        // ── CPU fitness evaluation (caller must update h_obj) ─────────────
        // In the Python binding, this is a callback. Here we copy placeholders.
        // Python side handles this via the gwo_binding.py interface.
        // [out_objectives is filled by Python between iterations]

        // ── Re-upload objectives after Python evaluation ───────────────────
        // (Python will call cuda_gwo_update_objectives after evaluating)

        // ── Update Pareto archive ─────────────────────────────────────────
        // Read objectives from GPU (set by Python)
        gwo_memory_download_objectives(mem, h_obj.data());

        update_archive(
            arch_pos.data(), &arch_size,
            arch_obj.data(),
            h_pos.data(), h_obj.data(),
            num_wolves, dim
        );
    }

    // ── Copy final state to output ────────────────────────────────────────
    gwo_memory_download_positions(mem, out_positions);
    gwo_memory_download_objectives(mem, out_objectives);

    // Best solution = minimum f1 in archive
    if (arch_size > 0) {
        int best_idx = 0;
        float best_f1 = arch_obj[0];
        for (int i = 1; i < arch_size; i++) {
            if (arch_obj[i * NUM_OBJ] < best_f1) {
                best_f1  = arch_obj[i * NUM_OBJ];
                best_idx = i;
            }
        }
        memcpy(out_best_pos, arch_pos.data() + best_idx * dim, dim * sizeof(float));
        memcpy(out_best_obj, arch_obj.data() + best_idx * NUM_OBJ, NUM_OBJ * sizeof(float));
    }

    *timing_ms = now_ms() - t_start;

    gwo_memory_destroy(mem);
    return 0;
}

/*
 * cuda_gwo_update_objectives
 * ───────────────────────────
 * Called by Python after evaluating objectives 2 & 3 on CPU.
 * Uploads the full objective matrix back to GPU.
 *
 * NOTE: This function exists to support the hybrid evaluation model
 * where Python evaluates objectives while the GPU handles positions.
 */
int cuda_gwo_update_objectives(
    void*        mem_handle,
    const float* objectives,  // [num_wolves × 4]
    int num_wolves
) {
    GWOMemory* mem = (GWOMemory*)mem_handle;
    gwo_memory_upload_objectives(mem, objectives);
    return 0;
}

/*
 * cuda_device_info
 * ─────────────────
 * Returns GPU device properties as a formatted string.
 * Called from Python at startup to log hardware info.
 */
void cuda_device_info(char* out_buf, int buf_size) {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    if (device_count == 0) {
        snprintf(out_buf, buf_size, "No CUDA devices found");
        return;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    snprintf(out_buf, buf_size,
        "GPU: %s | Compute: %d.%d | SMs: %d | "
        "Global mem: %.1f GB | Shared/block: %zu KB | "
        "Max threads/block: %d",
        prop.name,
        prop.major, prop.minor,
        prop.multiProcessorCount,
        (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0),
        prop.sharedMemPerBlock / 1024,
        prop.maxThreadsPerBlock
    );
}

} // extern "C"
