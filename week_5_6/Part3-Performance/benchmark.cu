/*
 * CUDA Matrix Multiplication Performance Benchmark
 * Benchmark Performa Perkalian Matriks
 *
 * This program compares the performance of three matrix multiplication implementations:
 * 1. CPU (serial) implementation
 * 2. Basic GPU implementation (global memory only)
 * 3. Tiled GPU implementation (shared memory optimization)
 *
 * Run this after completing Parts 1 and 2.
 * Students should observe significant speedups from basic GPU and especially tiled GPU.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "../common/matrix_io.h"
#include "../common/timer.h"

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s (line %d)\n", cudaGetErrorString(err), __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Tiled matrix multiplication tile width
#define TILE_WIDTH 16

// ============================================================================
// CPU Implementation (Reference)
// ============================================================================
void matmul_cpu(float *A, float *B, float *C, int M, int K, int N) {
    /*
     * CPU matrix multiplication: C = A * B
     * A: M x K matrix
     * B: K x N matrix
     * C: M x N matrix (output)
     */
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// ============================================================================
// Basic GPU Implementation (Global Memory)
// ============================================================================
__global__ void matmul_basic_kernel(float *A, float *B, float *C, int M, int K, int N) {
    /*
     * Basic GPU kernel using global memory
     * Thread (i, j) computes C[i*N + j]
     */
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void matmul_gpu_basic(float *A_h, float *B_h, float *C_h, int M, int K, int N) {
    /*
     * Wrapper for basic GPU matrix multiplication
     * Allocates device memory, copies data, launches kernel, and retrieves results
     */
    float *A_d, *B_d, *C_d;

    // Allocate device memory
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    CUDA_CHECK(cudaMalloc(&A_d, size_A));
    CUDA_CHECK(cudaMalloc(&B_d, size_B));
    CUDA_CHECK(cudaMalloc(&C_d, size_C));

    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(A_d, A_h, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B_h, size_B, cudaMemcpyHostToDevice));

    // Configure grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);

    // Launch kernel
    matmul_basic_kernel<<<gridDim, blockDim>>>(A_d, B_d, C_d, M, K, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(C_h, C_d, size_C, cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));
}

// ============================================================================
// Tiled GPU Implementation (Shared Memory Optimization)
// ============================================================================
__global__ void matmul_tiled_kernel(float *A, float *B, float *C, int M, int K, int N) {
    /*
     * Tiled matrix multiplication using shared memory
     * Tiles of size TILE_WIDTH x TILE_WIDTH are loaded into shared memory
     * This reduces global memory accesses significantly
     */
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float Cval = 0.0f;

    // Process K in tiles of size TILE_WIDTH
    for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; t++) {
        // Load A tile into shared memory
        if (row < M && t * TILE_WIDTH + tx < K) {
            As[ty][tx] = A[row * K + t * TILE_WIDTH + tx];
        } else {
            As[ty][tx] = 0.0f;
        }

        // Load B tile into shared memory
        if (t * TILE_WIDTH + ty < K && col < N) {
            Bs[ty][tx] = B[(t * TILE_WIDTH + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product using shared memory
        for (int k = 0; k < TILE_WIDTH; k++) {
            Cval += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    // Write result to global memory
    if (row < M && col < N) {
        C[row * N + col] = Cval;
    }
}

void matmul_gpu_tiled(float *A_h, float *B_h, float *C_h, int M, int K, int N) {
    /*
     * Wrapper for tiled GPU matrix multiplication
     * Same structure as basic GPU but uses optimized kernel
     */
    float *A_d, *B_d, *C_d;

    // Allocate device memory
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    CUDA_CHECK(cudaMalloc(&A_d, size_A));
    CUDA_CHECK(cudaMalloc(&B_d, size_B));
    CUDA_CHECK(cudaMalloc(&C_d, size_C));

    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(A_d, A_h, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B_h, size_B, cudaMemcpyHostToDevice));

    // Configure grid and block dimensions
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);

    // Launch kernel
    matmul_tiled_kernel<<<gridDim, blockDim>>>(A_d, B_d, C_d, M, K, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(C_h, C_d, size_C, cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));
}

// ============================================================================
// Verification and Utilities
// ============================================================================
bool verify_result(float *C_cpu, float *C_gpu, int M, int N, float epsilon = 1e-5f) {
    /*
     * Verify that GPU result matches CPU result within epsilon tolerance
     * Returns true if results match, false otherwise
     */
    for (int i = 0; i < M * N; i++) {
        float diff = fabsf(C_cpu[i] - C_gpu[i]);
        if (diff > epsilon) {
            fprintf(stderr, "Mismatch at index %d: CPU=%.6f, GPU=%.6f, diff=%.6f\n",
                    i, C_cpu[i], C_gpu[i], diff);
            return false;
        }
    }
    return true;
}

double compute_gflops(int M, int K, int N, double time_ms) {
    /*
     * Compute GFLOPS (Giga Floating Point Operations Per Second)
     * Matrix multiplication requires 2*M*K*N floating point operations
     * GFLOPS = (2*M*K*N) / (time_in_seconds * 1e9)
     */
    double flops = 2.0 * M * K * N;
    double time_s = time_ms / 1000.0;
    return (flops / (time_s * 1e9));
}

// ============================================================================
// Main Benchmark
// ============================================================================
int main() {
    printf("\n");
    printf("==============================================================\n");
    printf("  Matrix Multiplication Performance Benchmark\n");
    printf("  Benchmark Performa Perkalian Matriks\n");
    printf("==============================================================\n\n");

    // Test sizes: square matrices of varying dimensions
    int sizes[] = {128, 256, 512, 1024, 2048};  // optionally add 4096
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int num_runs = 3;  // Number of runs to average

    printf("  Size    |  CPU (ms)  | Basic GPU  | Tiled GPU  | Speedup\n");
    printf("--------------------------------------------------------------\n");

    for (int s = 0; s < num_sizes; s++) {
        int M = sizes[s];
        int K = sizes[s];
        int N = sizes[s];

        // Allocate host memory
        float *A_h = (float *)malloc(M * K * sizeof(float));
        float *B_h = (float *)malloc(K * N * sizeof(float));
        float *C_cpu = (float *)malloc(M * N * sizeof(float));
        float *C_basic = (float *)malloc(M * N * sizeof(float));
        float *C_tiled = (float *)malloc(M * N * sizeof(float));

        if (!A_h || !B_h || !C_cpu || !C_basic || !C_tiled) {
            fprintf(stderr, "Memory allocation failed for size %dx%d\n", M, N);
            continue;
        }

        // Initialize matrices with random values
        for (int i = 0; i < M * K; i++) {
            A_h[i] = (float)(rand() % 10);
        }
        for (int i = 0; i < K * N; i++) {
            B_h[i] = (float)(rand() % 10);
        }

        // Run CPU benchmark (only for smaller matrices)
        double cpu_time = 0.0;
        if (M <= 1024) {
            for (int run = 0; run < num_runs; run++) {
                struct timespec cpu_start = cpu_timer_start();
                matmul_cpu(A_h, B_h, C_cpu, M, K, N);
                cpu_time += cpu_timer_stop(cpu_start);
            }
            cpu_time /= num_runs;
        } else {
            printf("  %4dx%-3d | SKIPPED    |", M, N);
            cpu_time = -1.0;  // Mark as skipped
        }

        // Run basic GPU benchmark
        double basic_gpu_time = 0.0;
        for (int run = 0; run < num_runs; run++) {
            struct timespec basic_start = cpu_timer_start();
            matmul_gpu_basic(A_h, B_h, C_basic, M, K, N);
            basic_gpu_time += cpu_timer_stop(basic_start);
        }
        basic_gpu_time /= num_runs;

        // Run tiled GPU benchmark
        double tiled_gpu_time = 0.0;
        for (int run = 0; run < num_runs; run++) {
            struct timespec tiled_start = cpu_timer_start();
            matmul_gpu_tiled(A_h, B_h, C_tiled, M, K, N);
            tiled_gpu_time += cpu_timer_stop(tiled_start);
        }
        tiled_gpu_time /= num_runs;

        // Verify correctness at smallest size
        if (s == 0) {
            printf("\n  [Verification at size %dx%d]\n", M, N);

            // Run CPU version for verification
            matmul_cpu(A_h, B_h, C_cpu, M, K, N);

            if (verify_result(C_cpu, C_basic, M, N)) {
                printf("  Basic GPU:  PASSED\n");
            } else {
                printf("  Basic GPU:  FAILED\n");
            }

            if (verify_result(C_cpu, C_tiled, M, N)) {
                printf("  Tiled GPU:  PASSED\n");
            } else {
                printf("  Tiled GPU:  FAILED\n");
            }
            printf("\n");
        }

        // Print results
        if (cpu_time > 0) {
            double speedup = cpu_time / tiled_gpu_time;
            printf("  %4dx%-3d | %8.2f   | %8.2f   | %8.2f   | %6.1f x\n",
                   M, N, cpu_time, basic_gpu_time, tiled_gpu_time, speedup);
        } else {
            printf(" | %8.2f   | %8.2f   | (CPU skipped)\n",
                   basic_gpu_time, tiled_gpu_time);
        }

        // Free host memory
        free(A_h);
        free(B_h);
        free(C_cpu);
        free(C_basic);
        free(C_tiled);
    }

    printf("--------------------------------------------------------------\n");
    printf("==============================================================\n\n");

    // Compute and display GFLOPS for largest tested size
    int largest_size = sizes[num_sizes - 1];
    int M = largest_size;
    int K = largest_size;
    int N = largest_size;

    printf("  Effective GFLOPS at size %dx%dx%d:\n", M, K, N);
    printf("  (Higher is better - measures computational throughput)\n\n");

    // Allocate memory for GFLOPS computation
    float *A_h = (float *)malloc(M * K * sizeof(float));
    float *B_h = (float *)malloc(K * N * sizeof(float));
    float *C = (float *)malloc(M * N * sizeof(float));

    for (int i = 0; i < M * K; i++) A_h[i] = (float)(rand() % 10);
    for (int i = 0; i < K * N; i++) B_h[i] = (float)(rand() % 10);

    // Warm up runs
    matmul_gpu_basic(A_h, B_h, C, M, K, N);
    matmul_gpu_tiled(A_h, B_h, C, M, K, N);

    // Timed runs for GFLOPS
    struct timespec basic_start = cpu_timer_start();
    matmul_gpu_basic(A_h, B_h, C, M, K, N);
    double basic_time = cpu_timer_stop(basic_start);

    struct timespec tiled_start = cpu_timer_start();
    matmul_gpu_tiled(A_h, B_h, C, M, K, N);
    double tiled_time = cpu_timer_stop(tiled_start);

    double basic_gflops = compute_gflops(M, K, N, basic_time);
    double tiled_gflops = compute_gflops(M, K, N, tiled_time);

    printf("  Basic GPU:  %8.2f GFLOPS\n", basic_gflops);
    printf("  Tiled GPU:  %8.2f GFLOPS\n", tiled_gflops);
    printf("  Speedup:    %8.2f x\n\n", tiled_gflops / basic_gflops);

    free(A_h);
    free(B_h);
    free(C);

    printf("==============================================================\n");
    printf("  Benchmark Complete\n");
    printf("==============================================================\n\n");

    return EXIT_SUCCESS;
}
