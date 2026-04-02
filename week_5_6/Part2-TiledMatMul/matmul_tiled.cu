#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "../common/matrix_io.h"
#include "../common/timer.h"

#define CUDA_CHECK(call) { cudaError_t err = call; if(err != cudaSuccess) { fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); exit(1); } }

#define TILE_WIDTH 16

// ============================================================
// TODO 1: Implement the tiled matrix multiplication kernel
//
// This kernel uses shared memory to reduce global memory accesses.
//
// Key concepts:
// - Each thread block processes a tile of output matrix C
// - Data is loaded collaboratively into shared memory
// - __syncthreads() ensures all threads in block see consistent data
//
// Algorithm outline:
// 1. Declare shared memory arrays for A and B tiles
// 2. Loop over number of phases (tiles): num_phases = (K + TILE_WIDTH - 1) / TILE_WIDTH
// 3. For each phase:
//    a) Load tile of A from global memory to shared memory (with boundary check)
//    b) Load tile of B from global memory to shared memory (with boundary check)
//    c) Synchronize threads
//    d) Compute partial dot product using shared memory
//    e) Synchronize threads
// 4. Write final result to global memory (with boundary check)
//
// Hints:
// - Use blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y to identify thread position
// - For boundary check on loading: if (row < M && col < K) load A tile; else load 0
// - For boundary check on loading: if (row < K && col < N) load B tile; else load 0
// - Remember: A[row][col] is at A[row * K + col] in row-major storage
// - Remember: B[row][col] is at B[row * N + col] in row-major storage
// ============================================================
__global__ void matmul_tiled_kernel(float *A, float *B, float *C,
                                     int M, int K, int N) {
    // YOUR CODE HERE
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float Cval = 0.0f;

    for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; t++) {
        if (row < M && t * TILE_WIDTH + tx < K) {
            As[ty][tx] = A[row * K + t * TILE_WIDTH + tx];
        } else {
            As[ty][tx] = 0.0f;
        }

        if (t * TILE_WIDTH + ty < K && col < N) {
            Bs[ty][tx] = B[(t * TILE_WIDTH + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++) {
            Cval += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = Cval;
    }
}

// Basic kernel for comparison (non-tiled)
__global__ void matmul_basic_kernel(float *A, float *B, float *C,
                                     int M, int K, int N) {
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

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <matrix_A.bin> <matrix_B.bin> [expected_C.bin]\n", argv[0]);
        return 1;
    }

    // Load matrices from binary files
    Matrix A = matrix_load_bin(argv[1]);
    Matrix B = matrix_load_bin(argv[2]);

    printf("Matrix A: %d x %d\n", A.rows, A.cols);
    printf("Matrix B: %d x %d\n", B.rows, B.cols);

    // Validate dimensions
    if (A.cols != B.rows) {
        printf("Error: A.cols (%d) != B.rows (%d). Cannot multiply.\n", A.cols, B.rows);
        return 1;
    }

    int M = A.rows, K = A.cols, N = B.cols;

    // Allocate host output matrices
    Matrix C_tiled = matrix_alloc(M, N);
    Matrix C_basic = matrix_alloc(M, N);

    // ============================================================
    // TODO 2: Allocate device memory for A, B, C
    // ============================================================
    float *d_A, *d_B, *d_C_tiled, *d_C_basic;
    // YOUR CODE HERE
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    CUDA_CHECK(cudaMalloc(&d_A, size_A))
    CUDA_CHECK(cudaMalloc(&d_B, size_B))
    CUDA_CHECK(cudaMalloc(&d_C_tiled, size_C))
    CUDA_CHECK(cudaMalloc(&d_C_basic, size_C))

    // ============================================================
    // TODO 3: Copy A and B from host to device
    // ============================================================
    // YOUR CODE HERE
    CUDA_CHECK(cudaMemcpy(d_A, A.data, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data, size_B, cudaMemcpyHostToDevice));

    // ============================================================
    // TODO 4: Set up grid and block dimensions
    // Use TILE_WIDTH for block dimensions (TILE_WIDTH x TILE_WIDTH)
    // ============================================================
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    // dim3 gridDim(...);
    // YOUR CODE HERE
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);

    printf("Grid: %d x %d, Block: %d x %d\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y);

    // ============================================================
    // Benchmark: Run basic kernel for comparison
    // ============================================================
    printf("\n=== Running Basic (Non-Tiled) Kernel ===\n");
    GpuTimer timer_basic;
    gpu_timer_start(&timer_basic);

    // YOUR CODE HERE - launch basic kernel
    matmul_basic_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C_basic, M, K, N);

    float gpu_time_basic = gpu_timer_stop(&timer_basic);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("Basic Kernel Time: %.3f ms\n", gpu_time_basic);

    // ============================================================
    // TODO 5: Launch the tiled kernel and measure time
    // ============================================================
    printf("\n=== Running Tiled Kernel ===\n");
    GpuTimer timer_tiled;
    gpu_timer_start(&timer_tiled);

    // YOUR CODE HERE - launch tiled kernel
    matmul_tiled_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C_tiled, M, K, N);

    float gpu_time_tiled = gpu_timer_stop(&timer_tiled);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("Tiled Kernel Time: %.3f ms\n", gpu_time_tiled);

    // ============================================================
    // TODO 6: Copy results C from device to host
    // Copy both C_tiled and C_basic
    // ============================================================
    // YOUR CODE HERE
    CUDA_CHECK(cudaMemcpy(C_basic.data, d_C_basic, size_C, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(C_tiled.data, d_C_tiled, size_C, cudaMemcpyDeviceToHost));

    // Performance comparison
    printf("\n=== Performance Analysis ===\n");
    printf("Speedup (Basic -> Tiled): %.2fx\n", gpu_time_basic / gpu_time_tiled);

    // Verify results
    printf("\n=== Verification ===\n");
    if (argc >= 4) {
        Matrix C_expected = matrix_load_bin(argv[3]);
        int pass_tiled = matrix_compare(&C_tiled, &C_expected, 1e-3);
        int pass_basic = matrix_compare(&C_basic, &C_expected, 1e-3);
        printf("Tiled Kernel: %s\n", pass_tiled ? "PASSED" : "FAILED");
        printf("Basic Kernel: %s\n", pass_basic ? "PASSED" : "FAILED");
        matrix_free(&C_expected);
    } else {
        // Compare with CPU result
        printf("Computing CPU reference...\n");
        struct timespec cpu_start = cpu_timer_start();
        Matrix C_cpu = matrix_multiply_cpu(&A, &B);
        float cpu_time = cpu_timer_stop(cpu_start);
        printf("CPU Time: %.3f ms\n", cpu_time);

        int pass_tiled = matrix_compare(&C_tiled, &C_cpu, 1e-3);
        int pass_basic = matrix_compare(&C_basic, &C_cpu, 1e-3);
        printf("\nTiled Kernel: %s\n", pass_tiled ? "PASSED" : "FAILED");
        printf("Basic Kernel: %s\n", pass_basic ? "PASSED" : "FAILED");
        printf("Tiled Speedup (CPU): %.2fx\n", cpu_time / gpu_time_tiled);
        printf("Basic Speedup (CPU): %.2fx\n", cpu_time / gpu_time_basic);
        matrix_free(&C_cpu);
    }

    // Print first few elements for debugging
    printf("\nTiled Result C (first 4x4):\n");
    matrix_print(&C_tiled, 4);

    // ============================================================
    // TODO 7: Free device memory
    // ============================================================
    // YOUR CODE HERE
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C_basic));
    CUDA_CHECK(cudaFree(d_C_tiled));

    matrix_free(&A);
    matrix_free(&B);
    matrix_free(&C_tiled);
    matrix_free(&C_basic);

    return 0;
}
