#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "../common/matrix_io.h"
#include "../common/timer.h"

#define CUDA_CHECK(call) { cudaError_t err = call; if(err != cudaSuccess) { fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); exit(1); } }

// ============================================================
// TODO 1: Implement the basic matrix multiplication kernel
//
// Each thread computes ONE element of output matrix C
// C[row][col] = sum(A[row][k] * B[k][col]) for k = 0..K-1
//
// Parameters: A (MxK), B (KxN), C (MxN), M, K, N
// ============================================================
__global__ void matmul_basic_kernel(float *A, float *B, float *C,
                                     int M, int K, int N) {
    // YOUR CODE HERE
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

    // Allocate host output
    Matrix C = matrix_alloc(M, N);

    // ============================================================
    // TODO 2: Allocate device memory for A, B, C
    // Use: CUDA_CHECK(cudaMalloc(...))
    // Size in bytes: rows * cols * sizeof(float)
    // ============================================================
    float *d_A, *d_B, *d_C;
    // YOUR CODE HERE
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    // ============================================================
    // TODO 3: Copy A and B from host to device
    // Use: CUDA_CHECK(cudaMemcpy(..., cudaMemcpyHostToDevice))
    // ============================================================
    // YOUR CODE HERE
    CUDA_CHECK(cudaMemcpy(d_A, A.data, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data, size_B, cudaMemcpyHostToDevice));

    // ============================================================
    // TODO 4: Set up grid and block dimensions
    // Hint: Use 16x16 thread blocks
    // Calculate gridDim as (M + blockDim.x - 1) / blockDim.x
    // ============================================================
    // dim3 blockDim(...);
    // dim3 gridDim(...);
    // YOUR CODE HERE
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);

    // ============================================================
    // TODO 5: Launch the kernel and measure time
    // Use: matmul_basic_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    // ============================================================
    GpuTimer timer;
    gpu_timer_start(&timer);

    // YOUR CODE HERE - launch kernel
    matmul_basic_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);

    float gpu_time = gpu_timer_stop(&timer);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("GPU Kernel Time: %.3f ms\n", gpu_time);

    // ============================================================
    // TODO 6: Copy result C from device to host
    // Use: CUDA_CHECK(cudaMemcpy(C.data, d_C, ..., cudaMemcpyDeviceToHost))
    // ============================================================
    // YOUR CODE HERE
    CUDA_CHECK(cudaMemcpy(C.data, d_C, size_C, cudaMemcpyDeviceToHost));

    // Verify result
    if (argc >= 4) {
        Matrix C_expected = matrix_load_bin(argv[3]);
        int pass = matrix_compare(&C, &C_expected, 1e-3);
        printf("Verification: %s\n", pass ? "PASSED" : "FAILED");
        matrix_free(&C_expected);
    } else {
        // Compare with CPU result
        printf("Computing CPU reference...\n");
        struct timespec cpu_start = cpu_timer_start();
        Matrix C_cpu = matrix_multiply_cpu(&A, &B);
        float cpu_time = cpu_timer_stop(cpu_start);
        printf("CPU Time: %.3f ms\n", cpu_time);

        int pass = matrix_compare(&C, &C_cpu, 1e-3);
        printf("Verification: %s\n", pass ? "PASSED" : "FAILED");
        printf("Speedup: %.2fx\n", cpu_time / gpu_time);
        matrix_free(&C_cpu);
    }

    // Print first few elements for debugging
    printf("\nResult C (first 4x4):\n");
    matrix_print(&C, 4);

    // ============================================================
    // TODO 7: Free device memory
    // Use: CUDA_CHECK(cudaFree(...))
    // ============================================================
    // YOUR CODE HERE
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    matrix_free(&A);
    matrix_free(&B);
    matrix_free(&C);

    return 0;
}
