#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// -----------------------------------------------------------------------------
// Error-checking macro (always use this in production code)
// -----------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _err = (call);                                              \
        if (_err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d  ->  %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(_err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

__global__ void kernelMatrixAdd(const float *input_A, const float *input_B,
                                    float *output, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary guard: blocks at the edge may have excess threads
    if (col >= width || row >= height) return;

    int idx = row * width + col;
    output[idx] = input_A[idx] + input_B[idx];
}

// =============================================================================
// HOST HELPERS
// =============================================================================
void printMatrix(const char *label, const float *data, int width, int height) {
    printf("\n  [%s] (%d rows x %d cols)\n", label, height, width);
    for (int r = 0; r < height; r++) {
        printf("    row %d: ", r);
        for (int c = 0; c < width; c++) {
            printf("%7.2f ", data[r * width + c]);
        }
        printf("\n");
    }
}

// =============================================================================
// MAIN
// =============================================================================
int main(void) {

    {
        int W = 3, H = 5, N = W * H;
        float *d_A, *d_B, *d_C, *h_A, *h_B, *h_C;

        h_A = (float *)malloc(N * sizeof(float));
        h_B = (float *)malloc(N * sizeof(float));
        h_C = (float *)malloc(N * sizeof(float));
        for (int i = 0; i < N; i++) h_A[i] = (float)(1);
        for (int i = 0; i < N; i++) h_B[i] = (float)(2);

        CUDA_CHECK(cudaMalloc(&d_A,  N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_B,  N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C,  N * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice));

        dim3 block(16, 16);
        dim3 grid((W + block.x - 1) / block.x,
                  (H + block.y - 1) / block.y);

        kernelMatrixAdd<<<grid, block>>>(d_A, d_B, d_C, W, H);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost));

        printMatrix("Input A", h_A, W, H);
        printMatrix("Input B", h_B, W, H);
        printMatrix("Output C", h_C, W, H);

        free(h_A); free(h_B); free(h_C);
        CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
    }

    cudaDeviceReset();  // Force CUDA context cleanup (prevents hang on exit)
    return 0;
}
