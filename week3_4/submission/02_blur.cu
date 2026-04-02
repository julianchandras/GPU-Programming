#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                           \
                    __FILE__, __LINE__, cudaGetErrorString(_e));                 \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while(0)

__global__ void kernelBlur9pt(const float *in, float *out,
                              int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Skip border: need full 9-point neighbourhood
    if (col <= 0 || col >= width  - 1) return;
    if (row <= 0 || row >= height - 1) return;

    int idx = row * width + col;
    float centre    = in[idx];
    float north     = in[(row - 1) * width + col];
    float south     = in[(row + 1) * width + col];
    float west      = in[row * width + (col - 1)];
    float east      = in[row * width + (col + 1)];
    float northeast = in[(row - 1) * width + (col + 1)];
    float northwest = in[(row - 1) * width + (col - 1)];
    float southeast = in[(row + 1) * width + (col + 1)];
    float southwest = in[(row + 1) * width + (col - 1)];

    out[idx] = (centre + north + south + west + east + northeast + northwest + southeast + southwest) * (1.0f / 9.0f);
}

void printPatch(const char *label, const float *data, int width, int height,
                int r0, int c0, int patch_r, int patch_c) {
    printf("\n  [%s] patch at (%d,%d):\n", label, r0, c0);
    for (int r = r0; r < r0 + patch_r && r < height; r++) {
        printf("    ");
        for (int c = c0; c < c0 + patch_c && c < width; c++) {
            printf("%6.1f ", data[r * width + c]);
        }
        printf("\n");
    }
}

int main(void) {
    printf("============================================================\n");
    printf("  Week 3 (Modified): 9-Point Image Blur (stencil demo)\n");
    printf("============================================================\n");

    const int W = 8, H = 6;
    const int N = W * H;
    size_t bytes = N * sizeof(float);

    // Create test image: sharp vertical edge (left=0, right=100)
    float *h_in = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);
    for (int r = 0; r < H; r++) {
        for (int c = 0; c < W; c++) {
            h_in[r * W + c] = (c >= W / 2) ? 100.0f : 0.0f;
        }
    }

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    // Copy input to output for border (blur doesn't touch borders)
    CUDA_CHECK(cudaMemcpy(d_out, d_in, bytes, cudaMemcpyDeviceToDevice));

    dim3 block(4, 4);
    dim3 grid((W + block.x - 1) / block.x,
              (H + block.y - 1) / block.y);

    kernelBlur9pt<<<grid, block>>>(d_in, d_out, W, H);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    printPatch("Input (sharp edge)", h_in, W, H, 0, 0, H, W);
    printPatch("Output (1 blur pass)", h_out, W, H, 0, 0, H, W);

    // Verify: interior pixel (2,3): C=0, N=0, S=0, W=0, E=100, NE=100, NW=0, SE=100, SW=0 -> (0+0+0+0+100+100+0+100+0)/9 = 33.3
    float expected_23 = (0.0f + 0.0f + 0.0f + 0.0f + 100.0f + 100.0f + 0.0f + 100.0f + 0.0f) * (1.0f / 9.0f);
    float actual_23 = h_out[2 * W + 3];
    printf("\n  Verification: pixel (2,3) expected %.1f, got %.1f\n",
           expected_23, actual_23);
    if (fabsf(actual_23 - expected_23) < 0.01f) {
        printf("  [OK] Blur kernel correct!\n");
    } else {
        printf("  [FAIL] Mismatch\n");
    }

    printf("\n============================================================\n");
    printf("  Key: out[i][j] = (C+N+S+E+W+NE+NW+SE+SW)/9, border skipped\n");
    printf("============================================================\n");

    free(h_in); free(h_out);
    CUDA_CHECK(cudaFree(d_in)); CUDA_CHECK(cudaFree(d_out));
    cudaDeviceReset();  // Force CUDA context cleanup (prevents hang on exit)
    return 0;
}
