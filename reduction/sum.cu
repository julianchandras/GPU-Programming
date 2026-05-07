#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include "sum.h"

#define BLOCK_SIZE 256

/* Device function: in-warp reduction using shfl_down_sync (CUDA 9.0+) */
__device__ float warp_reduce(float val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void reduce_kernel(const float *input, float *output, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;
    
    // Shared memory for block-level reduction
    __shared__ float sdata[BLOCK_SIZE];
    
    float sum = 0.0f;
    while (idx < n) {
        sum += input[idx];
        idx += stride;
    }
    
    sdata[threadIdx.x] = sum;
    __syncthreads();
    
    // Block-level parallel reduction
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    // Efficiently reduce per-warp
    if (threadIdx.x < 32) {
        float warp_sum = sdata[threadIdx.x];
        if (blockDim.x > 32) warp_sum += sdata[threadIdx.x + 32];
        warp_sum = warp_reduce(warp_sum);
        
        if (threadIdx.x == 0) {
            output[blockIdx.x] = warp_sum;
        }
    }
}

/* Wrapper: GPU reduction with timing breakdown */
float reduce_gpu(const float *h_data, int n, 
                double *time_h2d_ms, double *time_kernel_ms, double *time_d2h_ms) {
    
    float *d_input = NULL, *d_partial = NULL;
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (num_blocks > 65536) num_blocks = 65536; /* Limit to max grid size */
    
    /* Allocate device memory */
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_partial, num_blocks * sizeof(float));
    
    if (!d_input || !d_partial) {
        fprintf(stderr, "GPU: memory allocation failed\n");
        cudaFree(d_input);
        cudaFree(d_partial);
        return 0.0f;
    }
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    /* H2D transfer */
    cudaEventRecord(start);
    cudaMemcpy(d_input, h_data, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_h2d = 0.0f;
    cudaEventElapsedTime(&ms_h2d, start, stop);
    *time_h2d_ms = ms_h2d;
    
    /* Kernel execution */
    cudaEventRecord(start);
    reduce_kernel<<<num_blocks, BLOCK_SIZE>>>(d_input, d_partial, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_kernel = 0.0f;
    cudaEventElapsedTime(&ms_kernel, start, stop);
    *time_kernel_ms = ms_kernel;
    
    /* D2H transfer and recursive reduction on CPU */
    float *h_partial = (float *)malloc(num_blocks * sizeof(float));
    cudaEventRecord(start);
    cudaMemcpy(h_partial, d_partial, num_blocks * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_d2h = 0.0f;
    cudaEventElapsedTime(&ms_d2h, start, stop);
    *time_d2h_ms = ms_d2h;
    
    /* Sum partial results on CPU */
    float result = 0.0f;
    for (int i = 0; i < num_blocks; i++) {
        result += h_partial[i];
    }
    
    free(h_partial);
    cudaFree(d_input);
    cudaFree(d_partial);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return result;
}
