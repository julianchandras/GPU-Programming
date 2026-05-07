#ifndef SUM_H
#define SUM_H

#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

// CPU function
float reduce_cpu(const float *data, int n);
float *load_dataset(const char *filename, int *out_n);
void benchmark_cpu(const float *data, int n, int runs, double *time_ms, double *std_dev);

// GPU function
float reduce_gpu(const float *h_data, int n, double *time_h2d_ms, double *time_kernel_ms, double *time_d2h_ms);

#ifdef __cplusplus
}
#endif

#endif
