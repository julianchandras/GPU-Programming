#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "sum.h"

float reduce_cpu(const float *data, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += data[i];
    }
    return sum;
}

/* Load dataset from binary file: [n (int32)] [float data...] */
float *load_dataset(const char *filename, int *out_n) {
    FILE *f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Error: cannot open file %s\n", filename);
        return NULL;
    }
    
    int n;
    if (fread(&n, sizeof(int), 1, f) != 1) {
        fprintf(stderr, "Error: cannot read dataset size\n");
        fclose(f);
        return NULL;
    }
    
    float *data = (float *)malloc(n * sizeof(float));
    if (!data) {
        fprintf(stderr, "Error: cannot allocate memory for %d elements\n", n);
        fclose(f);
        return NULL;
    }
    
    if (fread(data, sizeof(float), n, f) != (size_t)n) {
        fprintf(stderr, "Error: cannot read dataset (got %d elements)\n", n);
        free(data);
        fclose(f);
        return NULL;
    }
    
    fclose(f);
    *out_n = n;
    return data;
}

/* Benchmark CPU reduction: run multiple times and collect statistics */
void benchmark_cpu(const float *data, int n, int runs, double *time_ms, double *std_dev) {
    double *times = (double *)malloc(runs * sizeof(double));
    double sum_time = 0.0;
    
    for (int run = 0; run < runs; run++) {
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        
        float result = reduce_cpu(data, n);
        
        clock_gettime(CLOCK_MONOTONIC, &end);
        double elapsed = (end.tv_sec - start.tv_sec) * 1000.0 +
                        (end.tv_nsec - start.tv_nsec) / 1e6;
        times[run] = elapsed;
        sum_time += elapsed;
        
        if (run == 0) {
            printf("  CPU result: %.6f\n", result);
        }
    }
    
    *time_ms = sum_time / runs;
    
    // Calculate standard deviation
    double sum_sq_dev = 0.0;
    for (int i = 0; i < runs; i++) {
        double dev = times[i] - *time_ms;
        sum_sq_dev += dev * dev;
    }
    *std_dev = (runs > 1) ? sqrt(sum_sq_dev / (runs - 1)) : 0.0;
    
    free(times);
}
