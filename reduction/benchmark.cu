#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "sum.h"

#define BENCHMARK_RUNS 5

typedef struct {
    int n;
    double time_cpu_ms;
    double std_cpu_ms;
    double time_h2d_ms;
    double time_kernel_ms;
    double time_d2h_ms;
    double time_total_gpu_ms;
    float result_cpu;
    float result_gpu;
    double error_rel;
    double speedup;
} BenchmarkResult;

void print_results(BenchmarkResult *results, int num_results) {
    printf("\n=== BENCHMARK RESULTS ===\n\n");
    printf("%-12s %-15s %-15s %-12s %-12s %-12s %-15s %-12s %-15s %-10s\n",
           "n", "T_CPU (ms)", "T_Kernel (ms)", "H2D (ms)", "D2H (ms)", 
           "T_GPU (ms)", "Error (rel)", "Result_GPU", "Speedup", "T_GPU");
    printf("%-12s %-15s %-15s %-12s %-12s %-12s %-15s %-12s %-15s %-10s\n",
           "---", "---", "---", "---", "---", "---", "---", "---", "---", "---");
    
    for (int i = 0; i < num_results; i++) {
        BenchmarkResult *r = &results[i];
        printf("%-12d %-15.4f %-15.4f %-12.4f %-12.4f %-12.4f %-15.2e %-12.6f %-15.2f %-10.2f%%\n",
               r->n,
               r->time_cpu_ms,
               r->time_kernel_ms,
               r->time_h2d_ms,
               r->time_d2h_ms,
               r->time_total_gpu_ms,
               r->error_rel,
               r->result_gpu,
               r->speedup,
               100.0 * r->time_kernel_ms / r->time_total_gpu_ms);
    }
}

void save_csv(BenchmarkResult *results, int num_results, const char *filename) {
    FILE *f = fopen(filename, "w");
    if (!f) {
        fprintf(stderr, "Error: cannot open %s for writing\n", filename);
        return;
    }
    
    fprintf(f, "n,T_CPU_ms,Std_CPU_ms,T_H2D_ms,T_Kernel_ms,T_D2H_ms,T_Total_GPU_ms,");
    fprintf(f, "Result_CPU,Result_GPU,Error_rel,Speedup\n");
    
    for (int i = 0; i < num_results; i++) {
        BenchmarkResult *r = &results[i];
        fprintf(f, "%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,",
                r->n, r->time_cpu_ms, r->std_cpu_ms,
                r->time_h2d_ms, r->time_kernel_ms, r->time_d2h_ms,
                r->time_total_gpu_ms);
        fprintf(f, "%.10f,%.10f,%.6e,%.4f\n",
                r->result_cpu, r->result_gpu, r->error_rel, r->speedup);
    }
    
    fclose(f);
    printf("Results saved to: %s\n", filename);
}

int main(int argc, char *argv[]) {
    /* Dataset filenames (relative to bin directory) */
    const char *datasets[] = {
        "../datasets/data_1k.bin",
        "../datasets/data_100k.bin",
        "../datasets/data_1m.bin",
        "../datasets/data_100m.bin"
    };
    const int num_datasets = 4;
    
    BenchmarkResult results[4];
    int num_results = 0;
    
    printf("=== CUDA Reduction Benchmark ===\n\n");
    
    for (int i = 0; i < num_datasets; i++) {
        printf("Loading dataset: %s...\n", datasets[i]);
        
        int n;
        float *data = load_dataset(datasets[i], &n);
        
        if (!data) {
            printf("  SKIP: dataset not found or too large for current memory\n");
            continue;
        }
        
        printf("  Loaded %d elements\n", n);
        BenchmarkResult *r = &results[num_results];
        r->n = n;
        
        /* Benchmark CPU */
        printf("  CPU benchmark (%d runs)...\n", BENCHMARK_RUNS);
        benchmark_cpu(data, n, BENCHMARK_RUNS, &r->time_cpu_ms, &r->std_cpu_ms);
        printf("    Time: %.6f ms ± %.6f ms\n", r->time_cpu_ms, r->std_cpu_ms);
        
        /* Compute reference result (CPU) */
        r->result_cpu = reduce_cpu(data, n);
        
        /* Warmup GPU */
        printf("  GPU warmup...\n");
        reduce_gpu(data, n, &r->time_h2d_ms, &r->time_kernel_ms, &r->time_d2h_ms);
        
        /* Benchmark GPU (runs only kernel timing) */
        printf("  GPU benchmark (%d runs)...\n", BENCHMARK_RUNS);
        double h2d_total = 0, kernel_total = 0, d2h_total = 0;
        for (int run = 0; run < BENCHMARK_RUNS; run++) {
            float gpu_result = reduce_gpu(data, n, &r->time_h2d_ms, &r->time_kernel_ms, &r->time_d2h_ms);
            if (run == 0) r->result_gpu = gpu_result;
            h2d_total += r->time_h2d_ms;
            kernel_total += r->time_kernel_ms;
            d2h_total += r->time_d2h_ms;
        }
        r->time_h2d_ms = h2d_total / BENCHMARK_RUNS;
        r->time_kernel_ms = kernel_total / BENCHMARK_RUNS;
        r->time_d2h_ms = d2h_total / BENCHMARK_RUNS;
        r->time_total_gpu_ms = r->time_h2d_ms + r->time_kernel_ms + r->time_d2h_ms;
        
        /* Compute error and speedup */
        if (r->result_cpu != 0.0f) {
            r->error_rel = fabs(r->result_cpu - r->result_gpu) / fabs(r->result_cpu);
        } else {
            r->error_rel = fabs(r->result_cpu - r->result_gpu);
        }
        r->speedup = r->time_cpu_ms / r->time_total_gpu_ms;
        
        printf("    CPU result: %.6f, GPU result: %.6f, rel_error: %.2e\n",
               r->result_cpu, r->result_gpu, r->error_rel);
        printf("    H2D: %.4f ms, Kernel: %.4f ms, D2H: %.4f ms, Total: %.4f ms\n",
               r->time_h2d_ms, r->time_kernel_ms, r->time_d2h_ms, r->time_total_gpu_ms);
        printf("    Speedup: %.2f x\n\n", r->speedup);
        
        free(data);
        num_results++;
    }
    
    /* Save and print results */
    print_results(results, num_results);
    save_csv(results, num_results, "results.csv");
    
    return 0;
}
