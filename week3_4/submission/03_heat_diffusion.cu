// =============================================================================
// 03_heat_diffusion.cu
// Week 3: 2D Stencil -- Heat Diffusion Simulation
//
// Compile: nvcc -O2 -o 03_heat_diffusion 03_heat_diffusion.cu
// Run:     ./03_heat_diffusion
//
// Physics:
//   dT/dt = alpha * (d²T/dx² + d²T/dy²)        (heat equation)
//
// Discretised with explicit finite difference (5-point stencil):
//   T_new[i][j] = T[i][j]
//                 + alpha * dt/h² * (T[i-1][j] + T[i+1][j]
//                                  + T[i][j-1] + T[i][j+1]
//                                  - 4*T[i][j])
//
// Boundary condition: fixed temperature (Dirichlet)
//   - Edges stay at T_BOUNDARY = 0°C
//   - Initial hot spot at centre = T_INIT_HOT = 100°C
//
// Output: ASCII visualisation every N steps + timing report
// =============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

// -----------------------------------------------------------------------------
// Simulation parameters (feel free to tweak these)
// -----------------------------------------------------------------------------
#define GRID_W        256        // grid width  (columns)
#define GRID_H        256        // grid height (rows)
#define BLOCK_DIM_X    16        // threads per block in X
#define BLOCK_DIM_Y    16        // threads per block in Y
#define ALPHA         0.25f      // thermal diffusivity
#define DT            0.1f       // time step
#define DH            1.0f       // grid spacing
#define T_BOUNDARY    0.0f       // fixed boundary temperature (°C)
#define T_INIT_HOT  100.0f       // initial hot-spot temperature (°C)
#define HOT_RADIUS   20          // radius of initial hot spot (cells)
#define TOTAL_STEPS  40000        // total simulation steps
#define PRINT_EVERY   500        // print ASCII visualisation every N steps

// Stability criterion: alpha * dt / h² must be < 0.25 for 2D
#define STAB_COEFF   (ALPHA * DT / (DH * DH))

// -----------------------------------------------------------------------------
// Error macro
// -----------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                          \
                    __FILE__, __LINE__, cudaGetErrorString(_e));                 \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while(0)

// =============================================================================
// KERNEL: One step of 2D heat diffusion (5-point stencil)
//
// Grid layout: T[row * GRID_W + col]  (row-major, flat 1D)
// Boundary cells are SKIPPED (row=0, row=H-1, col=0, col=W-1)
// =============================================================================
__global__ void kernelHeatStep(const float *T_cur, float *T_new,
                                int width, int height, float coeff) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Skip boundary cells -- they are fixed
    if (col <= 0 || col >= width  - 1) return;
    if (row <= 0 || row >= height - 1) return;

    int idx = row * width + col;

    // 5-point stencil: north, south, east, west, centre
    float north = T_cur[(row - 1) * width + col];
    float south = T_cur[(row + 1) * width + col];
    float west  = T_cur[row * width + (col - 1)];
    float east  = T_cur[row * width + (col + 1)];
    float centre = T_cur[idx];

    T_new[idx] = centre + coeff * (north + south + east + west - 4.0f * centre);
}

// =============================================================================
// HOST: Initialise grid
//   - All cells at T_BOUNDARY
//   - Circular hot spot at centre
// =============================================================================
void initGrid(float *T, int width, int height) {
    int cx1 = width  / 3;
    int cy1 = height / 3;
    int cx2 = (2 * width)  / 3;
    int cy2 = (2 * height) / 3;

    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            int idx = r * width + c;
            float dist1 = sqrtf((float)((c - cx1)*(c - cx1) + (r - cy1)*(r - cy1)));
            float dist2 = sqrtf((float)((c - cx2)*(c - cx2) + (r - cy2)*(r - cy2)));
            T[idx] = (dist1 <= HOT_RADIUS || dist2 <= HOT_RADIUS) ? T_INIT_HOT : T_BOUNDARY;
        }
    }

    // Hard-set boundary rows/cols (Dirichlet)
    for (int c = 0; c < width;  c++) { T[c] = T_BOUNDARY; T[(height-1)*width+c] = T_BOUNDARY; }
    for (int r = 0; r < height; r++) { T[r*width] = T_BOUNDARY; T[r*width+width-1] = T_BOUNDARY; }
}

// =============================================================================
// HOST: ASCII visualisation -- heat map with ANSI colors (or plain if no color)
// =============================================================================
static int use_color = -1;  // -1 = not yet detected

static int supports_color(void) {
    if (use_color >= 0) return use_color;
    if (getenv("NO_COLOR")) { use_color = 0; return 0; }
    const char *term = getenv("TERM");
    use_color = (term && strcmp(term, "dumb") != 0 &&
                 (strstr(term, "256") || strstr(term, "xterm") ||
                  strstr(term, "color") || strstr(term, "linux")));
    return use_color;
}

// 256-color ANSI: blue(17) -> cyan(51) -> green(82) -> yellow(190) -> red(196)
static const int heat_colors[] = { 17, 27, 39, 51, 82, 154, 190, 208, 196 };
#define N_HEAT_COLORS 9

void asciiVisualize(const float *T, int width, int height,
                    int step, float elapsed_ms) {
    int step_c = width  / 64;
    int step_r = height / 24;
    if (step_c < 1) step_c = 1;
    if (step_r < 1) step_r = 1;

    int nlevels = 9;
    int has_color = supports_color();

    printf("\n");
    printf("\033[1m--- Step %5d  (%.1f ms elapsed) ---\033[0m\n", step, elapsed_ms);
    printf("+");
    for (int c = 0; c < width; c += step_c) printf("--");
    printf("+\n");

    for (int r = 0; r < height; r += step_r) {
        printf("|");
        for (int c = 0; c < width; c += step_c) {
            float val = T[r * width + c];
            int level = (int)(val / T_INIT_HOT * (nlevels - 1) + 0.5f);
            if (level < 0) level = 0;
            if (level >= nlevels) level = nlevels - 1;

            if (has_color) {
                printf("\033[48;5;%dm  \033[0m", heat_colors[level]);
            } else {
                // Fallback: use . : + * # @ for grayscale
                static const char *gray = " .:-=+*#@";
                putchar(gray[level]);
                putchar(gray[level]);
            }
        }
        printf("|\n");
    }

    printf("+");
    for (int c = 0; c < width; c += step_c) printf("--");
    printf("+\n");

    // Legend (color scale)
    if (has_color) {
        printf("  ");
        printf("\033[48;5;17m  \033[0m 0");
        printf("\302\260");  /* degree symbol UTF-8 */
        printf("C  ");
        printf("\033[48;5;82m  \033[0m 50");
        printf("\302\260");
        printf("C  ");
        printf("\033[48;5;196m  \033[0m 100");
        printf("\302\260");
        printf("C\n");
    }
}

// =============================================================================
// HOST: Compute max temperature (for monitoring)
// =============================================================================
float maxTemp(const float *T, int N) {
    float m = 0.0f;
    for (int i = 0; i < N; i++) if (T[i] > m) m = T[i];
    return m;
}

// =============================================================================
// MAIN
// =============================================================================
int main(void) {
    printf("============================================================\n");
    printf("  Week 3: 2D Heat Diffusion (5-point stencil)\n");
    printf("  Grid: %d x %d   Steps: %d\n", GRID_W, GRID_H, TOTAL_STEPS);
    printf("  Stability coeff alpha*dt/h^2 = %.4f (must be < 0.25)\n",
           STAB_COEFF);
    printf("============================================================\n");

    if (STAB_COEFF >= 0.25f) {
        fprintf(stderr, "ERROR: Stability condition violated! Reduce dt or alpha.\n");
        return 1;
    }

    int N = GRID_W * GRID_H;
    size_t bytes = N * sizeof(float);

    // ---- Allocate host memory ----
    float *h_T     = (float *)malloc(bytes);
    float *h_T_out = (float *)malloc(bytes);
    initGrid(h_T, GRID_W, GRID_H);

    // ---- Allocate device memory ----
    float *d_T_cur, *d_T_new;
    CUDA_CHECK(cudaMalloc(&d_T_cur, bytes));
    CUDA_CHECK(cudaMalloc(&d_T_new, bytes));

    // Copy initial condition to device (both buffers identical at start)
    CUDA_CHECK(cudaMemcpy(d_T_cur, h_T, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_T_new, h_T, bytes, cudaMemcpyHostToDevice));

    // ---- Configure kernel launch ----
    dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 gridDim((GRID_W  + blockDim.x - 1) / blockDim.x,
                 (GRID_H  + blockDim.y - 1) / blockDim.y);

    printf("\nBlock: (%d, %d)   Grid: (%d, %d)   Total threads: %d\n",
           blockDim.x, blockDim.y, gridDim.x, gridDim.y,
           gridDim.x * gridDim.y * blockDim.x * blockDim.y);

    // ---- Timing with CUDA events ----
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    // Initial visualisation
    asciiVisualize(h_T, GRID_W, GRID_H, 0, 0.0f);

    CUDA_CHECK(cudaEventRecord(ev_start));

    // ---- Main time-stepping loop ----
    for (int step = 1; step <= TOTAL_STEPS; step++) {
        kernelHeatStep<<<gridDim, blockDim>>>(d_T_cur, d_T_new,
                                              GRID_W, GRID_H, STAB_COEFF);

        // Ping-pong buffers: swap pointers (no memcpy needed)
        float *tmp = d_T_cur;
        d_T_cur    = d_T_new;
        d_T_new    = tmp;

        if (step % PRINT_EVERY == 0) {
            // Partial sync to get timing + copy for display
            float ms_so_far;
            CUDA_CHECK(cudaEventRecord(ev_stop));
            CUDA_CHECK(cudaEventSynchronize(ev_stop));
            CUDA_CHECK(cudaEventElapsedTime(&ms_so_far, ev_start, ev_stop));

            CUDA_CHECK(cudaMemcpy(h_T_out, d_T_cur, bytes, cudaMemcpyDeviceToHost));
            asciiVisualize(h_T_out, GRID_W, GRID_H, step, ms_so_far);
            printf("  Max temperature: %.2f°C\n", maxTemp(h_T_out, N));
        }
    }

    // ---- Final timing ----
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    float total_ms;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, ev_start, ev_stop));

    printf("\n============================================================\n");
    printf("  Simulation complete!\n");
    printf("  Total GPU time: %.2f ms for %d steps\n", total_ms, TOTAL_STEPS);
    printf("  Throughput: %.1f Mstencils/s\n",
           (double)TOTAL_STEPS * (GRID_W - 2) * (GRID_H - 2) / total_ms / 1e3);
    printf("============================================================\n");

    // ---- Cleanup ----
    free(h_T); free(h_T_out);
    CUDA_CHECK(cudaFree(d_T_cur));
    CUDA_CHECK(cudaFree(d_T_new));
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    cudaDeviceReset();  // Force CUDA context cleanup (prevents hang on exit)

    return 0;
}
