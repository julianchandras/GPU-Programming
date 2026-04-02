# =============================================================================
# Makefile -- Week 3 CUDA Lab
#
# Usage:
#   make           - compile all examples (00, 01, 02, 03)
#   make all       - same
#   make run_all   - compile and run all
#   make clean     - remove binaries
#
# Adjust ARCH if your GPU differs: sm_70=V100, sm_75=Turing, sm_80=A100, sm_86=RTX A5000, sm_89=L40
# =============================================================================

NVCC     = nvcc
ARCH     = sm_86
CFLAGS   = -O2 -arch=$(ARCH) --std=c++14 -lineinfo
WARN     = -Xcompiler -Wall

BIN_DIR  = bin
BINS     = $(addprefix $(BIN_DIR)/,00_hello_cuda 01_2d_indexing 02_blur 03_heat_diffusion)

.PHONY: all clean run_all run_session1 gpu_info

all: $(BIN_DIR) $(BINS)

$(BIN_DIR):
	mkdir -p $@

$(BIN_DIR)/00_hello_cuda: 00_hello_cuda.cu
	$(NVCC) $(CFLAGS) $(WARN) -o $@ $<
	@echo "  [OK] $@"

$(BIN_DIR)/01_2d_indexing: 01_2d_indexing.cu
	$(NVCC) $(CFLAGS) $(WARN) -o $@ $<
	@echo "  [OK] $@"

$(BIN_DIR)/02_blur: 02_blur.cu
	$(NVCC) $(CFLAGS) $(WARN) -o $@ $<
	@echo "  [OK] $@"

$(BIN_DIR)/03_heat_diffusion: 03_heat_diffusion.cu
	$(NVCC) $(CFLAGS) $(WARN) -lm -o $@ $<
	@echo "  [OK] $@"

run_all run_session1: all
	@echo "=== Running 00_hello_cuda ==="; ./$(BIN_DIR)/00_hello_cuda
	@echo "=== Running 01_2d_indexing ==="; ./$(BIN_DIR)/01_2d_indexing
	@echo "=== Running 02_blur ==="; ./$(BIN_DIR)/02_blur
	@echo "=== Running 03_heat_diffusion ==="; ./$(BIN_DIR)/03_heat_diffusion

clean:
	rm -rf $(BIN_DIR)
	@echo "  Cleaned."

gpu_info:
	nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader
