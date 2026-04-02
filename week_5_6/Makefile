# ============================================================================
# CUDA Matrix Multiplication Lab - Makefile
#
# This Makefile builds all three parts of the matrix multiplication lab:
# Part 1: Basic GPU matrix multiplication
# Part 2: Tiled GPU matrix multiplication
# Part 3: Performance benchmark comparing all implementations
#
# Usage:
#   make all          - Build all parts (part1, part2, part3)
#   make part1        - Build basic matrix multiplication
#   make part2        - Build tiled matrix multiplication
#   make part3        - Build performance benchmark
#   make benchmark    - Alias for 'make part3'
#   make datasets     - Generate test datasets
#   make solutions    - Build solution files (instructor only)
#   make test         - Run tests on part1 and part2
#   make clean        - Remove build artifacts
# ============================================================================

# Auto-detect NVCC compiler
NVCC := $(shell which nvcc)
ifeq ($(NVCC),)
    $(error NVCC not found. Please install CUDA Toolkit.)
endif

# Compiler and flags
NVCC_FLAGS := -O2
# NOTE: Change -arch=sm_60 based on your GPU compute capability:
#   sm_50 = Maxwell (GTX 750, GTX 960, etc.)
#   sm_60 = Pascal (GTX 1080, P100, etc.) - DEFAULT
#   sm_70 = Volta (V100, RTX 2080, etc.)
#   sm_80 = Ampere (RTX 30 series, A100, etc.)
#   sm_90 = Hopper (H100, etc.)
NVCC_ARCH := -arch=sm_60

# Directories
BIN_DIR := bin
PART1_DIR := Part1-BasicMatMul
PART2_DIR := Part2-TiledMatMul
PART3_DIR := Part3-Performance
COMMON_DIR := common
DATASETS_DIR := datasets

# Output files
PART1_EXE := $(BIN_DIR)/matmul_basic
PART2_EXE := $(BIN_DIR)/matmul_tiled
PART3_EXE := $(BIN_DIR)/benchmark
DATASET_GEN := $(BIN_DIR)/generate_datasets

# Source files
PART1_SRC := $(PART1_DIR)/matmul_basic.cu
PART2_SRC := $(PART2_DIR)/matmul_tiled.cu
PART3_SRC := $(PART3_DIR)/benchmark.cu

# Color output for better readability
RED := \033[0;31m
GREEN := \033[0;32m
BLUE := \033[0;34m
YELLOW := \033[1;33m
NC := \033[0m  # No Color

# ============================================================================
# Default target
# ============================================================================
.DEFAULT_GOAL := all

# ============================================================================
# Phony targets (not actual files)
# ============================================================================
.PHONY: all part1 part2 part3 benchmark datasets solutions test clean help

# ============================================================================
# Help target
# ============================================================================
help:
	@echo "$(BLUE)==============================================================$(NC)"
	@echo "$(BLUE)  CUDA Matrix Multiplication Lab - Build System$(NC)"
	@echo "$(BLUE)==============================================================$(NC)"
	@echo ""
	@echo "Available targets:"
	@echo "  $(GREEN)all$(NC)       - Build all parts (default)"
	@echo "  $(GREEN)part1$(NC)     - Build Part 1: Basic GPU implementation"
	@echo "  $(GREEN)part2$(NC)     - Build Part 2: Tiled GPU implementation"
	@echo "  $(GREEN)part3$(NC)     - Build Part 3: Performance benchmark"
	@echo "  $(GREEN)benchmark$(NC) - Alias for 'part3'"
	@echo "  $(GREEN)datasets$(NC)  - Generate test datasets"
	@echo "  $(GREEN)solutions$(NC) - Build solution files (instructor only)"
	@echo "  $(GREEN)test$(NC)      - Run tests on part1 and part2"
	@echo "  $(GREEN)clean$(NC)     - Remove build artifacts"
	@echo "  $(GREEN)help$(NC)      - Display this help message"
	@echo ""
	@echo "Configuration:"
	@echo "  CUDA Compiler: $(NVCC)"
	@echo "  NVCC Flags:    $(NVCC_FLAGS)"
	@echo "  GPU Arch:      $(NVCC_ARCH)"
	@echo ""
	@echo "================================================================"

# ============================================================================
# Create bin directory
# ============================================================================
$(BIN_DIR):
	@mkdir -p $(BIN_DIR)
	@echo "$(BLUE)[DIR]$(NC) Created $(BIN_DIR)/"

# ============================================================================
# Build Part 1: Basic GPU Matrix Multiplication
# ============================================================================
part1: $(PART1_EXE)

$(PART1_EXE): $(PART1_SRC) | $(BIN_DIR)
	@echo "$(YELLOW)[BUILD]$(NC) Part 1: Basic GPU implementation"
	@$(NVCC) $(NVCC_FLAGS) $(NVCC_ARCH) -o $@ $<
	@echo "$(GREEN)[OK]$(NC) $@"

# ============================================================================
# Build Part 2: Tiled GPU Matrix Multiplication
# ============================================================================
part2: $(PART2_EXE)

$(PART2_EXE): $(PART2_SRC) | $(BIN_DIR)
	@echo "$(YELLOW)[BUILD]$(NC) Part 2: Tiled GPU implementation"
	@$(NVCC) $(NVCC_FLAGS) $(NVCC_ARCH) -o $@ $<
	@echo "$(GREEN)[OK]$(NC) $@"

# ============================================================================
# Build Part 3: Performance Benchmark
# ============================================================================
part3: $(PART3_EXE)
benchmark: part3

$(PART3_EXE): $(PART3_SRC) | $(BIN_DIR)
	@echo "$(YELLOW)[BUILD]$(NC) Part 3: Performance benchmark"
	@$(NVCC) $(NVCC_FLAGS) $(NVCC_ARCH) -o $@ $<
	@echo "$(GREEN)[OK]$(NC) $@"

# ============================================================================
# Build All Targets
# ============================================================================
all: part1 part2 part3
	@echo ""
	@echo "$(GREEN)==============================================================$(NC)"
	@echo "$(GREEN)  All builds completed successfully!$(NC)"
	@echo "$(GREEN)==============================================================$(NC)"
	@echo ""
	@echo "Executables:"
	@echo "  $(PART1_EXE) - Run basic GPU implementation"
	@echo "  $(PART2_EXE) - Run tiled GPU implementation"
	@echo "  $(PART3_EXE) - Run performance benchmark"
	@echo ""

# ============================================================================
# Build Dataset Generator (if it exists)
# ============================================================================
datasets:
	@if [ -f "$(DATASETS_DIR)/generate.cu" ]; then \
		echo "$(YELLOW)[BUILD]$(NC) Dataset generator"; \
		$(NVCC) $(NVCC_FLAGS) $(NVCC_ARCH) -o $(DATASET_GEN) $(DATASETS_DIR)/generate.cu; \
		echo "$(GREEN)[OK]$(NC) $(DATASET_GEN)"; \
		echo "$(YELLOW)[RUN]$(NC) Generating datasets..."; \
		./$(DATASET_GEN); \
		echo "$(GREEN)[OK]$(NC) Datasets generated"; \
	else \
		echo "$(RED)[SKIP]$(NC) Dataset generator not found"; \
	fi

# ============================================================================
# Build Solutions (Instructor Files)
# ============================================================================
solutions:
	@if [ -d "Solutions" ]; then \
		echo "$(YELLOW)[BUILD]$(NC) Solution files"; \
		for file in Solutions/*.cu; do \
			if [ -f "$$file" ]; then \
				exe=$$(echo $$file | sed 's/\.cu//'); \
				$(NVCC) $(NVCC_FLAGS) $(NVCC_ARCH) -o $$exe $$file; \
				echo "$(GREEN)[OK]$(NC) $$exe"; \
			fi \
		done; \
	else \
		echo "$(RED)[INFO]$(NC) Solutions directory not found"; \
	fi

# ============================================================================
# Test Target: Run basic tests
# ============================================================================
test: part1 part2
	@echo ""
	@echo "$(BLUE)==============================================================$(NC)"
	@echo "$(BLUE)  Running Tests$(NC)"
	@echo "$(BLUE)==============================================================$(NC)"
	@echo ""

	@echo "$(YELLOW)[TEST]$(NC) Part 1: Basic GPU implementation"
	@./$(PART1_EXE) datasets/small/mat_128x128_A.bin datasets/small/mat_128x128_B.bin datasets/small/mat_128x128_C.bin 2>&1 | head -20
	@echo ""

	@echo "$(YELLOW)[TEST]$(NC) Part 2: Tiled GPU implementation"
	@./$(PART2_EXE) datasets/small/mat_128x128_A.bin datasets/small/mat_128x128_B.bin datasets/small/mat_128x128_C.bin 2>&1 | head -20
	@echo ""

	@echo "$(GREEN)[OK]$(NC) Tests completed"
	@echo ""

# ============================================================================
# Clean Target: Remove all build artifacts
# ============================================================================
clean:
	@echo "$(YELLOW)[CLEAN]$(NC) Removing build artifacts..."
	@rm -rf $(BIN_DIR)
	@rm -f *.o *.a
	@echo "$(GREEN)[OK]$(NC) Cleaned"
	@echo ""

# ============================================================================
# Print variables for debugging
# ============================================================================
print-%:
	@echo "$* = $($*)"
