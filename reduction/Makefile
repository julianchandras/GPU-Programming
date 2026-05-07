NVCC = /usr/bin/nvcc
NVCC_FLAGS = -O3 -std=c++14 -arch=sm_86 -m64
GCC = gcc
GCC_FLAGS = -O3 -lm -std=c99

# Directories
BIN_DIR = bin
DATASETS_DIR = datasets

# Targets
BENCHMARK = $(BIN_DIR)/benchmark

# All targets
all: $(BENCHMARK) datasets

# Main benchmark executable
$(BENCHMARK): benchmark.cu sum.cu sum.c
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LDFLAGS)
	@echo "Build successful: $@"

# Generate datasets
datasets:
	@python3 generate_dataset.py

# Run benchmark (requires datasets)
run: $(BENCHMARK) datasets
	cd $(BIN_DIR) && ./benchmark

# View results
view_results:
	@if [ -f bin/results.csv ]; then \
		echo "=== Results Summary ==="; \
		head -20 bin/results.csv; \
	else \
		echo "No results found. Run 'make run' first."; \
	fi

# Generate plots
plot: bin/results.csv
	python3 analyze_results.py

# Clean build artifacts
clean:
	rm -f $(BENCHMARK)
	@echo "Cleaned build artifacts"

# Clean everything (including datasets and results)
clean_all: clean
	rm -rf $(DATASETS_DIR)/*.bin bin/results.csv bin/*.png
	@echo "Cleaned all data and results"

.PHONY: all run clean clean_all datasets view_results plot
