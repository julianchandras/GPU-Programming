#!/usr/bin/env python3

import struct
import random
import os
import sys

def generate_dataset(filename, n, seed=42):
    """Generate a random float dataset and write to binary file."""
    random.seed(seed)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'wb') as f:
        # Write dataset size as int32
        f.write(struct.pack('i', n))
        
        # Write n random float values (uniform [0, 1))
        for _ in range(n):
            val = random.uniform(0.0, 1.0)
            f.write(struct.pack('f', val))
    
    print(f"Generated dataset: {filename} (n={n})")

def main():
    base_dir = "datasets"
    os.makedirs(base_dir, exist_ok=True)
    
    # Dataset sizes: 10^3, 10^5, 10^6, 10^8
    sizes = [
        (1000, "1k"),
        (100000, "100k"),
        (1000000, "1m"),
        (100000000, "100m"),
    ]
    
    for size, label in sizes:
        filepath = os.path.join(base_dir, f"data_{label}.bin")
        generate_dataset(filepath, size, seed=42)

if __name__ == "__main__":
    main()
