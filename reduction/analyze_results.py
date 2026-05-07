#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def load_results(csv_file):
    """Load benchmark results from CSV file."""
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. Run 'make run' first.")
        sys.exit(1)
    
    df = pd.read_csv(csv_file)
    return df

def plot_execution_time(df, output_file="execution_time.png"):
    """
    Plot 1: Execution time (CPU vs GPU) vs problem size.
    X-axis: n (log scale), Y-axis: time in ms (log scale)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.loglog(df['n'], df['T_CPU_ms'], 'o-', linewidth=2, markersize=8, label='CPU (Serial)', color='#d62728')
    ax.loglog(df['n'], df['T_Total_GPU_ms'], 's-', linewidth=2, markersize=8, label='GPU (Total)', color='#1f77b4')
    ax.loglog(df['n'], df['T_Kernel_ms'], '^--', linewidth=1.5, markersize=7, label='GPU (Kernel only)', color='#2ca02c', alpha=0.7)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax.set_xlabel('Problem Size (n)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Execution Time: CPU vs GPU Reduction', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def plot_speedup(df, output_file="speedup.png"):
    """
    Plot 2: Speedup (CPU time / GPU total time) vs problem size.
    X-axis: n (log scale), Y-axis: speedup (linear scale)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.semilogx(df['n'], df['Speedup'], 'o-', linewidth=2.5, markersize=10, color='#ff7f0e')
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='No speedup (speedup=1x)')
    
    ax.set_xlabel('Problem Size (n)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup (T_CPU / T_GPU)', fontsize=12, fontweight='bold')
    ax.set_title('GPU Speedup vs Problem Size', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def plot_memory_breakdown(df, output_file="memory_breakdown.png"):
    """
    Plot 3: GPU time breakdown (H2D, Kernel, D2H) as stacked bars.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(df))
    width = 0.6
    
    # Get data sizes as labels
    labels = [f"$10^{{{int(np.log10(n))}}}$" for n in df['n']]
    
    h2d = df['T_H2D_ms'].values
    kernel = df['T_Kernel_ms'].values
    d2h = df['T_D2H_ms'].values
    
    # Calculate percentages
    total = h2d + kernel + d2h
    h2d_pct = (h2d / total) * 100
    kernel_pct = (kernel / total) * 100
    d2h_pct = (d2h / total) * 100
    
    # Create stacked bars
    ax.bar(x, h2d, width, label='H2D Transfer', color='#1f77b4', alpha=0.8)
    ax.bar(x, kernel, width, bottom=h2d, label='Kernel', color='#2ca02c', alpha=0.8)
    ax.bar(x, d2h, width, bottom=h2d+kernel, label='D2H Transfer', color='#ff7f0e', alpha=0.8)
    
    max_total = float(total.max()) if len(total) > 0 else 0.0

    # Use one compact callout above each bar so the breakdown stays readable.
    # This avoids text collision inside narrow stacked segments.
    for i in range(len(x)):
        breakdown_text = (
            f"H2D {h2d_pct[i]:.1f}% | "
            f"K {kernel_pct[i]:.1f}% | "
            f"D2H {d2h_pct[i]:.1f}%"
        )
        ax.text(
            i,
            total[i] + max_total * 0.03,
            breakdown_text,
            ha='center',
            va='bottom',
            fontsize=7.5,
            color='black',
            bbox=dict(boxstyle='round,pad=0.22', facecolor='white', edgecolor='#999999', alpha=0.92),
            zorder=10,
            clip_on=False,
        )
    
    ax.set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Problem Size (n)', fontsize=12, fontweight='bold')
    ax.set_title('GPU Execution Time Breakdown', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max_total * 1.22 if max_total > 0 else 1.0)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def print_analysis(df):
    """Print summary statistics and analysis."""
    print("\n=== ANALYSIS SUMMARY ===\n")
    
    # Find breakeven point
    gpu_faster = df[df['Speedup'] > 1.0]
    if len(gpu_faster) > 0:
        min_speedup_row = gpu_faster[gpu_faster['Speedup'] == gpu_faster['Speedup'].min()].iloc[0]
        print(f"GPU becomes faster at n={min_speedup_row['n']:.0e} (speedup={min_speedup_row['Speedup']:.2f}x)")
    
    # Max speedup
    max_speedup_idx = df['Speedup'].idxmax()
    print(f"Maximum speedup: {df.loc[max_speedup_idx, 'Speedup']:.2f}x at n={df.loc[max_speedup_idx, 'n']:.0e}")
    
    # Average memory overhead percentage
    df['Memory_Overhead_Pct'] = ((df['T_H2D_ms'] + df['T_D2H_ms']) / df['T_Total_GPU_ms']) * 100
    print(f"Average memory transfer overhead: {df['Memory_Overhead_Pct'].mean():.1f}%")
    
    # Verification
    df['Within_Tolerance'] = df['Error_rel'] < 1e-3
    if df['Within_Tolerance'].all():
        print(f"✓ All GPU results within tolerance (relative error < 1e-3)")
    else:
        print(f"✗ Some GPU results exceed tolerance")
    
    print(f"\nDetailed timing statistics (ms):")
    print(df[['n', 'T_CPU_ms', 'T_Kernel_ms', 'T_Total_GPU_ms', 'Speedup']].to_string(index=False))

def main():
    # Try both bin/ and current directory
    csv_files = ["../bin/results.csv", "bin/results.csv", "results.csv"]
    csv_file = None
    for f in csv_files:
        if os.path.exists(f):
            csv_file = f
            break
    
    if csv_file is None:
        print("Error: results.csv not found in bin/, ./, or ../ directories")
        sys.exit(1)
    
    print(f"Loading results from: {csv_file}")
    df = load_results(csv_file)
    
    # Generate plots
    output_dir = os.path.dirname(csv_file) or "."
    plot_execution_time(df, os.path.join(output_dir, "execution_time.png"))
    plot_speedup(df, os.path.join(output_dir, "speedup.png"))
    plot_memory_breakdown(df, os.path.join(output_dir, "memory_breakdown.png"))
    
    # Print analysis
    print_analysis(df)
    
    print(f"\nPlots saved in: {os.path.abspath(output_dir)}/")

if __name__ == "__main__":
    main()
