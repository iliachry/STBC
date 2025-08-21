#!/usr/bin/env python3
"""
Extract key statistics from simulation results for the paper
"""

import pandas as pd
import numpy as np

# Read the performance summary
perf_df = pd.read_csv('results/20250821_130005/detector_performance_summary.csv')

# Read BER data for specific SNR points
opt_df = pd.read_csv('results/20250821_130005/ber_vs_snr_optimized_gamma.csv', skiprows=1)

print("=== Key Statistics for Paper ===\n")

# Performance ratios compared to ML
print("Performance Degradation Ratios (compared to ML):")
for _, row in perf_df.iterrows():
    detector = row['Detector']
    ratio = row['Ratio to ML']
    avg_ber = row['Avg BER']
    print(f"  {detector}: {ratio:.2f}× (Avg BER: {avg_ber:.4f})")

print("\n=== BER at Key SNR Points (Optimized γ = -i) ===\n")

# Get BER at specific SNR points
snr_points = [10, 12, 14]
for snr in snr_points:
    row = opt_df[opt_df['SNR (dB)'] == snr].iloc[0]
    print(f"SNR = {snr} dB:")
    print(f"  ML: {row['ML_BER']:.6f}")
    print(f"  MMSE: {row['MMSE_BER']:.6f}")
    print(f"  ZF: {row['ZF_BER']:.6f}")
    print(f"  ZF_REG: {row['ZF_REG_BER']:.6f}")
    print(f"  ADAPTIVE_MMSE: {row['ADAPTIVE_MMSE_BER']:.6f}")
    print(f"  HYBRID: {row['HYBRID_BER']:.6f}")
    print()

# Timing information
print("=== Average Detection Times ===\n")
for _, row in perf_df.iterrows():
    detector = row['Detector']
    avg_time = row['Avg Time(s)']
    print(f"  {detector}: {avg_time:.3f}s")

# Find where ML achieves error-free transmission
ml_ber_col = opt_df['ML_BER']
zero_ber_snr = opt_df[ml_ber_col == 0]['SNR (dB)'].min() if any(ml_ber_col == 0) else None
if zero_ber_snr is not None:
    print(f"\n=== ML Error-Free Transmission ===")
    print(f"ML achieves error-free transmission at SNR ≥ {zero_ber_snr} dB")

print("\n=== Simulation Parameters ===")
print("Number of trials: 1000")
print("SNR range: 0-20 dB with 2 dB steps")
print("Optimized γ: -i")
print("Standard γ: 1+i") 
print("Poor γ: 3+0.3i")