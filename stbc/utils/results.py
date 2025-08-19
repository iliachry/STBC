"""
Result management utilities.
"""

import csv
import datetime
import numpy as np
from pathlib import Path

def create_results_directory():
    """
    Create a timestamped results directory.
    
    Returns:
        Path: Path to the created directory
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("results") / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created results directory: {results_dir}")
    return results_dir

def save_results_to_csv(results_dir, snr_db_list, all_results_dict, gamma_opt, gamma_std, gamma_poor):
    """
    Save simulation results to CSV files.
    
    Args:
        results_dir: Directory to save results
        snr_db_list: List of SNR values
        all_results_dict: Dictionary of results
        gamma_opt: Optimized gamma value
        gamma_std: Standard gamma value
        gamma_poor: Poor gamma value
    """
    # Save BER vs SNR for optimized gamma
    csv_file = results_dir / "ber_vs_snr_optimized_gamma.csv"
    
    # Prepare headers
    detectors = list(all_results_dict.keys())
    headers = ["SNR (dB)"] + [f"{det.upper()}" for det in detectors]
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Optimized Gamma", f"{gamma_opt}"])
        writer.writerow(headers)
        
        # Write data for each SNR point
        for i, snr in enumerate(snr_db_list):
            row = [snr]
            for det in detectors:
                row.append(all_results_dict[det][0][i])  # First gamma (optimized)
            writer.writerow(row)
    
    print(f"Saved BER vs SNR data to {csv_file}")
    
    # Save BER vs SNR for standard gamma
    csv_file = results_dir / "ber_vs_snr_standard_gamma.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Standard Gamma", f"{gamma_std}"])
        writer.writerow(headers)
        
        # Write data for each SNR point
        for i, snr in enumerate(snr_db_list):
            row = [snr]
            for det in detectors:
                row.append(all_results_dict[det][1][i])  # Second gamma (standard)
            writer.writerow(row)
    
    print(f"Saved BER vs SNR data to {csv_file}")
    
    # Save BER vs SNR for poor gamma
    csv_file = results_dir / "ber_vs_snr_poor_gamma.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Poor Gamma", f"{gamma_poor}"])
        writer.writerow(headers)
        
        # Write data for each SNR point
        for i, snr in enumerate(snr_db_list):
            row = [snr]
            for det in detectors:
                row.append(all_results_dict[det][2][i])  # Third gamma (poor)
            writer.writerow(row)
    
    print(f"Saved BER vs SNR data to {csv_file}")
    
    # Save summary CSV with detector performance comparisons
    csv_file = results_dir / "detector_performance_summary.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Detector", "Avg BER", "Ratio to ML", "Best SNR Point", "Best BER"])
        
        # Get ML average for reference
        ml_data = all_results_dict['ml'][0]  # First gamma (optimized)
        ml_avg = np.mean([val for val in ml_data if val > 0])
        
        for det in detectors:
            det_data = all_results_dict[det][0]  # First gamma (optimized)
            det_avg = np.mean([val for val in det_data if val > 0])
            ratio = det_avg / ml_avg if ml_avg > 0 else 1.0
            
            # Find best SNR point
            det_data_array = np.array(det_data)
            best_idx = np.argmin(det_data_array)
            best_snr = snr_db_list[best_idx]
            best_ber = det_data[best_idx]
            
            writer.writerow([det.upper(), det_avg, ratio, best_snr, best_ber])
    
    print(f"Saved detector performance summary to {csv_file}")
