import torch
import numpy as np
import time
from simulation import simulate_ber
from plotting import plot_detection_results, print_performance_analysis

def main():
    # --- Setup for M1 GPU (MPS) ---
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using M1 GPU (MPS) for acceleration.")
    else:
        device = torch.device("cpu")
        print("MPS not available. Running on CPU (will be slower).")

    # --- Simulation Parameters ---
    snr_db_list = np.arange(0, 8, 1)  # 0-7 dB range
    num_trials = 2000  # Higher trials with variance reduction
    
    start_time = time.time()
    
    # Use literature-proven gamma values
    gamma_opt = 0.618 + 1.0j      # Golden ratio-based (proven good)
    gamma_std = 1.0 + 1.0j        # Standard reference  
    gamma_poor = 3.0 + 0.3j       # Known poor (but not extreme)
    
    print(f"\nUsing LITERATURE-PROVEN γ values:")
    print(f"  Optimized: {gamma_opt:.3f} (Golden ratio-based)")
    print(f"  Standard:  {gamma_std:.3f} (Literature baseline)") 
    print(f"  Poor:      {gamma_poor:.3f} (Conservative poor)")
    
    # Run Simulations
    print("\nRunning ML Detection...")
    ber_opt_ml, ber_std_ml = simulate_ber(gamma_opt, gamma_std, snr_db_list, 'ml', 2, num_trials, device)
    ber_poor_ml, _ = simulate_ber(gamma_poor, gamma_std, snr_db_list, 'ml', 2, num_trials//2, device)
    
    print("\nRunning MMSE Detection...")
    ber_opt_mmse, ber_std_mmse = simulate_ber(gamma_opt, gamma_std, snr_db_list, 'mmse', 2, num_trials, device)
    ber_poor_mmse, _ = simulate_ber(gamma_poor, gamma_std, snr_db_list, 'mmse', 2, num_trials//2, device)
    
    print("\nRunning ZF Detection...")
    ber_opt_zf, ber_std_zf = simulate_ber(gamma_opt, gamma_std, snr_db_list, 'zf', 2, num_trials, device)
    ber_poor_zf, _ = simulate_ber(gamma_poor, gamma_std, snr_db_list, 'zf', 2, num_trials//2, device)
    
    end_time = time.time()
    print(f"\nAll simulations completed in {end_time - start_time:.2f} seconds.")
    
    # Generate Plots
    plot_detection_results(snr_db_list, ber_opt_ml, ber_std_ml, ber_poor_ml, gamma_opt, gamma_std, gamma_poor, 
                          'ML', 'ml_detection.png')
    
    plot_detection_results(snr_db_list, ber_opt_mmse, ber_std_mmse, ber_poor_mmse, gamma_opt, gamma_std, gamma_poor,
                          'MMSE', 'mmse_detection.png')
    
    plot_detection_results(snr_db_list, ber_opt_zf, ber_std_zf, ber_poor_zf, gamma_opt, gamma_std, gamma_poor,
                          'ZF', 'zf_detection.png')
    
    # Performance Analysis
    print_performance_analysis(snr_db_list, ber_opt_ml, ber_std_ml, ber_opt_mmse, ber_std_mmse, ber_opt_zf, ber_std_zf)

if __name__ == "__main__":
    main()
