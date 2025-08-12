import os
import numpy as np
import torch
from optimize_gamma import optimize_gamma
from simulation import simulate_ber
from plotting import plot_detection_results, print_performance_analysis
from env_loader import load_dotenv


def main():
    # Load env
    load_dotenv()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Wider SNR range
    snr_db_list = np.arange(0, 21, 2)  # 0..20 dB step 2
    num_trials = int(os.getenv('NUM_TRIALS', '1000'))

    # Optimizer config (tunable via env)
    grid_steps = int(os.getenv('GAMMA_GRID_STEPS', '11'))
    refine_rounds = int(os.getenv('GAMMA_REFINE_ROUNDS', '2'))

    # Find best gamma via grid search (ML-based)
    print("Optimizing gamma (ML-based grid search)...")
    val_snr = np.arange(0, 13, 3)
    trials_per_gamma = int(os.getenv('GAMMA_TRIALS', '200'))
    best_gamma, best_score = optimize_gamma(
        initial_grid_steps=grid_steps,
        refine_rounds=refine_rounds,
        snr_db_list=val_snr,
        trials_per_gamma=trials_per_gamma,
        device=device
    )
    print(f"Best gamma found: {best_gamma} (avg BER: {best_score:.4e} over {val_snr} dB)")

    gamma_opt = best_gamma
    gamma_std = 1.0 + 1.0j
    gamma_poor = 3.0 + 0.3j

    # Run with best gamma over full SNR range for each detector
    print("\nRunning ML Detection with optimized gamma...")
    ber_opt_ml, ber_std_ml = simulate_ber(gamma_opt, gamma_std, snr_db_list, 'ml', 2, num_trials, device)
    ber_poor_ml, _ = simulate_ber(gamma_poor, gamma_std, snr_db_list, 'ml', 2, num_trials//2, device)

    print("\nRunning MMSE Detection with optimized gamma...")
    ber_opt_mmse, ber_std_mmse = simulate_ber(gamma_opt, gamma_std, snr_db_list, 'mmse', 2, num_trials, device)
    ber_poor_mmse, _ = simulate_ber(gamma_poor, gamma_std, snr_db_list, 'mmse', 2, num_trials//2, device)

    print("\nRunning ZF Detection with optimized gamma...")
    ber_opt_zf, ber_std_zf = simulate_ber(gamma_opt, gamma_std, snr_db_list, 'zf', 2, num_trials, device)
    ber_poor_zf, _ = simulate_ber(gamma_poor, gamma_std, snr_db_list, 'zf', 2, num_trials//2, device)

    # Plots
    plot_detection_results(snr_db_list, ber_opt_ml, ber_std_ml, ber_poor_ml, gamma_opt, gamma_std, gamma_poor, 'ML', 'ml_detection_optgamma.png')
    plot_detection_results(snr_db_list, ber_opt_mmse, ber_std_mmse, ber_poor_mmse, gamma_opt, gamma_std, gamma_poor, 'MMSE', 'mmse_detection_optgamma.png')
    plot_detection_results(snr_db_list, ber_opt_zf, ber_std_zf, ber_poor_zf, gamma_opt, gamma_std, gamma_poor, 'ZF', 'zf_detection_optgamma.png')

    print_performance_analysis(snr_db_list, ber_opt_ml, ber_std_ml, ber_opt_mmse, ber_std_mmse, ber_opt_zf, ber_std_zf)


if __name__ == '__main__':
    main() 