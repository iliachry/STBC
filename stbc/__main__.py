"""
Main entry point for STBC simulations.
"""

import os
import time
import argparse
import numpy as np
import torch
from pathlib import Path

from stbc.core.biquaternion import BiquaternionSTBC
from stbc.core.codewords import get_cached_codewords, clear_codeword_cache
from stbc.simulation.simulator import simulate_ber_all_detectors
from stbc.optimization.gamma_optimizer import optimize_gamma
from stbc.utils.device_utils import select_device
from stbc.utils.results import create_results_directory, save_results_to_csv
from stbc.visualization.plotting import plot_detection_results, plot_all_detectors_comparison
from stbc.visualization.tables import save_performance_table_png, save_all_detectors_table_png

def load_dotenv(env_path: str | None = None) -> None:
    path = Path(env_path) if env_path else Path.cwd() / ".env"
    if not path.exists():
        return
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                os.environ.setdefault(key, value)
    except Exception:
        pass

def parse_args():
    parser = argparse.ArgumentParser(description="STBC Simulation: Fixed vs Optimized gamma")
    parser.add_argument("--gamma-mode", choices=["optimize", "fixed"], default="optimize")
    parser.add_argument("--dry-run", action="store_true", help="Print effective config and exit")
    parser.add_argument("--snr-start", type=int, default=int(os.getenv("SNR_START", 0)))
    parser.add_argument("--snr-end", type=int, default=int(os.getenv("SNR_END", 20)))
    parser.add_argument("--snr-step", type=int, default=int(os.getenv("SNR_STEP", 2)))
    parser.add_argument("--num-trials", type=int, default=int(os.getenv("NUM_TRIALS", 1000)))
    parser.add_argument("--gamma-grid-steps", type=int, default=int(os.getenv("GAMMA_GRID_STEPS", 11)))
    parser.add_argument("--gamma-refine-rounds", type=int, default=int(os.getenv("GAMMA_REFINE_ROUNDS", 2)))
    parser.add_argument("--opt-snr-step", type=int, default=int(os.getenv("OPT_SNR_STEP", 3)))
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallel gamma evaluation")
    # Fixed-gamma options
    parser.add_argument("--gamma-preset", choices=["golden", "minusj"], default="golden", help="Preset gamma used when --gamma-mode=fixed")
    parser.add_argument("--gamma-fixed", type=str, default=os.getenv("GAMMA_FIXED", ""), help="Fixed gamma as a Python complex, e.g., 0.5+1j or -1j")
    parser.add_argument("--gamma-fixed-real", type=float, default=None, help="Fixed gamma real part (alternative to --gamma-fixed)")
    parser.add_argument("--gamma-fixed-imag", type=float, default=None, help="Fixed gamma imaginary part (alternative to --gamma-fixed)")
    parser.add_argument("--rate", type=int, default=2, choices=[1, 2], help="Code rate")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"], 
                        help="Computation device (default: auto-select)")
    return parser.parse_args()

def main():
    """Main function for running STBC simulations."""
    load_dotenv()
    args = parse_args()

    # Set device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA GPU ({torch.cuda.get_device_name(0)}) for acceleration.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using M1/M2 GPU (MPS) for acceleration.")
    else:
        device = torch.device("cpu")
        print("No GPU backend available. Running on CPU (will be slower).")
        
    # Generate SNR range
    snr_db_list = np.arange(args.snr_start, args.snr_end + 1, args.snr_step)
    num_trials = args.num_trials

    print("Config:")
    print(f"  gamma_mode={args.gamma_mode}")
    print(f"  snr_start={args.snr_start} snr_end={args.snr_end} snr_step={args.snr_step}")
    print(f"  num_trials={args.num_trials}")
    print(f"  gamma_grid_steps={args.gamma_grid_steps} gamma_refine_rounds={args.gamma_refine_rounds}")
    print(f"  parallel_enabled={not args.no_parallel}")
    if args.gamma_mode == "fixed":
        print(f"  gamma_preset={args.gamma_preset}")
        if args.gamma_fixed:
            print(f"  gamma_fixed={args.gamma_fixed}")
        if args.gamma_fixed_real is not None or args.gamma_fixed_imag is not None:
            print(f"  gamma_fixed_real={args.gamma_fixed_real} gamma_fixed_imag={args.gamma_fixed_imag}")
    if args.dry_run:
        return
        
    start_time = time.time()
    
    # Determine gamma values
    if args.gamma_mode == "optimize":
        print("Optimizing gamma parameter...")
        gamma_opt, best_score = optimize_gamma(initial_grid_steps=args.gamma_grid_steps,
                                              refine_rounds=args.gamma_refine_rounds,
                                              device=device,
                                              use_parallel=not args.no_parallel)
        print(f"Final optimized gamma: {gamma_opt} (min |det|^2: {best_score:.3e})")
        # Define standard and poor gamma for comparison
        gamma_std = 1.0 + 1.0j
        gamma_poor = 3.0 + 0.3j
    else:
        # Fixed gamma selection logic
        gamma_opt = None
        # Priority 1: explicit complex string
        if args.gamma_fixed:
            try:
                gamma_opt = complex(args.gamma_fixed)
                print(f"Using fixed gamma from --gamma-fixed: {gamma_opt}")
            except Exception:
                print(f"Warning: could not parse --gamma-fixed='{args.gamma_fixed}', falling back to other options")
        # Priority 2: real/imag parts
        if gamma_opt is None and (args.gamma_fixed_real is not None or args.gamma_fixed_imag is not None):
            real = args.gamma_fixed_real if args.gamma_fixed_real is not None else 0.0
            imag = args.gamma_fixed_imag if args.gamma_fixed_imag is not None else 0.0
            gamma_opt = complex(real, imag)
            print(f"Using fixed gamma from real/imag: {gamma_opt}")
        # Priority 3: presets
        if gamma_opt is None:
            if args.gamma_preset == "minusj":
                gamma_opt = -1j
                print(f"Using preset gamma: {gamma_opt}")
            else:
                gamma_opt = 0.618 + 1.0j
                print(f"Using preset gamma (golden): {gamma_opt:.3f}")
        
        # Define standard and poor gamma
        gamma_std = 1.0 + 1.0j
        gamma_poor = 3.0 + 0.3j
    
    print(f"\nRunning simulations with {num_trials} trials")
    gammas = [gamma_opt, gamma_std, gamma_poor]
    
    # Run simulation for all detectors
    results = simulate_ber_all_detectors(
        gammas, 
        snr_db_list,
        rate=args.rate,
        num_trials=num_trials,
        device=device
    )
    
    # Unpack results
    results_ml, results_mmse, results_zf, results_zf_reg, all_results_dict = results
    ber_opt_ml, ber_std_ml, ber_poor_ml = results_ml
    ber_opt_mmse, ber_std_mmse, ber_poor_mmse = results_mmse
    ber_opt_zf, ber_std_zf, ber_poor_zf = results_zf
    ber_opt_zf_reg, ber_std_zf_reg, ber_poor_zf_reg = results_zf_reg

    # Create results directory with timestamp
    results_dir = create_results_directory()
    
    # Generate individual detector plots for standard detectors
    plot_detection_results(snr_db_list, ber_opt_ml, ber_std_ml, ber_poor_ml, gamma_opt, gamma_std, gamma_poor, 'ML', 'ml_detection.png', results_dir)
    plot_detection_results(snr_db_list, ber_opt_mmse, ber_std_mmse, ber_poor_mmse, gamma_opt, gamma_std, gamma_poor, 'MMSE', 'mmse_detection.png', results_dir)
    plot_detection_results(snr_db_list, ber_opt_zf, ber_std_zf, ber_poor_zf, gamma_opt, gamma_std, gamma_poor, 'ZF', 'zf_detection.png', results_dir)
    plot_detection_results(snr_db_list, ber_opt_zf_reg, ber_std_zf_reg, ber_poor_zf_reg, gamma_opt, gamma_std, gamma_poor, 'ZF-Regularized', 'zf_reg_detection.png', results_dir)
    
    # Generate plots for enhanced detectors if available
    if len(results) == 5 and all_results_dict:
        # ML-Enhanced ZF
        if 'ml_zf' in all_results_dict:
            ber_opt_ml_zf = np.array(all_results_dict['ml_zf'][0])
            ber_std_ml_zf = np.array(all_results_dict['ml_zf'][1])
            ber_poor_ml_zf = np.array(all_results_dict['ml_zf'][2])
            plot_detection_results(snr_db_list, ber_opt_ml_zf, ber_std_ml_zf, ber_poor_ml_zf, gamma_opt, gamma_std, gamma_poor, 'ML-Enhanced ZF', 'ml_zf_detection.png', results_dir)
        
        # Adaptive MMSE
        if 'adaptive_mmse' in all_results_dict:
            ber_opt_adaptive = np.array(all_results_dict['adaptive_mmse'][0])
            ber_std_adaptive = np.array(all_results_dict['adaptive_mmse'][1])
            ber_poor_adaptive = np.array(all_results_dict['adaptive_mmse'][2])
            plot_detection_results(snr_db_list, ber_opt_adaptive, ber_std_adaptive, ber_poor_adaptive, gamma_opt, gamma_std, gamma_poor, 'Adaptive MMSE', 'adaptive_mmse_detection.png', results_dir)
        
        # Hybrid
        if 'hybrid' in all_results_dict:
            ber_opt_hybrid = np.array(all_results_dict['hybrid'][0])
            ber_std_hybrid = np.array(all_results_dict['hybrid'][1])
            ber_poor_hybrid = np.array(all_results_dict['hybrid'][2])
            plot_detection_results(snr_db_list, ber_opt_hybrid, ber_std_hybrid, ber_poor_hybrid, gamma_opt, gamma_std, gamma_poor, 'Hybrid', 'hybrid_detection.png', results_dir)

    # Generate comparison plot with all detectors
    if len(results) == 5 and all_results_dict:
        plot_all_detectors_comparison(snr_db_list, all_results_dict, gamma_opt, gamma_std, gamma_poor, 'all_detectors_comparison.png', results_dir)
        
        # Export results to CSV
        save_results_to_csv(results_dir, snr_db_list, all_results_dict, gamma_opt, gamma_std, gamma_poor)
    else:
        # Fallback for old format - create dict from individual results
        all_results_dict = {
            'ml': [ber_opt_ml, ber_std_ml, ber_poor_ml],
            'mmse': [ber_opt_mmse, ber_std_mmse, ber_poor_mmse],
            'zf': [ber_opt_zf, ber_std_zf, ber_poor_zf],
            'zf_reg': [ber_opt_zf_reg, ber_std_zf_reg, ber_poor_zf_reg]
        }
        plot_all_detectors_comparison(snr_db_list, all_results_dict, gamma_opt, gamma_std, gamma_poor, 'all_detectors_comparison.png', results_dir)
        
        # Export results to CSV
        save_results_to_csv(results_dir, snr_db_list, all_results_dict, gamma_opt, gamma_std, gamma_poor)

    # Generate performance table
    save_performance_table_png(snr_db_list, ber_opt_ml, ber_std_ml, ber_opt_mmse, ber_std_mmse, 
                              ber_opt_zf, ber_std_zf, ber_opt_zf_reg, ber_std_zf_reg, 
                              filename='performance_table.png', results_dir=results_dir)
    
    # Generate comprehensive table with all detectors
    if len(results) == 5 and all_results_dict:
        save_all_detectors_table_png(snr_db_list, all_results_dict, gamma_opt, 
                                    filename='all_detectors_table.png', results_dir=results_dir)
                                    
    # Save simulation parameters
    with open(results_dir / "simulation_parameters.txt", 'w') as f:
        f.write(f"Simulation Parameters:\n")
        f.write(f"SNR range: {min(snr_db_list)} to {max(snr_db_list)} dB\n")
        f.write(f"Number of trials: {args.num_trials}\n")
        f.write(f"Optimized gamma: {gamma_opt}\n")
        f.write(f"Standard gamma: {gamma_std}\n")
        f.write(f"Poor gamma: {gamma_poor}\n")
        f.write(f"Detectors: {', '.join(all_results_dict.keys())}\n")
        
    print(f"\nAll results saved to: {results_dir}")

    # Optional: Clear cache at end to free memory
    clear_codeword_cache()
    
    # Print total execution time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Total execution time: {int(hours):02}:{int(minutes):02}:{seconds:05.2f}")

if __name__ == "__main__":
    main()
