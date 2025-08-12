import torch
import numpy as np
import time
import os
from pathlib import Path
import argparse
from simulation import simulate_ber_three
from plotting import plot_detection_results, save_performance_table_png
from optimize_gamma import optimize_gamma


def load_dotenv(env_path: str | None = None) -> None:
    """Load key=value pairs from a .env file into os.environ (non-destructive for existing keys)."""
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
        # Do not fail app startup if .env parsing fails
        pass 


def parse_args():
    parser = argparse.ArgumentParser(description="STBC Simulation: Golden vs Optimized gamma")
    parser.add_argument(
        "--gamma-mode",
        choices=["optimize", "golden"],
        default="optimize",
        help="Select gamma strategy: optimize (grid search) or golden (0.618+1.0j). Default: optimize",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print effective config from CLI/.env and exit")
    # SNR range
    parser.add_argument("--snr-start", type=int, default=int(os.getenv("SNR_START", 0)))
    parser.add_argument("--snr-end", type=int, default=int(os.getenv("SNR_END", 20)))
    parser.add_argument("--snr-step", type=int, default=int(os.getenv("SNR_STEP", 2)))
    # Trials
    parser.add_argument("--num-trials", type=int, default=int(os.getenv("NUM_TRIALS", 1000)))
    # Optimizer controls
    parser.add_argument("--gamma-trials", type=int, default=int(os.getenv("GAMMA_TRIALS", 200)))
    parser.add_argument("--gamma-grid-steps", type=int, default=int(os.getenv("GAMMA_GRID_STEPS", 11)))
    parser.add_argument("--gamma-refine-rounds", type=int, default=int(os.getenv("GAMMA_REFINE_ROUNDS", 2)))
    parser.add_argument("--opt-snr-step", type=int, default=int(os.getenv("OPT_SNR_STEP", 3)), help="Step to subsample the run SNR range for optimizer validation grid")
    return parser.parse_args()


def main():
    # Load environment variables from .env if present
    load_dotenv()
    args = parse_args()

    # --- Device selection: prefer CUDA (e.g., RTX 3060), then MPS, else CPU ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA GPU ({torch.cuda.get_device_name(0)}) for acceleration.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using M1/M2 GPU (MPS) for acceleration.")
    else:
        device = torch.device("cpu")
        print("No GPU backend available. Running on CPU (will be slower).")

    # --- Simulation Parameters ---
    snr_db_list = np.arange(args.snr_start, args.snr_end + 1, args.snr_step)
    num_trials = args.num_trials

    # Echo effective configuration
    print("Config:")
    print(f"  gamma_mode={args.gamma_mode}")
    print(f"  snr_start={args.snr_start} snr_end={args.snr_end} snr_step={args.snr_step}")
    print(f"  num_trials={args.num_trials}")
    print(f"  gamma_trials={args.gamma_trials} gamma_grid_steps={args.gamma_grid_steps} gamma_refine_rounds={args.gamma_refine_rounds}")
    print(f"  opt_snr_step={args.opt_snr_step}")
    if args.dry_run:
        return

    start_time = time.time()

    if args.gamma_mode == "optimize":
        print("Optimizing gamma (ML-based grid search)...")
        val_snr = np.arange(args.snr_start, args.snr_end + 1, args.opt_snr_step)
        gamma_opt, best_score = optimize_gamma(
            initial_grid_steps=args.gamma_grid_steps,
            refine_rounds=args.gamma_refine_rounds,
            snr_db_list=val_snr,
            trials_per_gamma=args.gamma_trials,
            device=device,
        )
        print(f"Best gamma found: {gamma_opt} (avg BER: {best_score:.4e} over {val_snr} dB)")
    else:
        gamma_opt = 0.618 + 1.0j  # Golden ratio-based (proven good)
        print(f"Using golden-ratio gamma: {gamma_opt:.3f}")

    gamma_std = 1.0 + 1.0j        # Standard reference
    gamma_poor = 3.0 + 0.3j       # Known poor (but not extreme)

    print(f"\nUsing gamma values:")
    print(f"  Selected (mode={args.gamma_mode}): {gamma_opt:.3f}")
    print(f"  Standard:  {gamma_std:.3f}")
    print(f"  Poor:      {gamma_poor:.3f}")

    # Run Simulations with common random numbers across all three gammas
    print("\nRunning ML Detection...")
    ber_opt_ml, ber_std_ml, ber_poor_ml = simulate_ber_three(gamma_opt, gamma_std, gamma_poor, snr_db_list, 'ml', 2, num_trials, device)

    print("\nRunning MMSE Detection...")
    ber_opt_mmse, ber_std_mmse, ber_poor_mmse = simulate_ber_three(gamma_opt, gamma_std, gamma_poor, snr_db_list, 'mmse', 2, num_trials, device)

    print("\nRunning ZF Detection...")
    ber_opt_zf, ber_std_zf, ber_poor_zf = simulate_ber_three(gamma_opt, gamma_std, gamma_poor, snr_db_list, 'zf', 2, num_trials, device)

    end_time = time.time()
    print(f"\nAll simulations completed in {end_time - start_time:.2f} seconds.")

    # Generate Plots
    plot_detection_results(snr_db_list, ber_opt_ml, ber_std_ml, ber_poor_ml, gamma_opt, gamma_std, gamma_poor,
                          'ML', 'ml_detection.png')

    plot_detection_results(snr_db_list, ber_opt_mmse, ber_std_mmse, ber_poor_mmse, gamma_opt, gamma_std, gamma_poor,
                          'MMSE', 'mmse_detection.png')

    plot_detection_results(snr_db_list, ber_opt_zf, ber_std_zf, ber_poor_zf, gamma_opt, gamma_std, gamma_poor,
                          'ZF', 'zf_detection.png')

    # Save performance table as PNG instead of CLI prints
    save_performance_table_png(
        snr_db_list,
        ber_opt_ml, ber_std_ml,
        ber_opt_mmse, ber_std_mmse,
        ber_opt_zf, ber_std_zf,
        filename='performance_table.png'
    )


if __name__ == "__main__":
    main()
