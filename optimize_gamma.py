import numpy as np
import cmath
import torch
from simulation import simulate_ber_for_gamma
import os


def complex_grid(r_ranges, i_ranges, steps):
    r_min, r_max = r_ranges
    i_min, i_max = i_ranges
    for r in np.linspace(r_min, r_max, steps):
        for im in np.linspace(i_min, i_max, steps):
            yield complex(float(r), float(im))


def evaluate_gamma(gamma, snr_db_list, num_trials, device):
    ber = simulate_ber_for_gamma(gamma, snr_db_list, detector='ml', rate=2, num_trials=num_trials, device=device)
    return np.mean(ber), ber


def optimize_gamma(initial_grid_steps=11, refine_rounds=2, refine_factor=0.4, snr_db_list=None, trials_per_gamma=400, device=None):
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if snr_db_list is None:
        snr_db_list = np.arange(2, 9, 2)  # validation SNRs: 2,4,6,8 dB (if extended)

    print(f"[GammaOpt] SNR validation grid: {snr_db_list}")

    # Coarse search region
    r_bounds = (-1.0, 2.0)
    i_bounds = (0.0, 2.0)

    best_gamma = None
    best_score = float('inf')

    for round_idx in range(refine_rounds + 1):
        steps = initial_grid_steps if round_idx == 0 else 7
        total_candidates = steps * steps
        print(f"[GammaOpt] Round {round_idx+1}/{refine_rounds+1}: bounds R{r_bounds} I{i_bounds}, steps={steps} ({total_candidates} candidates)")

        candidate_scores = []
        for idx, gamma in enumerate(complex_grid(r_bounds, i_bounds, steps), start=1):
            if idx % max(1, total_candidates // 10) == 0:
                print(f"  [GammaOpt] Candidate {idx}/{total_candidates}...")
            score, _ = evaluate_gamma(gamma, snr_db_list, num_trials=trials_per_gamma, device=device)
            candidate_scores.append((score, gamma))
        candidate_scores.sort(key=lambda x: x[0])
        best_score, best_gamma = candidate_scores[0]
        print(f"[GammaOpt] Best so far: gamma={best_gamma:.4f}, avg BER={best_score:.6e}")

        # refine bounds around best
        r_center, i_center = best_gamma.real, best_gamma.imag
        r_span = (r_bounds[1] - r_bounds[0]) * 0.4
        i_span = (i_bounds[1] - i_bounds[0]) * 0.4
        r_bounds = (r_center - r_span/2, r_center + r_span/2)
        i_bounds = (i_center - i_span/2, i_center + i_span/2)

    return best_gamma, best_score 