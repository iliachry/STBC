import numpy as np
import cmath
import torch
from biquaternion import BiquaternionSTBC, generate_all_codewords_biquaternion


def complex_grid(r_ranges, i_ranges, steps):
    r_min, r_max = r_ranges
    i_min, i_max = i_ranges
    for r in np.linspace(r_min, r_max, steps):
        for im in np.linspace(i_min, i_max, steps):
            yield complex(float(r), float(im))


def _min_det_squared_for_gamma(gamma: complex, rate: int, device: torch.device) -> float:
    stbc = BiquaternionSTBC(gamma, device)
    X, _bits = generate_all_codewords_biquaternion(stbc, rate=rate)
    diffs = X[:, None, :, :] - X[None, :, :, :]
    dets = torch.linalg.det(diffs)
    det_pow2 = torch.abs(dets) ** 2
    N = det_pow2.shape[0]
    tri_mask = torch.triu(torch.ones((N, N), dtype=torch.bool, device=device), diagonal=1)
    vals = det_pow2[tri_mask]
    eps = 1e-12
    vals = vals[vals > eps]
    if vals.numel() == 0:
        return 0.0
    return float(torch.min(vals).item())


def optimize_gamma(initial_grid_steps=11, refine_rounds=2, refine_factor=0.4, snr_db_list=None, trials_per_gamma=400, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    r_bounds = (-1.0, 2.0)
    i_bounds = (0.0, 2.0)

    best_gamma = None
    best_score = -float('inf')
    min_abs_gamma = 1e-3

    for round_idx in range(refine_rounds + 1):
        steps = initial_grid_steps if round_idx == 0 else 7
        candidate_scores = []
        for gamma in complex_grid(r_bounds, i_bounds, steps):
            if abs(gamma) < min_abs_gamma:
                continue
            min_det_sq = _min_det_squared_for_gamma(gamma, rate=2, device=device)
            candidate_scores.append((min_det_sq, gamma))
        if not candidate_scores:
            # If all filtered, relax slightly
            min_abs_gamma *= 0.1
            continue
        candidate_scores.sort(key=lambda x: x[0], reverse=True)
        best_score, best_gamma = candidate_scores[0]

        r_center, i_center = best_gamma.real, best_gamma.imag
        r_span = (r_bounds[1] - r_bounds[0]) * refine_factor
        i_span = (i_bounds[1] - i_bounds[0]) * refine_factor
        r_bounds = (r_center - r_span/2, r_center + r_span/2)
        i_bounds = (i_center - i_span/2, i_center + i_span/2)

    return best_gamma, best_score 