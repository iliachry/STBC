"""
Gamma parameter optimization for Biquaternion STBC.
"""

import time
import random
from multiprocessing import Pool, cpu_count
from functools import partial
import numpy as np
import torch

from ..core.biquaternion import BiquaternionSTBC
from ..core.codewords import generate_all_codewords_biquaternion
from ..utils.device_utils import select_device

def is_valid_gamma(gamma: complex) -> bool:
    """
    Check if gamma creates a valid division algebra.
    
    Args:
        gamma: Complex gamma value
        
    Returns:
        bool: True if gamma creates valid division algebra
    """
    # Avoid gamma values that make Q2 identical to Q1 = (-1,-1/F)
    if abs(gamma - (-1+0j)) < 1e-3:
        return False
    
    # Avoid zero or very small gamma
    if abs(gamma) < 1e-3:
        return False
    
    # Avoid purely real negative values close to -1
    if abs(gamma.imag) < 1e-3 and gamma.real < -0.5:
        return False
    
    return True

def min_det_squared_for_gamma_worker(gamma: complex, rate: int, device_str: str) -> tuple:
    """
    Worker function for parallel gamma evaluation.
    
    Args:
        gamma: Complex gamma value
        rate: Code rate
        device_str: String representation of device
        
    Returns:
        tuple: (gamma, min_det_squared)
    """
    try:
        # Recreate device from string (needed for multiprocessing)
        device = torch.device(device_str)
        
        # Check division algebra validity first
        if not is_valid_gamma(gamma):
            return (gamma, 0.0)
        
        stbc = BiquaternionSTBC(gamma, device)
        
        # Additional runtime check
        if not stbc.is_valid_division_algebra():
            return (gamma, 0.0)
        
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
            return (gamma, 0.0)
        return (gamma, float(torch.min(vals).item()))
    except Exception as e:
        print(f"Error evaluating gamma={gamma}: {e}")
        return (gamma, 0.0)

def min_det_squared_for_gamma(gamma: complex, rate: int, device: torch.device) -> float:
    """
    Serial gamma evaluation (fallback).
    
    Args:
        gamma: Complex gamma value
        rate: Code rate
        device: Computation device
        
    Returns:
        float: Minimum determinant squared
    """
    try:
        if not is_valid_gamma(gamma):
            return 0.0
        
        stbc = BiquaternionSTBC(gamma, device)
        
        if not stbc.is_valid_division_algebra():
            return 0.0
        
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
    except Exception as e:
        print(f"Error in min_det calculation for gamma={gamma}: {e}")
        return 0.0

def optimize_gamma(initial_grid_steps=11, refine_rounds=2, refine_factor=0.4, device=None, use_parallel=True):
    """Enhanced gamma optimization with parallel evaluation and caching"""
    device = select_device(device)
    
    # Expanded search bounds to avoid boundary solutions
    r_bounds = (-2.0, 3.0)  # Expanded from (-1.0, 2.0)
    i_bounds = (-1.0, 3.0)  # Expanded from (0.0, 2.0) to include negative imaginary
    
    best_gamma, best_score = None, -float('inf')
    max_relaxation_attempts = 5
    
    # Start with theoretically motivated candidates
    theory_candidates = [
        0.618 + 1.0j,   # Golden ratio based
        1.0 + 1.0j,     # Standard literature choice
        0.5 + 1.5j,     # Moderate complex value
        1.5 + 0.8j,     # Alternative choice
    ]
    
    print("Evaluating theory-based candidates...")
    for gamma in theory_candidates:
        score = min_det_squared_for_gamma(gamma, rate=2, device=device)
        print(f"  γ = {gamma:.3f}: min |det|^2 = {score:.3e}")
        if score > best_score:
            best_score, best_gamma = score, gamma
    
    print(f"Best theory candidate: γ = {best_gamma:.3f} (score: {best_score:.3e})")
    
    # Determine parallel processing setup
    if use_parallel and device.type == 'cpu':
        max_workers = min(cpu_count(), 8)  # Limit workers to avoid overload
        print(f"Using parallel evaluation with {max_workers} workers")
    else:
        use_parallel = False
        print("Using serial evaluation (GPU detected or parallel disabled)")
    
    for round_idx in range(refine_rounds + 1):
        steps = initial_grid_steps if round_idx == 0 else 7
        candidate_scores = []
        relaxation_count = 0
        
        print(f"Optimization round {round_idx + 1}/{refine_rounds + 1}")
        print(f"  Search bounds: Re({r_bounds[0]:.2f}, {r_bounds[1]:.2f}), Im({i_bounds[0]:.2f}, {i_bounds[1]:.2f})")
        
        while not candidate_scores and relaxation_count < max_relaxation_attempts:
            # Generate all gamma candidates for this round
            gamma_candidates = []
            for r in np.linspace(r_bounds[0], r_bounds[1], steps):
                for im in np.linspace(i_bounds[0], i_bounds[1], steps):
                    gamma = complex(float(r), float(im))
                    if is_valid_gamma(gamma):
                        gamma_candidates.append(gamma)
            
            if not gamma_candidates:
                print(f"  No valid candidates found, attempt {relaxation_count + 1}")
                relaxation_count += 1
                continue
            
            print(f"  Evaluating {len(gamma_candidates)} gamma candidates...")
            
            if use_parallel:
                # Parallel evaluation
                worker_func = partial(min_det_squared_for_gamma_worker, rate=2, device_str=str(device))
                with Pool(max_workers) as pool:
                    results = pool.map(worker_func, gamma_candidates)
                
                for gamma, score in results:
                    if score > 0:
                        candidate_scores.append((score, gamma))
            else:
                # Serial evaluation
                for gamma in gamma_candidates:
                    score = min_det_squared_for_gamma(gamma, rate=2, device=device)
                    if score > 0:
                        candidate_scores.append((score, gamma))
            
            if not candidate_scores:
                print(f"  No valid candidates found, attempt {relaxation_count + 1}")
                relaxation_count += 1
        
        if not candidate_scores:
            print("  Warning: No valid candidates found after all attempts")
            break
            
        candidate_scores.sort(key=lambda x: x[0], reverse=True)
        round_best_score, round_best_gamma = candidate_scores[0]
        
        print(f"  Round best: γ = {round_best_gamma:.3f} (score: {round_best_score:.3e})")
        
        if round_best_score > best_score:
            best_score, best_gamma = round_best_score, round_best_gamma
        
        # Refine bounds around best candidate
        r_center, i_center = best_gamma.real, best_gamma.imag
        r_span = (r_bounds[1] - r_bounds[0]) * refine_factor
        i_span = (i_bounds[1] - i_bounds[0]) * refine_factor
        r_bounds = (r_center - r_span/2, r_center + r_span/2)
        i_bounds = (i_center - i_span/2, i_center + i_span/2)
    
    return best_gamma, best_score
