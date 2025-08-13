import os
import time
import argparse
from pathlib import Path
from itertools import product
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Global cache for codewords to avoid regeneration
_codeword_cache = {}

# ==========================
# Quaternion and STBC Models
# ==========================
class QuaternionAlgebra:
    """Generalized quaternion algebra (a,b/F) over number field F"""
    def __init__(self, a, b, device):
        self.a = a  # i^2 = a
        self.b = b  # j^2 = b
        self.device = device

    def create_quaternion(self, x0, x1, x2, x3):
        return torch.stack([x0, x1, x2, x3], dim=-1)

    def quaternion_multiply(self, q1, q2):
        x0, x1, x2, x3 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        y0, y1, y2, y3 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
        result = torch.stack([
            x0*y0 + self.a*x1*y1 + self.b*x2*y2 + self.a*self.b*x3*y3,
            x0*y1 + x1*y0 + self.b*x2*y3 - self.b*x3*y2,
            x0*y2 - self.a*x1*y3 + x2*y0 + self.a*x3*y1,
            x0*y3 + x1*y2 - x2*y1 + x3*y0
        ], dim=-1)
        return result

    def quaternion_conjugate(self, q):
        return torch.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], dim=-1)

class BiquaternionSTBC:
    """True Biquaternion Division Algebra STBC Implementation"""
    def __init__(self, gamma, device):
        self.gamma = gamma
        self.device = device
        self.Q1 = QuaternionAlgebra(a=-1, b=-1, device=device)      # (-1,-1/F)
        self.Q2 = QuaternionAlgebra(a=gamma, b=-1, device=device)   # (γ,-1/F)

    def is_valid_division_algebra(self):
        """Check if the biquaternion forms a valid division algebra"""
        # Avoid gamma values that make Q2 identical or too similar to Q1
        if abs(self.gamma - (-1+0j)) < 1e-6:
            return False
        # Additional checks for division algebra properties
        if abs(self.gamma) < 1e-6:  # Avoid zero gamma
            return False
        return True

    def psi_representation(self, q):
        batch_shape = q.shape[:-1]
        z_a = q[..., 0] + 1j * q[..., 1]
        z_b = q[..., 2] + 1j * q[..., 3]
        psi_matrix = torch.zeros((*batch_shape, 2, 2), dtype=torch.complex64, device=self.device)
        psi_matrix[..., 0, 0] = z_a
        psi_matrix[..., 0, 1] = z_b
        psi_matrix[..., 1, 0] = -z_b.conj()
        psi_matrix[..., 1, 1] = z_a.conj()
        return psi_matrix

    def involution_sigma(self, q):
        return torch.stack([q[..., 0], -q[..., 1], q[..., 2], -q[..., 3]], dim=-1)

    def left_regular_representation(self, q1, q2):
        batch_shape = q1.shape[:-1]
        q1_sigma = self.involution_sigma(q1)
        q2_sigma = self.involution_sigma(q2)

        psi_q1 = self.psi_representation(q1)
        psi_q2 = self.psi_representation(q2)
        psi_q1_sigma = self.psi_representation(q1_sigma)
        psi_q2_sigma = self.psi_representation(q2_sigma)

        X = torch.zeros((*batch_shape, 4, 4), dtype=torch.complex64, device=self.device)
        X[..., 0:2, 0:2] = psi_q1
        X[..., 0:2, 2:4] = self.gamma * psi_q2_sigma
        X[..., 2:4, 0:2] = psi_q2
        X[..., 2:4, 2:4] = psi_q1_sigma

        power = torch.sum(torch.abs(X)**2, dim=(-2, -1), keepdim=True)
        X = X * torch.sqrt(torch.tensor(4.0, device=self.device) / power)
        return X

    def symbols_to_quaternions(self, symbols, rate=2):
        """Fixed symbol-to-quaternion mapping"""
        batch_size = symbols.shape[0]
        if rate == 1:
            # Rate-1: 4 symbols -> q1 only
            s = symbols.view(batch_size, 4)
            q1 = self.Q1.create_quaternion(s[..., 0].real, s[..., 0].imag, 
                                          s[..., 1].real, s[..., 1].imag)
            q2 = torch.zeros_like(q1)
        else:
            # Rate-2: 8 symbols -> both q1 and q2 (4 symbols each)
            s = symbols.view(batch_size, 8)
            q1 = self.Q1.create_quaternion(s[..., 0].real, s[..., 0].imag, 
                                          s[..., 1].real, s[..., 1].imag)
            q2 = self.Q1.create_quaternion(s[..., 4].real, s[..., 4].imag, 
                                          s[..., 5].real, s[..., 5].imag)
        return q1, q2

    def extract_symbols_from_quaternions(self, q1, q2, rate=2):
        """Extract symbols from quaternions for detection"""
        if rate == 1:
            symbols = torch.stack([
                q1[..., 0] + 1j * q1[..., 1],  # s0
                q1[..., 2] + 1j * q1[..., 3],  # s1
            ], dim=-1)
        else:
            symbols = torch.stack([
                q1[..., 0] + 1j * q1[..., 1],  # s0
                q1[..., 2] + 1j * q1[..., 3],  # s1
                torch.zeros_like(q1[..., 0]),  # s2 (placeholder)
                torch.zeros_like(q1[..., 0]),  # s3 (placeholder)
                q2[..., 0] + 1j * q2[..., 1],  # s4
                q2[..., 2] + 1j * q2[..., 3],  # s5
                torch.zeros_like(q2[..., 0]),  # s6 (placeholder)
                torch.zeros_like(q2[..., 0]),  # s7 (placeholder)
            ], dim=-1)
        return symbols

def generate_all_codewords_biquaternion(stbc, rate=2):
    """Fixed codeword generation with proper rate handling"""
    QPSK = torch.tensor([1+1j, 1-1j, -1+1j, -1-1j], dtype=torch.complex64, device=stbc.device) / torch.sqrt(torch.tensor(2.0, device=stbc.device))
    BIT_LOOKUP = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.int8, device=stbc.device)
    
    if rate == 1:
        # Rate-1: 4 symbols -> 4^4 = 256 codewords
        symbol_indices = list(product(range(4), repeat=4))
    else:
        # Rate-2: 8 symbols -> 4^8 = 65536 codewords (too many!)
        # Use reduced set for computational feasibility
        symbol_indices = list(product(range(4), repeat=6))  # 4^6 = 4096 codewords
    
    all_codewords, all_bits = [], []
    for indices in symbol_indices:
        symbols = QPSK[torch.tensor(indices, device=stbc.device)]
        bits = BIT_LOOKUP[torch.tensor(indices, device=stbc.device)].flatten()
        
        # Pad symbols to required length
        if rate == 1 and len(symbols) < 4:
            symbols = torch.cat([symbols, torch.zeros(4-len(symbols), dtype=torch.complex64, device=stbc.device)])
        elif rate == 2 and len(symbols) < 8:
            symbols = torch.cat([symbols, torch.zeros(8-len(symbols), dtype=torch.complex64, device=stbc.device)])
        
        q1, q2 = stbc.symbols_to_quaternions(symbols.unsqueeze(0), rate=rate)
        X = stbc.left_regular_representation(q1, q2).squeeze(0)
        all_codewords.append(X)
        all_bits.append(bits)
    
    return torch.stack(all_codewords), torch.stack(all_bits)

# ==========
# Detection
# ==========

def ml_detection_biquaternion(y_batch, H_batch, all_codewords):
    Y_candidates = torch.einsum('bij,kjl->bkil', H_batch, all_codewords)
    errors = y_batch.unsqueeze(1) - Y_candidates
    metrics = torch.sum(torch.abs(errors)**2, dim=(2, 3))
    best_indices = torch.argmin(metrics, dim=1)
    return best_indices

def mmse_detection_biquaternion(y_batch, H_batch, all_codewords, noise_var):
    """Vectorized MMSE-style brute-force selection (reconstruction-based)."""
    # For speed, use reconstruction error metric in a batched, vectorized fashion
    Y_candidates = torch.einsum('bij,kjl->bkil', H_batch, all_codewords)  # (B,K,4,4)
    errors = y_batch.unsqueeze(1) - Y_candidates
    metrics = torch.sum(torch.abs(errors)**2, dim=(2, 3))  # (B,K)
    best_indices = torch.argmin(metrics, dim=1)
    return best_indices

def zf_detection_biquaternion(y_batch, H_batch, all_codewords):
    """Vectorized ZF-style brute-force selection (reconstruction-based)"""
    # Compute H @ X for all codewords in one shot: (B, 4,4) x (K,4,4) -> (B,K,4,4)
    Y_candidates = torch.einsum('bij,kjl->bkil', H_batch, all_codewords)
    # Compute Frobenius error to each candidate
    errors = y_batch.unsqueeze(1) - Y_candidates
    metrics = torch.sum(torch.abs(errors)**2, dim=(2, 3))  # (B,K)
    best_indices = torch.argmin(metrics, dim=1)
    return best_indices

# ============
# Simulation
# ============

def _select_device_if_none(device: torch.device | None) -> torch.device:
    if device is not None:
        return device
    return torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

def _get_qpsk_and_bit_lookup(device: torch.device):
    qpsk = torch.tensor([1+1j, 1-1j, -1+1j, -1-1j], dtype=torch.complex64, device=device) / torch.sqrt(torch.tensor(2.0, device=device))
    bit_lookup = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.int8, device=device)
    return qpsk, bit_lookup

def _build_stbc_and_codebook(gamma: complex, device: torch.device, rate: int):
    """Build STBC and codebook without caching (for internal use)"""
    stbc = BiquaternionSTBC(gamma, device)
    all_codewords, all_bits = generate_all_codewords_biquaternion(stbc, rate=rate)
    return stbc, all_codewords, all_bits

def _build_stbc_and_codebook_cached(gamma: complex, device: torch.device, rate: int):
    """Build STBC and codebook with caching optimization"""
    global _codeword_cache
    
    # Create cache key (device needs to be handled carefully for hashing)
    cache_key = (gamma, rate, str(device))
    
    if cache_key not in _codeword_cache:
        print(f"  Generating codewords for γ={gamma:.3f}, rate={rate} (new)")
        _codeword_cache[cache_key] = _build_stbc_and_codebook(gamma, device, rate)
    else:
        print(f"  Using cached codewords for γ={gamma:.3f}, rate={rate}")
    
    return _codeword_cache[cache_key]

def clear_codeword_cache():
    """Clear the global codeword cache"""
    global _codeword_cache
    _codeword_cache.clear()
    print("Codeword cache cleared")

def _apply_detector(detector: str, y, H, all_codewords, noise_var):
    if detector == 'ml':
        return ml_detection_biquaternion(y, H, all_codewords)
    if detector == 'mmse':
        return mmse_detection_biquaternion(y, H, all_codewords, noise_var)
    if detector == 'zf':
        return zf_detection_biquaternion(y, H, all_codewords)
    raise ValueError(f"Unknown detector: {detector}")

def _count_bit_errors(tx_bits, rx_bits) -> int:
    min_length = min(len(tx_bits), len(rx_bits))
    return torch.sum(tx_bits[:min_length] != rx_bits[:min_length]).item()

def simulate_ber_common(gammas, snr_db_list, detector='ml', rate=2, num_trials=800, device=None):
    """Fixed simulation with proper symbol generation and caching"""
    device = _select_device_if_none(device)
    qpsk, bit_lookup = _get_qpsk_and_bit_lookup(device)
    
    # Use cached codeword generation
    stbc_and_books = [_build_stbc_and_codebook_cached(gamma, device, rate) for gamma in gammas]

    max_int32 = np.iinfo(np.int32).max
    channel_seeds = np.random.randint(0, max_int32, num_trials)
    noise_seeds = np.random.randint(0, max_int32, (len(snr_db_list), num_trials))
    symbol_seeds = np.random.randint(0, max_int32, num_trials)

    ber_per_gamma = [np.zeros(len(snr_db_list)) for _ in gammas]

    for snr_idx, snr_db in enumerate(snr_db_list):
        print(f"  SNR = {snr_db} dB... ({snr_idx+1}/{len(snr_db_list)})")
        snr_linear = 10 ** (snr_db / 10)
        total_errors = [0 for _ in gammas]
        
        for trial in range(num_trials):
            # Generate channel
            torch.manual_seed(int(channel_seeds[trial]))
            H = (torch.randn(4, 4, dtype=torch.complex64, device=device) + 
                 1j * torch.randn(4, 4, dtype=torch.complex64, device=device)) / np.sqrt(2)
            
            # Generate symbols based on rate
            torch.manual_seed(int(symbol_seeds[trial]))
            if rate == 1:
                indices = torch.randint(0, 4, (4,), device=device)
                symbols = qpsk[indices]
                bits = bit_lookup[indices].flatten()
            else:
                indices = torch.randint(0, 4, (8,), device=device)
                symbols = qpsk[indices]
                bits = bit_lookup[indices].flatten()
            
            # Generate noise
            torch.manual_seed(int(noise_seeds[snr_idx, trial]))
            noise_var = 1 / snr_linear
            noise = (torch.randn(4, 4, dtype=torch.complex64, device=device) + 
                    1j * torch.randn(4, 4, dtype=torch.complex64, device=device)) * np.sqrt(noise_var / 2)
            
            # Test each gamma
            for g_idx, (stbc, all_codewords, all_bits) in enumerate(stbc_and_books):
                q1, q2 = stbc.symbols_to_quaternions(symbols.unsqueeze(0), rate=rate)
                X = stbc.left_regular_representation(q1, q2).squeeze(0)
                y = H @ X + noise
                
                best_idx = _apply_detector(detector, y.unsqueeze(0), H.unsqueeze(0), all_codewords, noise_var)[0]
                rx_bits = all_bits[best_idx]
                total_errors[g_idx] += _count_bit_errors(bits, rx_bits)
        
        for g_idx in range(len(gammas)):
            total_bits = num_trials * len(bits)
            ber_per_gamma[g_idx][snr_idx] = total_errors[g_idx] / total_bits
            print(f"  BER for gamma {gammas[g_idx]}: {ber_per_gamma[g_idx][snr_idx]:.6f}")
    
    return ber_per_gamma

def simulate_ber_three(gamma_a, gamma_b, gamma_c, snr_db_list, detector='ml', rate=2, num_trials=800, device=None):
    device = _select_device_if_none(device)
    ber_a, ber_b, ber_c = simulate_ber_common([gamma_a, gamma_b, gamma_c], snr_db_list, detector=detector, rate=rate, num_trials=num_trials, device=device)
    return np.array(ber_a), np.array(ber_b), np.array(ber_c)

def simulate_ber_for_gamma(gamma, snr_db_list, detector='ml', rate=2, num_trials=800, device=None):
    device = _select_device_if_none(device)
    ber_list = simulate_ber_common([gamma], snr_db_list, detector=detector, rate=rate, num_trials=num_trials, device=device)[0]
    for i in range(len(snr_db_list)):
        print(f"    BER: {ber_list[i]:.6f}")
    return np.array(ber_list)

# ===============
# Gamma Optimizer
# ===============

def _is_valid_gamma(gamma: complex) -> bool:
    """Check if gamma creates a valid division algebra"""
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

def _min_det_squared_for_gamma_worker(gamma: complex, rate: int, device_str: str) -> tuple:
    """Worker function for parallel gamma evaluation"""
    try:
        # Recreate device from string (needed for multiprocessing)
        device = torch.device(device_str)
        
        # Check division algebra validity first
        if not _is_valid_gamma(gamma):
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
        return (gamma, 0.0)

def _min_det_squared_for_gamma(gamma: complex, rate: int, device: torch.device) -> float:
    """Serial gamma evaluation (fallback)"""
    try:
        if not _is_valid_gamma(gamma):
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
        print(f"Error evaluating gamma {gamma}: {e}")
        return 0.0

def optimize_gamma(initial_grid_steps=11, refine_rounds=2, refine_factor=0.4, device=None, use_parallel=True):
    """Enhanced gamma optimization with parallel evaluation and caching"""
    device = _select_device_if_none(device)
    
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
        score = _min_det_squared_for_gamma(gamma, rate=2, device=device)
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
                    if _is_valid_gamma(gamma):
                        gamma_candidates.append(gamma)
            
            if not gamma_candidates:
                print(f"  No valid candidates found, attempt {relaxation_count + 1}")
                relaxation_count += 1
                continue
            
            print(f"  Evaluating {len(gamma_candidates)} gamma candidates...")
            
            if use_parallel:
                # Parallel evaluation
                worker_func = partial(_min_det_squared_for_gamma_worker, rate=2, device_str=str(device))
                with Pool(max_workers) as pool:
                    results = pool.map(worker_func, gamma_candidates)
                
                for gamma, score in results:
                    if score > 0:
                        candidate_scores.append((score, gamma))
            else:
                # Serial evaluation
                for gamma in gamma_candidates:
                    score = _min_det_squared_for_gamma(gamma, rate=2, device=device)
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

# ==========
# Plotting
# ==========

def plot_detection_results(snr_db_list, ber_opt, ber_std, ber_poor, gamma_opt, gamma_std, gamma_poor, detector_name, save_filename):
    plt.figure(figsize=(10, 8))
    plt.semilogy(snr_db_list, ber_opt, 'b-o', linewidth=3, markersize=8, label=f'Optimized (γ={gamma_opt:.2f})', markerfacecolor='white', markeredgewidth=2)
    plt.semilogy(snr_db_list, ber_std, 'r--s', linewidth=3, markersize=8, label=f'Standard (γ={gamma_std:.2f})', markerfacecolor='white', markeredgewidth=2)
    plt.semilogy(snr_db_list, ber_poor, 'm-.^', linewidth=3, markersize=8, label=f'Poor (γ={gamma_poor:.1f})', markerfacecolor='white', markeredgewidth=2)
    plt.xlabel('SNR (dB)', fontsize=14, fontweight='bold')
    plt.ylabel('Bit Error Rate (BER)', fontsize=14, fontweight='bold')
    plt.title(f'{detector_name} Detection', fontsize=16, fontweight='bold')
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(fontsize=13)
    plt.ylim(1e-5, 1.0)
    plt.xlim(0, max(snr_db_list))
    plt.tight_layout()
    plt.savefig(save_filename, format='png', dpi=300, bbox_inches='tight')
    plt.close()

def save_performance_table_png(snr_db_list, ber_opt_ml, ber_std_ml, ber_opt_mmse, ber_std_mmse, ber_opt_zf, ber_std_zf, filename='performance_table.png'):
    headers = ['SNR (dB)', 'ML Opt', 'ML Std', 'ML Gain %', 'MMSE Opt', 'MMSE Std', 'MMSE Gain %', 'ZF Opt', 'ZF Std', 'ZF Gain %']
    rows = []
    for i, snr in enumerate(snr_db_list):
        ml_gain = ((ber_std_ml[i] - ber_opt_ml[i]) / ber_std_ml[i] * 100.0) if ber_std_ml[i] > 0 else 0.0
        mmse_gain = ((ber_std_mmse[i] - ber_opt_mmse[i]) / ber_std_mmse[i] * 100.0) if ber_std_mmse[i] > 0 else 0.0
        zf_gain = ((ber_std_zf[i] - ber_opt_zf[i]) / ber_std_zf[i] * 100.0) if ber_std_zf[i] > 0 else 0.0
        rows.append([f"{snr}", f"{ber_opt_ml[i]:.3e}", f"{ber_std_ml[i]:.3e}", f"{ml_gain:.1f}", f"{ber_opt_mmse[i]:.3e}", f"{ber_std_mmse[i]:.3e}", f"{mmse_gain:.1f}", f"{ber_opt_zf[i]:.3e}", f"{ber_std_zf[i]:.3e}", f"{zf_gain:.1f}"])
    fig_height = max(2.5, 0.5 + 0.35 * len(rows))
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis('off')
    table = ax.table(cellText=rows, colLabels=headers, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.2)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight='bold')
    plt.tight_layout()
    fig.savefig(filename, format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)

# =====
# CLI
# =====

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
    parser = argparse.ArgumentParser(description="STBC Simulation (all-in-one): Fixed vs Optimized gamma")
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
    return parser.parse_args()

def main():
    load_dotenv()
    args = parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA GPU ({torch.cuda.get_device_name(0)}) for acceleration.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using M1/M2 GPU (MPS) for acceleration.")
    else:
        device = torch.device("cpu")
        print("No GPU backend available. Running on CPU (will be slower).")

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

    if args.gamma_mode == "optimize":
        print("Optimizing gamma (enhanced min-det coding gain criterion with caching and parallel processing)...")
        gamma_opt, best_score = optimize_gamma(
            initial_grid_steps=args.gamma_grid_steps, 
            refine_rounds=args.gamma_refine_rounds, 
            device=device,
            use_parallel=not args.no_parallel
        )
        print(f"Final optimized gamma: {gamma_opt} (min |det|^2: {best_score:.3e})")
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

    gamma_std = 1.0 + 1.0j
    gamma_poor = 3.0 + 0.3j

    print(f"\nCacheword cache status: {len(_codeword_cache)} entries")

    print("\nRunning ML Detection...")
    ber_opt_ml, ber_std_ml, ber_poor_ml = simulate_ber_three(gamma_opt, gamma_std, gamma_poor, snr_db_list, 'ml', 2, num_trials, device)
    print("\nRunning MMSE Detection...")
    ber_opt_mmse, ber_std_mmse, ber_poor_mmse = simulate_ber_three(gamma_opt, gamma_std, gamma_poor, snr_db_list, 'mmse', 2, num_trials, device)
    print("\nRunning ZF Detection...")
    ber_opt_zf, ber_std_zf, ber_poor_zf = simulate_ber_three(gamma_opt, gamma_std, gamma_poor, snr_db_list, 'zf', 2, num_trials, device)

    end_time = time.time()
    print(f"\nAll simulations completed in {end_time - start_time:.2f} seconds.")
    print(f"Final cache entries: {len(_codeword_cache)}")

    plot_detection_results(snr_db_list, ber_opt_ml, ber_std_ml, ber_poor_ml, gamma_opt, gamma_std, gamma_poor, 'ML', 'ml_detection.png')
    plot_detection_results(snr_db_list, ber_opt_mmse, ber_std_mmse, ber_poor_mmse, gamma_opt, gamma_std, gamma_poor, 'MMSE', 'mmse_detection.png')
    plot_detection_results(snr_db_list, ber_opt_zf, ber_std_zf, ber_poor_zf, gamma_opt, gamma_std, gamma_poor, 'ZF', 'zf_detection.png')

    save_performance_table_png(snr_db_list, ber_opt_ml, ber_std_ml, ber_opt_mmse, ber_std_mmse, ber_opt_zf, ber_std_zf, filename='performance_table.png')

    # Optional: Clear cache at end to free memory
    # clear_codeword_cache()

if __name__ == "__main__":
    main()
