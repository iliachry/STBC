"""
Additional simulation functions for STBC.
"""

import numpy as np
import torch

from ..core.biquaternion import BiquaternionSTBC
from ..core.codewords import get_cached_codewords
from ..utils.device_utils import select_device, get_qpsk_and_bit_lookup
from .simulator import apply_detector

def _count_bit_errors(tx_bits, rx_bits) -> int:
    """Count bit errors between transmitted and received bits"""
    min_length = min(len(tx_bits), len(rx_bits))
    return torch.sum(tx_bits[:min_length] != rx_bits[:min_length]).item()

def simulate_ber_common(gammas, snr_db_list, detector='ml', rate=2, num_trials=800, device=None):
    """
    Fixed simulation with proper symbol generation and caching.
    
    Args:
        gammas: List of gamma values to test
        snr_db_list: List of SNR values in dB
        detector: Detector to use
        rate: Code rate
        num_trials: Number of simulation trials
        device: Device to run on
        
    Returns:
        list: BER values for each gamma and SNR point
    """
    device = select_device(device)
    qpsk, bit_lookup = get_qpsk_and_bit_lookup(device)
    
    # Build codebooks for all gammas
    stbc_and_books = []
    for gamma in gammas:
        stbc = BiquaternionSTBC(gamma, device)
        all_codewords, all_bits = get_cached_codewords(stbc, rate)
        stbc_and_books.append((stbc, all_codewords, all_bits))

    # Generate random seeds for reproducibility
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
                # Rate-2 codebook uses 4 symbols
                indices = torch.randint(0, 4, (4,), device=device)
                symbols = qpsk[indices]
                symbols = torch.cat([symbols, torch.zeros(4, dtype=torch.complex64, device=device)])
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
                
                best_idx = apply_detector(detector, y.unsqueeze(0), H.unsqueeze(0), all_codewords, noise_var)[0]
                rx_bits = all_bits[best_idx]
                total_errors[g_idx] += _count_bit_errors(bits, rx_bits)
        
        for g_idx in range(len(gammas)):
            total_bits = num_trials * len(bits)
            ber_per_gamma[g_idx][snr_idx] = total_errors[g_idx] / total_bits
            print(f"  BER for gamma {gammas[g_idx]}: {ber_per_gamma[g_idx][snr_idx]:.6f}")
    
    return ber_per_gamma

def simulate_ber_three(gamma_a, gamma_b, gamma_c, snr_db_list, detector='ml', rate=2, num_trials=800, device=None):
    """
    Simulate BER for three gamma values.
    
    Args:
        gamma_a: First gamma value
        gamma_b: Second gamma value
        gamma_c: Third gamma value
        snr_db_list: List of SNR values in dB
        detector: Detector to use
        rate: Code rate
        num_trials: Number of simulation trials
        device: Device to run on
        
    Returns:
        tuple: (BER for gamma_a, BER for gamma_b, BER for gamma_c)
    """
    device = select_device(device)
    ber_a, ber_b, ber_c = simulate_ber_common([gamma_a, gamma_b, gamma_c], snr_db_list, detector=detector, rate=rate, num_trials=num_trials, device=device)
    return np.array(ber_a), np.array(ber_b), np.array(ber_c)
