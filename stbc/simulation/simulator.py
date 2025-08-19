"""
Simulation functions for STBC.
"""

import time
import numpy as np
import torch

from ..core.biquaternion import BiquaternionSTBC
from ..core.codewords import get_cached_codewords
from ..utils.device_utils import select_device
from ..detectors.basic_detectors import (
    ml_detection_biquaternion,
    mmse_detection_biquaternion,
    zf_detection_biquaternion
)
from ..detectors.enhanced_detectors import (
    regularized_zf_detection_biquaternion,
    ml_enhanced_zf_detection_biquaternion,
    adaptive_mmse_detection_biquaternion,
    hybrid_detection_biquaternion
)

def apply_detector(detector: str, y, H, all_codewords, noise_var, stbc=None, rate=2):
    """
    Apply the specified detector to the received signal.
    
    Args:
        detector: Name of the detector to use
        y: Received signal
        H: Channel matrix
        all_codewords: All possible codewords
        noise_var: Noise variance
        stbc: STBC object (for fast quantization)
        rate: Code rate
        
    Returns:
        torch.Tensor: Indices of the detected codewords
    """
    if detector == 'ml':
        return ml_detection_biquaternion(y, H, all_codewords)
    if detector == 'mmse':
        return mmse_detection_biquaternion(y, H, all_codewords, noise_var, stbc, rate)
    if detector == 'zf':
        return zf_detection_biquaternion(y, H, all_codewords, stbc, rate)
    if detector == 'zf_reg':
        return regularized_zf_detection_biquaternion(y, H, all_codewords, noise_var, stbc, rate)
    if detector == 'ml_zf':  # ML-enhanced ZF
        return ml_enhanced_zf_detection_biquaternion(y, H, all_codewords)
    if detector == 'adaptive_mmse':  # Adaptive MMSE
        return adaptive_mmse_detection_biquaternion(y, H, all_codewords, noise_var)
    if detector == 'hybrid':  # Hybrid detector
        return hybrid_detection_biquaternion(y, H, all_codewords, noise_var)
    raise ValueError(f"Unknown detector: {detector}")

def simulate_ber(gamma, snr_db_list, detector='ml', rate=2, num_trials=800, device=None):
    """
    Simulate BER for a given gamma and SNR range.
    
    Args:
        gamma: The gamma parameter for STBC
        snr_db_list: List of SNR values in dB
        detector: Detector to use
        rate: Code rate
        num_trials: Number of simulation trials
        device: Device to run on
        
    Returns:
        numpy.ndarray: BER for each SNR point
    """
    device = select_device(device)
    batch_size = 100  # Process in batches for memory efficiency
    batches = (num_trials + batch_size - 1) // batch_size
    
    print(f"Simulating {detector.upper()} detection for γ={gamma}, {num_trials} trials, rate={rate}")
    ber_list = []
    
    for snr_db in snr_db_list:
        # Convert SNR from dB to linear
        snr = 10 ** (snr_db / 10)
        noise_var = 1 / snr
        # Convert noise_var to tensor for PyTorch operations
        noise_var_tensor = torch.tensor(noise_var, device=device, dtype=torch.float32)
        
        # Build STBC and get codebook
        stbc = BiquaternionSTBC(gamma, device)
        all_codewords, all_bits = get_cached_codewords(stbc, rate)
        
        # Stats for current SNR point
        bit_errors = 0
        total_bits = 0
        
        start_time = time.time()
        for b in range(batches):
            current_batch_size = min(batch_size, num_trials - b * batch_size)
            if current_batch_size <= 0:
                break
                
            # Randomly select codewords for this batch
            codeword_indices = torch.randint(0, all_codewords.shape[0], (current_batch_size,), device=device)
            X = all_codewords[codeword_indices]
            true_bits = all_bits[codeword_indices]
            
            # Generate random channel (Rayleigh fading)
            H = (torch.randn(current_batch_size, 4, 4, device=device) +
                1j * torch.randn(current_batch_size, 4, 4, device=device)) / torch.sqrt(torch.tensor(2.0, device=device))
                
            # Generate received signal: Y = H @ X + N
            Y = torch.matmul(H, X)
            N = torch.sqrt(noise_var_tensor/2) * (torch.randn_like(Y.real) + 1j * torch.randn_like(Y.real))
            Y_noisy = Y + N
            
            # Apply detector
            detected_indices = apply_detector(detector, Y_noisy, H, all_codewords, noise_var_tensor, stbc, rate)
            detected_bits = all_bits[detected_indices]
            
            # Count bit errors
            errors = (true_bits != detected_bits).sum().item()
            bit_errors += errors
            total_bits += true_bits.numel()
            
        # Calculate BER
        ber = bit_errors / total_bits if total_bits > 0 else 1.0
        print(f"  SNR = {snr_db} dB, BER = {ber:.6f}, time: {time.time() - start_time:.2f}s")
        ber_list.append(ber)
        
    return np.array(ber_list)

def simulate_ber_for_gamma(gamma, snr_db_list, detector='ml', rate=2, num_trials=800, device=None):
    """
    Wrapper function to simulate BER for a single gamma value.
    
    Args:
        gamma: The gamma parameter for STBC
        snr_db_list: List of SNR values in dB
        detector: Detector to use
        rate: Code rate
        num_trials: Number of simulation trials
        device: Device to run on
        
    Returns:
        numpy.ndarray: BER for each SNR point
    """
    device = select_device(device)
    ber_list = simulate_ber(gamma, snr_db_list, detector=detector, rate=rate, num_trials=num_trials, device=device)
    for i in range(len(snr_db_list)):
        print(f"    BER: {ber_list[i]:.6f}")
    return np.array(ber_list)

def simulate_ber_all_detectors(gammas, snr_db_list, rate=2, num_trials=800, device=None, detectors=None):
    """
    Simulate BER for all detectors with multiple gamma values.
    
    Args:
        gammas: List of gamma values to test
        snr_db_list: List of SNR values in dB
        rate: Code rate
        num_trials: Number of simulation trials
        device: Device to run on
        detectors: List of detectors to evaluate (if None, use all)
        
    Returns:
        tuple: Results for each gamma value and detector (includes timing data)
    """
    device = select_device(device)
    if detectors is None:
        detectors = ['ml', 'mmse', 'zf', 'zf_reg', 'ml_zf', 'adaptive_mmse', 'hybrid']
    
    results = {det: [[], [], []] for det in detectors}
    timing_results = {det: [[], [], []] for det in detectors}
    
    for i, gamma in enumerate(gammas):
        print(f"\nSimulating gamma = {gamma}")
        
        # Create STBC instance and cached codewords once per gamma
        stbc = BiquaternionSTBC(gamma, device)
        all_codewords, all_bits = get_cached_codewords(stbc, rate)
        
        # Process in batches for memory efficiency
        batch_size = 100  
        batches = (num_trials + batch_size - 1) // batch_size
        
        # Loop over SNR points first, then detectors
        # This ensures all detectors use the same random data for fair comparison
        for snr_idx, snr_db in enumerate(snr_db_list):
            print(f"\nSNR = {snr_db} dB:")
            
            # Convert SNR from dB to linear
            snr = 10 ** (snr_db / 10)
            noise_var = 1 / snr
            # Convert noise_var to tensor for PyTorch operations
            noise_var_tensor = torch.tensor(noise_var, device=device, dtype=torch.float32)
            
            # Set fixed random seed for this SNR point to ensure reproducibility
            # while still having different random data across SNR points
            torch.manual_seed(42 + snr_idx)
            
            # Generate all random data once
            all_batches_data = []
            
            # Pre-generate all random data
            for b in range(batches):
                current_batch_size = min(batch_size, num_trials - b * batch_size)
                if current_batch_size <= 0:
                    break
                
                # Randomly select codewords
                codeword_indices = torch.randint(0, all_codewords.shape[0], (current_batch_size,), device=device)
                X = all_codewords[codeword_indices]
                true_bits = all_bits[codeword_indices]
                
                # Generate random channel (Rayleigh fading)
                H = (torch.randn(current_batch_size, 4, 4, device=device) +
                     1j * torch.randn(current_batch_size, 4, 4, device=device)) / torch.sqrt(torch.tensor(2.0, device=device))
                     
                # Generate received signal: Y = H @ X + N
                Y = torch.matmul(H, X)
                N = torch.sqrt(noise_var_tensor/2) * (torch.randn_like(Y.real) + 1j * torch.randn_like(Y.real))
                Y_noisy = Y + N
                
                # Store all data for this batch
                all_batches_data.append((X, H, Y_noisy, true_bits))
            
            # Now run all detectors with the same data
            for det in detectors:
                bit_errors = 0
                total_bits = 0
                start_time = time.time()
                
                # Process each batch
                for X, H, Y_noisy, true_bits in all_batches_data:
                    # Apply detector (using the same noisy received signal)
                    detected_indices = apply_detector(det, Y_noisy, H, all_codewords, noise_var_tensor, stbc, rate)
                    detected_bits = all_bits[detected_indices]
                    
                    # Count bit errors
                    errors = (true_bits != detected_bits).sum().item()
                    bit_errors += errors
                    total_bits += true_bits.numel()
                
                # Calculate BER and timing
                ber = bit_errors / total_bits if total_bits > 0 else 1.0
                elapsed_time = time.time() - start_time
                print(f"  {det.upper()}: BER = {ber:.6f}, time: {elapsed_time:.2f}s")
                
                # Store result at the correct position
                if len(results[det][i]) <= snr_idx:
                    results[det][i].append(ber)
                    timing_results[det][i].append(elapsed_time)
                else:
                    results[det][i][snr_idx] = ber
                    timing_results[det][i][snr_idx] = elapsed_time
            
    # Return all results and dictionary for easy access (including timing)
    return (
        (np.array(results['ml'][0]), np.array(results['ml'][1]), np.array(results['ml'][2])),
        (np.array(results['mmse'][0]), np.array(results['mmse'][1]), np.array(results['mmse'][2])),
        (np.array(results['zf'][0]), np.array(results['zf'][1]), np.array(results['zf'][2])),
        (np.array(results['zf_reg'][0]), np.array(results['zf_reg'][1]), np.array(results['zf_reg'][2])),
        results,
        timing_results
    )
