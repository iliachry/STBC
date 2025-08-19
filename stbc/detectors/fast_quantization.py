"""
Fast quantization functions for linear detectors.
Fixed implementation that properly inverts the STBC construction process.
"""

import torch

def fast_linear_detection(X_estimate, gamma, rate=2):
    """
    Fast quantization for linear detectors (MMSE/ZF).
    
    This algorithm reverses the STBC construction process:
    1. Extract psi matrices from 4x4 STBC matrix
    2. Extract quaternions from psi matrices  
    3. Extract symbols from quaternions
    4. Quantize symbols to QPSK constellation
    5. Convert to codeword indices
    
    Args:
        X_estimate: Estimated STBC matrix from linear detector [batch_size, 4, 4]
        gamma: STBC gamma parameter
        rate: Code rate (1 or 2)
        
    Returns:
        torch.Tensor: Codeword indices [batch_size]
    """
    batch_size = X_estimate.shape[0]
    device = X_estimate.device
    
    # Handle gamma conversion
    if isinstance(gamma, torch.Tensor):
        gamma_val = gamma.item() if gamma.numel() == 1 else gamma
    else:
        gamma_val = complex(gamma)
    
    # Avoid division by zero
    if abs(gamma_val) < 1e-12:
        gamma_val = 1e-12 + 0j
    
    # Step 1: Extract psi matrices from STBC structure
    # Based on left_regular_representation:
    # X[0:2, 0:2] = psi_q1
    # X[0:2, 2:4] = gamma * psi_q2_sigma  
    # X[2:4, 0:2] = psi_q2
    # X[2:4, 2:4] = psi_q1_sigma
    
    psi_q1 = X_estimate[..., 0:2, 0:2]           # Direct extraction
    psi_q2 = X_estimate[..., 2:4, 0:2]           # Direct extraction  
    psi_q1_sigma = X_estimate[..., 2:4, 2:4]     # Direct extraction
    psi_q2_sigma = X_estimate[..., 0:2, 2:4] / gamma_val  # Undo gamma scaling
    
    # Step 2: Extract quaternions from psi matrices
    # Psi matrix structure: [[z_a, z_b], [-z_b*, z_a*]]
    # So z_a = psi[0,0], z_b = psi[0,1]
    
    # Extract q1 from psi_q1
    z_a1 = psi_q1[..., 0, 0]  # Complex number
    z_b1 = psi_q1[..., 0, 1]  # Complex number
    
    q1_real = torch.stack([
        z_a1.real,  # q1[0] = Re(z_a1)
        z_a1.imag,  # q1[1] = Im(z_a1)  
        z_b1.real,  # q1[2] = Re(z_b1)
        z_b1.imag   # q1[3] = Im(z_b1)
    ], dim=-1)
    
    # Extract q2 from psi_q2
    z_a2 = psi_q2[..., 0, 0]  # Complex number
    z_b2 = psi_q2[..., 0, 1]  # Complex number
    
    q2_real = torch.stack([
        z_a2.real,  # q2[0] = Re(z_a2)
        z_a2.imag,  # q2[1] = Im(z_a2)
        z_b2.real,  # q2[2] = Re(z_b2) 
        z_b2.imag   # q2[3] = Im(z_b2)
    ], dim=-1)
    
    # Step 3: Extract symbols from quaternions
    # Based on symbols_to_quaternions mapping:
    # q1 = Q1.create_quaternion(s0.real, s0.imag, s1.real, s1.imag)
    # q2 = Q1.create_quaternion(s2.real, s2.imag, s3.real, s3.imag)
    
    if rate == 1:
        # Rate-1: Only use q1 for 2 symbols
        symbols = torch.stack([
            q1_real[..., 0] + 1j * q1_real[..., 1],  # s0 = q1[0] + j*q1[1]
            q1_real[..., 2] + 1j * q1_real[..., 3],  # s1 = q1[2] + j*q1[3]
        ], dim=-1)
    else:
        # Rate-2: Use both q1 and q2 for 4 symbols
        symbols = torch.stack([
            q1_real[..., 0] + 1j * q1_real[..., 1],  # s0 = q1[0] + j*q1[1]
            q1_real[..., 2] + 1j * q1_real[..., 3],  # s1 = q1[2] + j*q1[3]
            q2_real[..., 0] + 1j * q2_real[..., 1],  # s2 = q2[0] + j*q2[1] 
            q2_real[..., 2] + 1j * q2_real[..., 3],  # s3 = q2[2] + j*q2[3]
        ], dim=-1)
    
    # Step 4: Quantize symbols to QPSK constellation
    # QPSK constellation: {1+j, 1-j, -1+j, -1-j} / sqrt(2)
    # Decision regions based on sign of real and imaginary parts
    
    real_part = symbols.real
    imag_part = symbols.imag
    
    # Map to constellation indices based on signs
    # Positive real -> 0, negative real -> 1 (in binary: bit 1)
    # Positive imag -> 0, negative imag -> 1 (in binary: bit 0)
    real_bits = (real_part < 0).long()  # 0 if positive, 1 if negative
    imag_bits = (imag_part < 0).long()  # 0 if positive, 1 if negative
    
    # Combine bits: indices = real_bit * 2 + imag_bit
    # This gives: ++->0, +-->1, -+->2, -->3
    symbol_indices = real_bits * 2 + imag_bits
    
    # Step 5: Convert symbol indices to codeword index
    if rate == 1:
        # Rate-1: 2 symbols, each 0-3 -> total 16 codewords
        # Index = s0*4 + s1
        codeword_indices = (symbol_indices[..., 0] * 4 + 
                           symbol_indices[..., 1])
    else:
        # Rate-2: 4 symbols, each 0-3 -> total 256 codewords  
        # Index = s0*64 + s1*16 + s2*4 + s3
        codeword_indices = (symbol_indices[..., 0] * 64 + 
                           symbol_indices[..., 1] * 16 + 
                           symbol_indices[..., 2] * 4 + 
                           symbol_indices[..., 3])
    
    return codeword_indices

def verify_fast_quantization_accuracy(stbc, all_codewords, rate=2, num_test=100):
    """
    Verify that fast quantization produces correct results by comparing 
    with exhaustive search on known codewords.
    
    Args:
        stbc: BiquaternionSTBC instance
        all_codewords: All possible codewords [N, 4, 4]
        rate: Code rate
        num_test: Number of codewords to test
        
    Returns:
        tuple: (accuracy, error_count, total_count)
    """
    device = stbc.device
    num_codewords = all_codewords.shape[0]
    
    # Test on random subset of codewords
    test_indices = torch.randperm(num_codewords, device=device)[:num_test]
    test_codewords = all_codewords[test_indices]
    
    # Apply fast quantization
    fast_indices = fast_linear_detection(test_codewords, stbc.gamma, rate)
    
    # Compare with expected indices
    expected_indices = test_indices
    
    # Count correct predictions
    correct = (fast_indices == expected_indices).sum().item()
    total = num_test
    accuracy = correct / total
    
    return accuracy, total - correct, total

def benchmark_fast_quantization_speed(X_batch, gamma, rate=2, num_runs=100):
    """
    Benchmark the speed of fast quantization vs exhaustive search.
    
    Args:
        X_batch: Batch of STBC matrices [batch_size, 4, 4]
        gamma: STBC gamma parameter
        rate: Code rate
        num_runs: Number of timing runs
        
    Returns:
        tuple: (fast_time, fast_std)
    """
    import time
    
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        _ = fast_linear_detection(X_batch, gamma, rate)
        end_time = time.time()
        times.append(end_time - start_time)
    
    times = torch.tensor(times)
    return times.mean().item(), times.std().item()