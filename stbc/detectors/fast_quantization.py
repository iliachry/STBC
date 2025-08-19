"""
Fast quantization functions for linear detectors.
"""

import torch

def extract_quaternions_from_stbc_matrix(X, gamma):
    """Extract quaternions q1, q2 from 4x4 STBC matrix"""
    device = X.device
    batch_shape = X.shape[:-2]
    
    # Extract psi matrices from STBC structure (be more careful with gamma)
    psi_q1 = X[..., 0:2, 0:2]
    psi_q2 = X[..., 2:4, 0:2]
    psi_q1_sigma = X[..., 2:4, 2:4]
    
    # Handle gamma division carefully
    if isinstance(gamma, torch.Tensor):
        gamma_val = gamma.item() if gamma.numel() == 1 else gamma
    else:
        gamma_val = complex(gamma)
    
    # Avoid division by zero
    if abs(gamma_val) < 1e-12:
        gamma_val = 1e-12
    
    psi_q2_sigma = X[..., 0:2, 2:4] / gamma_val
    
    # Extract quaternions from psi matrices
    # For psi_q1: z_a = psi[0,0], z_b = psi[0,1]
    z_a1 = psi_q1[..., 0, 0]
    z_b1 = psi_q1[..., 0, 1]
    
    q1 = torch.stack([
        z_a1.real,  # q1[0]
        z_a1.imag,  # q1[1] 
        z_b1.real,  # q1[2]
        z_b1.imag   # q1[3]
    ], dim=-1)
    
    # For psi_q2: z_a = psi[0,0], z_b = psi[0,1]
    z_a2 = psi_q2[..., 0, 0]
    z_b2 = psi_q2[..., 0, 1]
    
    q2 = torch.stack([
        z_a2.real,  # q2[0]
        z_a2.imag,  # q2[1]
        z_b2.real,  # q2[2] 
        z_b2.imag   # q2[3]
    ], dim=-1)
    
    return q1, q2

def extract_symbols_from_quaternions(q1, q2, rate=2):
    """Extract symbols from quaternions"""
    if rate == 1:
        symbols = torch.stack([
            q1[..., 0] + 1j * q1[..., 1],  # s0
            q1[..., 2] + 1j * q1[..., 3],  # s1
        ], dim=-1)
    else:
        symbols = torch.stack([
            q1[..., 0] + 1j * q1[..., 1],  # s0
            q1[..., 2] + 1j * q1[..., 3],  # s1
            q2[..., 0] + 1j * q2[..., 1],  # s2
            q2[..., 2] + 1j * q2[..., 3],  # s3
        ], dim=-1)
    return symbols

def quantize_qpsk_symbols(symbols):
    """Fast QPSK quantization using sign-based decision"""
    # Much faster than distance computation
    # QPSK decision regions based on real/imaginary parts
    
    real_part = symbols.real
    imag_part = symbols.imag
    
    # Quantize based on signs (faster than distance calculation)
    real_bits = (real_part >= 0).long()  # 0 if negative, 1 if positive
    imag_bits = (imag_part >= 0).long()  # 0 if negative, 1 if positive
    
    # Map to QPSK indices: 00->0, 01->1, 10->2, 11->3
    indices = real_bits * 2 + imag_bits
    
    # QPSK constellation points
    sqrt2 = torch.sqrt(torch.tensor(2.0, device=symbols.device))
    qpsk_points = torch.tensor([1+1j, 1-1j, -1+1j, -1-1j],
                              dtype=torch.complex64, device=symbols.device) / sqrt2
    
    # Get quantized symbols
    quantized_symbols = qpsk_points[indices]
    
    return quantized_symbols, indices

def symbols_to_codeword_index(symbol_indices, rate=2):
    """Convert symbol indices to codeword index"""
    if rate == 1:
        # Rate-1: 2 symbols, each can be 0,1,2,3 -> index = s0*4 + s1
        codeword_idx = symbol_indices[..., 0] * 4 + symbol_indices[..., 1]
    else:
        # Rate-2: 4 symbols, each can be 0,1,2,3 -> index = s0*64 + s1*16 + s2*4 + s3
        codeword_idx = (symbol_indices[..., 0] * 64 + 
                       symbol_indices[..., 1] * 16 + 
                       symbol_indices[..., 2] * 4 + 
                       symbol_indices[..., 3])
    return codeword_idx

def fast_linear_detection(X_estimate, gamma, rate=2):
    """Fast quantization for linear detectors (MMSE/ZF)"""
    # Step 1: Extract quaternions from STBC matrix
    q1, q2 = extract_quaternions_from_stbc_matrix(X_estimate, gamma)
    
    # Step 2: Extract symbols from quaternions
    symbols = extract_symbols_from_quaternions(q1, q2, rate)
    
    # Step 3: Quantize symbols to QPSK
    quantized_symbols, symbol_indices = quantize_qpsk_symbols(symbols)
    
    # Step 4: Convert to codeword index
    codeword_indices = symbols_to_codeword_index(symbol_indices, rate)
    
    return codeword_indices