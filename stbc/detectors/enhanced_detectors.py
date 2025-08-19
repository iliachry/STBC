"""
Enhanced detection algorithms for STBC.
"""

import torch
from .fast_quantization import fast_linear_detection

def adaptive_reg_factor(noise_var):
    """Fixed regularization - should be proportional to noise variance"""
    # Handle tensor input
    if isinstance(noise_var, torch.Tensor):
        noise_var_val = noise_var.item()
    else:
        noise_var_val = float(noise_var)
    
    # Ensure positive noise variance
    noise_var_val = max(noise_var_val, 1e-10)
    
    # Return a multiplicative factor, not absolute values
    # This will be multiplied by noise_var later
    snr_linear = 1.0 / noise_var_val
    
    if snr_linear > 100:  # High SNR (> 20 dB)
        return 0.01   # Very small factor for high SNR
    elif snr_linear > 10:  # Medium SNR (10-20 dB)
        return 0.1    # Small factor
    elif snr_linear > 1:   # Low-Medium SNR (0-10 dB)
        return 1.0    # Standard factor
    else:  # Very Low SNR (< 0 dB)
        return 5.0    # Large factor for very low SNR
        
def regularized_zf_detection_biquaternion(y_batch, H_batch, all_codewords, noise_var, stbc=None, rate=2):
    """Regularized ZF with fast quantization"""
    batch_size = y_batch.shape[0]
    device = y_batch.device
    
    # Convert noise_var to scalar
    if isinstance(noise_var, torch.Tensor):
        noise_var_val = noise_var.item()
    else:
        noise_var_val = float(noise_var)
    
    # Moderate regularization
    H_h = H_batch.transpose(-2, -1).conj()
    HH = torch.matmul(H_h, H_batch)
    
    reg_value = 1.0 * noise_var_val  # Increase from 0.1 to 1.0 for better performance
    I_batch = torch.eye(4, device=device, dtype=H_batch.dtype).unsqueeze(0).expand(batch_size, -1, -1)
    HH_reg = HH + reg_value * I_batch
    
    try:
        # Standard regularized ZF computation
        HH_reg_inv = torch.linalg.inv(HH_reg)
        W_reg = torch.matmul(HH_reg_inv, H_h)
        X_zf_reg = torch.matmul(W_reg, y_batch)
        
        # Fast quantization - newly fixed algorithm
        if stbc is not None:
            try:
                best_indices = fast_linear_detection(X_zf_reg, stbc.gamma, rate)
                return best_indices
            except Exception as e:
                # Fallback to exhaustive search if fast quantization fails
                print(f"Warning: Fast quantization failed ({e}), using exhaustive search")
        
        # Fallback: exhaustive search
        X_zf_exp = X_zf_reg.unsqueeze(1)
        all_codewords_exp = all_codewords.unsqueeze(0)
        distances = torch.sum(torch.abs(X_zf_exp - all_codewords_exp)**2, dim=(2, 3))
        best_indices = torch.argmin(distances, dim=1)
        
    except Exception as e:
        # Fallback to basic ZF
        from .basic_detectors import zf_detection_biquaternion
        best_indices = zf_detection_biquaternion(y_batch, H_batch, all_codewords, stbc, rate)
    
    return best_indices
    
def ml_enhanced_zf_detection_biquaternion(y_batch, H_batch, all_codewords):
    """Enhanced ZF that should perform better than basic ZF"""
    batch_size = y_batch.shape[0]
    device = y_batch.device
    
    # Use more robust pseudo-inverse with slight regularization
    H_h = H_batch.transpose(-2, -1).conj()
    HH = torch.matmul(H_h, H_batch)
    
    # Add small regularization to improve conditioning
    eps = 1e-6
    I_batch = torch.eye(4, device=device, dtype=H_batch.dtype).unsqueeze(0).expand(batch_size, -1, -1)
    HH_reg = HH + eps * I_batch
    
    # Compute enhanced ZF solution
    HH_inv = torch.linalg.inv(HH_reg)
    W_enhanced = torch.matmul(HH_inv, H_h)
    X_enhanced = torch.matmul(W_enhanced, y_batch)
    
    # Find closest codeword
    X_exp = X_enhanced.unsqueeze(1)
    all_codewords_exp = all_codewords.unsqueeze(0)
    distances = torch.sum(torch.abs(X_exp - all_codewords_exp)**2, dim=(2, 3))
    best_indices = torch.argmin(distances, dim=1)
    
    return best_indices

def adaptive_mmse_detection_biquaternion(y_batch, H_batch, all_codewords, noise_var):
    """Adaptive MMSE with different regularization than basic MMSE"""
    batch_size = y_batch.shape[0]
    device = y_batch.device
    
    # Convert noise_var to scalar
    if isinstance(noise_var, torch.Tensor):
        noise_var_val = noise_var.item()
    else:
        noise_var_val = float(noise_var)
    
    # Use different regularization strategy than basic MMSE
    H_h = H_batch.transpose(-2, -1).conj()
    HH = torch.matmul(H_h, H_batch)
    
    # Adaptive regularization - stronger than basic MMSE
    reg_factor = 2.0 * noise_var_val  # Double the regularization
    I_batch = torch.eye(4, device=device, dtype=H_batch.dtype).unsqueeze(0).expand(batch_size, -1, -1)
    HH_reg = HH + reg_factor * I_batch
    
    # Compute adaptive MMSE solution
    HH_reg_inv = torch.linalg.inv(HH_reg)
    X_adaptive = torch.matmul(torch.matmul(HH_reg_inv, H_h), y_batch)
    
    # Find closest codeword
    X_exp = X_adaptive.unsqueeze(1)
    all_codewords_exp = all_codewords.unsqueeze(0)
    distances = torch.sum(torch.abs(X_exp - all_codewords_exp)**2, dim=(2, 3))
    best_indices = torch.argmin(distances, dim=1)
    
    return best_indices

def hybrid_detection_biquaternion(y_batch, H_batch, all_codewords, noise_var):
    """Hybrid detector with intermediate regularization"""
    batch_size = y_batch.shape[0]
    device = y_batch.device
    
    # Convert noise_var to scalar
    if isinstance(noise_var, torch.Tensor):
        noise_var_val = noise_var.item()
    else:
        noise_var_val = float(noise_var)
    
    # Hybrid approach: regularization between MMSE and Adaptive MMSE
    H_h = H_batch.transpose(-2, -1).conj()
    HH = torch.matmul(H_h, H_batch)
    
    # Intermediate regularization - between basic MMSE (1x) and adaptive MMSE (2x)
    reg_factor = 1.5 * noise_var_val
    I_batch = torch.eye(4, device=device, dtype=H_batch.dtype).unsqueeze(0).expand(batch_size, -1, -1)
    HH_reg = HH + reg_factor * I_batch
    
    # Compute hybrid solution
    HH_reg_inv = torch.linalg.inv(HH_reg)
    X_hybrid = torch.matmul(torch.matmul(HH_reg_inv, H_h), y_batch)
    
    # Find closest codeword
    X_exp = X_hybrid.unsqueeze(1)
    all_codewords_exp = all_codewords.unsqueeze(0)
    distances = torch.sum(torch.abs(X_exp - all_codewords_exp)**2, dim=(2, 3))
    best_indices = torch.argmin(distances, dim=1)
    
    return best_indices
