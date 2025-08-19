"""
Basic detection algorithms for STBC.
"""

import torch
from .fast_quantization import fast_linear_detection

def ml_detection_biquaternion(y_batch, H_batch, all_codewords):
    """Fully vectorized ML detection - optimal performance"""
    # Compute all possible received signals: H @ X for all codewords
    Y_candidates = torch.einsum('bij,kjl->bkil', H_batch, all_codewords)
    
    # Compute error norms for all combinations at once
    errors = y_batch.unsqueeze(1) - Y_candidates
    metrics = torch.sum(torch.abs(errors)**2, dim=(2, 3))
    
    # Find best codeword for each batch element
    best_indices = torch.argmin(metrics, dim=1)
    return best_indices

def mmse_detection_biquaternion(y_batch, H_batch, all_codewords, noise_var, stbc=None, rate=2):
    """Fast MMSE detection with symbol quantization"""
    batch_size = y_batch.shape[0]
    device = y_batch.device
    
    # Compute MMSE estimate: X_mmse = (H^H H + σ²I)^(-1) H^H y
    H_h = H_batch.transpose(-2, -1).conj()
    HH = torch.matmul(H_h, H_batch)
    
    # Convert noise_var to scalar for consistent handling
    if isinstance(noise_var, torch.Tensor):
        noise_var_val = noise_var.item()
    else:
        noise_var_val = float(noise_var)
    
    # Fixed regularization - use same value as enhanced detectors work
    I = torch.eye(4, device=device, dtype=H_batch.dtype).unsqueeze(0).expand(batch_size, -1, -1)
    HH_reg = HH + noise_var_val * I
    
    try:
        # Direct inversion (most stable)
        HH_inv = torch.linalg.inv(HH_reg)
        X_mmse = torch.matmul(torch.matmul(HH_inv, H_h), y_batch)
    except:
        # Fallback methods
        try:
            L = torch.linalg.cholesky(HH_reg)
            H_h_y = torch.matmul(H_h, y_batch)
            z = torch.linalg.solve_triangular(L, H_h_y, upper=False)
            X_mmse = torch.linalg.solve_triangular(L.transpose(-2, -1).conj(), z, upper=True)
        except:
            X_mmse = torch.matmul(torch.linalg.pinv(H_batch), y_batch)
    
    # Fast quantization may not be optimal for noisy MMSE estimates
    # Use exhaustive search for better accuracy
    # if stbc is not None:
    #     try:
    #         best_indices = fast_linear_detection(X_mmse, stbc.gamma, rate)
    #         return best_indices
    #     except Exception as e:
    #         print(f"Warning: Fast quantization failed in MMSE ({e}), using exhaustive search")
    
    # Fallback: exhaustive search (slow but accurate)
    X_mmse_exp = X_mmse.unsqueeze(1)
    all_codewords_exp = all_codewords.unsqueeze(0)
    distances = torch.sum(torch.abs(X_mmse_exp - all_codewords_exp)**2, dim=(2, 3))
    best_indices = torch.argmin(distances, dim=1)
    return best_indices

def zf_detection_biquaternion(y_batch, H_batch, all_codewords, stbc=None, rate=2):
    """Fast Zero-Forcing detection with symbol quantization"""
    batch_size = y_batch.shape[0]
    device = y_batch.device
    
    try:
        # Compute ZF estimate: X = H^+ y
        H_pinv_batch = torch.linalg.pinv(H_batch)
        X_zf_batch = torch.matmul(H_pinv_batch, y_batch)
        
        # Fast quantization - now fixed and working
        if stbc is not None:
            try:
                best_indices = fast_linear_detection(X_zf_batch, stbc.gamma, rate)
                return best_indices
            except Exception as e:
                print(f"Warning: Fast quantization failed in ZF ({e}), using exhaustive search")
        
        # Fallback: exhaustive search (slow but accurate)
        X_zf_exp = X_zf_batch.unsqueeze(1)
        all_codewords_exp = all_codewords.unsqueeze(0)
        distances = torch.sum(torch.abs(X_zf_exp - all_codewords_exp)**2, dim=(2, 3))
        best_indices = torch.argmin(distances, dim=1)
        
    except Exception as e:
        # Last resort: fallback to ML
        Y_candidates = torch.einsum('bij,kjl->bkil', H_batch, all_codewords)
        errors = y_batch.unsqueeze(1) - Y_candidates
        metrics = torch.sum(torch.abs(errors)**2, dim=(2, 3))
        best_indices = torch.argmin(metrics, dim=1)
    
    return best_indices
