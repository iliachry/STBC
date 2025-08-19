"""
Basic detection algorithms for STBC.
"""

import torch

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

def mmse_detection_biquaternion(y_batch, H_batch, all_codewords, noise_var):
    """True MMSE detection: first compute MMSE estimate, then find closest codeword"""
    batch_size = y_batch.shape[0]
    device = y_batch.device
    
    # This is the theoretically correct MMSE approach:
    # 1. Compute MMSE estimate: X_mmse = (H^H H + σ²I)^(-1) H^H y
    # 2. Find closest valid codeword to X_mmse
    
    # This is different from ML which directly minimizes ||y - HX||²
    
    H_h = H_batch.transpose(-2, -1).conj()
    HH = torch.matmul(H_h, H_batch)
    
    # Regularization based on noise level
    # This is the key difference from ZF - we always add noise variance
    I = torch.eye(4, device=device, dtype=H_batch.dtype).unsqueeze(0).expand(batch_size, -1, -1)
    HH_reg = HH + noise_var * I
    
    try:
        # Method 1: Direct inversion (more stable for well-conditioned matrices)
        HH_inv = torch.linalg.inv(HH_reg)
        X_mmse = torch.matmul(torch.matmul(HH_inv, H_h), y_batch)
    except:
        # Method 2: Cholesky decomposition (if direct inversion fails)
        try:
            L = torch.linalg.cholesky(HH_reg)
            H_h_y = torch.matmul(H_h, y_batch)
            z = torch.linalg.solve_triangular(L, H_h_y, upper=False)
            X_mmse = torch.linalg.solve_triangular(L.transpose(-2, -1).conj(), z, upper=True)
        except:
            # Fallback to pseudo-inverse
            X_mmse = torch.matmul(torch.linalg.pinv(H_batch), y_batch)
    
    # Now find closest codeword to MMSE estimate
    # This is where MMSE differs from ML - we're finding closest to X_mmse, not minimizing ||y - HX||²
    X_mmse_exp = X_mmse.unsqueeze(1)  # (batch, 1, 4, 4)
    all_codewords_exp = all_codewords.unsqueeze(0)  # (1, num_codewords, 4, 4)
    
    # Euclidean distance in codeword space
    distances = torch.sum(torch.abs(X_mmse_exp - all_codewords_exp)**2, dim=(2, 3))
    
    best_indices = torch.argmin(distances, dim=1)
    return best_indices

def zf_detection_biquaternion(y_batch, H_batch, all_codewords):
    """Pure Zero-Forcing detection without enhancements"""
    batch_size = y_batch.shape[0]
    device = y_batch.device
    
    try:
        # Simple pseudo-inverse without SVD thresholding
        # This is true ZF - it will amplify noise for ill-conditioned channels
        H_pinv_batch = torch.linalg.pinv(H_batch)
        
        # Direct ZF estimate: X = H^+ y
        X_zf_batch = torch.matmul(H_pinv_batch, y_batch)
        
        # Find closest codeword to ZF estimate
        # No power normalization - just direct distance
        X_zf_exp = X_zf_batch.unsqueeze(1)  # (batch, 1, 4, 4)
        all_codewords_exp = all_codewords.unsqueeze(0)  # (1, num_codewords, 4, 4)
        
        # Simple Euclidean distance
        distances = torch.sum(torch.abs(X_zf_exp - all_codewords_exp)**2, dim=(2, 3))
        best_indices = torch.argmin(distances, dim=1)
        
    except Exception as e:
        # Fallback to ML if inversion fails
        Y_candidates = torch.einsum('bij,kjl->bkil', H_batch, all_codewords)
        errors = y_batch.unsqueeze(1) - Y_candidates
        metrics = torch.sum(torch.abs(errors)**2, dim=(2, 3))
        best_indices = torch.argmin(metrics, dim=1)
    
    return best_indices
