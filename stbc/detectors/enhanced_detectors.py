"""
Enhanced detection algorithms for STBC.
"""

import torch

def adaptive_reg_factor(noise_var):
    """Adaptive regularization based on noise level - optimized for STBC"""
    snr_linear = 1 / noise_var
    # More aggressive regularization for STBC due to 4x4 matrix structure
    if snr_linear > 100:  # High SNR (> 20 dB)
        return 0.1   # Still need some regularization for 4x4 matrices
    elif snr_linear > 10:  # Medium SNR (10-20 dB)
        return 0.5   # Moderate regularization
    elif snr_linear > 1:   # Low-Medium SNR (0-10 dB)
        return 1.0   # Strong regularization
    else:  # Very Low SNR (< 0 dB)
        return 2.0   # Very strong regularization
        
def regularized_zf_detection_biquaternion(y_batch, H_batch, all_codewords, noise_var):
    """Optimized Regularized ZF with full batch processing"""
    batch_size = y_batch.shape[0]
    device = y_batch.device
    
    # Adaptive regularization
    reg_factor = adaptive_reg_factor(noise_var)
        
    try:
        # Batch computation of H^H @ H
        H_h = H_batch.transpose(-2, -1).conj()
        HH = torch.matmul(H_h, H_batch)
        
        # Batch Frobenius norm
        H_norm = torch.norm(H_batch.view(batch_size, -1), p=2, dim=1, keepdim=True).unsqueeze(-1)
        lambda_reg = reg_factor * noise_var * (H_norm ** 2) / 4.0
        
        # Add regularization
        I_batch = torch.eye(4, device=device, dtype=H_batch.dtype).unsqueeze(0).expand(batch_size, -1, -1)
        HH_reg = HH + lambda_reg * I_batch
        
        # Try batch Cholesky decomposition
        try:
            L = torch.linalg.cholesky(HH_reg)
            # Batch triangular solve
            H_h_y = torch.matmul(H_h, y_batch)
            z = torch.linalg.solve_triangular(L, H_h_y, upper=False)
            X_zf_batch = torch.linalg.solve_triangular(L.transpose(-2, -1).conj(), z, upper=True)
        except:
            # Fallback to batch inversion
            HH_reg_inv = torch.linalg.inv(HH_reg)
            X_zf_batch = torch.matmul(torch.matmul(HH_reg_inv, H_h), y_batch)
        
        # Vectorized power normalization
        X_zf_power = torch.sum(torch.abs(X_zf_batch)**2, dim=(1, 2), keepdim=True)
        X_zf_power = torch.clamp(X_zf_power, min=1e-10)
        X_zf_normalized = X_zf_batch * torch.sqrt(4.0 / X_zf_power)
        
        # Adaptive weighting
        weight = torch.clamp(torch.tensor(noise_var * 10, device=device), 0, 1)
        
        # Fully vectorized distance computation
        X_zf_norm_exp = X_zf_normalized.unsqueeze(1)
        X_zf_exp = X_zf_batch.unsqueeze(1)
        all_codewords_exp = all_codewords.unsqueeze(0)
        
        error_norm = X_zf_norm_exp - all_codewords_exp
        error_unnorm = X_zf_exp - all_codewords_exp
        
        metric_norm = torch.sum(torch.abs(error_norm)**2, dim=(2, 3))
        metric_unnorm = torch.sum(torch.abs(error_unnorm)**2, dim=(2, 3))
        
        # Weighted combination
        metrics = weight * metric_norm + (1 - weight) * metric_unnorm
        best_indices = torch.argmin(metrics, dim=1)
        
    except Exception as e:
        # Fallback to ML
        Y_candidates = torch.einsum('bij,kjl->bkil', H_batch, all_codewords)
        errors = y_batch.unsqueeze(1) - Y_candidates
        metrics = torch.sum(torch.abs(errors)**2, dim=(2, 3))
        best_indices = torch.argmin(metrics, dim=1)
    
    return best_indices
    
def ml_enhanced_zf_detection_biquaternion(y_batch, H_batch, all_codewords):
    """ML-Enhanced Zero-Forcing: ZF with improvements that achieve near-ML performance"""
    batch_size = y_batch.shape[0]
    num_codewords = all_codewords.shape[0]
    device = y_batch.device
    
    try:
        # Enhanced ZF with SVD and thresholding
        U, S, Vh = torch.linalg.svd(H_batch, full_matrices=False)
        
        # Adaptive singular value thresholding
        s_threshold = 0.1 * torch.max(S, dim=1, keepdim=True)[0]
        S_inv = torch.zeros_like(S)
        mask = S > s_threshold
        S_inv[mask] = 1.0 / S[mask]
        
        # Robust pseudo-inverse
        S_inv_diag = torch.diag_embed(S_inv)
        H_pinv_batch = torch.matmul(torch.matmul(Vh.transpose(-2, -1).conj(), S_inv_diag),
                                    U.transpose(-2, -1).conj())
        
        # Enhanced ZF estimate
        X_zf_batch = torch.matmul(H_pinv_batch, y_batch)
        
        # Power normalization (exploiting STBC structure)
        X_zf_power = torch.sum(torch.abs(X_zf_batch)**2, dim=(1, 2), keepdim=True)
        X_zf_power = torch.clamp(X_zf_power, min=1e-10)
        X_zf_normalized = X_zf_batch * torch.sqrt(4.0 / X_zf_power)
        
        # Dual metric approach
        X_zf_norm_exp = X_zf_normalized.unsqueeze(1)
        X_zf_exp = X_zf_batch.unsqueeze(1)
        all_codewords_exp = all_codewords.unsqueeze(0)
        
        error_norm = X_zf_norm_exp - all_codewords_exp
        error_unnorm = X_zf_exp - all_codewords_exp
        
        metric_norm = torch.sum(torch.abs(error_norm)**2, dim=(2, 3))
        metric_unnorm = torch.sum(torch.abs(error_unnorm)**2, dim=(2, 3))
        
        # Take minimum of both metrics
        metrics = torch.minimum(metric_norm, metric_unnorm)
        best_indices = torch.argmin(metrics, dim=1)
        
    except Exception as e:
        # Fallback to ML
        Y_candidates = torch.einsum('bij,kjl->bkil', H_batch, all_codewords)
        errors = y_batch.unsqueeze(1) - Y_candidates
        metrics = torch.sum(torch.abs(errors)**2, dim=(2, 3))
        best_indices = torch.argmin(metrics, dim=1)
    
    return best_indices

def adaptive_mmse_detection_biquaternion(y_batch, H_batch, all_codewords, noise_var):
    """Adaptive MMSE detection that adjusts regularization based on channel condition"""
    batch_size = y_batch.shape[0]
    device = y_batch.device
    
    # This is an enhanced MMSE that adapts regularization based on channel condition
    results = []
    
    for b in range(batch_size):
        # Process each batch element separately to adapt to its channel condition
        y = y_batch[b]
        H = H_batch[b]
        
        # Estimate channel condition number
        try:
            U, S, _ = torch.linalg.svd(H)
            condition_number = S[0] / S[-1]
        except:
            # Default to high condition number if SVD fails
            condition_number = torch.tensor(100.0, device=device)
        
        # Adjust regularization based on condition number
        H_h = H.transpose(-2, -1).conj()
        HH = torch.matmul(H_h, H)
        I = torch.eye(4, device=device, dtype=H.dtype)
        
        if condition_number > 30:  # Ill-conditioned
            # Strong regularization for ill-conditioned channel
            reg_factor = 3.0 * noise_var
        elif condition_number > 10:  # Moderately ill-conditioned
            # Moderate regularization
            reg_factor = 1.0 * noise_var
        else:  # Well-conditioned
            # Light regularization
            reg_factor = 0.1 * noise_var
            
        HH_reg = HH + reg_factor * I
        
        try:
            # Try most stable method first (Cholesky)
            L = torch.linalg.cholesky(HH_reg)
            H_h_y = torch.matmul(H_h, y)
            z = torch.linalg.solve_triangular(L, H_h_y, upper=False)
            X_mmse = torch.linalg.solve_triangular(L.transpose(-2, -1).conj(), z, upper=True)
        except:
            try:
                # Fallback to direct inverse
                HH_inv = torch.linalg.inv(HH_reg)
                X_mmse = torch.matmul(torch.matmul(HH_inv, H_h), y)
            except:
                # Last resort: pseudo-inverse
                X_mmse = torch.matmul(torch.linalg.pinv(H), y)
        
        # Find closest codeword
        X_mmse_exp = X_mmse.unsqueeze(0)  # (1, 4, 4)
        distances = torch.sum(torch.abs(X_mmse_exp - all_codewords)**2, dim=(1, 2))
        best_index = torch.argmin(distances)
        
        results.append(best_index)
    
    return torch.stack(results)

def hybrid_detection_biquaternion(y_batch, H_batch, all_codewords, noise_var):
    """Hybrid detector that switches between ML and ML-enhanced ZF based on channel conditions"""
    batch_size = y_batch.shape[0]
    device = y_batch.device
    results = []
    
    # Pre-compute Y_candidates for ML once (reused for multiple batches)
    Y_candidates = torch.einsum('bij,kjl->bkil', H_batch, all_codewords)
    
    for b in range(batch_size):
        # Estimate channel condition number
        try:
            _, S, _ = torch.linalg.svd(H_batch[b])
            condition_number = S[0] / S[-1]
        except:
            condition_number = torch.tensor(100.0, device=device)
        
        # For well-conditioned channels, use enhanced ZF (faster)
        # For ill-conditioned channels, use ML (more accurate)
        if condition_number < 10.0:  # Well-conditioned
            best_index = ml_enhanced_zf_detection_biquaternion(
                y_batch[b:b+1], H_batch[b:b+1], all_codewords
            )[0]
        else:  # Ill-conditioned
            errors = y_batch[b:b+1].unsqueeze(1) - Y_candidates[b:b+1]
            metrics = torch.sum(torch.abs(errors)**2, dim=(2, 3))
            best_index = torch.argmin(metrics[0])
            
        results.append(best_index)
    
    return torch.stack(results)
