import torch

def ml_detection_biquaternion(y_batch, H_batch, all_codewords):
    """Maximum Likelihood detection (optimal)"""
    Y_candidates = torch.einsum('bij,kjl->bkil', H_batch, all_codewords)
    errors = y_batch.unsqueeze(1) - Y_candidates
    metrics = torch.sum(torch.abs(errors)**2, dim=(2, 3))
    best_indices = torch.argmin(metrics, dim=1)
    return best_indices

def mmse_detection_biquaternion(y_batch, H_batch, all_codewords, noise_var):
    """MMSE Linear Detection (practical compromise)"""
    batch_size = y_batch.shape[0]
    device = y_batch.device
    best_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
    
    # QPSK constellation
    QPSK = torch.tensor([1+1j, 1-1j, -1+1j, -1-1j], dtype=torch.complex64, device=device) / torch.sqrt(torch.tensor(2.0, device=device))
    
    for b in range(batch_size):
        try:
            H = H_batch[b]
            y = y_batch[b]
            
            # MMSE solution: (H^H * H + σ²I)^(-1) * H^H * y
            H_H = H.conj().transpose(-2, -1)
            reg_matrix = H_H @ H + noise_var * torch.eye(4, device=device)
            
            y_flat = y.flatten()
            H_H_y = H_H @ y_flat.view(-1, 1)
            x_mmse = torch.linalg.solve(reg_matrix, H_H_y.squeeze())
            
            # Map to nearest QPSK symbols
            x_qpsk = torch.zeros(4, dtype=torch.complex64, device=device)
            for i in range(4):
                distances = torch.abs(x_mmse[i] - QPSK)
                x_qpsk[i] = QPSK[torch.argmin(distances)]
            
            # Find closest codeword
            min_dist = float('inf')
            best_idx = 0
            for c in range(len(all_codewords)):
                codeword_symbols = torch.stack([
                    all_codewords[c, 0, 0], all_codewords[c, 0, 1],
                    all_codewords[c, 0, 2], all_codewords[c, 0, 3]
                ])
                dist = torch.sum(torch.abs(x_qpsk - codeword_symbols)**2).item()
                if dist < min_dist:
                    min_dist = dist
                    best_idx = c
            best_indices[b] = best_idx
            
        except:
            # Fallback to ML
            y_single = y_batch[b:b+1]
            H_single = H_batch[b:b+1]
            best_indices[b] = ml_detection_biquaternion(y_single, H_single, all_codewords)[0]
    
    return best_indices

def zf_detection_biquaternion(y_batch, H_batch, all_codewords):
    """Zero-Forcing Linear Detection (simple but suboptimal)"""
    batch_size = y_batch.shape[0]
    device = y_batch.device
    best_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
    
    for b in range(batch_size):
        try:
            H = H_batch[b]
            H_H = H.conj().transpose(-2, -1)
            normal_matrix = H_H @ H
            
            # Small regularization for numerical stability
            reg_term = 1e-6 * torch.eye(4, device=device)
            normal_matrix += reg_term
            
            y_flat = y_batch[b].flatten()
            H_H_y = H_H @ y_flat.view(-1, 1)
            x_hat = torch.linalg.solve(normal_matrix, H_H_y.squeeze())
            
            # Find closest codeword by brute force
            min_dist = float('inf')
            best_idx = 0
            for c in range(len(all_codewords)):
                X_reconstructed = H @ all_codewords[c]
                dist = torch.sum(torch.abs(y_batch[b] - X_reconstructed)**2).item()
                if dist < min_dist:
                    min_dist = dist
                    best_idx = c
            best_indices[b] = best_idx
            
        except:
            # Fallback to ML
            y_single = y_batch[b:b+1]
            H_single = H_batch[b:b+1]
            best_indices[b] = ml_detection_biquaternion(y_single, H_single, all_codewords)[0]
    
    return best_indices
