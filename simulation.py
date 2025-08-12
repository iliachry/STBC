import torch
import numpy as np
from scipy.signal import savgol_filter
from biquaternion import BiquaternionSTBC, generate_all_codewords_biquaternion
from detection import ml_detection_biquaternion, mmse_detection_biquaternion, zf_detection_biquaternion


def _print_device(device):
    if device.type == 'cuda':
        print(f"[Device] CUDA ({torch.cuda.get_device_name(0)})")
    elif device.type == 'mps':
        print("[Device] MPS")
    else:
        print("[Device] CPU")


def simulate_ber_with_common_random_numbers(gamma_opt, gamma_std, snr_db_list, detector='ml', rate=2, num_trials=1000, device=None):
    """
    Common Random Numbers
    Use identical channel realizations and noise for both gamma values to reduce variance
    """
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    _print_device(device)
    
    # QPSK constellation
    QPSK = torch.tensor([1+1j, 1-1j, -1+1j, -1-1j], dtype=torch.complex64, device=device) / torch.sqrt(torch.tensor(2.0, device=device))
    BIT_LOOKUP = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.int8, device=device)
    
    stbc_opt = BiquaternionSTBC(gamma_opt, device)
    stbc_std = BiquaternionSTBC(gamma_std, device)
    all_codewords_opt, all_bits_opt = generate_all_codewords_biquaternion(stbc_opt, rate=rate)
    all_codewords_std, all_bits_std = generate_all_codewords_biquaternion(stbc_std, rate=rate)
    
    # Pre-generate all random seeds for reproducibility (stay within int32 bounds)
    max_int32 = np.iinfo(np.int32).max
    channel_seeds = np.random.randint(0, max_int32, num_trials)
    noise_seeds = np.random.randint(0, max_int32, (len(snr_db_list), num_trials))
    symbol_seeds = np.random.randint(0, max_int32, num_trials)
    
    ber_opt = np.zeros(len(snr_db_list))
    ber_std = np.zeros(len(snr_db_list))
    
    print(f"Using Common Random Numbers for {detector.upper()} detection...")
    
    for i, snr_db in enumerate(snr_db_list):
        snr_linear = 10**(snr_db / 10)
        total_errors_opt = 0
        total_errors_std = 0
        
        print(f"  SNR = {snr_db} dB... ({i+1}/{len(snr_db_list)})")
        
        progress_step = max(1, num_trials // 10)
        for trial in range(num_trials):
            # if (trial + 1) % progress_step == 0:
            #     print(f"    Trial {trial+1}/{num_trials}")
            # IDENTICAL channel realization for both gamma values
            torch.manual_seed(int(channel_seeds[trial]))
            H = (torch.randn(4, 4, dtype=torch.complex64, device=device) + 
                 1j * torch.randn(4, 4, dtype=torch.complex64, device=device)) / np.sqrt(2)
            
            # IDENTICAL symbol pattern for both gamma values
            torch.manual_seed(int(symbol_seeds[trial]))
            if rate == 1:
                indices = torch.randint(0, 4, (4,), device=device)
            else:
                indices = torch.randint(0, 4, (4,), device=device)
            symbols = QPSK[indices]
            bits = BIT_LOOKUP[indices].flatten()
            
            # IDENTICAL noise realization for both gamma values
            torch.manual_seed(int(noise_seeds[i, trial]))
            noise_var = 1 / snr_linear
            noise = (torch.randn(4, 4, dtype=torch.complex64, device=device) + 
                    1j * torch.randn(4, 4, dtype=torch.complex64, device=device)) * np.sqrt(noise_var / 2)
            
            # Pad symbols to 8 for rate=2 processing
            if rate == 2:
                symbols = torch.cat([symbols, symbols], dim=0)
            
            # Test both gamma values with identical conditions
            for gamma, stbc, all_codewords, all_bits, error_counter in [
                (gamma_opt, stbc_opt, all_codewords_opt, all_bits_opt, 'opt'),
                (gamma_std, stbc_std, all_codewords_std, all_bits_std, 'std')
            ]:
                q1, q2 = stbc.symbols_to_quaternions(symbols.unsqueeze(0), rate=rate)
                X = stbc.left_regular_representation(q1, q2).squeeze(0)
                y = H @ X + noise
                
                # Apply detector
                if detector == 'ml':
                    best_idx = ml_detection_biquaternion(y.unsqueeze(0), H.unsqueeze(0), all_codewords)[0]
                elif detector == 'mmse':
                    best_idx = mmse_detection_biquaternion(y.unsqueeze(0), H.unsqueeze(0), all_codewords, noise_var)[0]
                elif detector == 'zf':
                    best_idx = zf_detection_biquaternion(y.unsqueeze(0), H.unsqueeze(0), all_codewords)[0]
                
                rx_bits = all_bits[best_idx]
                
                # Ensure bit sizes match before comparison
                if bits.shape[0] != rx_bits.shape[0]:
                    min_bits = min(bits.shape[0], rx_bits.shape[0])
                    error_count = torch.sum(bits[:min_bits] != rx_bits[:min_bits]).item()
                else:
                    error_count = torch.sum(bits != rx_bits).item()
                
                if error_counter == 'opt':
                    total_errors_opt += error_count
                else:
                    total_errors_std += error_count
        
        ber_opt[i] = total_errors_opt / (num_trials * len(bits))
        print(f"    BER_opt: {ber_opt[i]:.6f}")
        ber_std[i] = total_errors_std / (num_trials * len(bits))
        print(f"    BER_std: {ber_std[i]:.6f}")
    
    return ber_opt, ber_std


def simulate_ber_for_gamma(gamma, snr_db_list, detector='ml', rate=2, num_trials=800, device=None):
    """Compute BER for a single gamma across an SNR list using the chosen detector."""
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    _print_device(device)

    # QPSK constellation
    QPSK = torch.tensor([1+1j, 1-1j, -1+1j, -1-1j], dtype=torch.complex64, device=device) / torch.sqrt(torch.tensor(2.0, device=device))
    BIT_LOOKUP = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.int8, device=device)

    stbc = BiquaternionSTBC(gamma, device)
    all_codewords, all_bits = generate_all_codewords_biquaternion(stbc, rate=rate)

    # Seeds within int32 bounds
    max_int32 = np.iinfo(np.int32).max
    channel_seeds = np.random.randint(0, max_int32, num_trials)
    noise_seeds = np.random.randint(0, max_int32, (len(snr_db_list), num_trials))
    symbol_seeds = np.random.randint(0, max_int32, num_trials)

    ber = np.zeros(len(snr_db_list))

    for i, snr_db in enumerate(snr_db_list):
        snr_linear = 10**(snr_db / 10)
        total_errors = 0
        progress_step = max(1, num_trials // 10)

        print(f"  [γ={gamma:.3f}] SNR {snr_db} dB ({i+1}/{len(snr_db_list)})")
        for trial in range(num_trials):
            # if (trial + 1) % progress_step == 0:
            #     print(f"    Trial {trial+1}/{num_trials}")
            torch.manual_seed(int(channel_seeds[trial]))
            H = (torch.randn(4, 4, dtype=torch.complex64, device=device) +
                 1j * torch.randn(4, 4, dtype=torch.complex64, device=device)) / np.sqrt(2)

            torch.manual_seed(int(symbol_seeds[trial]))
            indices = torch.randint(0, 4, (4,), device=device)
            symbols = QPSK[indices]
            bits = BIT_LOOKUP[indices].flatten()

            torch.manual_seed(int(noise_seeds[i, trial]))
            noise_var = 1 / snr_linear
            noise = (torch.randn(4, 4, dtype=torch.complex64, device=device) +
                     1j * torch.randn(4, 4, dtype=torch.complex64, device=device)) * np.sqrt(noise_var / 2)

            if rate == 2:
                symbols = torch.cat([symbols, symbols], dim=0)

            q1, q2 = stbc.symbols_to_quaternions(symbols.unsqueeze(0), rate=rate)
            X = stbc.left_regular_representation(q1, q2).squeeze(0)
            y = H @ X + noise

            if detector == 'ml':
                best_idx = ml_detection_biquaternion(y.unsqueeze(0), H.unsqueeze(0), all_codewords)[0]
            elif detector == 'mmse':
                best_idx = mmse_detection_biquaternion(y.unsqueeze(0), H.unsqueeze(0), all_codewords, noise_var)[0]
            elif detector == 'zf':
                best_idx = zf_detection_biquaternion(y.unsqueeze(0), H.unsqueeze(0), all_codewords)[0]

            rx_bits = all_bits[best_idx]
            if bits.shape[0] != rx_bits.shape[0]:
                min_bits = min(bits.shape[0], rx_bits.shape[0])
                error_count = torch.sum(bits[:min_bits] != rx_bits[:min_bits]).item()
            else:
                error_count = torch.sum(bits != rx_bits).item()

            total_errors += error_count

        ber[i] = total_errors / (num_trials * len(bits))
        print(f"    BER: {ber[i]:.6f}")

    return ber


def smooth_ber_curves(snr_db_list, ber_curves, method='savgol'):
    """
    Post-Processing Smoothing
    Apply smoothing to reduce statistical noise in final results
    """
    smoothed_curves = []
    
    for ber in ber_curves:
        if len(ber) < 5:
            smoothed_curves.append(ber)
            continue
            
        if method == 'savgol':
            window_length = min(5, len(ber) if len(ber) % 2 == 1 else len(ber)-1)
            if window_length >= 3:
                ber_smooth = savgol_filter(ber, window_length=window_length, polyorder=2)
            else:
                ber_smooth = ber
        elif method == 'moving_average':
            window = min(3, len(ber))
            ber_smooth = np.convolve(ber, np.ones(window)/window, mode='same')
        elif method == 'exponential':
            alpha = 0.3
            ber_smooth = np.zeros_like(ber)
            ber_smooth[0] = ber[0]
            for i in range(1, len(ber)):
                ber_smooth[i] = alpha * ber[i] + (1 - alpha) * ber_smooth[i-1]
        else:
            ber_smooth = ber
        
        ber_smooth = np.maximum(ber_smooth, 1e-6)
        smoothed_curves.append(ber_smooth)
    
    return smoothed_curves


def simulate_ber(gamma_opt, gamma_std, snr_db_list, detector='ml', rate=2, num_trials=800, device=None):
    """
    Combined Enhanced Simulation
    Combines all variance reduction techniques for maximum smoothness
    """
    print(f"Simulation for {detector.upper()} detection...")
    
    # Use common random numbers for comparison
    ber_opt_raw, ber_std_raw = simulate_ber_with_common_random_numbers(
        gamma_opt, gamma_std, snr_db_list, detector, rate, num_trials, device
    )
    
    # Apply post-processing smoothing
    # ber_opt_smooth, ber_std_smooth = smooth_ber_curves(
    #     snr_db_list, [ber_opt_raw, ber_std_raw], method='savgol'
    # )
    
    #return ber_opt_smooth, ber_std_smooth
    return ber_opt_raw, ber_std_raw
