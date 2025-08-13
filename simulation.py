import torch
import numpy as np
from scipy.signal import savgol_filter
from biquaternion import BiquaternionSTBC, generate_all_codewords_biquaternion
from detection import ml_detection_biquaternion, mmse_detection_biquaternion, zf_detection_biquaternion


def _select_device_if_none(device: torch.device | None) -> torch.device:
    if device is not None:
        return device
    return torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    )


def _get_qpsk_and_bit_lookup(device: torch.device):
    qpsk = (
        torch.tensor([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j], dtype=torch.complex64, device=device)
        / torch.sqrt(torch.tensor(2.0, device=device))
    )
    bit_lookup = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.int8, device=device)
    return qpsk, bit_lookup


def _build_stbc_and_codebook(gamma: complex, device: torch.device, rate: int):
    stbc = BiquaternionSTBC(gamma, device)
    all_codewords, all_bits = generate_all_codewords_biquaternion(stbc, rate=rate)
    return stbc, all_codewords, all_bits


def _apply_detector(detector: str, y, H, all_codewords, noise_var):
    if detector == "ml":
        return ml_detection_biquaternion(y, H, all_codewords)
    if detector == "mmse":
        return mmse_detection_biquaternion(y, H, all_codewords, noise_var)
    if detector == "zf":
        return zf_detection_biquaternion(y, H, all_codewords)
    raise ValueError(f"Unknown detector: {detector}")


def _count_bit_errors(tx_bits, rx_bits) -> int:
    if tx_bits.shape[0] != rx_bits.shape[0]:
        min_bits = min(tx_bits.shape[0], rx_bits.shape[0])
        return torch.sum(tx_bits[:min_bits] != rx_bits[:min_bits]).item()
    return torch.sum(tx_bits != rx_bits).item()


def simulate_ber_common(gammas, snr_db_list, detector="ml", rate=2, num_trials=800, device=None):
    """Simulate BER for one or more gammas using common random numbers.
    Returns a list of BER arrays, one per gamma, in the same order as `gammas`.
    """
    device = _select_device_if_none(device)
    qpsk, bit_lookup = _get_qpsk_and_bit_lookup(device)

    stbc_and_books = [
        _build_stbc_and_codebook(gamma, device, rate) for gamma in gammas
    ]

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
            torch.manual_seed(int(channel_seeds[trial]))
            H = (
                torch.randn(4, 4, dtype=torch.complex64, device=device)
                + 1j * torch.randn(4, 4, dtype=torch.complex64, device=device)
            ) / np.sqrt(2)

            torch.manual_seed(int(symbol_seeds[trial]))
            indices = torch.randint(0, 4, (4,), device=device)
            symbols = qpsk[indices]
            bits = bit_lookup[indices].flatten()

            torch.manual_seed(int(noise_seeds[snr_idx, trial]))
            noise_var = 1 / snr_linear
            noise = (
                torch.randn(4, 4, dtype=torch.complex64, device=device)
                + 1j * torch.randn(4, 4, dtype=torch.complex64, device=device)
            ) * np.sqrt(noise_var / 2)

            if rate == 2:
                symbols = torch.cat([symbols, symbols], dim=0)

            for g_idx, (stbc, all_codewords, all_bits) in enumerate(stbc_and_books):
                q1, q2 = stbc.symbols_to_quaternions(symbols.unsqueeze(0), rate=rate)
                X = stbc.left_regular_representation(q1, q2).squeeze(0)
                y = H @ X + noise

                best_idx = _apply_detector(
                    detector, y.unsqueeze(0), H.unsqueeze(0), all_codewords, noise_var
                )[0]
                rx_bits = all_bits[best_idx]
                total_errors[g_idx] += _count_bit_errors(bits, rx_bits)

        for g_idx in range(len(gammas)):
            ber_per_gamma[g_idx][snr_idx] = total_errors[g_idx] / (num_trials * len(bits))
            print(f"  BER for gamma {gammas[g_idx]}: {ber_per_gamma[g_idx][snr_idx]:.6f}")

    return ber_per_gamma


def simulate_ber_three(gamma_a, gamma_b, gamma_c, snr_db_list, detector='ml', rate=2, num_trials=800, device=None):
    """Compute BER for three gammas with common random numbers. Returns (ber_a, ber_b, ber_c)."""
    device = _select_device_if_none(device)
    ber_a, ber_b, ber_c = simulate_ber_common(
        [gamma_a, gamma_b, gamma_c], snr_db_list, detector=detector, rate=rate, num_trials=num_trials, device=device
    )
    return np.array(ber_a), np.array(ber_b), np.array(ber_c)


def simulate_ber_for_gamma(gamma, snr_db_list, detector='ml', rate=2, num_trials=800, device=None):
    """Compute BER for a single gamma across an SNR list using the chosen detector."""
    device = _select_device_if_none(device)

    ber_list = simulate_ber_common(
        [gamma], snr_db_list, detector=detector, rate=rate, num_trials=num_trials, device=device
    )[0]

    for i in range(len(snr_db_list)):
        print(f"    BER: {ber_list[i]:.6f}")

    return np.array(ber_list)


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
