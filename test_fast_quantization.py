#!/usr/bin/env python3
"""
Test script to validate the fixed fast quantization algorithm.
"""

import torch
import time
import numpy as np
from stbc.core.biquaternion import BiquaternionSTBC
from stbc.core.codewords import get_cached_codewords
from stbc.detectors.fast_quantization import fast_linear_detection, verify_fast_quantization_accuracy

def test_fast_quantization_accuracy():
    """Test fast quantization accuracy against exhaustive search"""
    print("=" * 60)
    print("Testing Fast Quantization Accuracy")
    print("=" * 60)
    
    device = torch.device('cpu')
    gamma = 0.618 + 1.0j
    rate = 2
    
    # Create STBC and get codewords
    stbc = BiquaternionSTBC(gamma, device)
    all_codewords, all_bits = get_cached_codewords(stbc, rate)
    
    print(f"STBC Parameters:")
    print(f"  Gamma: {gamma}")
    print(f"  Rate: {rate}")
    print(f"  Total codewords: {all_codewords.shape[0]}")
    print(f"  Codeword shape: {all_codewords.shape[1:]}")
    
    # Test on first 50 codewords for comprehensive validation
    num_test = min(50, all_codewords.shape[0])
    test_codewords = all_codewords[:num_test]
    expected_indices = torch.arange(num_test, device=device)
    
    print(f"\nTesting on first {num_test} codewords...")
    
    # Apply fast quantization
    try:
        fast_indices = fast_linear_detection(test_codewords, gamma, rate)
        print(f"Fast quantization completed successfully")
        print(f"Expected indices: {expected_indices[:10].tolist()}...")
        print(f"Fast indices:     {fast_indices[:10].tolist()}...")
        
        # Check accuracy
        correct = (fast_indices == expected_indices).sum().item()
        accuracy = correct / num_test
        
        print(f"\nAccuracy Results:")
        print(f"  Correct predictions: {correct}/{num_test}")
        print(f"  Accuracy: {accuracy:.2%}")
        
        if accuracy == 1.0:
            print("✅ PERFECT ACCURACY - Fast quantization is working correctly!")
        elif accuracy > 0.9:
            print("⚠️  High accuracy but not perfect - minor issues remain")
        else:
            print("❌ Low accuracy - significant issues in algorithm")
            
        return accuracy
        
    except Exception as e:
        print(f"❌ Fast quantization failed with error: {e}")
        return 0.0

def test_fast_quantization_speed():
    """Test fast quantization speed"""
    print("\n" + "=" * 60)
    print("Testing Fast Quantization Speed")
    print("=" * 60)
    
    device = torch.device('cpu')
    gamma = 0.618 + 1.0j
    rate = 2
    batch_size = 100
    
    # Create STBC and random test data
    stbc = BiquaternionSTBC(gamma, device)
    all_codewords, _ = get_cached_codewords(stbc, rate)
    
    # Generate random STBC matrices for speed test
    test_matrices = torch.randn(batch_size, 4, 4, dtype=torch.complex64, device=device)
    
    print(f"Speed test parameters:")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of runs: 100")
    
    # Time fast quantization
    num_runs = 100
    fast_times = []
    
    for _ in range(num_runs):
        start_time = time.time()
        _ = fast_linear_detection(test_matrices, gamma, rate)
        end_time = time.time()
        fast_times.append(end_time - start_time)
    
    fast_times = np.array(fast_times)
    fast_mean = fast_times.mean()
    fast_std = fast_times.std()
    
    # Time exhaustive search for comparison
    exhaustive_times = []
    
    for _ in range(10):  # Fewer runs since it's slower
        start_time = time.time()
        
        # Simulate exhaustive search
        X_exp = test_matrices.unsqueeze(1)  # [batch, 1, 4, 4]
        all_codewords_exp = all_codewords.unsqueeze(0)  # [1, N, 4, 4]
        distances = torch.sum(torch.abs(X_exp - all_codewords_exp)**2, dim=(2, 3))
        _ = torch.argmin(distances, dim=1)
        
        end_time = time.time()
        exhaustive_times.append(end_time - start_time)
    
    exhaustive_times = np.array(exhaustive_times)
    exhaustive_mean = exhaustive_times.mean()
    
    speedup = exhaustive_mean / fast_mean
    
    print(f"\nSpeed Results:")
    print(f"  Fast quantization: {fast_mean*1000:.2f} ± {fast_std*1000:.2f} ms")
    print(f"  Exhaustive search: {exhaustive_mean*1000:.2f} ms")
    print(f"  Speedup: {speedup:.1f}x faster")
    
    if speedup > 5:
        print("✅ Excellent speedup achieved!")
    elif speedup > 2:
        print("✅ Good speedup achieved!")
    else:
        print("⚠️  Limited speedup - algorithm may need optimization")
    
    return speedup

def test_detector_performance():
    """Test detector performance with fixed fast quantization"""
    print("\n" + "=" * 60)
    print("Testing Detector Performance")
    print("=" * 60)
    
    from stbc.simulation.simulator import simulate_ber_all_detectors
    
    device = torch.device('cpu')
    gamma = 0.618 + 1.0j
    snr_db_list = [10]  # Single SNR point for quick test
    num_trials = 100
    
    print(f"Running quick detector test:")
    print(f"  SNR: {snr_db_list[0]} dB")
    print(f"  Trials: {num_trials}")
    
    try:
        results = simulate_ber_all_detectors(
            [gamma], snr_db_list, rate=2, num_trials=num_trials, device=device
        )
        
        # Unpack results
        results_ml, results_mmse, results_zf, results_zf_reg, all_results_dict, timing_results = results
        
        print(f"\nBER Results at {snr_db_list[0]} dB:")
        for detector in ['ml', 'mmse', 'zf', 'zf_reg']:
            if detector in all_results_dict:
                ber = all_results_dict[detector][0][0]  # First gamma, first SNR
                time_taken = timing_results[detector][0][0]
                print(f"  {detector.upper()}: BER = {ber:.4f}, Time = {time_taken:.3f}s")
        
        # Check if ZF_REG is working (should have reasonable BER, not 80%+)
        zf_reg_ber = all_results_dict['zf_reg'][0][0]
        if zf_reg_ber < 0.1:  # Less than 10% BER is reasonable
            print("✅ ZF-Reg detector is working correctly!")
            return True
        else:
            print(f"❌ ZF-Reg detector still has high BER: {zf_reg_ber:.1%}")
            return False
            
    except Exception as e:
        print(f"❌ Detector test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Fast Quantization Algorithm Validation")
    print("=" * 60)
    
    # Test accuracy
    accuracy = test_fast_quantization_accuracy()
    
    # Test speed only if accuracy is reasonable
    if accuracy > 0.5:
        speedup = test_fast_quantization_speed()
    else:
        speedup = 0
        print("⚠️  Skipping speed test due to low accuracy")
    
    # Test detector performance only if algorithm is working
    if accuracy > 0.8:
        detector_working = test_detector_performance()
    else:
        detector_working = False
        print("⚠️  Skipping detector test due to low accuracy")
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Accuracy: {accuracy:.1%}")
    if speedup > 0:
        print(f"Speedup: {speedup:.1f}x")
    print(f"Detector working: {'✅ Yes' if detector_working else '❌ No'}")
    
    if accuracy == 1.0 and speedup > 2 and detector_working:
        print("\n🎉 SUCCESS: Fast quantization algorithm is fully fixed!")
    elif accuracy > 0.9 and detector_working:
        print("\n✅ GOOD: Fast quantization is mostly working")
    else:
        print("\n❌ ISSUES: Fast quantization needs more work")

if __name__ == "__main__":
    main()