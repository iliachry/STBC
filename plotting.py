import matplotlib.pyplot as plt
import numpy as np

def plot_detection_results(snr_db_list, ber_opt, ber_std, ber_poor, gamma_opt, gamma_std, gamma_poor, detector_name, save_filename):
    """Plot BER results for a specific detector"""
    plt.figure(figsize=(10, 8))
    plt.semilogy(snr_db_list, ber_opt, 'b-o', linewidth=3, markersize=8, 
                 label=f'Optimized (γ={gamma_opt:.2f})', markerfacecolor='white', markeredgewidth=2)
    plt.semilogy(snr_db_list, ber_std, 'r--s', linewidth=3, markersize=8, 
                 label='Standard (γ=1+j)', markerfacecolor='white', markeredgewidth=2)
    plt.semilogy(snr_db_list, ber_poor, 'm-.^', linewidth=3, markersize=8, 
                 label=f'Poor (γ={gamma_poor:.1f})', markerfacecolor='white', markeredgewidth=2)
    
    plt.xlabel('SNR (dB)', fontsize=14, fontweight='bold')
    plt.ylabel('Bit Error Rate (BER)', fontsize=14, fontweight='bold')
    plt.title(f'{detector_name} Detection: True Biquaternion STBC\n', 
              fontsize=16, fontweight='bold')
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(fontsize=13)
    plt.ylim(1e-5, 1.0)
    plt.xlim(0, max(snr_db_list))
    
    # Add enhancement annotations
    if detector_name == 'ML':
        plt.text(0.02, 0.02, 
                 'Variance Reduction Applied:\n' +
                 '✓ Common Random Numbers\n' +
                 '✓ Golden Ratio γ Values\n' +
                 '✓ Savitzky-Golay Smoothing\n' +
                 '✓ Enhanced Statistical Accuracy',
                 transform=plt.gca().transAxes, fontsize=11,
                 verticalalignment='bottom', bbox=dict(boxstyle="round,pad=0.4", facecolor='lightgreen', alpha=0.8))
    
    # Add annotations for optimization benefits
    if detector_name == 'MMSE' and len(snr_db_list) > 4:
        mid_snr = snr_db_list[len(snr_db_list)//2]
        mid_idx = len(snr_db_list)//2
        if ber_std[mid_idx] > ber_opt[mid_idx]:
            plt.annotate('Clear Optimization\nBenefit Visible', 
                        xy=(mid_snr, ber_std[mid_idx]), 
                        xytext=(mid_snr + 3, ber_std[mid_idx] * 3),
                        arrowprops=dict(arrowstyle='->', color='darkgreen', alpha=0.9, lw=3),
                        fontsize=12, ha='center', fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.4", facecolor='lightgreen', alpha=0.8))
    
    if detector_name == 'ZF' and len(snr_db_list) > 3:
        high_snr = snr_db_list[-4]
        high_idx = -4
        if ber_std[high_idx] > ber_opt[high_idx]:
            plt.annotate('Maximum\nOptimization\nBenefit', 
                        xy=(high_snr, ber_opt[high_idx]), 
                        xytext=(high_snr - 4, ber_opt[high_idx] * 0.1),
                        arrowprops=dict(arrowstyle='->', color='red', alpha=0.9, lw=3),
                        fontsize=12, ha='center', fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.4", facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_filename, format='png', dpi=300, bbox_inches='tight')
    plt.show()

def print_performance_analysis(snr_db_list, ber_opt_ml, ber_std_ml, ber_opt_mmse, ber_std_mmse, ber_opt_zf, ber_std_zf):
    """Print detailed performance analysis"""
    print(f"\n{'='*80}")
    print("ANALYSIS: Detection Complexity vs Optimization Sensitivity")
    print(f"{'='*80}")

    for snr_val in snr_db_list:
        if snr_val < len(snr_db_list):
            idx = snr_val // 2  # Adjust for SNR step size
            print(f"\nAt {snr_db_list[idx]} dB SNR:")
            
            print(f"  ML Detection:")
            print(f"    Optimized:    {ber_opt_ml[idx]:.3e}")
            print(f"    Standard:     {ber_std_ml[idx]:.3e}")
            if ber_std_ml[idx] > 0:
                ml_improvement = (ber_std_ml[idx] - ber_opt_ml[idx]) / ber_std_ml[idx] * 100
                print(f"    → Gain:       {ml_improvement:.1f}%")
            
            print(f"  MMSE Detection:")
            print(f"    Optimized:    {ber_opt_mmse[idx]:.3e}")
            print(f"    Standard:     {ber_std_mmse[idx]:.3e}")
            if ber_std_mmse[idx] > 0:
                mmse_improvement = (ber_std_mmse[idx] - ber_opt_mmse[idx]) / ber_std_mmse[idx] * 100
                print(f"    → Gain:       {mmse_improvement:.1f}%")
            
            print(f"  ZF Detection:")
            print(f"    Optimized:    {ber_opt_zf[idx]:.3e}")
            print(f"    Standard:     {ber_std_zf[idx]:.3e}")
            if ber_std_zf[idx] > 0:
                zf_improvement = (ber_std_zf[idx] - ber_opt_zf[idx]) / ber_std_zf[idx] * 100
                print(f"    → Gain:       {zf_improvement:.1f}%")
