"""
Plotting functions for visualizing STBC simulation results.
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_detection_results(snr_db_list, ber_opt, ber_std, ber_poor, gamma_opt, gamma_std, gamma_poor, 
                          detector_name, save_filename, results_dir=None):
    """
    Plot BER vs SNR for a single detector with different gamma values.
    
    Args:
        snr_db_list: List of SNR values in dB
        ber_opt: BER results for optimized gamma
        ber_std: BER results for standard gamma
        ber_poor: BER results for poor gamma
        gamma_opt: Optimized gamma value
        gamma_std: Standard gamma value
        gamma_poor: Poor gamma value
        detector_name: Name of detector
        save_filename: Filename to save plot
        results_dir: Optional directory to save results
    """
    plt.figure(figsize=(10, 8))
    plt.semilogy(snr_db_list, ber_opt, 'b-o', linewidth=3, markersize=8, label=f'Optimized (γ={gamma_opt:.2f})', markerfacecolor='white', markeredgewidth=2)
    plt.semilogy(snr_db_list, ber_std, 'r--s', linewidth=3, markersize=8, label=f'Standard (γ={gamma_std:.2f})', markerfacecolor='white', markeredgewidth=2)
    plt.semilogy(snr_db_list, ber_poor, 'm-.^', linewidth=3, markersize=8, label=f'Poor (γ={gamma_poor:.1f})', markerfacecolor='white', markeredgewidth=2)
    plt.xlabel('SNR (dB)', fontsize=14, fontweight='bold')
    plt.ylabel('Bit Error Rate (BER)', fontsize=14, fontweight='bold')
    plt.title(f'{detector_name} Detection', fontsize=16, fontweight='bold')
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(fontsize=13)
    plt.ylim(1e-5, 1.0)
    plt.xlim(0, max(snr_db_list))
    plt.tight_layout()
    
    # Save to results directory if specified
    if results_dir is not None:
        output_path = results_dir / save_filename
    else:
        output_path = save_filename
        
    plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_all_detectors_comparison(snr_db_list, all_results_dict, gamma_opt, gamma_std, gamma_poor, 
                                 save_filename='all_detectors_comparison.png', results_dir=None):
    """
    Plot comparison of all detectors including enhanced versions.
    
    Args:
        snr_db_list: List of SNR values in dB
        all_results_dict: Dictionary containing results for all detectors
        gamma_opt: Optimized gamma value
        gamma_std: Standard gamma value
        gamma_poor: Poor gamma value
        save_filename: Filename to save plot
        results_dir: Optional directory to save results
    """
    plt.figure(figsize=(14, 10))
    
    # Define colors and markers for each detector
    detector_styles = {
        'ml': {'color': 'b', 'marker': 'o', 'linestyle': '-', 'label': 'ML'},
        'mmse': {'color': 'g', 'marker': 's', 'linestyle': '-', 'label': 'MMSE'},
        'zf': {'color': 'r', 'marker': '^', 'linestyle': '-', 'label': 'ZF'},
        'zf_reg': {'color': 'm', 'marker': 'd', 'linestyle': '-', 'label': 'ZF-Reg'},
        'ml_zf': {'color': 'c', 'marker': 'v', 'linestyle': '--', 'label': 'ML-Enhanced ZF'},
        'adaptive_mmse': {'color': 'orange', 'marker': 'p', 'linestyle': '--', 'label': 'Adaptive MMSE'},
        'hybrid': {'color': 'brown', 'marker': 'h', 'linestyle': '--', 'label': 'Hybrid'}
    }
    
    # Plot each detector (optimized gamma only)
    for det_name, style in detector_styles.items():
        if det_name in all_results_dict:
            ber_data = all_results_dict[det_name][0]  # Optimized gamma
            plt.semilogy(
                snr_db_list, 
                ber_data, 
                color=style['color'], 
                marker=style['marker'], 
                linestyle=style['linestyle'],
                linewidth=2,
                markersize=8,
                label=style['label'],
                markerfacecolor='white',
                markeredgewidth=1.5
            )
    
    plt.xlabel('SNR (dB)', fontsize=14, fontweight='bold')
    plt.ylabel('Bit Error Rate (BER)', fontsize=14, fontweight='bold')
    plt.title('All Detectors Performance Comparison', fontsize=16, fontweight='bold')
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(fontsize=11, ncol=2, loc='upper right')
    plt.ylim(1e-5, 1.0)
    plt.xlim(0, max(snr_db_list))
    plt.tight_layout()
    
    # Save to results directory if specified
    if results_dir is not None:
        output_path = results_dir / save_filename
    else:
        output_path = save_filename
        
    plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()

    # Create a second plot showing detector groups
    plt.figure(figsize=(14, 10))
    
    # Group 1: Standard detectors
    for detector in ['ml', 'mmse', 'zf', 'zf_reg']:
        if detector in all_results_dict:
            style = detector_styles[detector]
            ber_values = all_results_dict[detector][0]
            plt.semilogy(snr_db_list, ber_values,
                        color=style['color'],
                        marker=style['marker'],
                        linestyle='-',
                        linewidth=3,
                        markersize=8,
                        label=style['label'],
                        markerfacecolor='white',
                        markeredgewidth=2)
    
    # Group 2: Enhanced detectors
    for detector in ['ml_zf', 'adaptive_mmse', 'hybrid']:
        if detector in all_results_dict:
            style = detector_styles[detector]
            ber_values = all_results_dict[detector][0]
            plt.semilogy(snr_db_list, ber_values,
                        color=style['color'],
                        marker=style['marker'],
                        linestyle='--',
                        linewidth=2,
                        markersize=7,
                        label=style['label'],
                        markerfacecolor='white',
                        markeredgewidth=1.5,
                        alpha=0.8)
    
    plt.xlabel('SNR (dB)', fontsize=14, fontweight='bold')
    plt.ylabel('Bit Error Rate (BER)', fontsize=14, fontweight='bold')
    plt.title('Standard vs Enhanced Detectors', fontsize=16, fontweight='bold')
    plt.grid(True, which="both", alpha=0.3)
    
    # Add text annotations for detector groups
    plt.text(0.02, 0.98, 'Standard Detectors', transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.text(0.02, 0.88, 'Enhanced Detectors', transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.legend(fontsize=11, loc='lower left')
    plt.ylim(1e-5, 1.0)
    plt.xlim(0, max(snr_db_list))
    plt.tight_layout()
    
    # Save grouped comparison plot
    grouped_filename = 'detectors_grouped_comparison.png'
    if results_dir is not None:
        output_path = results_dir / grouped_filename
    else:
        output_path = grouped_filename
        
    plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()
