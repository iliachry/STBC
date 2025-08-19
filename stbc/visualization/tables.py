"""
Table generation functions for visualizing STBC performance results.
"""

import numpy as np
import matplotlib.pyplot as plt

def save_performance_table_png(snr_db_list, ber_opt_ml, ber_std_ml, ber_opt_mmse, ber_std_mmse, 
                              ber_opt_zf, ber_std_zf, ber_opt_zf_reg=None, ber_std_zf_reg=None, 
                              all_results_dict=None, gamma_opt=None, filename='performance_table.png', 
                              results_dir=None):
    """
    Generate a table showing performance of different detectors.
    
    Args:
        snr_db_list: List of SNR values in dB
        ber_opt_ml: BER for ML with optimized gamma
        ber_std_ml: BER for ML with standard gamma
        ber_opt_mmse: BER for MMSE with optimized gamma
        ber_std_mmse: BER for MMSE with standard gamma
        ber_opt_zf: BER for ZF with optimized gamma
        ber_std_zf: BER for ZF with standard gamma
        ber_opt_zf_reg: BER for ZF-Reg with optimized gamma
        ber_std_zf_reg: BER for ZF-Reg with standard gamma
        all_results_dict: Optional dictionary with all results
        gamma_opt: Optimized gamma value
        filename: Filename to save table
        results_dir: Optional directory to save results
    """
    headers = ['SNR (dB)', 'ML Opt', 'ML Std', 'ML Gain %', 'MMSE Opt', 'MMSE Std', 'MMSE Gain %', 'ZF Opt', 'ZF Std', 'ZF Gain %']
    
    if ber_opt_zf_reg is not None and ber_std_zf_reg is not None:
        headers.extend(['ZF-Reg Opt', 'ZF-Reg Std', 'ZF-Reg Gain %'])
    
    rows = []
    for i, snr in enumerate(snr_db_list):
        ml_gain = ((ber_std_ml[i] - ber_opt_ml[i]) / ber_std_ml[i] * 100.0) if ber_std_ml[i] > 0 else 0.0
        mmse_gain = ((ber_std_mmse[i] - ber_opt_mmse[i]) / ber_std_mmse[i] * 100.0) if ber_std_mmse[i] > 0 else 0.0
        zf_gain = ((ber_std_zf[i] - ber_opt_zf[i]) / ber_std_zf[i] * 100.0) if ber_std_zf[i] > 0 else 0.0
        
        row = [f"{snr}", f"{ber_opt_ml[i]:.3e}", f"{ber_std_ml[i]:.3e}", f"{ml_gain:.1f}", 
               f"{ber_opt_mmse[i]:.3e}", f"{ber_std_mmse[i]:.3e}", f"{mmse_gain:.1f}", 
               f"{ber_opt_zf[i]:.3e}", f"{ber_std_zf[i]:.3e}", f"{zf_gain:.1f}"]
        
        if ber_opt_zf_reg is not None and ber_std_zf_reg is not None:
            zf_reg_gain = ((ber_std_zf_reg[i] - ber_opt_zf_reg[i]) / ber_std_zf_reg[i] * 100.0) if ber_std_zf_reg[i] > 0 else 0.0
            row.extend([f"{ber_opt_zf_reg[i]:.3e}", f"{ber_std_zf_reg[i]:.3e}", f"{zf_reg_gain:.1f}"])
        
        rows.append(row)
    
    fig_height = max(2.5, 0.5 + 0.35 * len(rows))
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.axis('off')
    table = ax.table(cellText=rows, colLabels=headers, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.2)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight='bold')
    plt.tight_layout()
    
    # Save to results directory if specified
    if results_dir is not None:
        output_path = results_dir / filename
    else:
        output_path = filename
        
    fig.savefig(output_path, format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)

def save_all_detectors_table_png(snr_db_list, all_results_dict, gamma_opt, filename='all_detectors_table.png', results_dir=None):
    """
    Create comprehensive performance table for all detectors.
    
    Args:
        snr_db_list: List of SNR values in dB
        all_results_dict: Dictionary containing results for all detectors
        gamma_opt: Optimized gamma value
        filename: Filename to save table
        results_dir: Optional directory to save results
    """
    # Prepare headers
    headers = ['SNR (dB)']
    detector_names = {
        'ml': 'ML',
        'mmse': 'MMSE',
        'zf': 'ZF',
        'zf_reg': 'ZF-Reg',
        'ml_zf': 'ML-ZF',
        'adaptive_mmse': 'Adapt-MMSE',
        'hybrid': 'Hybrid'
    }
    
    # Add headers for each detector
    for det_key, det_name in detector_names.items():
        if det_key in all_results_dict:
            headers.append(f'{det_name} BER')
    
    # Prepare rows
    rows = []
    for i, snr in enumerate(snr_db_list):
        row = [f"{snr} dB"]
        for det_key in detector_names.keys():
            if det_key in all_results_dict:
                # Use optimized gamma results (index 0)
                row.append(f"{all_results_dict[det_key][0][i]:.3e}")
        rows.append(row)
    
    # Create table
    fig_height = max(2.5, 0.5 + 0.35 * len(rows))
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.axis('off')
    table = ax.table(cellText=rows, colLabels=headers, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # Color-coding for performance comparison
    # Get ML performance as reference
    if 'ml' in all_results_dict:
        ml_ber = all_results_dict['ml'][0]  # Optimized gamma results
        
        # Color cells based on performance relative to ML
        for i, snr in enumerate(snr_db_list):
            ml_value = ml_ber[i]
            
            # Start from column 1 (skip SNR column)
            col_idx = 1
            for det_key in detector_names.keys():
                if det_key in all_results_dict:
                    # Skip coloring the ML column itself
                    if det_key != 'ml':
                        det_value = all_results_dict[det_key][0][i]
                        ratio = det_value / ml_value if ml_value > 0 else 1.0
                        
                        # Get the cell and color it
                        try:
                            cell = table.get_celld()[(i+1, col_idx)]
                            # Green: within 10% of ML
                            # Yellow: within 2x of ML
                            # Pink: more than 2x worse than ML
                            if ratio < 1.1:
                                cell.set_facecolor('#C8E6C9')  # Light green for good
                            elif ratio < 2.0:
                                cell.set_facecolor('#FFF9C4')  # Light yellow for medium
                            else:
                                cell.set_facecolor('#FFCDD2')  # Light red for poor
                        except:
                            pass
                    col_idx += 1
    
    plt.title(f'All Detectors Performance (γ={gamma_opt:.2f})', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    # Save to results directory if specified
    if results_dir is not None:
        output_path = results_dir / filename
    else:
        output_path = filename
        
    fig.savefig(output_path, format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)
