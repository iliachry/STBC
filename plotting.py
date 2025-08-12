import matplotlib.pyplot as plt
import numpy as np

def plot_detection_results(snr_db_list, ber_opt, ber_std, ber_poor, gamma_opt, gamma_std, gamma_poor, detector_name, save_filename):
    """Plot BER results for a specific detector"""
    plt.figure(figsize=(10, 8))
    plt.semilogy(snr_db_list, ber_opt, 'b-o', linewidth=3, markersize=8, 
                 label=f'Optimized (γ={gamma_opt:.2f})', markerfacecolor='white', markeredgewidth=2)
    plt.semilogy(snr_db_list, ber_std, 'r--s', linewidth=3, markersize=8, 
                 label=f'Standard (γ={gamma_std:.2f})', markerfacecolor='white', markeredgewidth=2)
    plt.semilogy(snr_db_list, ber_poor, 'm-.^', linewidth=3, markersize=8, 
                 label=f'Poor (γ={gamma_poor:.1f})', markerfacecolor='white', markeredgewidth=2)
    
    plt.xlabel('SNR (dB)', fontsize=14, fontweight='bold')
    plt.ylabel('Bit Error Rate (BER)', fontsize=14, fontweight='bold')
    plt.title(f'{detector_name} Detection', 
              fontsize=16, fontweight='bold')
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(fontsize=13)
    plt.ylim(1e-5, 1.0)
    plt.xlim(0, max(snr_db_list))
    
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
            idx = snr_val  # Adjust for SNR step size
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


def save_performance_table_png(snr_db_list, ber_opt_ml, ber_std_ml, ber_opt_mmse, ber_std_mmse, ber_opt_zf, ber_std_zf, filename='performance_table.png'):
    """Render performance analysis as a table and save to PNG (no CLI prints)."""
    # Build table data
    headers = [
        'SNR (dB)',
        'ML Opt', 'ML Std', 'ML Gain %',
        'MMSE Opt', 'MMSE Std', 'MMSE Gain %',
        'ZF Opt', 'ZF Std', 'ZF Gain %'
    ]
    rows = []
    for i, snr in enumerate(snr_db_list):
        # Gains with safe guards
        ml_gain = ((ber_std_ml[i] - ber_opt_ml[i]) / ber_std_ml[i] * 100.0) if ber_std_ml[i] > 0 else 0.0
        mmse_gain = ((ber_std_mmse[i] - ber_opt_mmse[i]) / ber_std_mmse[i] * 100.0) if ber_std_mmse[i] > 0 else 0.0
        zf_gain = ((ber_std_zf[i] - ber_opt_zf[i]) / ber_std_zf[i] * 100.0) if ber_std_zf[i] > 0 else 0.0
        rows.append([
            f"{snr}",
            f"{ber_opt_ml[i]:.3e}", f"{ber_std_ml[i]:.3e}", f"{ml_gain:.1f}",
            f"{ber_opt_mmse[i]:.3e}", f"{ber_std_mmse[i]:.3e}", f"{mmse_gain:.1f}",
            f"{ber_opt_zf[i]:.3e}", f"{ber_std_zf[i]:.3e}", f"{zf_gain:.1f}"
        ])

    # Figure setup sized by number of rows
    fig_height = max(2.5, 0.5 + 0.35 * len(rows))
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis('off')

    table = ax.table(cellText=rows, colLabels=headers, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.2)

    # Bold header row
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight='bold')

    plt.tight_layout()
    fig.savefig(filename, format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)
