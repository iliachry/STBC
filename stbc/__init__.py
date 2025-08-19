# STBC Module Package
"""
Space-Time Block Code (STBC) library for simulation and performance evaluation
of various detection methods for Biquaternion Division Algebra STBC.
"""

__version__ = '1.0.0'

# Core components
from .core import (
    QuaternionAlgebra,
    BiquaternionSTBC,
    BiquaternionModulation,
    generate_all_codewords_biquaternion,
    get_cached_codewords,
    clear_codeword_cache
)

# Detectors
from .detectors import (
    ml_detection_biquaternion,
    mmse_detection_biquaternion,
    zf_detection_biquaternion,
    adaptive_reg_factor,
    regularized_zf_detection_biquaternion,
    ml_enhanced_zf_detection_biquaternion,
    adaptive_mmse_detection_biquaternion,
    hybrid_detection_biquaternion
)

# Optimization
from .optimization import (
    is_valid_gamma,
    min_det_squared_for_gamma_worker,
    min_det_squared_for_gamma,
    optimize_gamma
)

# Simulation
from .simulation import (
    apply_detector,
    simulate_ber,
    simulate_ber_for_gamma,
    simulate_ber_all_detectors,
    simulate_ber_common,
    simulate_ber_three
)

# Utils
from .utils import (
    select_device,
    get_qpsk_and_bit_lookup,
    create_results_directory,
    save_results_to_csv
)

# Visualization
from .visualization import (
    plot_detection_results,
    plot_all_detectors_comparison,
    save_performance_table_png,
    save_all_detectors_table_png
)

__all__ = [
    # Core
    'QuaternionAlgebra',
    'BiquaternionSTBC',
    'BiquaternionModulation',
    'generate_all_codewords_biquaternion',
    'get_cached_codewords',
    'clear_codeword_cache',
    # Detectors
    'ml_detection_biquaternion',
    'mmse_detection_biquaternion',
    'zf_detection_biquaternion',
    'adaptive_reg_factor',
    'regularized_zf_detection_biquaternion',
    'ml_enhanced_zf_detection_biquaternion',
    'adaptive_mmse_detection_biquaternion',
    'hybrid_detection_biquaternion',
    # Optimization
    'is_valid_gamma',
    'min_det_squared_for_gamma_worker',
    'min_det_squared_for_gamma',
    'optimize_gamma',
    # Simulation
    'apply_detector',
    'simulate_ber',
    'simulate_ber_for_gamma',
    'simulate_ber_all_detectors',
    'simulate_ber_common',
    'simulate_ber_three',
    # Utils
    'select_device',
    'get_qpsk_and_bit_lookup',
    'create_results_directory',
    'save_results_to_csv',
    # Visualization
    'plot_detection_results',
    'plot_all_detectors_comparison',
    'save_performance_table_png',
    'save_all_detectors_table_png'
]
