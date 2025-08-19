# STBC - Space-Time Block Code Simulation Framework

This package provides a comprehensive framework for simulating and evaluating the performance of Space-Time Block Codes (STBC) based on Biquaternion Division Algebras.

## Features

- **Quaternion Algebra**: Implementation of generalized quaternion algebras over number fields
- **Biquaternion STBC**: Implementation of Biquaternion Division Algebra STBC
- **Multiple Detectors**:
  - Maximum Likelihood (ML) - optimal but computationally intensive
  - Minimum Mean Square Error (MMSE) - balances performance and complexity
  - Zero-Forcing (ZF) - simple but suffers at low SNR
  - Regularized Zero-Forcing (ZF-Reg) - improved ZF with regularization
  - ML-Enhanced Zero-Forcing (ML-ZF) - ZF variant approaching ML performance
  - Adaptive MMSE - MMSE with adaptive regularization
  - Hybrid - switches between ML and enhanced ZF based on channel condition
- **Gamma Optimization**: Optimizes the gamma parameter for best STBC performance
- **Comprehensive Visualization**: BER curves, performance tables, and comparisons
- **Result Management**: Automatically saves all results in timestamped directories with CSV exports

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/STBC.git
cd STBC

# Install the package
pip install -e .
```

## Usage

### Basic usage:

```bash
# Run with default parameters
python -m stbc

# Specify SNR range and number of trials
python -m stbc --snr-start 0 --snr-end 20 --snr-step 2 --num-trials 2000

# Use fixed gamma instead of optimization
python -m stbc --gamma-mode fixed --gamma-fixed 0.5+0.5j

# Use preset gamma values
python -m stbc --gamma-mode fixed --gamma-preset minusj

# Specify device
python -m stbc --device cuda

# Just show configuration (without running)
python -m stbc --dry-run
```

### Command-line arguments:

- `--gamma-mode`: Choose between "optimize" or "fixed" gamma (default: optimize)
- `--dry-run`: Print effective config and exit
- `--snr-start`: Minimum SNR in dB (default: 0)
- `--snr-end`: Maximum SNR in dB (default: 20)
- `--snr-step`: SNR step size in dB (default: 2)
- `--num-trials`: Number of simulation trials (default: 1000)
- `--gamma-grid-steps`: Number of grid steps for gamma optimization (default: 11)
- `--gamma-refine-rounds`: Number of refinement rounds for gamma optimization (default: 2)
- `--opt-snr-step`: SNR step size for optimization (default: 3)
- `--no-parallel`: Disable parallel gamma evaluation
- `--gamma-preset`: Preset gamma for fixed mode: "golden" or "minusj" (default: golden)
- `--gamma-fixed`: Fixed gamma as a Python complex string, e.g., "0.5+1j" or "-1j"
- `--gamma-fixed-real`: Fixed gamma real part (alternative to --gamma-fixed)
- `--gamma-fixed-imag`: Fixed gamma imaginary part (alternative to --gamma-fixed)
- `--rate`: Code rate, 1 or 2 (default: 2)
- `--device`: Computation device: cpu, cuda, mps (default: auto-select)

### Environment Variables

You can also use environment variables (via .env file) to set parameters:

- `SNR_START`: Starting SNR value in dB
- `SNR_END`: Ending SNR value in dB
- `SNR_STEP`: SNR step size in dB
- `NUM_TRIALS`: Number of simulation trials
- `GAMMA_GRID_STEPS`: Grid steps for gamma optimization
- `GAMMA_REFINE_ROUNDS`: Refinement rounds for gamma optimization
- `OPT_SNR_STEP`: SNR step for optimization
- `GAMMA_FIXED`: Fixed gamma value (as complex string)

## Results

All results are automatically saved in a timestamped directory under `results/YYYYMMDD_HHMMSS/` with:

- Individual detector BER curves
- Detector comparison plots
- Performance tables
- CSV files with raw data
- Simulation parameters record

## Package Structure

```
stbc/
  ├── core/            # Core STBC implementation
  │   ├── quaternion.py
  │   ├── biquaternion.py
  │   ├── modulation.py
  │   └── codewords.py
  ├── detectors/       # Detection algorithms
  │   ├── basic_detectors.py
  │   └── enhanced_detectors.py
  ├── optimization/    # Gamma parameter optimization
  │   └── gamma_optimizer.py
  ├── simulation/      # Simulation framework
  │   └── simulator.py
  ├── visualization/   # Results visualization
  │   ├── plotting.py
  │   └── tables.py
  ├── utils/           # Utility functions
  │   ├── device_utils.py
  │   └── results.py
  └── __main__.py      # Command-line interface
```

## Implementation Notes

### Modularity

The package follows a modular structure with clear separation of concerns:

- **Core Module**: Quaternion algebra, biquaternion implementation, and codeword generation
- **Detectors Module**: Various detection algorithms from basic (ML, MMSE, ZF) to enhanced variants
- **Optimization Module**: Gamma parameter optimization functions
- **Simulation Module**: BER simulation framework
- **Visualization Module**: Plotting and table generation utilities
- **Utils Module**: Helper functions for device selection and result management

### Class Integration

The `BiquaternionSTBC` class integrates with `BiquaternionModulation` through delegation:

```python
def symbols_to_quaternions(self, symbols, rate=2):
    """Convert complex symbols to quaternions (delegates to BiquaternionModulation)"""
    return BiquaternionModulation.symbols_to_quaternions(self, symbols, rate=rate)
```

This pattern allows for better separation of concerns while maintaining the original API.

## License

This project is licensed under the MIT License - see the LICENSE file for details.