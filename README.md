# STBC - Space-Time Block Coding Simulation

A PyTorch-based implementation of Space-Time Block Coding (STBC) with Maximum Likelihood (ML) detection, optimized for Apple Silicon (M1/M2) GPUs using the Metal Performance Shaders (MPS) backend.

## Overview

This project implements and simulates the performance of Space-Time Block Coding schemes for MIMO wireless communication systems. The implementation includes:

- **QPSK Modulation**: Quadrature Phase Shift Keying constellation
- **4×4 STBC Matrix Construction**: Using quaternion-based approach
- **ML Detection**: Brute-force Maximum Likelihood detection with GPU acceleration
- **BER Performance Analysis**: Bit Error Rate simulation across different SNR values
- **GPU Acceleration**: Optimized for Apple Silicon using PyTorch MPS backend

## Features

- ✅ **GPU Accelerated**: Leverages Apple Silicon M1/M2 GPUs for fast computation
- ✅ **Vectorized Operations**: Efficient batch processing of STBC matrices
- ✅ **Multiple Rate Support**: Rate-1 (robust) and Rate-2 (high-throughput) configurations
- ✅ **Configurable Parameters**: Adjustable SNR ranges, trial counts, and optimization parameters
- ✅ **Visualization**: Automatic BER plot generation and saving

## Requirements

- Python 3.8+
- PyTorch 2.0+ with MPS support
- NumPy
- Matplotlib
- Apple Silicon Mac (M1/M2) for GPU acceleration

## Installation

1. **Clone the repository**:

2. **Create and activate virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip3 install torch torchvision numpy matplotlib
   ```

## Usage

### Running the Simulation

Execute the main simulation script:

```bash
python main.py
```

The script will:
1. Detect and use Apple Silicon GPU (MPS) if available
2. Run BER simulations for three different configurations:
   - Optimized Rate-2 (γ = 0.4 + 1.1j)
   - Non-optimized Rate-2 (γ = 1 + j)
   - Robust Rate-1 (γ = 0)
3. Generate and save BER performance plots

### Configuration

You can modify the simulation parameters in `main.py`:

```python
# SNR range (in dB)
snr_db_list = np.arange(0, 21, 3)

# Number of Monte Carlo trials
num_trials = 200

# STBC parameters
gamma_optimized = 0.4 + 1.1j  # Optimized parameter
gamma_standard = 1 + 1j       # Standard parameter
```

### Output

The simulation generates:
- Console output showing progress and intermediate results
- BER performance plots (e.g., `ml_detection.png`, `mmse_detection.png`, `zf_detection.png`)
- Performance comparison across different STBC configurations

### Configuration via .env

You can configure runs without exporting shell variables. Create a `.env` in the project root (already git-ignored) with e.g.:

```
NUM_TRIALS=1000
GAMMA_TRIALS=200
GAMMA_GRID_STEPS=11
GAMMA_REFINE_ROUNDS=2
MPLBACKEND=Agg
```

Scripts (`main.py`, `optimize_and_run.py`, `optimize_gamma.py`) will load `.env` automatically. Process env overrides `.env` if both are set.

## Technical Details

### STBC Matrix Structure

The 4×4 STBC matrix is constructed using a quaternion-based approach:

```
X = [Ψ(q₁)    γΨ(q₂)]
    [Ψ(q₂)    Ψ(q₁) ]
```

Where Ψ(q) represents the quaternion matrix transformation.

### ML Detection

The Maximum Likelihood detector finds the transmitted symbol vector by:

```
ŝ = argmin ||Y - HX||²_F
```

Where:
- Y is the received signal matrix
- H is the channel matrix
- X is the candidate STBC matrix

### Performance Metrics

- **Rate-1**: 2 bits per channel use, higher reliability
- **Rate-2**: 4 bits per channel use, higher throughput
- **BER**: Bit Error Rate as a function of Signal-to-Noise Ratio

## Project Structure

```
STBC/
├── main.py              # Main simulation script
├── .gitignore          # Git ignore file
├── README.md           # This file
├── venv/               # Virtual environment (ignored)
└── ber_plot_m1_optimized.eps  # Generated plot
```

## Performance

The implementation is optimized for Apple Silicon GPUs:
- **Vectorized Operations**: Batch processing of multiple STBC matrices
- **GPU Memory Management**: Efficient tensor operations on MPS
- **Parallel ML Detection**: Simultaneous evaluation of all symbol candidates

Typical performance on M1 MacBook Pro:
- ~200 trials per SNR point
- Multiple SNR values (0-20 dB)
- Execution time: ~30-60 seconds depending on configuration

## Troubleshooting

### Common Issues

1. **MPS Not Available**:
   - Ensure you're running on Apple Silicon (M1/M2)
   - Update to macOS 12.3+ and PyTorch 2.0+

2. **Complex Number Operations**:
   - The code handles MPS limitations with complex operations
   - Uses `.abs()` for norm calculations when needed

3. **Memory Issues**:
   - Reduce `num_trials` if running out of GPU memory
   - Consider smaller SNR ranges for initial testing

### Debug Mode

To run with reduced trials for testing:

```python
num_trials = 50  # Reduced for faster testing
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

- Tarokh, V., Seshadri, N., & Calderbank, A. R. (1998). Space-time codes for high data rate wireless communication
- PyTorch MPS Backend Documentation
- Space-Time Block Coding: From Theory to Practice

## Acknowledgments

- PyTorch team for MPS backend support
- Apple for Metal Performance Shaders framework
- Research community for STBC theoretical foundations 