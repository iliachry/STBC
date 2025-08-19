"""
Device utility functions for STBC simulations.
"""

import torch

def select_device(device: torch.device | None = None) -> torch.device:
    """
    Select the appropriate device for computations.
    
    Args:
        device: Optional device specification. If None, auto-select.
        
    Returns:
        torch.device: The selected device.
    """
    if device is not None:
        return device
    return torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

def get_qpsk_and_bit_lookup(device: torch.device):
    """
    Get QPSK modulation points and bit lookup tables.
    
    Args:
        device: Device to place tensors on
        
    Returns:
        tuple: (qpsk, bit_lookup) - QPSK symbols and bit lookup tables
    """
    qpsk = torch.tensor([1+1j, 1-1j, -1+1j, -1-1j], dtype=torch.complex64, device=device) / torch.sqrt(torch.tensor(2.0, device=device))
    bit_lookup = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.int8, device=device)
    return qpsk, bit_lookup
