"""
Codeword generation functions for STBC.
"""

import torch
from itertools import product

# Global cache for codewords to avoid regeneration
_codeword_cache = {}

def generate_all_codewords_biquaternion(stbc, rate=2):
    """Fixed codeword generation with proper rate handling"""
    QPSK = torch.tensor([1+1j, 1-1j, -1+1j, -1-1j], dtype=torch.complex64, device=stbc.device) / torch.sqrt(torch.tensor(2.0, device=stbc.device))
    BIT_LOOKUP = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.int8, device=stbc.device)
    
    if rate == 1:
        # Rate-1: 4 symbols -> 4^4 = 256 codewords
        symbol_indices = list(product(range(4), repeat=4))
    else:
        # Rate-2: 4 symbols -> 4^4 = 256 codewords
        symbol_indices = list(product(range(4), repeat=4))
    
    all_codewords, all_bits = [], []
    for indices in symbol_indices:
        symbols = QPSK[torch.tensor(indices, device=stbc.device)]
        bits = BIT_LOOKUP[torch.tensor(indices, device=stbc.device)].flatten()
        
        # Pad symbols to required length
        if rate == 1 and len(symbols) < 4:
            symbols = torch.cat([symbols, torch.zeros(4-len(symbols), dtype=torch.complex64, device=stbc.device)])
        elif rate == 2 and len(symbols) < 8:
            symbols = torch.cat([symbols, torch.zeros(8-len(symbols), dtype=torch.complex64, device=stbc.device)])
        
        q1, q2 = stbc.symbols_to_quaternions(symbols.unsqueeze(0), rate=rate)
        X = stbc.left_regular_representation(q1, q2).squeeze(0)
        all_codewords.append(X)
        all_bits.append(bits)
    
    return torch.stack(all_codewords), torch.stack(all_bits)

def get_cached_codewords(stbc, rate=2):
    """Get or create codewords with caching for performance"""
    global _codeword_cache
    
    # Use gamma value, rate, and device as cache key (device-aware caching)
    cache_key = (stbc.gamma, rate, str(stbc.device))
    
    if cache_key not in _codeword_cache:
        all_codewords, all_bits = generate_all_codewords_biquaternion(stbc, rate=rate)
        _codeword_cache[cache_key] = (all_codewords, all_bits)
    
    return _codeword_cache[cache_key]

def clear_codeword_cache():
    """Clear the codeword cache to free memory"""
    global _codeword_cache
    _codeword_cache.clear()
