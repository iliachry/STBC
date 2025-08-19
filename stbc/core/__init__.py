# Core STBC components including quaternion algebra and STBC implementations

from .quaternion import QuaternionAlgebra
from .biquaternion import BiquaternionSTBC
from .modulation import BiquaternionModulation
from .codewords import (
    generate_all_codewords_biquaternion,
    get_cached_codewords,
    clear_codeword_cache
)

__all__ = [
    'QuaternionAlgebra',
    'BiquaternionSTBC',
    'BiquaternionModulation',
    'generate_all_codewords_biquaternion',
    'get_cached_codewords',
    'clear_codeword_cache'
]
