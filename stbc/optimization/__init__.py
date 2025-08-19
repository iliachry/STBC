# STBC gamma optimization module

from .gamma_optimizer import (
    is_valid_gamma,
    min_det_squared_for_gamma_worker,
    min_det_squared_for_gamma,
    optimize_gamma
)

__all__ = [
    'is_valid_gamma',
    'min_det_squared_for_gamma_worker',
    'min_det_squared_for_gamma',
    'optimize_gamma'
]
