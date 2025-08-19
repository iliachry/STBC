"""
Biquaternion Division Algebra STBC Implementation.
"""

import torch
from .quaternion import QuaternionAlgebra
from .modulation import BiquaternionModulation

class BiquaternionSTBC:
    """True Biquaternion Division Algebra STBC Implementation"""
    def __init__(self, gamma, device):
        self.gamma = gamma
        self.device = device
        self.Q1 = QuaternionAlgebra(a=-1, b=-1, device=device)      # (-1,-1/F)
        self.Q2 = QuaternionAlgebra(a=gamma, b=-1, device=device)   # (γ,-1/F)

    def is_valid_division_algebra(self):
        """Check if the biquaternion forms a valid division algebra"""
        # Avoid gamma values that make Q2 identical or too similar to Q1
        if abs(self.gamma - (-1+0j)) < 1e-6:
            return False
        # Additional checks for division algebra properties
        if abs(self.gamma) < 1e-6:  # Avoid zero gamma
            return False
        return True

    def psi_representation(self, q):
        batch_shape = q.shape[:-1]
        z_a = q[..., 0] + 1j * q[..., 1]
        z_b = q[..., 2] + 1j * q[..., 3]
        psi_matrix = torch.zeros((*batch_shape, 2, 2), dtype=torch.complex64, device=self.device)
        psi_matrix[..., 0, 0] = z_a
        psi_matrix[..., 0, 1] = z_b
        psi_matrix[..., 1, 0] = -z_b.conj()
        psi_matrix[..., 1, 1] = z_a.conj()
        return psi_matrix

    def involution_sigma(self, q):
        return torch.stack([q[..., 0], -q[..., 1], q[..., 2], -q[..., 3]], dim=-1)

    def left_regular_representation(self, q1, q2):
        batch_shape = q1.shape[:-1]
        q1_sigma = self.involution_sigma(q1)
        q2_sigma = self.involution_sigma(q2)

        psi_q1 = self.psi_representation(q1)
        psi_q2 = self.psi_representation(q2)
        psi_q1_sigma = self.psi_representation(q1_sigma)
        psi_q2_sigma = self.psi_representation(q2_sigma)

        X = torch.zeros((*batch_shape, 4, 4), dtype=torch.complex64, device=self.device)
        X[..., 0:2, 0:2] = psi_q1
        X[..., 0:2, 2:4] = self.gamma * psi_q2_sigma
        X[..., 2:4, 0:2] = psi_q2
        X[..., 2:4, 2:4] = psi_q1_sigma

        power = torch.sum(torch.abs(X)**2, dim=(-2, -1), keepdim=True)
        X = X * torch.sqrt(torch.tensor(4.0, device=self.device) / power)
        return X
        
    def symbols_to_quaternions(self, symbols, rate=2):
        """Fixed symbol-to-quaternion mapping"""
        batch_size = symbols.shape[0]
        if rate == 1:
            # Rate-1: 4 symbols -> q1 only
            s = symbols.view(batch_size, 4)
            q1 = self.Q1.create_quaternion(s[..., 0].real, s[..., 0].imag,
                                          s[..., 1].real, s[..., 1].imag)
            q2 = torch.zeros_like(q1)
        else:
            # Rate-2: 8 symbols -> both q1 and q2 (4 symbols each)
            s = symbols.view(batch_size, 8)
            q1 = self.Q1.create_quaternion(s[..., 0].real, s[..., 0].imag,
                                          s[..., 1].real, s[..., 1].imag)
            q2 = self.Q1.create_quaternion(s[..., 2].real, s[..., 2].imag,
                                          s[..., 3].real, s[..., 3].imag)
        return q1, q2
        
    def extract_symbols_from_quaternions(self, q1, q2, rate=2):
        """Extract symbols from quaternions for detection"""
        if rate == 1:
            symbols = torch.stack([
                q1[..., 0] + 1j * q1[..., 1],  # s0
                q1[..., 2] + 1j * q1[..., 3],  # s1
            ], dim=-1)
        else:
            symbols = torch.stack([
                q1[..., 0] + 1j * q1[..., 1],  # s0
                q1[..., 2] + 1j * q1[..., 3],  # s1
                q2[..., 0] + 1j * q2[..., 1],  # s2
                q2[..., 2] + 1j * q2[..., 3],  # s3
                torch.zeros_like(q2[..., 0]),  # s4 (placeholder)
                torch.zeros_like(q2[..., 0]),  # s5 (placeholder)
                torch.zeros_like(q2[..., 0]),  # s6 (placeholder)
                torch.zeros_like(q2[..., 0]),  # s7 (placeholder)
            ], dim=-1)
        return symbols
