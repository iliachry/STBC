import torch
from itertools import product
from quaternion import QuaternionAlgebra

class BiquaternionSTBC:
    """True Biquaternion Division Algebra STBC Implementation"""
    def __init__(self, gamma, device):
        self.gamma = gamma
        self.device = device
        self.Q1 = QuaternionAlgebra(a=-1, b=-1, device=device)  # (-1,-1/F)
        self.Q2 = QuaternionAlgebra(a=gamma, b=-1, device=device)  # (γ,-1/F)
    
    def psi_representation(self, q):
        """Standard irreducible representation ψ: Q₁ → M₂(ℂ)"""
        batch_shape = q.shape[:-1]
        
        z_a = q[..., 0] + 1j * q[..., 1]  # x₀ + x₁i
        z_b = q[..., 2] + 1j * q[..., 3]  # x₂ + x₃i
        
        psi_matrix = torch.zeros((*batch_shape, 2, 2), dtype=torch.complex64, device=self.device)
        psi_matrix[..., 0, 0] = z_a
        psi_matrix[..., 0, 1] = z_b
        psi_matrix[..., 1, 0] = -z_b.conj()
        psi_matrix[..., 1, 1] = z_a.conj()
        
        return psi_matrix
    
    def involution_sigma(self, q):
        """Involution σ induced by Q₂ structure"""
        return torch.stack([q[..., 0], -q[..., 1], q[..., 2], -q[..., 3]], dim=-1)
    
    def left_regular_representation(self, q1, q2):
        """Left regular representation of biquaternion element"""
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
        """Map complex symbols to quaternions"""
        batch_size = symbols.shape[0]
        
        if rate == 1:
            s = symbols.view(batch_size, 4)
            q1 = self.Q1.create_quaternion(
                s[..., 0].real, s[..., 0].imag,
                s[..., 1].real, s[..., 1].imag
            )
            q2 = torch.zeros_like(q1)
        else:
            s = symbols.view(batch_size, 8)
            q1 = self.Q1.create_quaternion(
                s[..., 0].real, s[..., 0].imag,
                s[..., 1].real, s[..., 1].imag
            )
            q2 = self.Q1.create_quaternion(
                s[..., 2].real, s[..., 2].imag,
                s[..., 3].real, s[..., 3].imag
            )
        
        return q1, q2

def generate_all_codewords_biquaternion(stbc, rate=2):
    """Generate all possible biquaternion STBC codewords"""
    # QPSK constellation
    QPSK = torch.tensor([1+1j, 1-1j, -1+1j, -1-1j], dtype=torch.complex64, device=stbc.device) / torch.sqrt(torch.tensor(2.0, device=stbc.device))
    BIT_LOOKUP = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.int8, device=stbc.device)
    
    if rate == 1:
        symbol_indices = list(product(range(4), repeat=4))
        num_bits = 8
    else:
        symbol_indices = list(product(range(4), repeat=4))  # Reduced for feasibility
        num_bits = 8
    
    all_codewords = []
    all_bits = []
    
    for indices in symbol_indices:
        symbols = QPSK[torch.tensor(indices, device=stbc.device)]
        bits = BIT_LOOKUP[torch.tensor(indices, device=stbc.device)].flatten()
        
        if rate == 2 and len(symbols) < 8:
            symbols = torch.cat([symbols, symbols], dim=0)
        
        q1, q2 = stbc.symbols_to_quaternions(symbols.unsqueeze(0), rate=rate)
        X = stbc.left_regular_representation(q1, q2).squeeze(0)
        
        all_codewords.append(X)
        all_bits.append(bits)
    
    return torch.stack(all_codewords), torch.stack(all_bits)
