import torch

class QuaternionAlgebra:
    """Generalized quaternion algebra (a,b/F) over number field F"""
    def __init__(self, a, b, device):
        self.a = a  # i² = a
        self.b = b  # j² = b  
        self.device = device
    
    def create_quaternion(self, x0, x1, x2, x3):
        """Create quaternion x₀ + x₁i + x₂j + x₃k"""
        return torch.stack([x0, x1, x2, x3], dim=-1)
    
    def quaternion_multiply(self, q1, q2):
        """Multiply two quaternions using algebra rules"""
        x0, x1, x2, x3 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        y0, y1, y2, y3 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
        
        result = torch.stack([
            x0*y0 + self.a*x1*y1 + self.b*x2*y2 + self.a*self.b*x3*y3,  # 1 term
            x0*y1 + x1*y0 + self.b*x2*y3 - self.b*x3*y2,                # i term  
            x0*y2 - self.a*x1*y3 + x2*y0 + self.a*x3*y1,                # j term
            x0*y3 + x1*y2 - x2*y1 + x3*y0                               # k term
        ], dim=-1)
        
        return result
    
    def quaternion_conjugate(self, q):
        """Compute quaternion conjugate"""
        return torch.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], dim=-1)
