import torch
from stbc.core.biquaternion import BiquaternionSTBC  
from stbc.core.codewords import get_cached_codewords
from stbc.detectors.basic_detectors import mmse_detection_biquaternion, ml_detection_biquaternion
import time

device = torch.device('cpu')
gamma = -1-1j
stbc = BiquaternionSTBC(gamma, device)
all_codewords, all_bits = get_cached_codewords(stbc, rate=2)

# Test data
batch_size = 10
H = torch.randn(batch_size, 4, 4, dtype=torch.complex64, device=device)
y = torch.randn(batch_size, 4, 4, dtype=torch.complex64, device=device)
noise_var = torch.tensor(0.1, device=device)

print('Testing detector speeds...')

# Test ML
start = time.time()
ml_result = ml_detection_biquaternion(y, H, all_codewords)
ml_time = time.time() - start
print(f'ML: {ml_time:.4f}s')

# Test MMSE with fast quantization
start = time.time()  
mmse_result = mmse_detection_biquaternion(y, H, all_codewords, noise_var, stbc, rate=2)
mmse_time = time.time() - start
print(f'MMSE: {mmse_time:.4f}s')

print(f'Speed ratio (MMSE/ML): {mmse_time/ml_time:.2f}x')