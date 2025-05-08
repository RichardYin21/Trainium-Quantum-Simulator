from qiskit.quantum_info import random_unitary as qiskit_random_unitary
import torch
import numpy as np

def random_unitary(n, dtype=torch.float16, device="xla"):
	unitary = qiskit_random_unitary(2**n).data
	# Stack the real and imaginary parts along a new dimension
	real_imag_tensor = torch.tensor(np.stack([unitary.real, unitary.imag], axis=0), dtype=dtype).to(device)
	return (unitary, real_imag_tensor)
