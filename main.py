from QuantumCircuitSimulator import QuantumCircuitSimulator
from QiskitSimulator import QiskitSimulator
from gates import *

import random
import torch
import numpy as np
from qiskit.quantum_info import random_unitary as qiskit_random_unitary

import torch_xla.core.xla_model as xm

from timeit import default_timer as timer

def random_unitary(n, dtype=torch.float16):
    unitary = qiskit_random_unitary(2**n).data
    # Stack the real and imaginary parts along a new dimension
    real_imag_tensor = torch.tensor(np.stack([unitary.real, unitary.imag], axis=0), dtype=dtype)
    return (unitary, real_imag_tensor)

if __name__ == "__main__":
	device = "xla"
	dtype = torch.float16

	check_with_qiskit = False

	n = 3
	m = 1
	steps = 1

	# define circuit
	print("defining circuit")
	targets = [random.sample(range(n), k=m) for i in range(steps)]
	unitaries = [random_unitary(m, dtype=dtype) for _ in range(steps)]
	qiskit_gates = [unitary[0] for unitary in unitaries]
	xla_gates = [unitary[1] for unitary in unitaries]

	# PyTorch implementation
	print("starting pytorch implementation")
	start = timer()
	with torch.no_grad():
		state = torch.zeros([2] + [2]*n, dtype=dtype).to(device) # store real and imaginary part separately
		state[tuple([0] + [0]*n)] = 1
		qc = QuantumCircuitSimulator(n, targets, xla_gates).to(device)
		result = qc(state, print_state=False)
		result_real, result_imag = result[0], result[1]
		xm.mark_step()
	end = timer()
	pytorch_time = end - start

	# use Qiskit as reference implementation
	if check_with_qiskit:
		print("starting qiskit implementation")
		start = timer()
		simulator = QiskitSimulator(n, targets, qiskit_gates)
		end = timer()
		qiskit_build_time = end - start

		start = timer()
		reference_result = simulator.simulate(device="CPU")
		end = timer()
		qiskit_time = end - start

	# compare results
	# print(result)
	# print(reference_result)
	print("pytorch:", pytorch_time)

	if check_with_qiskit:
		print("qiskit build:", qiskit_build_time)
		print("qiskit:", qiskit_time)
		# print(torch.linalg.norm(result - torch.tensor(reference_result.data, dtype=torch.complex128).to(device)))
		# print(torch.linalg.norm(result.to("cpu") - torch.tensor(reference_result.data, dtype=torch.complex128)))
		print(result.to("cpu"))
