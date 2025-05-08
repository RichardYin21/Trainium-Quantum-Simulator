from QuantumCircuitSimulator2 import QuantumCircuitSimulator2 as QuantumCircuitSimulator
from RandomUnitary import random_unitary

import random
import torch

import torch_xla.core.xla_model as xm

if __name__ == "__main__":
	# import os
	# os.system()

	device = "xla"
	dtype = torch.float16

	n = 10
	m = 7
	steps = 10

	# define circuit
	print("defining circuit")
	targets = [random.sample(range(n), k=m) for i in range(steps)]
	unitaries = [random_unitary(m, dtype=dtype, device=device) for _ in range(steps)]
	qiskit_gates = [unitary[0] for unitary in unitaries]
	xla_gates = [unitary[1] for unitary in unitaries]

	# implementation
	print("starting pytorch implementation")
	with torch.no_grad():
		print("init state")
		state = torch.zeros([2] + [2]*n, dtype=dtype).to(device) # store real and imaginary part separately
		state[tuple([0] + [0]*n)] = 1
		xm.mark_step()
		qc = QuantumCircuitSimulator(n, m).to(device)
		result = qc(state, targets, xla_gates).to("cpu")
		result_real, result_imag = result[0], result[1]
		result = result_real + (1j) * result_imag

		print(result.flatten())
