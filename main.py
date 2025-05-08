from QuantumCircuitSimulator import QuantumCircuitSimulator
from QiskitSimulator import QiskitSimulator
from gates import *
from RandomUnitary import random_unitary

import random
import torch
import numpy as np
from qiskit.quantum_info import random_unitary as qiskit_random_unitary

import torch_neuronx
import torch_xla.core.xla_model as xm

from timeit import default_timer as timer

if __name__ == "__main__":
	# import os
	# os.environ['NEURON_FRAMEWORK_DEBUG'] = "1"

	device = "xla"
	dtype = torch.float16

	check_with_qiskit = False

	n = 6
	m = 2
	steps = 10

	# define circuit
	print("defining circuit")
	targets = [random.sample(range(n), k=m) for i in range(steps)]
	unitaries = [random_unitary(m, dtype=dtype) for _ in range(steps)]
	qiskit_gates = [unitary[0] for unitary in unitaries]
	xla_gates = [unitary[1] for unitary in unitaries]

	# sim = QuantumCircuitSimulator(n, targets, xla_gates)
	# sim.eval()
	# state = torch.zeros([2] + [2]*n, dtype=dtype)
	# state[tuple([0] + [0]*n)] = 1

	# print("tracing")
	# start = timer()
	# trace = torch_neuronx.trace(sim, state, compiler_args = "--optlevel 1")
	# torch.jit.save(trace, "sim.pt")
	# end = timer()
	# print("trace time:", end - start)

	# PyTorch implementation
	print("starting pytorch implementation")
	start = timer()
	with torch.no_grad():
		print("init state")
		state = torch.zeros([2] + [2]*n, dtype=dtype).to(device) # store real and imaginary part separately
		state[tuple([0] + [0]*n)] = 1
		xm.mark_step()
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
		reference_result = simulator.simulate(device="GPU")
		end = timer()
		qiskit_time = end - start

	print("pytorch:", pytorch_time)

	if check_with_qiskit:
		print("qiskit build:", qiskit_build_time)
		print("qiskit:", qiskit_time)
		result = result_real + (1j) * result_imag
		result = result.flatten()
		print(torch.linalg.norm(result - torch.tensor(reference_result.data, dtype=torch.complex128).to(device)))
		print(torch.linalg.norm(result.to("cpu") - torch.tensor(reference_result.data, dtype=torch.complex128)))
		print(result.to("cpu"))
		print(reference_result)
