from QuantumCircuitSimulator import QuantumCircuitSimulator
from QiskitSimulator import QiskitSimulator

from RandomUnitary import random_unitary
import random

import torch
import numpy as np

from timeit import default_timer as timer

import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met

if __name__ == "__main__":
	device = "xla" # use neuroncore device for matrix mult
	dtype = torch.float16 # neuroncore v2 devices support "cFP8, FP16, BF16, TF32, FP32, INT8, INT16, and INT32"

	n = 12 # number of qubits
	m = 3 # gate size
	steps = 1000 # number of gates in quantum circuit

	# torch.set_num_threads(8)

	# generate random circuit for testing
	print("generating circuit")
	targets = [random.sample(range(n), k=m) for i in range(steps)]
	unitaries = [random_unitary(m, dtype=dtype, device=device) for _ in range(steps)]
	qiskit_gates = [unitary[0] for unitary in unitaries]
	xla_gates = [unitary[1] for unitary in unitaries]

	# custom simulator
	print("starting pytorch implementation")
	with torch.no_grad(): # not intending to perform backprop or anything
		# represent the state vector as an order n tensor
		# each dim is size 2, representing a certain qubit's |0> and |1>
		# since neuroncore doesn't natively support complex numbers, we store the real and imaginary parts separately
		# state[0,:] stores the real part, state[1,:] stores the imaginary part
		print("init state")
		state = torch.zeros([2] + [2]*n, dtype=dtype, device=device) # directly init state vector on xla device to avoid absurdly long XLA compilation
		state[tuple([0] + [0]*n)] = 1 # init the real part of |0^n> to 1
		xm.mark_step()

		start = timer()
		# simulate
		qc = QuantumCircuitSimulator(n, m) # simulator for an n-qubit program with m-qubit gates
		result_xla = qc(state, targets, xla_gates) # perform simulation
		end = timer()
		print(end - start)

		# print result
		result_xla = result_xla.to("cpu") # move to cpu to be able to work with complex numeric type
		result_real, result_imag = result_xla[0], result_xla[1]
		result_xla = result_real + (1j) * result_imag
		result_xla = result_xla.flatten()
		print(result_xla)

	# qiskit simulator
	print("startin qiskit simulator")
	qiskit_sim = QiskitSimulator(n, targets, qiskit_gates)
	result_qiskit = qiskit_sim.simulate()
	print(result_qiskit)

	diff = np.linalg.norm(result_xla.type(torch.complex128).numpy() - result_qiskit.data)
	print(diff)
