import torch
import torch.nn as nn
from ComplexMatrixMult import complex_matrix_mult

from gates import *

import torch_xla.core.xla_model as xm

class QuantumCircuitSimulator2(nn.Module):
	def __init__(self, num_qubits, num_qubits_gates):
		super().__init__()
		self.num_qubits = num_qubits
		self.num_qubits_gates = num_qubits_gates
		self.dims = list(range(self.num_qubits))

	def forward(self, state, targets, gates):
		for step, target, gate in zip(range(len(targets)), targets, gates):
			print("step:", step)
			# align with qiskit convention
			# i don't actually know why this works I just asked ChatGPT
			target_axes = [self.num_qubits - 1 - q for q in target]
			target_axes = list(reversed(target_axes))

			permutation = [i for i in self.dims if i not in target_axes] + target_axes
			permutation = [0] + [i + 1 for i in permutation]
			inv_permutation = [0]*(self.num_qubits+1)
			for i, dim in enumerate(permutation):
				inv_permutation[dim] = i

			state = torch.permute(state, permutation)
			state = state.reshape((2, -1, 2**self.num_qubits_gates))

			state = complex_matrix_mult(state, gate.transpose(1, 2))

			state = state.reshape([2] + [2]*self.num_qubits)
			state = torch.permute(state, inv_permutation)

			xm.mark_step()
		return state
