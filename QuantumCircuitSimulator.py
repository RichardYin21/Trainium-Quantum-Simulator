import torch
import torch.nn as nn
from ComplexMatrixMult import complex_matrix_mult
from nki_gate_apply import nki_gate_apply

import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met

class QuantumCircuitSimulator(nn.Module):
	def __init__(self, num_qubits, num_qubits_gate, device="xla"):
		super().__init__()
		self.num_qubits = num_qubits
		self.num_qubits_gate = num_qubits_gate
		self.dims = list(range(self.num_qubits))

		# specify the device to perform major computations on
		# (in case you want to try on gpu, but there's still xla/neuroncore specific code in this file that needs to be removed before running)
		self.device = device

	def forward(self, state, targets, gates):
		xm.mark_step()
		for step, target, gate in zip(range(len(targets)), targets, gates):
			print("step:", step)

			# align with qiskit convention for easier checking of results
			target_axes = [self.num_qubits - 1 - q for q in target]
			# target_axes = [8, 0, 5, 15, 1, 11, 6]
			target_axes = list(reversed(target_axes))

			# determine how to permute the qubits to prepare for efficient application of the gate matrix
			permutation = [i for i in self.dims if i not in target_axes] + target_axes
			permutation = [0] + [i + 1 for i in permutation] # don't permute the real/dim axis

			# suboptimal way of computing the inverse permutation
			inv_permutation = [0]*(self.num_qubits+1)
			for i, dim in enumerate(permutation):
				inv_permutation[dim] = i

			# hack: different permutations result in different xla computation graphs (which take forever to compile!!!)
			# move permutation to CPU to avoid needing to compile all 7 gigajillion possible permutations
			# the permutation operation itself is efficient on CPU as it only edits the tensor stride metadata
			# see: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.stride.html, https://github.com/pytorch/pytorch/blob/a6170573c898a1367517d8daf8e777abaf96f752/aten/src/ATen/native/TensorShape.cpp#L367-L385
			# (note that tensors on NeuronCore may or may not have the same/similar implementation for permute)
			# a potential concern with this hack is that the moving of tensors between devices could be quite expensive

			# alternative idea discussed: chain multiple smaller permutations together
			# if we permute (swap) only 2 axes at a time, a total of num_qubits_gate swaps would be needed per permutation
			# num_qubits * num_qubits_gate swap operations to compile, instead of 7 gigajillion permutations
			# state = torch.permute(state.to("cpu"), permutation).to(self.device)
			state = torch.permute(state.to("cpu"), permutation)
			# state = torch.permute(state, permutation)

			# apply $I_2^{\otimes n-m} \otimes U_m$ on the state vector
			# because $I_2^{\otimes n-m} \otimes U_m$ is a block diagonal matrix, this operation can be performed more efficiently
			# as U applied to each 2^m-sized partition of the state vector

			# reshape so that each state vector partition is a row vector
			state = state.reshape((2, -1, 2**self.num_qubits_gate)).to(self.device)
			# state = state.reshape((2, -1, 2**self.num_qubits_gate))

			# right-multiply by U^T
			# note that we transpose dims 1 and 2 of the gate because dim 0 determines the real/imaginary part
			# also note that transposing on CPU is probably more efficient (currently we are transposing on XLA)
			# this is _the_ place where we would parallelize across multiple NeuronCores
			# state = complex_matrix_mult(state, gate.transpose(1, 2))
			state = nki_gate_apply(gate, state.transpose(1, 2)).transpose(1, 2)
			# xm.mark_step()

			# reshape back to original shape
			state = state.to("cpu").reshape([2] + [2]*self.num_qubits)
			# state = state.reshape([2] + [2]*self.num_qubits)

			# restore the qubit order back to normal (using the same permutation hack as before)
			# state = torch.permute(state.to("cpu"), inv_permutation).to(self.device)
			# state = torch.permute(state, inv_permutation).to(self.device)
			state = torch.permute(state, inv_permutation)

			# compile one iteration of the loop for reuse
			# (currently probably unncessary due to the permute hacks forcing earlier compilation from moving tensors to CPU)
			xm.mark_step()
		# return state
		return state.to(self.device)
