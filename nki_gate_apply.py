import neuronxcc.nki as nki
import neuronxcc.nki.language as nl

@nki.jit()
def nki_gate_apply(gate, state):
	assert state.ndim == 3
	assert state.shape[0] == 2 # state should have real and imag component
	assert state.shape[1] <= 128 # contraction axis
	assert gate.ndim == 3
	assert gate.shape[0] == 2 # gate should have real and imag component
	assert gate.shape[1] <= 128 # contraction axis
	assert gate.shape[2] <= 128 # layout constraint
	assert gate.shape[1] == gate.shape[2] # gate matrix should be square
	assert gate.shape[1] == state.shape[1] # contraction axes should match in size

	result = nl.ndarray((2, gate.shape[2], state.shape[2]), dtype=state.dtype, buffer=nl.shared_hbm)

	gate_tile_real = nl.ndarray(gate.shape, dtype=gate.dtype, buffer=nl.sbuf)
	gate_tile_imag = nl.ndarray(gate.shape, dtype=gate.dtype, buffer=nl.sbuf)

	i_gate_tile_p, i_gate_tile_f = nl.mgrid[0:gate.shape[1], 0:gate.shape[2]]
	gate_tile_real = nl.load(gate[0, i_gate_tile_p, i_gate_tile_f])
	gate_tile_imag = nl.load(gate[1, i_gate_tile_p, i_gate_tile_f])

	for i in nl.affine_range(state.shape[2] // 512): # layout constraint
		state_tile_real = nl.ndarray((state.shape[1], 512), dtype=gate.dtype, buffer=nl.sbuf)
		state_tile_imag = nl.ndarray((state.shape[1], 512), dtype=gate.dtype, buffer=nl.sbuf)
		state_tile_real[...] = nl.load(state[0, 0:state.shape[1], i*512:(i+1)*512])
		state_tile_imag[...] = nl.load(state[1, 0:state.shape[1], i*512:(i+1)*512])

		result_psum = nl.matmul(gate_tile_real, state_tile_real, transpose_x=False)
		P1 = nl.copy(result_psum, dtype=result.dtype)

		result_psum = nl.matmul(gate_tile_imag, state_tile_imag, transpose_x=False)
		P2 = nl.copy(result_psum, dtype=result.dtype)

		result_psum = nl.matmul(nl.add(gate_tile_real, gate_tile_imag), nl.add(state_tile_real, state_tile_imag), transpose_x=False)
		P3 = nl.copy(result_psum, dtype=result.dtype)

		result_real = nl.subtract(P1, P2)
		result_imag = nl.subtract(nl.subtract(P3, P1), P2)
		nl.store(result[0, 0:gate.shape[2], i*512:(i+1)*512], value=result_real)
		nl.store(result[1, 0:gate.shape[2], i*512:(i+1)*512], value=result_imag)

	return result
