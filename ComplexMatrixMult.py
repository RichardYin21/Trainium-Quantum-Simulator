# GPT-4o output, 3M algorithm

import torch

def complex_matrix_mult(A, B):
    # Split real and imaginary parts
    A_real, A_imag = A[0], A[1]
    B_real, B_imag = B[0], B[1]

    # Compute intermediate products
    P1 = A_real @ B_real  # Real(A) * Real(B)
    P2 = A_imag @ B_imag  # Imag(A) * Imag(B)
    P3 = (A_real + A_imag) @ (B_real + B_imag)  # (Real(A) + Imag(A)) * (Real(B) + Imag(B))

    # Compute the real and imaginary parts of the result
    C_real = P1 - P2
    C_imag = P3 - P1 - P2

    # Stack the real and imaginary parts
    C = torch.stack([C_real, C_imag], dim=0)

    return C
