import numpy as np
import math

# Givens rotation computation
def givens_rotation(a, b):
    if b == 0:
        c = np.sign(a)
        s = 0
        r = np.abs(a)
    elif a == 0:
        c = 0
        s = -np.sign(b)
        r = np.abs(b)
    elif np.abs(a) > np.abs(b):
        tau = b / a
        u = np.sign(a) * math.sqrt(1 + tau * tau)
        c = 1 / u
        s = -c * tau
        r = a * u
    else:
        tau = a / b
        u = np.sign(b) * math.sqrt(1 + tau * tau)
        s = -1 / u
        c = tau / u
        r = b * u

    return c, s, r

# Applies Givens rotation to the matrix A
def apply_givens(A, c, s, i, k):
    n = A.shape[0]
    for j in range(n):
        tau1 = A[i, j]
        tau2 = A[k, j]

        A[i, j] = c * tau1 - s * tau2
        A[k, j] = s * tau1 + c * tau2
    
    return A

# Wilkinson shift for 2x2 matrix
def wilkinson_shift(A):
    n = A.shape[0]
    a = A[n-2, n-2]
    b = A[n-1, n-2]
    c = A[n-1, n-1]
    d = (a - c) / 2
    sign_d = 1 if d >= 0 else -1
    shift = c - (b * b) / (d + sign_d * np.sqrt(d * d + b * b))

    return shift

# Updates e_1^T Q by applying a single Givens rotation
def apply_givens_to_evec_row(evec_row, c, s, i, k):
    tau1 = evec_row[i]
    tau2 = evec_row[k]

    evec_row[i] = c * tau1 - s * tau2
    evec_row[k] = s * tau1 + c * tau2

    return evec_row

# QR iteration with Wilkinson shift
def qr_iteration_wilkinson(A, max_iter=1000, tol=1e-10):
    n = A.shape[0]
    A = A.copy()

    # List to store Givens rotations
    givens_rotations = []

    for iter in range(max_iter):
        # Compute Wilkinson shift
        shift = wilkinson_shift(A)

        # Apply shift
        A -= shift * np.eye(n)

        # QR decomposition using Givens rotations
        for i in range(n - 1):
            if abs(A[i+1, i]) > tol:
                c, s, r = givens_rotation(A[i+1, i], A[i+2, i])
#                print("before \n", A)
                A = apply_givens(A, c, s, i+1, i+2)
                A = apply_givens(A.T, c, s, i+1, i+2).T
#                print("after \n", A)
                givens_rotations.append((c, s, i, i+1))

        # Shift back
        A += shift * np.eye(n)

        # Check for convergence
        off_diag_norm = np.sqrt(np.sum(A[np.triu_indices(n, k=1)]**2))
        if off_diag_norm < tol:
            break

    # Extract eigenvalues
    eigvals = np.diag(A)

    # Compute e_1^T Q
    evec_row = np.zeros(n)
    evec_row[0] = 1.0
    for c, s, i, k in reversed(givens_rotations):
        evec_row = apply_givens_to_evec_row(evec_row, c, s, i, k)

    return eigvals, evec_row
