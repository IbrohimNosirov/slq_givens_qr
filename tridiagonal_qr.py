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

# Wilkinson shift for 2x2 matrix
def wilkinson_shift(a, b):
    d = (a[-2] - a[-1]) / 2
    sign_d = 1 if d >= 0 else -1
    shift = a[-1] - (b[-1]**2) / (d + sign_d * np.sqrt(d**2 + b[-1]**2))
    return shift

# Apply a Givens rotation to the tridiagonal vectors
def apply_givens(a, b, c, s, i):
    if i+1 < b.size:
        block = np.array([[a[i], b[i],   0],
                          [b[i], a[i+1], b[i+1]],
	                      [0,    b[i+1], a[i+2]]])

        givens_rotation = np.array([[c,-s, 0],
        	                        [s, c, 0],
            		                [0, 0, 1]])
    else:
        block = np.array([[a[i], b[i]],
	                      [b[i], a[i+1]]])

        givens_rotation = np.array([[c,-s],
            		                [s, c]])


#    print("givens rotation ", givens_rotation)
#    print("block ", block)
    result = givens_rotation @ block @ givens_rotation.T
#    result = givens_rotation @ block
#    print("Result ", result)

    a[i] = result[0,0]
    a[i+1] = result[1,1]
    b[i] = result[1,0]
    if i+1 < b.size:
        b[i+1] = result[2,1]

    return a, b

# Updates e_1^T Q by applying a single Givens rotation
def apply_givens_to_evec_row(evec_row, c, s, i):
    tau1 = evec_row[i]
    tau2 = evec_row[i+1]

    evec_row[i] = c * tau1 - s * tau2
    evec_row[i+1] = s * tau1 + c * tau2

    return evec_row

# QR iteration with Wilkinson shift for tridiagonal matrices
def qr_tridiag(a, b, max_iter=1000, tol=1e-10):
    n = len(a)
    a = a.copy()
    b = b.copy()

    # List to store Givens rotations
    givens_rotations = []

    for iter in range(max_iter):
        # Compute Wilkinson shift
        shift = wilkinson_shift(a, b)

        # Apply shift
        a -= shift

        # QR decomposition using Givens rotations
        for i in range(n - 1):
            if abs(b[i]) > tol:
                c, s, r = givens_rotation(a[i], b[i])
                a, b = apply_givens(a, b, c, s, i)
                givens_rotations.append((c, s, i))

        # Shift back
        a += shift

        # Check for convergence
        if np.linalg.norm(b, ord=2) < tol:
            break

    # Extract eigenvalues
    eigvals = a

    # Compute e_1^T Q
    evec_row = np.zeros(n)
    evec_row[0] = 1.0
    for c, s, i in reversed(givens_rotations):
        evec_row = apply_givens_to_evec_row(evec_row, c, s, i)

    return eigvals, evec_row
