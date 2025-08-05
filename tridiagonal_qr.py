# status update as of 2025-07-29: givens rotation seems to work (checked with
                                                                 # wikipedia)
# bulge is created, propagated, and deleted properly.

import numpy as np
from numpy import linalg
import scipy as sc
import math

# Givens rotation computation
def givens_rotation(a_i, b_i):
    if b_i == 0:
        c = np.sign(a_i)
        s = 0
        r = np.abs(a_i)
    elif a_i == 0:
        c = 0
        s = -np.sign(b_i)
        r = np.abs(b_i)
    elif np.abs(a_i) > np.abs(b_i):
        tau = b_i / a_i
        u = np.sign(a_i) * math.sqrt(1 + tau * tau)
        c = 1 / u
        s = -c * tau
        r = a_i * u
    else:
        tau = a_i / b_i
        u = np.sign(b_i) * math.sqrt(1 + tau * tau)
        s = -1 / u
        c = tau / u
        r = b_i * u
    return c, s, r

# Wilkinson shift for 2x2 matrix
def wilkinson_shift(a, b):
    d = (a[-2] - a[-1])/2
    if abs(d) < 1e-14 and abs(b[-1]) < 1e-14:
        return a[-1]
    denominator = d + np.sign(d)*np.sqrt(d*d + b[-1]**2)
    if abs(denominator) < 1e-14:
        return a[-1]
    shift = a[-1] - (b[-1]*b[-1])/denominator
    return shift

def make_bulge(a, b, givens_mtrx):
    """first 3x3 block to make bulge."""
    T_block = np.diag(a[:3]) 
    T_block += np.diag(b[:2], k=-1) + np.diag(b[:2], k=1)
    #print("make bulge before ", T_block)
    givens_block = np.eye(3)
    givens_block[:2,:2] = givens_mtrx
    result = givens_block.T @ T_block @ givens_block
    a[:3] = np.diag(result)
    b[:2] = np.diag(result, k=-1)
    #print("make bulge after ", result)
    return result[2,0]

def cancel_bulge(a, b, bulge, givens_mtrx):
    """last 3x3 block to get rid of bulge."""
    T_block = np.diag(a[-3:])
    T_block += np.diag(b[-2:], k=-1) + np.diag(b[-2:], k=1)
    T_block[2,0] = bulge
    T_block[0,2] = T_block[2,0]
    #print("cancel bulge before ", T_block)
    givens_block = np.eye(3)
    givens_block[1:,1:] = givens_mtrx
    result = givens_block.T @ T_block @ givens_block
    a[-3:] = np.diag(result)
    b[-2:] = np.diag(result, k=-1)
    return result[2,0]

def move_bulge(a, b, bulge, i, givens_mtrx):
    # assume i >= 1 because i == 0 is only for make_bulge()
    assert i >= 1, "index too small."
    assert i < b.size-1, "index too large."
    j = i-1
    T_block = np.diag(a[j:j+4])
    T_block += np.diag(b[j:j+3], k=-1) + np.diag(b[j:j+3], k=1)
    T_block[2,0] = bulge
    T_block[0,2] = T_block[2,0]
    #print("T_block before ", T_block)
    givens_block = np.eye(4)
    givens_block[1:-1,1:-1] = givens_mtrx
    #print("givens_block", givens_block)
    result = givens_block.T @ T_block @ givens_block
    #print("T_block after ", result)
    a[j:j+4] = np.diag(result)
    b[j:j+3] = np.diag(result, k=-1)
    return result[3,1]

# Apply a Givens rotation to the tridiagonal vectors
def do_bulge_chasing(a, b, evec_row):
    """a givens rotation moves makes a bulge in the first iteration and cancels
    it in the last iteration. As such iter 1, n are 3x3 operations. everything in
    between are 4x4 operations. """

    assert a.size - b.size == 1

    shift = wilkinson_shift(a, b)
    x = a[0] - shift
    z = b[0]
    c, s, _ = givens_rotation(x, z)
    apply_givens_to_evec_row(evec_row, c, s, 0)
    givens_mtrx = np.array([[c, s], [-s, c]])
    bulge = make_bulge(a, b, givens_mtrx)
    x = b[0]
    z = bulge

    for i in range(1, b.size-1):
        c, s, _ = givens_rotation(x, z)
        apply_givens_to_evec_row(evec_row, c, s, i)
        givens_mtrx = np.array([[c, s],
                               [-s, c]])

        bulge = move_bulge(a, b, bulge, i, givens_mtrx)

        x = b[i]
        z = bulge

    c, s, _ = givens_rotation(x, z)
    apply_givens_to_evec_row(evec_row, c, s, b.size-1)
    givens_mtrx = np.array([[c, s], [-s, c]])

    bulge = cancel_bulge(a, b, bulge, givens_mtrx)

    assert np.abs(bulge) < 1e-15
    return a, b

# Updates Q (Gm Gm-1 ... G1).T e_1 by applying a single Givens rotation
def apply_givens_to_evec_row(evec_row, c, s, i):
    # refer to pg. 241 GVL 4th Ed.
    tau1 = evec_row[i]
    tau2 = evec_row[i+1]
    evec_row[i]   = c*tau1 - s*tau2
    evec_row[i+1] = s*tau1 + c*tau2

def qr_tridiag(a, b, max_iter=1000, tol=1e-8):
    """QR iteration with Wilkinson shift for tridiagonal matrices."""
    n = len(a)
    a = a.copy()
    b = b.copy()

    evec_row = np.zeros(n)
    evec_row[0] = 1.0

    for iter in range(max_iter):
        if iter == max_iter-1:
            print("hit max_iteration")
            break
        if (np.max(np.abs(b)) < tol):
            print("stopped at iteration", iter)
            break
        # [1:n-1] but with zero indexing -> [0:n-1)
        a, b = do_bulge_chasing(a, b, evec_row)
#        print("b ", b)
    eigvals = a 
    return eigvals, evec_row

a = np.linspace(1, 100, 100)
b = np.ones(99)
print(sc.linalg.eigh_tridiagonal(a,b, eigvals_only=False, lapack_driver='stemr'))
print(qr_tridiag(a,b))
