import numpy as np

def givens_rotation(a, b):
    """
    Compute the Givens rotation matrix parameters for a given (a, b) pair.
    Returns cos(theta) and sin(theta).
    """
    if b == 0:
        return 1.0, 0.0
    elif abs(b) > abs(a):
        tau = -a/b
        s = 1/np.sqrt(1 + tau*tau)
        c = s*tau
        return c, s
    else:
        tau = -b/a
        c = 1/np.sqrt(1 + tau*tau)
        s = c*tau
        return c, s

def apply_givens_rotation(A, c, s, i, k):
    """
    Apply Givens rotation to matrix A in-place.
    """
    n = A.shape[0]
    for j in range(n):
        temp = c * A[i,j] - s * A[k,j]
        A[k,j] = s * A[i,j] + c * A[k,j]
        A[i,j] = temp
    return A

def wilkinson_shift(A, n):
    """
    Compute the Wilkinson shift for a 2x2 matrix.
    """
    a = A[n-2,n-2]
    b = A[n-2,n-1]
    c = A[n-1,n-1]
    d = (a - c) / 2
    sign_d = 1 if d >= 0 else -1
    shift = c - sign_d * b * b / (abs(d) + np.sqrt(d*d + b*b))
    return shift

def qr_iteration_wilkinson(A, max_iter=100, tol=1e-10):
    """
    Perform QR iteration with Wilkinson shift using Givens rotations.
    
    Parameters:
    A : numpy.ndarray
        Input matrix (must be real and symmetric)
    max_iter : int
        Maximum number of iterations
    tol : float
        Convergence tolerance for off-diagonal elements
    
    Returns:
    numpy.ndarray
        Matrix containing eigenvalues on the diagonal
    int
        Number of iterations performed
    """
    n = A.shape[0]
    A = A.copy()
    
    for iter in range(max_iter):
        # Check if matrix is essentially upper triangular
        if np.all(np.abs(np.tril(A, -1)) < tol):
            return A, iter

        # Compute Wilkinson shift
        shift = wilkinson_shift(A, n)
        
        # Apply shift
        A -= shift * np.eye(n)
        
        # QR decomposition using Givens rotations
        for i in range(n-1):
            for j in range(i+1, n):
                if abs(A[j,i]) > tol:
                    c, s = givens_rotation(A[i,i], A[j,i])
                    A = apply_givens_rotation(A, c, s, i, j)
                    A = apply_givens_rotation(A.T, c, s, i, j).T
        
        # Shift back
        A += shift * np.eye(n)
    
    return A, max_iter

# Example usage
def test_qr_iteration():
    # Create a symmetric test matrix
    n = 4
    np.random.seed(42)
    X = np.random.rand(n, n)
    A = X @ X.T  # Ensure matrix is symmetric
    
    # Run QR iteration
    result, iterations = qr_iteration_wilkinson(A)
    
    # Get eigenvalues from diagonal
    computed_eigenvals = np.sort(np.diagonal(result))
    
    # Compare with numpy's eigenvalue solver
    true_eigenvals = np.sort(np.linalg.eigvals(A))
    
    print("Computed eigenvalues:", computed_eigenvals)
    print("True eigenvalues:", true_eigenvals)
    print("Number of iterations:", iterations)
    print("Maximum error:", np.max(np.abs(computed_eigenvals - true_eigenvals)))

if __name__ == "__main__":
    test_qr_iteration()
