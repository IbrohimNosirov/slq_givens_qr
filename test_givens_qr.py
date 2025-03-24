import unittest
import numpy as np
from scipy.linalg import eigh_tridiagonal
import matplotlib.pyplot as plt


class TestGivensQR(unittest.TestCase):
    def test_qr_iteration_wilkinson_tridiagonal(self):
        from tridiagonal_qr import qr_tridiag 

        def generate_matrix_with_eigenvalues(n):
            """Generate a dense matrix with eigenvalues evenly spaced from 1 to
            n.
            """
            evals = np.linspace(1, n, n) + 2
            D = np.diag(evals)
            Q, _ = np.linalg.qr(np.random.rand(n, n))
            return Q @ D @ Q.T

        def householder_tridiagonalize(A):
            """
            Reduce a symmetric matrix to tridiagonal form using Householder
            transformations.
            """
            n = A.shape[0]
            A = A.copy()
            
            for k in range(n-2):
                x = A[k+1:, k]
                norm_x = np.linalg.norm(x)
                
                if norm_x > 0:
                    # Compute Householder vector
                    v = x.copy()
                    v[0] += np.sign(x[0]) * norm_x
                    v = v / np.linalg.norm(v)
                    
                    # Apply Householder transformation
                    tau = 2 * np.outer(v, v)
                    temp = A[k+1:, k:]
                    A[k+1:, k:] -= tau @ temp
                    temp = A[k:, k+1:]
                    A[k:, k+1:] -= temp @ tau.T
            return np.diag(A), np.diag(A, 1)

        def generate_tridiagonal_vectors(n):
            """
            Helper function to generate test vectors for tridiagonal matrix.
            """
            A = generate_matrix_with_eigenvalues(n)
            return householder_tridiagonalize(A)

        # Generate test cases
        test_cases = [
            generate_tridiagonal_vectors(3),
            generate_tridiagonal_vectors(4),
            generate_tridiagonal_vectors(10),
            generate_tridiagonal_vectors(20),
            generate_tridiagonal_vectors(50),
            generate_tridiagonal_vectors(100),
        ]

        for idx, (main_diag, off_diag) in enumerate(test_cases):
            with self.subTest(matrix_index=idx):
                computed_evals, computed_evec_row = qr_tridiag(main_diag, off_diag)
                ref_evals, ref_evecs = eigh_tridiagonal(main_diag, off_diag)
                computed_evals = np.sort(computed_evals)
                ref_evals = np.sort(ref_evals)

                np.testing.assert_allclose(
                    computed_evals,
                    ref_evals,
                    rtol=1e-3,
                    atol=1e-4,
                    err_msg=f"Eigenvalues mismatch for test case {idx}"
                )
                
                np.testing.assert_allclose(
                    np.sort(computed_evec_row),
                    np.sort(ref_evecs[0, :]),
                    rtol=1e-3,
                    atol=1e-4,
                    err_msg=f"Eigenvector mismatch for test case {idx}"
                )

if __name__ == "__main__":
    unittest.main()