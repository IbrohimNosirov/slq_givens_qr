import unittest
import math
import numpy as np
from scipy.linalg import eigh_tridiagonal

class TestGivensQR(unittest.TestCase):

    def test_givens_rotation(self):
        from tridiagonal_qr import givens_rotation

        test_cases = [
            (3, 4),   # Standard Pythagorean triple
            (0, 5),   # One zero component
            (5, 0),   # Another zero component
            (1, 1),   # Arbitrary values
        ]
        
        for a, b in test_cases:
            with self.subTest(a=a, b=b):
                c, s, r = givens_rotation(a, b)
                
                # Check orthonormality
                self.assertAlmostEqual(c**2 + s**2, 1.0, msg="Not orthogonal.")
    
                # Check rotation works
                r_true = math.sqrt(a**2 + b**2)
                self.assertAlmostEqual(r_true, r, msg="Rotation incorrect.")
    
#    def test_apply_givens(self):
#        from tridiagonal_qr import apply_givens
#        from tridiagonal_qr import givens_rotation
#
#        # Initial tridiagonal matrix representation
#        a = np.array([6.0, 1.0, 3.0])  # Diagonal
#        b = np.array([5.0, 4.0])  # Subdiagonal
#        c, s, r = givens_rotation(a[0], b[0])
#        
#        # Apply Givens rotation
#        a1, b1 = apply_givens(a.copy(), b.copy(), c, s, 0)
#        # Expected transformation
#        expected_a1 = np.array([7.8102, -2.4327, 3.0])
#        # Check that the modified diagonal and subdiagonal match expectations
#        np.testing.assert_almost_equal(a1, expected_a1, decimal=4,
#                                       err_msg="Diagonal elements incorrect.")
#
#        c, s, r = givens_rotation(a1[1], b1[1])
#        a2, b2 = apply_givens(a1, b1, c, s, 1)
#        expected_a2 = np.array([7.8102, 4.6817, -4.1843])
#        expected_b = np.array([0.0, 1.0])
#        np.testing.assert_almost_equal(a2, expected_a2, decimal=4,
#                                       err_msg="Diagonal elements incorrect.")
#        np.testing.assert_almost_equal(b_new, expected_b, decimal=1,
#                                    err_msg="Subdiagonal elements incorrect.")

    def test_qr_iteration_wilkinson_tridiagonal(self):
        from tridiagonal_qr import qr_tridiag 

        # Helper function to generate main diagonal and off-diagonal
        def generate_tridiagonal_vectors(n, min_val=0, max_val=1):
            np.random.seed(42)  # For reproducibility
            main_diag = np.linspace(1.0, n, num=n)
            off_diag = np.random.uniform(min_val, max_val, n - 1)
            return main_diag, off_diag

        # Generate test cases
        test_cases = [
            generate_tridiagonal_vectors(3),
            generate_tridiagonal_vectors(5),
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

                print("evec_actual ", computed_evec_row)
                print("evec_desired ", ref_evecs[0, :])

                np.testing.assert_allclose(
                    computed_evals,
                    ref_evals,
                    rtol=1e-5,
                    atol=1e-6,
                    err_msg=f"Eigenvalues mismatch for test case {idx}"
                )

                np.testing.assert_allclose(
                    computed_evec_row,
                    ref_evecs[0, :],
                    rtol=1e-5,
                    atol=1e-6,
                    err_msg=f"Eigenvector mismatch for test case {idx}"
                )

if __name__ == "__main__":
    unittest.main()
