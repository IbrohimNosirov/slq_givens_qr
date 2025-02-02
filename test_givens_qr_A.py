import unittest
import math
import numpy as np

class TestGivensQR(unittest.TestCase):

    def test_givens_rotation(self):
        from tridiagonal_qr_A import givens_rotation

        test_cases = [
            (3, 4),   # Standard Pythagorean triple
            (0, 5),   # One zero component
            (5, 0),   # Another zero component
            (1, 1),   # Arbitrary values
        ]
        
        for a, b in test_cases:
            with self.subTest(a=a, b=b):

                c, s, r = givens_rotation(a, b)
                # check orthonormality.
                self.assertAlmostEqual(c**2 + s**2, 1.0, msg="Not orthogonal.")
    
                # check rotation works.
                r_true = math.sqrt(a**2 + b**2)
    
                self.assertAlmostEqual(r_true,r, msg="rotation incorrect.")

    def test_qr_iteration_wilkinson(self):
        from tridiagonal_qr_A import qr_iteration_wilkinson

        np.random.seed(42)
    # Generate test cases: random tridiagonal matrices
        def generate_tridiagonal_matrix(n, min_val=0, max_val=1):
            main_diag = np.linspace(1.0, n, num=n)
            off_diag = np.random.uniform(min_val, max_val, n - 1)
            A = np.diag(main_diag) 
            A += np.diag(off_diag, k=1)
            A += np.diag(off_diag, k=-1)

            return A

        # List of test matrices
        test_cases = [
#            generate_tridiagonal_matrix(3),
            generate_tridiagonal_matrix(5),
#            generate_tridiagonal_matrix(10),
#            generate_tridiagonal_matrix(20),
#            generate_tridiagonal_matrix(50),
#            generate_tridiagonal_matrix(100)
        ]

        for idx, A in enumerate(test_cases):
            with self.subTest(matrix_index=idx):
                # Compute eigenvalues using qr_iteration_wilkinson
                computed_evals, computed_evecs = qr_iteration_wilkinson(A)

                # Compute eigenvalues using NumPy's eig function
                ref_evals, ref_evecs = np.linalg.eigh(A)

                # Sort computed eigenvalues and eigenvectors
                sort_indices = np.argsort(computed_evals)
                computed_evals = computed_evals[sort_indices]
                computed_evecs = computed_evecs[sort_indices]

                # Sort reference eigenvalues for comparison
                sort_indices = np.argsort(ref_evals)
                ref_evals = ref_evals[sort_indices]
                ref_evecs = ref_evecs[:, sort_indices]

                #print(computed_evals, ref_evals)
                print(computed_evecs, ref_evecs[0,:])
                # Assert that the eigenvalues are close
#                print(np.linalg.cond(A))
                np.testing.assert_allclose(
                    computed_evals,
                    ref_evals,
                    rtol=1e-5,
                    atol=1e-6,
                    err_msg=f"Eigenvalues mismatch for test case {idx}"
                )            

if __name__ == "__main__":
    unittest.main()
