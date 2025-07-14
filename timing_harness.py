import numpy as np
from scipy import linalg
import time
import matplotlib.pyplot as plt
from tridiagonal_qr import *

def generate_tridiagonal_vectors(n, min_val=0, max_val=1):
    np.random.seed(42)
    x = np.diag(np.linspace(0.1, n, num=n))

    givens_rotations_angle = np.random.uniform(0, 1.0, (n-1,))
    givens_rotations = np.array([np.cos(givens_rotations_angle),
                                 np.sin(givens_rotations_angle)]).T

    for i in range(n-1):
        c,s = givens_rotations[i,:]
        givens_matrix = np.eye(n,n)
        givens_matrix[i,i], givens_matrix[i,i+1] = c, -s
        givens_matrix[i+1,i], givens_matrix[i+1,i+1] = s, c
        x = givens_matrix.T @ x @ givens_matrix

    print(x)
    main_diag = np.diag(x, k=0)
    off_diag = np.diag(x, k=-1)

    return main_diag, off_diag

def benchmark_solvers(sizes):
    """
    Benchmark custom QR solver against SciPy's solver for different matrix sizes.
    
    Args:
        sizes: List of matrix sizes to test
    Returns:
        times_custom: List of execution times for custom solver
        times_scipy: List of execution times for SciPy solver
        max_errors: List of maximum absolute differences between solutions
    """
    times_custom = []
    times_scipy = []
    max_errors = []
    
    for n in sizes:
        print(f"Testing size {n}x{n}")
        
        # Generate test matrix
        main_diag, off_diag = generate_tridiagonal_vectors(n)
        
        # Time custom QR solver
        start_time = time.time()
        eigvals_custom, evec_custom = qr_tridiag(main_diag, off_diag)
        custom_time = time.time() - start_time
        times_custom.append(custom_time)
        
        
        # Time scipy solver
        start_time = time.time()
        eigvals_scipy, evecs_scipy =linalg.eigh_tridiagonal(main_diag,off_diag,
                                                            select='a')
        scipy_time = time.time() - start_time
        times_scipy.append(scipy_time)
        
        # Compute maximum error in eigenvalues
        max_error = np.max(np.abs(np.sort(eigvals_custom) - np.sort(eigvals_scipy)))
        max_errors.append(max_error)
        
        print(f"Custom solver time: {custom_time:.4f}s")
        print(f"SciPy solver time: {scipy_time:.4f}s")
        print(f"Max eigenvalue difference: {max_error:.2e}\n")
    
    return times_custom, times_scipy, max_errors

def plot_results(sizes, times_custom, times_scipy, max_errors):
    """Plot timing and error results."""
    plt.figure(figsize=(12, 5))
    
    # Timing plot
    plt.subplot(1, 2, 1)
    plt.loglog(sizes, times_custom, 'o-', label='Custom QR')
    plt.loglog(sizes, times_scipy, 's-', label='SciPy stemr')
    plt.xlabel('Matrix Size')
    plt.ylabel('Time (seconds)')
    plt.title('Solver Performance Comparison')
    plt.legend()
    plt.grid(True)
    
    # Error plot
    plt.subplot(1, 2, 2)
    plt.semilogy(sizes, max_errors, 'o-')
    plt.xlabel('Matrix Size')
    plt.ylabel('Maximum Absolute Error')
    plt.title('Solution Accuracy')
    plt.grid(True)
    
    plt.tight_layout()
    return plt.gcf()

# Run the benchmark
if __name__ == "__main__":
    # Test sizes from 100 to 10000
#    sizes = [100, 200, 400, 800, 1600]
    sizes = [10]
    
    # Run benchmark
    times_custom, times_scipy, max_errors = benchmark_solvers(sizes)
    
    # Plot results
    plot_results(sizes, times_custom, times_scipy, max_errors)
    plt.show()
    
    # Print summary
    print("\nSummary:")
    print(f"Largest matrix tested: {sizes[-1]}x{sizes[-1]}")
    print(f"Custom solver final time: {times_custom[-1]:.2f}s")
    print(f"SciPy solver final time: {times_scipy[-1]:.2f}s")
    print(f"Maximum error across all sizes: {max(max_errors):.2e}")
