## Performant implementation of Stochastic Lanczos Quadrature with robust stopping criteria.
1. [almost done; need fix a bug.] QR iteration with Wilkinson, bulge chasing, and deflation. Use Givens rotations and apply them to ith canonical basis vector to obtain the ith row of an evector matrix.
2. [work in progress] Lanczos with selective orthogonalization as a criteria for deflation (project off select "good" Ritz pairs) and switching to the usual Monte Carlo regime.
