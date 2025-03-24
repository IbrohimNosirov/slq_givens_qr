import numpy as np

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

main_diag = np.diag(x, k=0)
off_diag = np.diag(x, k=-1)

x = np.diag(main_diag) + np.diag(off_diag, k=-1) + np.diag(off_diag, k=1)

print("x: ", x)
print(np.linalg.eigh(x))
print(main_diag, off_diag)
