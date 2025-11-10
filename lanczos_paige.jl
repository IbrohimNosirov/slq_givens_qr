using LinearAlgebra
using Random
include("matrix_gallery.jl")

#struct Lanczos <: ConvergenceInfo
#    A :: AbstractMatrix
#    q :: AbstractVector
#    k :: Int64
#    Q :: AbstractMatrix
#end

# TODO: make a constructor, put lots of asserts here.

function lanczos_paige(A :: AbstractMatrix, q :: AbstractVector, k :: Int64)
    n = size(Q, 1)
    a = zeros(k)
    b = zeros(k)
    Q = zeros(n,k)
    q /= norm(q)

    #z = A * q
    #a[1] = q' * z
    #z = z - a[1]*q
    #b[1] = norm(z)
    q_prev = q
    alpha  = 0.0
    beta   = 0.0

    orth_tol = 1.0e-08
    wn = 0.0

    for j = 1:k
        z = A*q - beta * q_prev
        alpha = q' * z
        z = z - alpha*q
        beta = norm(z)

        if beta*j < orth_tol*wn
            break
        end

        z -= view(Q,:,1:j) * (view(Q,:,1:j)' * z)
        z -= view(Q,:,1:j) * (view(Q,:,1:j)' * z)

        q_prev = q

        q = z / b[j]
        Q[:,j] = q

    end
    return a, b[1:end-1]
end

let
Random.seed!(3)
n = 100
k = 100
evals = collect(range(1, n))
A = mtrx_make(evals)

q = randn(n)
w_full, v_full = lanczos_paige(A, q, k)
T = SymTridiagonal(w_full, v_full)
println("Relative error ", norm((eigvals(T) - eigvals(A))./norm(A), Inf))
end
