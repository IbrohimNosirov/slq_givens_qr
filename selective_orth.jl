# Author: Ibrohim Nosirov
# Date: 2025-06-26
# from https://www.netlib.org/utk/people/JackDongarra/etemplates/node110.html#wrecursion
using LinearAlgebra
using Random
using Profile
using StatProfilerHTML

include("matrix_gallery.jl")
include("tridiagonal_qr.jl")

# TODO: add tests.
# TODO: add timing.
# TODO: show how storage scales.
# TODO: at convergence, provide ConvergenceInfo

struct Lanczos
end

function lanczos_full(A :: AbstractMatrix, q :: AbstractVector, k :: Int64,
                 Q :: AbstractMatrix)
    a = zeros(k)
    b = zeros(k)
    q /= norm(q)

    W = diagm(0  => ones(k+1),
              -1 => eps(Float64)*ones(k), 
              1  => eps(Float64)*ones(k))
    norm_A = norm(A)

    z = A * q
    a[1] = q' * z
    z = z - a[1]*q
    b[1] = norm(z)

    for j = 2:k
        q_prev = q
        q = z / b[j-1]
        Q[:,j-1] = q

        z = A*q - b[j-1] * q_prev
        a[j] = q' * z
        z = z - a[j]*q
        b[j] = norm(z)

        if b[j] == 0
            break
        end

        z -= view(Q,:,1:j) * (view(Q,:,1:j)' * z)
        z -= view(Q,:,1:j) * (view(Q,:,1:j)' * z)
    end
    return a, b[1:end-1]
end

function lanczos_selective(A :: AbstractMatrix, q :: AbstractVector,
                           k :: Int64, Q :: AbstractMatrix)
    a = zeros(k)
    b = zeros(k)
    q /= norm(q)

    W = diagm(0  => ones(k+1),
              -1 => eps(Float64)*ones(k), 
              1  => eps(Float64)*ones(k))
    # TODO: should guess the norm.
    norm_A = norm(A)

    z = A * q
    a[1] = q' * z
    z = z - a[1]*q
    b[1] = norm(z)

    for j = 2:k
        q_prev = q
        q = z / b[j-1]
        Q[:,j-1] = q

        z = A*q - b[j-1] * q_prev
        a[j] = q' * z
        z = z - a[j]*q
        b[j] = norm(z)

        if b[j] == 0
            break
        end

        for k = 2:j
            w_tilde  = b[k]*W[j,k+1] + (a[k] - a[j])*W[j,k] 
            w_tilde += b[k-1]*W[j,k-1] - b[j-1]*W[j-1,k]
            W[j+1,k] = (w_tilde + 2*sign(w_tilde)*eps(Float64)*norm_A)/b[j]
            if W[j+1,k] > sqrt(eps(Float64))
                #println("converged at step ", j)
                @views Q[:,1:j-1] = Matrix(qr(Q[:,1:j-1]).Q)
                z -= view(Q,:,1:j-1) * (view(Q,:,1:j-1)' * z)
                z -= view(Q,:,1:j-1) * (view(Q,:,1:j-1)' * z)
                W[j+1,k] = eps(Float64)
                W[j,k] = eps(Float64)
            end
        end
    end
    return a, b[1:end-1]
end

# run Lanczos until the first reorthogonalization.
function lanczos(A :: AbstractMatrix, q :: AbstractVector, k :: Int64)
    a = zeros(k)
    b = zeros(k)
    q /= norm(q)

    n = size(A)[1]
    Q = zeros(n,k)

    W = diagm(0  => ones(k+1),
              -1 => eps(Float64)*ones(k), 
              1  => eps(Float64)*ones(k))

    norm_A = norm(A)

    z = A * q
    a[1] = q' * z
    z = z - a[1]*q
    b[1] = norm(z)

    for j = 2:k
        q_prev = q
        q = z / b[j-1]
        Q[:,j-1] = q

        z = A*q - b[j-1] * q_prev
        a[j] = q' * z
        z = z - a[j]*q
        b[j] = norm(z)

        if b[j] == 0
            break
        end

        for k = 2:j
            w_tilde  = b[k]*W[j,k+1] + (a[k] - a[j])*W[j,k] 
            w_tilde += b[k-1]*W[j,k-1] - b[j-1]*W[j-1,k]
            W[j+1,k] = (w_tilde + 2*sign(w_tilde)*eps(Float64)*norm_A)/b[j]
            if W[j+1,k] > sqrt(eps(Float64))
                #println("converged at step ", j)
                @views Q[:,1:j-1] = Matrix(qr(Q[:,1:j-1]).Q)
                z -= view(Q,:,1:j-1) * (view(Q,:,1:j-1)' * z)
                z -= view(Q,:,1:j-1) * (view(Q,:,1:j-1)' * z)
                W[j+1,k] = eps(Float64)
                W[j,k] = eps(Float64)
                return a[1:j], b[1:j-1]
                break
            end
        end
    end
    return a, b[1:end-1]
end

# set up the lanczos recurrence
A = diagm(collect(exp.(-range(1,n))))
let
Random.seed!(3)
n = 1000
k = 1000
evals = collect(range(1, n))
a, b = tridiag_mtrx_make(evals)
A = diagm(0 => a, -1 => b, 1 => b)

println("condition number ", cond(A))
q = randn(n)
q = q/norm(q)
Q = zeros(n,k)

x, y = lanczos(A, q, k)
println(" x ", x)
println(" y ", y)
w_selective, v_selective = lanczos_selective(A, q, k, Q)
Q = zeros(n,k)
w_full, v_full = lanczos_full(A, q, k, Q)
T = SymTridiagonal(deepcopy(w_selective), deepcopy(v_selective))
evec_row = qr_tridiag!(w_selective,v_selective)

println("Relative error ", norm((sort!(w_selective) - eigvals(A))./norm(A), Inf))
println("Relative error ", norm((eigvals(T) - eigvals(A))./norm(A), Inf))
##println("T evals ", eigvals(T))
##println("A evals ", eigvals(A))
end
