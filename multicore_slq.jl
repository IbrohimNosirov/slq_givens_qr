using LinearAlgebra
using Base.Threads

include("selective_orth.jl")
include("tridiagonal_qr.jl")
include("matrix_gallery.jl")

# output approximate trace Γ of f(A)
#for l = 1:nv
#    generate Rademacher random vector u_l, form unit vector v_l = u_l/||u_l||_2
#    T = Lanczos(A, v_l, m+1)
#    [Y, Theta] = eig(T)
#    τ_k = [e_1^Ty_k] for k = 0, ..., m
#    Γ += Γ + sum up to m of τ_k^2 f(θ_k)
#end

function slq(A :: AbstractMatrix, nv :: Int64, m :: Int64)
    n = size(A)[1]
    Γ = zeros(nv)
    Threads.@threads for l = 1:nv
        v_l = randn(n)
        Q = zeros(n,m)
        evals, b = lanczos_full(A, v_l, m, Q)
        evec_row = qr_tridiag!(evals, b)
        Γ[l] = (evec_row.*evec_row)' * evals
    end

    sum(Γ)/nv
end

function slq_LAPACK(A :: AbstractMatrix, nv :: Int64, m :: Int64)
    n = size(A)[1]
    Γ = zeros(nv)
    Threads.@threads for l = 1:nv
        v_l = randn(n)
        Q = zeros(n,m)
        evals, b = lanczos_full(A, v_l, m, Q)
        evec_row = eigen!(evals, b)
        Γ[l] = (evec_row.*evec_row)' * evals
    end

    sum(Γ)/nv
end

n  = 100  # number of evals (size of matrix A).
p  = 3    # number of standalone evals.
nv = 30   # number of Monte Carlo trials.
m  = 1000 # number of Krylov steps.

evals = spectrum_make(n, p)
A = mtrx_make(evals)

println("slq trace ", slq(A, nv, m))
println("trace of A ", tr(A))
