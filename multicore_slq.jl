using LinearAlgebra
using Base.Threads

include("lanczos.jl")
include("tridiagonal_qr.jl")
include("matrix_gallery.jl")

NUM_THREADS = Threads.nthreads()

function slq_full(A :: AbstractMatrix, nv :: Int64, m :: Int64)
    n = size(A)[1]
    Γ = zeros(nv)
    for j = 1:NUM_THREADS:nv
        Threads.@threads for i = 1:NUM_THREADS
            idx = j + i - 1
            if idx > nv
                continue
            end
            v = sign.(randn((n)))
            evals, b = lanczos_full(A, v, m)
            evec_row = qr_tridiag!(evals, b)
            Γ[idx] = (evec_row.*evec_row)' * evals
        end
    end

    n*sum(Γ)/nv
end

function slq_selective(A :: AbstractMatrix, nv :: Int64, m :: Int64)
    n = size(A)[1]
    Γ = zeros(nv)
    for j = 1:NUM_THREADS:nv
        Threads.@threads for i = 1:NUM_THREADS
            idx = j + i - 1
            if idx > nv
                continue
            end
            v = sign.(randn((n)))
            a, b = lanczos_selective(A, v, m)
            E = eigen!(SymTridiagonal(a, b))
            evec_row = E.vectors[1,:]
            evals = E.values
            Γ[idx] = (evec_row.*evec_row)' * evals
        end
    end

    n*sum(Γ)/nv
end

function slq_iteration(A :: AbstractMatrix, j :: Int64, m :: Int64)
    n = size(A, 1)
    v = sign.(kronecker_quasirand_vec(n, j))
    a, b = lanczos_full(A, v, m)
    E = eigen!(SymTridiagonal(a, b))
    evec_row = E.vectors[1,:]
    evals = E.values

    (evec_row.*evec_row)' * evals
end

function slq_LAPACK(A :: AbstractMatrix, nv :: Int64, m :: Int64)
    n = size(A)[1]
    Γ = zeros(nv)
    Threads.@threads for j = 1:nv
        Γ[j] = slq_iteration(A, j, m)
    end

    n*sum(Γ)/nv
end

function slq_LAPACK_seq(A :: AbstractMatrix, nv :: Int64, m :: Int64)
    n = size(A)[1]
    Γ = zeros(nv)
    for j = 1:NUM_THREADS:nv
        for i = 1:NUM_THREADS
            idx = j + i - 1
            if idx > nv
                continue
            end
            v = sign.(randn((n)))
            a, b = lanczos_full(A, v, m)
            E = eigen!(SymTridiagonal(a, b))
            evec_row = E.vectors[1,:]
            # TODO: print out evec_row
            # TODO evec_row[i] = 1/sqrt(n)
            evals = E.values
            Γ[idx] = (evec_row.*evec_row)' * evals
        end
    end

    n*sum(Γ)/nv
end

n  = 1000  # number of evals (size of matrix A).
p  = 3     # number of standalone evals.
nv = 200   # number of Monte Carlo trials.
m  = 50   # number of Krylov steps.

evals = spectrum_make(n, p)
plot!(evals)
A = mtrx_make(evals)

@time println("slq trace LAPACK sequential ", slq_LAPACK_seq(A, nv, m))
@time println("slq trace LAPACK ", slq_LAPACK(A, nv, m))
@time println("slq trace mine (full) ", slq_full(A, nv, m))
@time println("slq trace mine (selective) ", slq_selective(A, nv, m))
println("trace of A ", tr(A))
