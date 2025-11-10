using LinearAlgebra
using Base.Threads
using ProgressBars

include("lanczos.jl")
include("tridiagonal_qr.jl")
include("matrix_gallery.jl")

NUM_THREADS = Threads.nthreads()

function slq(A :: AbstractMatrix, nv :: Int64, m :: Int64)
    n = size(A)[1]
    avg_evecs = zeros(9)
    Γ = zeros(nv)
    A_tmp = copy(A)
    tridiag_reduce!(A_tmp)
    evals, β = tridiag_params(A_tmp)
    evec_row = qr_tridiag!(evals, β)
    ν = discretemeasure(evals)

    for j in ProgressBar(1:NUM_THREADS:nv)
        Threads.@threads for i = 1:NUM_THREADS
            idx = j + i - 1
            if idx > nv
                continue
            end
            v = sign.(randn((n)))
            ctx = LanczosContext(A, v/norm(v), m, ν, 'u')
            steps_taken = lanczos(ctx)
            evals = get_a(ctx, steps_taken)
            b = get_b(ctx, steps_taken-1)
            evec_row = qr_tridiag!(evals, b)
            Γ[idx] = (evec_row.*evec_row)' * evals
            avg_evecs += evec_row
        end
    end
    println(avg_evecs / nv)
    n*sum(Γ)/nv
end

let
n  = 1000  # number of evals (size of matrix A).
p  = 3     # number of standalone evals.
nv = 200   # number of Monte Carlo trials.
m  = 100   # number of Krylov steps.

evals = spectrum_make(n, p)
gr()
plot!(evals)
A = mtrx_make(evals)

@time println("slq trace mine (full) ", slq(A, nv, m))
println("trace of A ", tr(A))
end
