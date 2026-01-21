using LinearAlgebra
#using Base.Threads
#using ProgressBars
#using Distributed
#using Profile
using Plots
#using StatProfilerHTML
using OptimalTransport
using Distributions
using DataStructures

#= TODO: need much higher assert coverage; on the order of 40 (avg 2 per
 function.)
=#
# TODO: remove all dependencies except LinearAlgebra + Base.Threads.

const MACHEPS = eps(Float64)

include("matrix_gallery.jl")
include("lanczos.jl")

NUM_THREADS = Threads.nthreads()

function slq(A :: AbstractMatrix, nv :: Int64, m :: Int64, orth :: Char)
    # nv - number of vectors (monte carlo samples).
    # m  - size of krylov subspace.
    N = size(A)[1]
    avg_evecs = zeros(m)
    Γ = zeros(nv)
    A_tmp = copy(A)
    tridiag_reduce!(A_tmp)
    evals, β = tridiag_params(A_tmp)
    evec_row = qr_tridiag!(evals, β, 1)
    ν = discretemeasure(evals)
    v = sign.(randn((N)))
    ctx = LanczosContext(A, v/norm(v), m, ν, orth)

    for idx in 1:nv
        v = sign.(randn((N)))
        ctx = LanczosContext(A, v/norm(v), m, ν, orth)
        steps_taken = lanczos(ctx)
        evals = get_a(ctx, steps_taken)
        b = get_b(ctx, steps_taken-1)
        evec_row = qr_tridiag!(evals, b, 1)
        Γ[idx] = (evec_row.*evec_row)' * evals
        avg_evecs[1:steps_taken] .+= evec_row
#        end
    end
    println("average of the evec rows ", avg_evecs ./ nv)
    #n*sum(Γ)/nv
    ctx
end

let
println("linear decay, 2 gaps, n = 1000")
n  = 1000  # number of evals (size of matrix A).
p  = 2     # number of standalone evals.
nv = 1   # number of Monte Carlo trials.
m  = 200   # number of Krylov steps.

evals = spectrum_linear_make(n, p)
A = mtrx_make(evals)
@time ctx = slq(A, nv, m, 'o')

println("exponential decay, 2 gaps, n = 1000")
n  = 1000 # number of evals (size of matrix A).
p  = 2    # number of standalone evals.
nv = 1   # number of Monte Carlo trials.
m  = 200   # number of Krylov steps.

evals = spectrum_exponential_make(n, p)
A = mtrx_make(evals)
@time ctx = slq(A, nv, m, 'o')

println("linear decay, 3 gaps, n = 1000")
n  = 1000  # number of evals (size of matrix A).
p  = 3     # number of standalone evals.
nv = 1   # number of Monte Carlo trials.
m  = 200   # number of Krylov steps.

evals = spectrum_linear_make(n, p)
A = mtrx_make(evals)
@time ctx = slq(A, nv, m, 'o')

println("exponential decay, 3 gaps, n = 1000")
n  = 1000 # number of evals (size of matrix A).
p  = 3    # number of standalone evals.
nv = 1   # number of Monte Carlo trials.
m  = 200   # number of Krylov steps.

evals = spectrum_exponential_make(n, p)
A = mtrx_make(evals)
@time ctx = slq(A, nv, m, 'o')

println("linear decay, no gaps, n = 1000")
n  = 1000 # number of evals (size of matrix A).
p  = 0    # number of standalone evals.
nv = 1 # number of Monte Carlo trials.
m  = 200   # number of Krylov steps.

evals = spectrum_linear_make(n, p)
A = mtrx_make(evals)
#@time ctx = slq(A, nv, m, 's')
@time ctx = slq(A, nv, m, 'o')
#@time ctx = slq(A, nv, m, 'u')
#@time ctx = slq(A, nv, m, 'f')
gr()
scatter(evals)

#println("trace of A ", tr(A))
end
