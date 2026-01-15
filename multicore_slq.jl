using LinearAlgebra
using Base.Threads
using ProgressBars
using Distributed
using Profile
using Plots
using StatProfilerHTML
using OptimalTransport
using Distributions
using DataStructures

#= TODO: need much higher assert coverage; on the order of 40 (avg 2 per
 function.)
=#
# TODO: remove all dependencies except LinearAlgebra + Base.Threads.

include("matrix_gallery.jl")

NUM_THREADS = Threads.nthreads()

#=
QR Iteration
=#
const MACHEPS = eps(Float64)

# NEED AT MINIMUM 20 asserts, HAVE 5 asserts.

function givens_rotation!(x :: Float64, z :: Float64)
    c, s = LinearAlgebra.givensAlgorithm(x, z)
    s = -conj(s)
    c, s
end

function wilkinson_shift(a1:: Float64, a2 :: Float64, b :: Float64)
    # a1 last index, a2 second to last index, b last index.
    d = (a2 - a1)/2.0
    if abs(d) < MACHEPS && abs(b) < MACHEPS
        return a1
    end
    denominator = d + sign(d)*sqrt(d*d + b*b)
    if abs(denominator) < 5.0*MACHEPS
        return a1
    end
    shift = a1 - (b*b)/denominator

    shift
end

function make_bulge!(a::AbstractVector{Float64}, b::AbstractVector{Float64},
    c::Float64, s::Float64)

    @assert size(a)[1] == 2   "input is not a 2x2 block due to input a."
    @assert size(b)[1] == 2   "input is not a 2x2 block due to input b."

    a1_tmp = c*(a[1]*c - b[1]*s) - s*(b[1]*c - a[2]*s)
    a2_tmp = s*(a[1]*s + b[1]*c) + c*(b[1]*s + a[2]*c)
    b[1]   = c*(a[1]*s + b[1]*c) - s*(b[1]*s + a[2]*c)
    bulge  = s*-b[2]
    b[2]   = c* b[2]
    a[1]   = a1_tmp
    a[2]   = a2_tmp

    bulge
end

function cancel_bulge!(a::AbstractVector{Float64}, b::AbstractVector{Float64},
                       c::Float64, s::Float64,
                       bulge::Float64)
#    @assert abs(bulge) >= MACHEPS "bulge cannot be zero before \
#    cancellation."
    @assert size(a)[1] == 2     "input is not a 3x3 block due to a."
    @assert size(b)[1] == 2     "input is not a 3x3 block due to b."

    a1_tmp = c*(a[1]*c - b[2]*s) - s*(b[2]*c - a[2]*s)
    a2_tmp = s*(a[1]*s + b[2]*c) + c*(b[2]*s + a[2]*c)
    b1_tmp = b[1]*c - bulge*s

    b[2]   = c*(a[1]*s + b[2]*c) - s*(b[2]*s + a[2]*c)
    bulge  = b[1]*s + bulge*c
    a[1]   = a1_tmp
    a[2]   = a2_tmp
    b[1]   = b1_tmp

#    @assert abs(bulge) < 10.0*MACHEPS "bulge must be zero after cancellation."
end

function move_bulge!(a::AbstractVector{Float64}, b::AbstractVector{Float64},
                     c::Float64, s::Float64,
                     bulge::Float64)
#    @assert b[1]*s + bulge*c < 1000.0*MACHEPS (b[1], c, s, bulge,
#b[1]*s + bulge*c)

    a1_tmp = c*(a[1]*c - b[2]*s) - s*(b[2]*c - a[2]*s)
    a2_tmp = s*(a[1]*s + b[2]*c) + c*(b[2]*s + a[2]*c)
    b[1]   = b[1]*c - bulge*s
    b[2]   = c*(a[1]*s + b[2]*c) - s*(b[2]*s + a[2]*c)
    bulge  = -s*b[3]
    b[3]   =  c*b[3]
    a[1]   =  a1_tmp
    a[2]   =  a2_tmp

    bulge
end

function apply_givens_to_evec_row!(evec_row :: AbstractVector,
                                   c :: Float64, s :: Float64,
                                   i :: Int64)
    tau1          = evec_row[i]
    tau2          = evec_row[i+1]
    evec_row[i]   = c*tau1 + s*tau2
    evec_row[i+1] = -s*tau1 + c*tau2
end

# 2x2 matrix case
function apply_evec_to_evec_row!(evec_row :: AbstractVector,
                                 v1 :: Float64, v2 :: Float64,
                                 v3 :: Float64, v4 :: Float64,
                                 i :: Int64)
    tau1          = evec_row[i]
    tau2          = evec_row[i+1]
    evec_row[i]   = v1*tau1 - v3*tau2
    evec_row[i+1] = v2*tau1 + v4*tau2
end

function do_bulge_chasing!(a::AbstractVector, b::AbstractVector,
                           evec_row::AbstractVector,
                           s_idx::Int64, f_idx::Int64)
    @assert size(a)[1] - size(b)[1] == 1 "a, b dimension mismatch."

    # s_idx and f_idx represent bounds on the big submatrix.
    # We chase the bulge through this big submatrix without any checks.
    # a1 last index, a2 second to last index, b last index.
    shift = wilkinson_shift(a[f_idx], a[f_idx-1], b[f_idx-1])
    x = a[s_idx] - shift
    z = b[s_idx]
    c, s = givens_rotation(x, z)
    apply_givens_to_evec_row!(evec_row, c, s, s_idx)

    bulge = make_bulge!(view(a, s_idx:s_idx+1), view(b, s_idx:s_idx+1), c, s)

    x = b[s_idx]
    z = bulge

    p = s_idx
    q = f_idx

    for i = p+1:q-2
        c, s = givens_rotation!(x, z)
        apply_givens_to_evec_row!(evec_row, c, s, i)

        bulge = move_bulge!(view(a, i:i+1), view(b, i-1:i+1), c, s, bulge)

        if abs(b[i-1]) < MACHEPS*(abs(a[i-1]) + abs(a[i])) 
            b[i-1] = 0.0;
#            println("deflation!")
        end

        x = b[i]
        z = bulge
    end

    c, s = givens_rotation!(x, z)
    apply_givens_to_evec_row!(evec_row, c, s, q-1)

    cancel_bulge!(view(a, q-1:q), view(b, q-2:q-1), c, s, bulge)

    if abs(b[q-2]) < 2*MACHEPS*(abs(a[q-2]) + abs(a[q-1])) 
        b[q-2] = 0.0
#        println("deflation!")
    end

    if abs(b[q-1]) < 2*MACHEPS*(abs(a[q-1]) + abs(a[q]))
        b[q-1] = 0.0
#        println("deflation!")
    end
end

function qr_tridiag!(a :: AbstractVector, b :: AbstractVector, IDX :: Int64)
    N = size(a, 1)
    MAX_ITER = 3*N
    evec_row = zeros(N)
    evec_row[IDX] = 1.0
    converged = false
    s_idx = 1
    f_idx = N

    # TODO: an iteration is 1 to N.
    for i = 1:MAX_ITER
        @assert i != MAX_ITER "hit max iteration."
        # indices of a, not b.

        # sweep off-diagonal looking for non-zeros.
        for j = 1:N-1
            if b[j] != 0.0
                s_idx = j # s_idx -> j in a[j]; a[j] sits right above b[j].
                break
            end

            if j == N-1
                converged = true
            end
        end

        if converged
            break
        end

        for k = s_idx:N-1
            if b[k] == 0.0
                f_idx = k
                break
            end
        end

        @assert f_idx - s_idx > 0 "not a matrix."

        if f_idx - s_idx == 1
            # 2-by-2 matrix
#            println("triggers")
            evals, evecs = eigen!(SymTridiagonal(view(a, s_idx:f_idx+1),
                                                 view(b, s_idx:f_idx)))
            a[s_idx:f_idx+1] = evals
            b[s_idx] = 0.0
            apply_evec_to_evec_row!(evec_row, evecs[1,1], evecs[1,2],
                                              evecs[2,1], evecs[2,2],s_idx)
        end

#        println("s_idx: ", s_idx)
#        println("f_idx: ", f_idx)
        if f_idx-s_idx > 1
            do_bulge_chasing!(a, b, evec_row, s_idx, f_idx)
        end
    end

    p = sortperm(a)
    sort!(a)
    evec_row[p]
end

#=
Lanczos
=# 
struct LanczosContext
    A       :: Matrix{Float64}
    Qstore  :: Union{Matrix{Float64}, Nothing}
    Wstore  :: Union{Matrix{Float64}, Nothing}
    Rstore  :: Matrix{Float64}                  # Ritz residuals
    a       :: Vector{Float64}
    b       :: Vector{Float64}
    q       :: Vector{Float64}
    w_dists :: Vector{Float64}
    ν       :: DiscreteNonParametric            # true distribution
    k       :: Int64
    n       :: Int64
    orth    :: Char
end

getA(ctx :: LanczosContext) = view(ctx.A, :, 1:ctx.n)
getQstore(ctx :: LanczosContext) = view(ctx.Qstore, :, 1:ctx.k)
getQstore(ctx :: LanczosContext, j) = view(ctx.Qstore, :, 1:j)
getRstore(ctx :: LanczosContext) = view(ctx.Rstore, 1:ctx.k, 1:ctx.k)
getRstore(ctx :: LanczosContext, j) = view(ctx.Rstore, 1:j, j) # jth column.

get_a(ctx :: LanczosContext) = view(ctx.a, 1:ctx.k)
get_b(ctx :: LanczosContext) = view(ctx.b, 1:ctx.k)
get_a(ctx :: LanczosContext, j) = view(ctx.a, 1:j)
get_b(ctx :: LanczosContext, j) = view(ctx.b, 1:j)
get_q(ctx :: LanczosContext) = view(ctx.q, 1:ctx.n)

get_w_dists(ctx :: LanczosContext) = view(ctx.w_dists, 1:ctx.k)
get_w_dists(ctx :: LanczosContext, j) = view(ctx.w_dists, 1:j)

function get_residuals(ctx :: LanczosContext, j :: Int64)
    # b[j] |s[j]|
    a = get_a(ctx, j)
    b = get_b(ctx, j-1)
    evec_row = qr_tridiag!(deepcopy(a), deepcopy(b), j)

    R = getRstore(ctx, j)
    R[:] = abs.(evec_row) .* b[j-1]
    if norm(R, Inf) < sqrt(eps(Float64))
        println("converged to an eval at iteration ", j)
    end
end

function compute_μ(ctx :: LanczosContext, j)
    a = get_a(ctx, j)
    evals = copy(a)
    b = get_b(ctx, j-1)
    evec_row = qr_tridiag!(evals, copy(b), 1) # first row of evals.
    evec_row = evec_row.^2
    evec_row = evec_row / norm(evec_row, 1)

    discretemeasure(evals, evec_row)
end

function getWstore(ctx :: LanczosContext)
    if !(ctx.orth == 'f')
        return view(ctx.Wstore, :, 1:ctx.k+1)
    else
        return nothing
    end
end

function LanczosContext(A :: Matrix{Float64}, q :: Vector{Float64},
                        k :: Int64, ν :: DiscreteNonParametric, orth :: Char)
    @assert norm(q) ≈ 1.0 "pass a unit vector. "
    n = size(A, 1)
    Qstore = nothing
    Wstore = nothing

    # s -> selective orthogonalization
    # f -> full orthogonalization
    # u -> until the first orthogonalization
    if !(orth == 'u')
        Qstore = zeros(n, k)
    end

    if !(orth == 'f')
        Wstore = diagm(0  => ones(k+1),
                       -1 => eps(Float64)*ones(k), 
                       1  => eps(Float64)*ones(k))
    end

    Rstore = zeros(k, k)
    a = zeros(k)
    b = zeros(k)
    w_dists = zeros(k)

    LanczosContext(A, Qstore, Wstore, Rstore, a, b, q, w_dists, ν, k, n, orth)
end

function lanczos(ctx :: LanczosContext)
    if ctx.orth == 'f'
        lanczos_f(ctx)
    elseif ctx.orth == 's' 
        lanczos_s(ctx)
    elseif ctx.orth == 'u'
        lanczos_u(ctx)
    else
        error("not a valid orthogonalization scheme.")
    end
end

# Lanczos with full orthogonalization (orthogonalization at every step).
function lanczos_f(ctx :: LanczosContext)
    A = getA(ctx)
    q = get_q(ctx)
    a = get_a(ctx)
    b = get_b(ctx)

    z = A * q
    a[1] = q' * z
    z = z - a[1]*q
    b[1] = norm(z)

    for j = 2:ctx.k
        q_prev = q
        q = z / b[j-1]
        Q = getQstore(ctx, j)
        Q[:,j-1] = q

        z = A*q - b[j-1] * q_prev
        a[j] = q' * z
        z = z - a[j]*q
        b[j] = norm(z)

        if b[j] == 0
            break
        end

        z -= view(Q,:,1:j-1) * (view(Q,:,1:j-1)' * z)
        z -= view(Q,:,1:j-1) * (view(Q,:,1:j-1)' * z)
        # residuals
        get_residuals(ctx, j)        
        # Wasserstein distance.
        μ = compute_μ(ctx, j)
        ctx.w_dists[j] = wasserstein(μ, ctx.ν; p=Val(1))
    end
    ctx.k
end

# Lanczos with selective orthogonalization.
function lanczos_s(ctx :: LanczosContext)
    A = getA(ctx)
    q = get_q(ctx)
    a = get_a(ctx)
    b = get_b(ctx)
    W = getWstore(ctx)

    norm_A = norm(A)

    z = A * q
    a[1] = q' * z
    z = z - a[1]*q
    b[1] = norm(z)

    for j = 2:ctx.k
        q_prev = q
        q = z / b[j-1]
        Q = getQstore(ctx, j)
        Q[:,j-1] = q

        z = A*q - b[j-1] * q_prev
        a[j] = q' * z
        z = z - a[j]*q
        b[j] = norm(z)

        if b[j] == 0.0
            break
        end

        orthogonalized = false
        for i = 2:j
            w_tilde  = b[i]*W[j,i+1] + (a[i] - a[j])*W[j,i]
            w_tilde += b[i-1]*W[j,i-1] - b[j-1]*W[j-1,i]
            W[j+1,i] = (w_tilde + 2*sign(w_tilde)*eps(Float64)*norm_A)/b[j]

            if W[j+1,i] > sqrt(eps(Float64))
                if orthogonalized == false
                    U = @view Q[:,1:j-1]
                    U = Matrix(qr(U).Q)
                    @assert U' * U ≈ I(j-1)
                    z -= U * (U' * z)
                    z -= U * (U' * z)
                    orthogonalized = true
                end
                W[j+1,i] = eps(Float64)
                W[j,i] = eps(Float64)
                #break
            end
        end
        #residuals
        get_residuals(ctx, j)        
        # Wasserstein distance.
        μ = compute_μ(ctx, j)
        ctx.w_dists[j] = wasserstein(μ, ctx.ν; p=Val(1))
    end

    ctx.k
end

# Lanczos until the first reorthogonalization.
function lanczos_u(ctx :: LanczosContext)
    A = getA(ctx)
    q = get_q(ctx)
    a = get_a(ctx)
    b = get_b(ctx)
    W = getWstore(ctx)

    norm_A = norm(getA(ctx))

    z = A * q
    a[1] = q' * z
    z = z - a[1]*q
    b[1] = norm(z)

    for j = 2:ctx.k
        q_prev = q
        q = z / b[j-1]

        z = A*q - b[j-1] * q_prev
        a[j] = q' * z
        z = z - a[j]*q
        b[j] = norm(z)

        if b[j] == 0
            break
        end

        #residuals
        get_residuals(ctx, j)        
        # Wasserstein distance.
        μ = compute_μ(ctx, j)
        ctx.w_dists[j] = wasserstein(μ, ctx.ν; p=Val(1))

        for i = 2:j
            w_tilde  = b[i]*W[j,i+1] + (a[i] - a[j])*W[j,i]
            w_tilde += b[i-1]*W[j,i-1] - b[j-1]*W[j-1,i]
            W[j+1,i] = (w_tilde + 2*sign(w_tilde)*eps(Float64)*norm_A)/b[j]
            if W[j+1,i] > sqrt(eps(Float64))
                println("converged at iteration ", j)
                return j
            end
        end
    end

    ctx.k
end

function slq(A :: AbstractMatrix, nv :: Int64, m :: Int64)
    # nv - number of vectors (monte carlo samples).
    # m  - size of krylov subspace.
    n = size(A)[1]
    avg_evecs = zeros(m)
    Γ = zeros(nv)
    A_tmp = copy(A)
    tridiag_reduce!(A_tmp)
    evals, β = tridiag_params(A_tmp)
    evec_row = qr_tridiag!(evals, β, 1)
    ν = discretemeasure(evals)
    v = sign.(randn((n)))
    ctx = LanczosContext(A, v/norm(v), m, ν, 'u')

    for idx in 1:nv
#    for j in ProgressBar(1:NUM_THREADS:nv)
#        Threads.@threads for i = 1:NUM_THREADS
#            idx = j + i - 1
#            if idx > nv
#                continue
#            end
        v = sign.(randn((n)))
        ctx = LanczosContext(A, v/norm(v), m, ν, 'u')
        steps_taken = lanczos(ctx)
        evals = get_a(ctx, steps_taken)
        b = get_b(ctx, steps_taken-1)
        evec_row = qr_tridiag!(evals, b, 1)
        Γ[idx] = (evec_row.*evec_row)' * evals
        avg_evecs[1:steps_taken] .+= evec_row
#        end
    end
    #println("average of the evec rows ", avg_evecs / nv)
    #n*sum(Γ)/nv
    ctx
end

let
println("linear decay, 2 gaps, n = 1000")
n  = 1000  # number of evals (size of matrix A).
p  = 2     # number of standalone evals.
nv = 20   # number of Monte Carlo trials.
m  = 20   # number of Krylov steps.

evals = spectrum_linear_make(n, p)
A = mtrx_make(evals)
@time ctx = slq(A, nv, m)

println("exponential decay, 2 gaps, n = 1000")
n  = 1000 # number of evals (size of matrix A).
p  = 2    # number of standalone evals.
nv = 20   # number of Monte Carlo trials.
m  = 20   # number of Krylov steps.

evals = spectrum_exponential_make(n, p)
A = mtrx_make(evals)
@time ctx = slq(A, nv, m)

println("linear decay, 3 gaps, n = 1000")
n  = 1000  # number of evals (size of matrix A).
p  = 3     # number of standalone evals.
nv = 20   # number of Monte Carlo trials.
m  = 20   # number of Krylov steps.

evals = spectrum_linear_make(n, p)
A = mtrx_make(evals)
@time ctx = slq(A, nv, m)

println("exponential decay, 3 gaps, n = 1000")
n  = 1000 # number of evals (size of matrix A).
p  = 3    # number of standalone evals.
nv = 20   # number of Monte Carlo trials.
m  = 20   # number of Krylov steps.

evals = spectrum_exponential_make(n, p)
A = mtrx_make(evals)
@time ctx = slq(A, nv, m)

println("linear decay, no gaps, n = 1000")
n  = 1000 # number of evals (size of matrix A).
p  = 0    # number of standalone evals.
nv = 20   # number of Monte Carlo trials.
m  = 20   # number of Krylov steps.

evals = spectrum_linear_make(n, p)
A = mtrx_make(evals)
@time ctx = slq(A, nv, m)
gr()
scatter(evals)

#println("trace of A ", tr(A))
end
