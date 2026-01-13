# Author: Ibrohim Nosirov
# Date: 2025-06-26
# from https://www.netlib.org/utk/people/JackDongarra/etemplates/node110.html#wrecursion
using LinearAlgebra

include("matrix_gallery.jl")
include("tridiagonal_qr.jl")

# Needs a lot more assertions.

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
    evec_row = qr_tridiag!(evals, copy(b)) # first row of evals.
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


#let
#    n = 50
#    k = 50
#    evals_true = collect(range(1, n))
#    #evals = collect(exp.(-range(1, n)))
#    A = mtrx_make(evals_true)
#    A_tmp = copy(A)
#    tridiag_reduce!(A_tmp)
#    evals, β = tridiag_params(A_tmp)
#    evec_row = qr_tridiag!(evals, β)
#    ν = discretemeasure(evals)
#    q = kronecker_quasirand_vec(n)
##    display(A)
#
#    ctx = LanczosContext(A, q/norm(q), k, ν,'f')
#    evals_lanczos, β = lanczos(ctx)
#    evec_row_lanczos = qr_tridiag!(evals_lanczos, β)
##    println(evals_lanczos)
#    relative_diff = abs.(evals_lanczos - evals_true)./abs.(evals_true)
##    display(maximum(relative_diff))
##    display(getRstore(ctx))
#
#    ctx_selective = LanczosContext(A, q/norm(q), k, ν, 's')
#    evals_lanczos, β = lanczos(ctx_selective)
#    evec_row_lanczos = qr_tridiag!(evals_lanczos, β)
##    println(evals_lanczos)
##    display(getRstore(ctx_selective))
#    relative_diff = abs.(evals_lanczos - evals_true)./abs.(evals_true)
##    display(maximum(relative_diff))
#
#    ctx_until_first = LanczosContext(A, q/norm(q), k, ν,'u')
#    evals_lanczos, β, j = lanczos(ctx_until_first)
#    evec_row_lanczos = qr_tridiag!(evals_lanczos, β)
#    display(get_w_dists(ctx_until_first, j))
##    display(getRstore(ctx_until_first))
##    relative_diff = abs.(evals_lanczos - evals_true)./abs.(evals_true)
##    display(maximum(relative_diff))
#end
