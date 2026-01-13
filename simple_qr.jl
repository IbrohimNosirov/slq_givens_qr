using LinearAlgebra
using Profile
using StatProfilerHTML
include("matrix_gallery.jl")

const MACHEPS = eps(Float64)

# NEED AT MINIMUM 20 asserts, HAVE 5 asserts.

function givens_rotation(x :: Float64, z :: Float64)
    G, r = givens(x, z, 1, 2)
    g = G*[1.; 0.]
    g[1], g[2]
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
    shift = wilkinson_shift(a[s_idx], a[f_idx-1], b[f_idx-1])
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
        c, s = givens_rotation(x, z)
        apply_givens_to_evec_row!(evec_row, c, s, i)

        bulge = move_bulge!(view(a, i:i+1), view(b, i-1:i+1), c, s, bulge)

        abs(b[i-1]) < MACHEPS*(abs(a[i-1]) + abs(a[i])) && (b[i-1] = 0.0;
println("deflation!"))

        x = b[i]
        z = bulge
    end

    c, s = givens_rotation(x, z)
    apply_givens_to_evec_row!(evec_row, c, s, q-1)

    cancel_bulge!(view(a, q-1:q), view(b, q-2:q-1), c, s, bulge)

    abs(b[q-2]) < 2*MACHEPS*(abs(a[q-2]) + abs(a[q-1])) && (b[q-2] = 0.0;
println("deflation!"))
    abs(b[q-1]) < 2*MACHEPS*(abs(a[q-1]) + abs(a[q]))   && (b[q-1] = 0.0;
println("deflation!"))
end

function qr_tridiag!(a :: AbstractVector, b :: AbstractVector, IDX :: Int64)
    N = size(a, 1)
    MAX_ITER = 2*N
    evec_row = zeros(N)
    evec_row[IDX] = 1.0

    for i = 1:MAX_ITER
        s_idx = 1
        f_idx = N
        for j = 1:N-2
#            if b[j] != 0.0
#                s_idx = j
#                break
#            end
            b[j] != 0.0 && (s_idx = j; break)
        end
        for k = s_idx:N-2
#            if b[k+1] == 0.0
#                println("triggered")
#                f_idx = k
#                break
#            end
            b[k+1] == 0.0 && (f_idx = k; break)
        end

        # TODO: this never triggers.
        if f_idx - s_idx == 0
            # 2-by-2 matrix
            println("triggers")
            evals, evecs = eigen!(SymTridiagonal(view(a, s_idx:f_idx+1),
                                                 view(b, s_idx:f_idx)))
            a[s_idx:f_idx+1] = evals
            b[s_idx] = 0.0
            apply_evec_to_evec_row!(evec_row, evecs[1,1], evecs[1,2],
                                              evecs[2,1], evecs[2,2],s_idx)
        end

        println("s_idx: ", s_idx)
        println("f_idx: ", f_idx)
        if f_idx-s_idx > 0 
            do_bulge_chasing!(a, b, evec_row, s_idx, f_idx)
        end
    end

    p = sortperm(a)
    sort!(a)
    evec_row[p]
end

let
    n = 100
    evals = spectrum_linear_make(n, 0)
    a, b = tridiag_mtrx_make(evals)
    evals_lapack, evecs_lapack = eigen!(SymTridiagonal(a, b))
    evecs_lapack = evecs_lapack[1,:]
    p = sortperm(evals_lapack)
    evals_lapack = evals_lapack[p]
    evecs_lapack = evecs_lapack[p]

    evals_mine, b = tridiag_mtrx_make(evals)
    evecs_mine = @profilehtml qr_tridiag!(evals_mine, b, 1)
#    evecs_mine = qr_tridiag!(a, b, 1)
    p = sortperm(evals_mine)
    evals_mine = evals_mine[p]
    evecs_mine = evecs_mine[p]

#    evec_err = norm(abs.(evecs_mine) - abs.(evecs_lapack), Inf)
#    println("max evec error ", evec_err)
#
    evals_lapack_err = maximum(abs.(evals .- evals_lapack)./abs.(evals))
    println("max lapack eval error ", evals_lapack_err)

    evals_mine_err = maximum(abs.(evals .- evals_mine)./abs.(evals))
    println("max mine eval errror ", evals_mine_err)
end
