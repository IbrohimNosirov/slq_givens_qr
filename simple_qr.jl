using LinearAlgebra
using DataStructures

function givens_rotation(x :: Float64, z :: Float64)
    if z == 0.0
        c = sign(x)
        if c == 0.0
            c = 1.0
        end
        s = 0.0
    elseif x == 0.0
        c = 0.0
        s = -sign(z)
    elseif abs(x) > abs(z)
        tau = z / x
        u = sign(x) * sqrt(1 + tau * tau)
        c = 1.0 / u
        s = -c * tau
    else
        tau = x / z
        u = sign(z) * sqrt(1 + tau*tau)
        s = -1.0 / u
        c = tau / u
    end

    c, s
end

function wilkinson_shift(a1:: Float64, a2 :: Float64, b :: Float64)
    # a1 last index, a2 second to last index, b last index.
    d = (a2 - a1)/2
    if abs(d) < eps(Float64) && abs(b) < eps(Float64)
        return a1
    end
    denominator = d + sign(d)*sqrt(d*d + b*b)
    if abs(denominator) < eps(Float64)
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
#    @assert abs(bulge) >= eps(Float64) "bulge cannot be zero before \
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

    @assert abs(bulge) < eps(Float64) "bulge must be zero after cancellation."
end

function move_bulge!(a::AbstractVector{Float64}, b::AbstractVector{Float64},
                     c::Float64, s::Float64,
                     bulge::Float64)
    @assert b[1]*s + bulge*c < 1e-14 b[1]*s + bulge*c "bulge didn't move properly."

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

function apply_evec_to_evec_row!(evec_row :: AbstractVector,
                                 v1 :: Float64, v2 :: Float64,
                                 v3 :: Float64, v4 :: Float64,
                                 i :: Int64)
    tau1          = evec_row[i]
    tau2          = evec_row[i+1]
    evec_row[i]   = v1*tau1 - v3*tau2
    evec_row[i+1] = v2*tau1 + v4*tau2
end

function do_bulge_chasing!(a :: AbstractVector, b :: AbstractVector,
                           p :: Int64, q :: Int64,
                           evec_row :: AbstractVector,
                           b_max :: Int64)
    @assert size(a)[1] - size(b)[1] == 1
    @assert q - p > 0 "one block; invalid input."

    if q - p == 1
        #println("reached base case!")
        evals, evecs = eigen!(SymTridiagonal(view(a,p:p+1), view(b,p:p)))
        a[p:p+1] = evals
        b[p] = 0.0
        apply_evec_to_evec_row!(evec_row, evecs[1,1], evecs[1,2],
                                          evecs[2,1], evecs[2,2], p)
        return 0
    end

    q_tmp = q
    shift = wilkinson_shift(a[q], a[q-1], b[q-1])
    x = a[p] - shift
    z = b[p]
    c, s = givens_rotation(x, z)
    apply_givens_to_evec_row!(evec_row, c, s, p)

    bulge = make_bulge!(view(a, p:p+1), view(b, p:p+1), c, s)

    x = b[p]
    z = bulge

    for i = p+1:q-2
        c, s = givens_rotation(x, z)
        apply_givens_to_evec_row!(evec_row, c, s, i)

        bulge = move_bulge!(view(a, i:i+1), view(b, i-1:i+1), c, s, bulge)

        if abs(b[i-1]) < eps(Float64)*(abs(a[i-1]) + abs(a[i]))
            #println("trigger deflation.")
            b[i-1] = 0.0
            # if there are multiple deflations on one sweep.
            if q == q_tmp
                q_tmp = i-1 # TODO: not sure what the right index should be.
            end
        end

        x = b[i]
        z = bulge
    end

    c, s = givens_rotation(x, z)
    apply_givens_to_evec_row!(evec_row, c, s, q-1)

    cancel_bulge!(view(a, q-1:q), view(b, q-2:q-1), c, s, bulge)

    if abs(b[q-2]) < eps(Float64)*(abs(a[q-2]) + abs(a[q-1]))
        b[q-2] = 0.0
        # if there are multiple deflations in one sweep. 
        if q == q_tmp
            q_tmp = q-2
        end
    end

    if q != q_tmp
        return q_tmp
    end
end

function find_new_interval(b :: AbstractVector)
    # TODO: find the first two non-zeros in b.
end

function qr_tridiag!(a :: AbstractVector, b :: AbstractVector)
    n = size(a)[1]
    evec_row = zeros(n)
    evec_row[1] = 1.0

    p = 1
    q = n
    b_max = maximum(b)

    while b_max > sqrt(eps(Float64))
        q = do_bulge_chasing!(a, b, p, q, evec_row, b_max)
        if q == 0
            p, q = find_new_interval(b)
        end
    end
    evec_row
end

let
    n = 10
    a = 2 * ones(n)
    b = -1 * ones(n-1)
    evals_stemr, evecs_stemr = eigen!(SymTridiagonal(a, b))
    a = 2 * ones(n)
    b = -1 * ones(n-1)
    evec_row = qr_tridiag!(a, b)
    sort!(a)
    println("evec difference ",
        norm(sort!(abs.(evec_row)) - sort!(abs.(evecs_stemr[1,:])), Inf))
    y = (a .- evals_stemr)./norm(evals_stemr, Inf)
    x = range(1, n, n)
    plot(x, abs.(y), yaxis=:log, seriestype=:scatter)
end

let
    a = collect(range(1, 10, 10)) .+ 10.0
    b = ones(9)
    evals_stemr, evecs_stemr = @time eigen!(SymTridiagonal(a, b))
    a = collect(range(1, 10, 10)) .+ 10.0
    b = ones(9)
    evec_row = @time qr_tridiag!(a, b)
    println("evec_row error ", (evecs_stemr[1,:] - evec_row))
    println("stemr evals ", evals_stemr)
    println("evals ", a)
    y = (sort!(a) .- sort!(evals_stemr))./norm(evals_stemr, Inf)
    x = range(1, 10, 10)
    plot(x, abs.(y), yaxis=:log, seriestype=:scatter)
end
