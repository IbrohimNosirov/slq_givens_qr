using LinearAlgebra
using DataStructures

function givens_rotation(x :: Float64, z :: Float64)
    if z == 0.0
        c = sign(x)
        s = 0.0
    elseif x == 0.0
        c = 0.0
        s = -sign(z)
    elseif abs(x) > abs(z)
        tau = z / x
        u = sign(x) * sqrt(1 + tau * tau)
        c = 1 / u
        s = -c * tau
    else
        tau = x / z
        u = sign(z) * sqrt(1 + tau*tau)
        s = -1 / u
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
    if abs(denominator) < 1e-14
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
    a2_tmp = s*(a[1]*s - b[1]*c) + c*(b[1]*s + a[2]*c)
    b[1]   = c*(a[1]*s + b[1]*c) - s*(b[1]*s + a[2]*c)
    bulge  = s*-b[2]
    b[2]   = c* b[2]
    a[1]   = a1_tmp
    a[2]   = a2_tmp

#    @assert abs(bulge) > eps(Float64) "bulge was not properly initialized."
    #TODO: b[2] can't be zero since it should have been deflated away.
    bulge
end

function cancel_bulge!(a::AbstractVector{Float64}, b::AbstractVector{Float64},
    c::Float64, s::Float64, bulge::Float64, bounds_stack::Stack, p::Int64, q::Int64)

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

    if abs(b[2]) <= eps(Float64)*(abs(a[1]) + abs(a[2]))
        println("trigger deflation.")
        b[2] = 0.0

        p_large = p
        q_large = q-1
        pop!(bounds_stack)
        push!(bounds_stack, (p_large, q_large))
    end

    @assert abs(bulge) < eps(Float64) "bulge must be zero after cancellation."
end

function move_bulge!(a::AbstractVector{Float64}, b::AbstractVector{Float64},
                     c::Float64, s::Float64,
                     bulge::Float64)
#    @assert abs(bulge) >= eps(Float64) "bulge cannot be 0 before movement."

    a1_tmp = c*(a[1]*c - b[2]*s) - s*(b[2]*c - a[2]*s)
    a2_tmp = s*(a[1]*s + b[2]*c) + c*(b[2]*s + a[2]*c)
    b[1]   = b[1]*c - bulge*s
    b[2]   = c*(a[1]*s + b[2]*c) - s*(b[2]*s + a[2]*c)
    bulge  = -s*b[3]
    b[3]   = c*b[3]
    a[1]   = a1_tmp
    a[2]   = a2_tmp

#    @assert abs(bulge) >= 1e-16 "bulge cannot be 0 after movement."
    bulge
end

function apply_givens_to_evec_row!(evec_row :: AbstractVector, c :: Float64,
    s :: Float64, i :: Int64)
    tau1          = evec_row[i]
    tau2          = evec_row[i+1]
    evec_row[i]   = c*tau1 - s*tau2
    evec_row[i+1] = s*tau1 + c*tau2
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

function do_bulge_chasing!(a::AbstractVector, b::AbstractVector,
                           p::Int64, q::Int64,
                           evec_row::AbstractVector,
                           bounds_stack::Stack)
    @assert size(a)[1] - size(b)[1] == 1

    if q - p == 0
        pop!(bounds_stack)
        return
    end

    if q - p == 1
        println("reached base case!")
        evals, evecs = eigen!(SymTridiagonal(view(a,p:p+1), view(b,p:p)))
        a[p:p+1] = evals
        b[p] = 0.0
        apply_evec_to_evec_row!(evec_row, evecs[1,1], evecs[1,2],
                                          evecs[2,1], evecs[2,2], p)
        pop!(bounds_stack)
        return
    end

    shift = wilkinson_shift(a[q], a[q-1], b[q-1])
    x = a[p] - shift
    z = b[p]
    c, s = givens_rotation(x, z)
    apply_givens_to_evec_row!(evec_row, c, s, p)

    bulge = make_bulge!(view(a, p:p+1), view(b, p:p+1), c, s)

    x = b[p]
    z = bulge

    for i = p:q-3
        c, s = givens_rotation(x, z)
        apply_givens_to_evec_row!(evec_row, c, s, i+1)

        bulge = move_bulge!(view(a, i:i+1), view(b, i:i+2), c, s, bulge)

        if abs(b[i]) <= eps(Float64)*(abs(a[i]) + abs(a[i+1]))
            println("trigger deflation.")
            b[i] = 0.0

            p_large = p
            q_large = i
            p_small = i+1
            q_small = q
            pop!(bounds_stack)
            push!(bounds_stack, (p_small, q_small)) # stacks are LIFO
            push!(bounds_stack, (p_large, q_large))
        end

        x = b[i+1]
        z = bulge
    end

    c, s = givens_rotation(x, z)
    apply_givens_to_evec_row!(evec_row, c, s, q-1)

    cancel_bulge!(view(a, q-1:q), view(b, q-2:q-1), c, s, bulge, bounds_stack, p, q)
end

function qr_tridiag!(a :: AbstractVector, b :: AbstractVector)
    n = size(a)[1]
    evec_row = zeros(n)
    evec_row[1] = 1.0

    # TODO: make stack for each step of deflation [[p,q], [p,q], ...].
    # iterate through each element in the array and run bulge chasing on each
    # bound. if deflation occurs, add the resulting [p,q] to the stack.
    bounds_stack = Stack{Tuple{Int64, Int64}}()
    push!(bounds_stack, (1,n))

    while norm(b, Inf) > sqrt(eps(Float64))
        @assert !isempty(bounds_stack) display(b)
        println("bounds stack ", bounds_stack)
        p, q = first(bounds_stack)
        do_bulge_chasing!(a, b, p, q, evec_row, bounds_stack)
    end

    evec_row
end
