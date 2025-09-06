using LinearAlgebra

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
    if abs(d) < 1e-14 && abs(b) < 1e-14
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
    bulge  = -b[2]*s
    b[2]   = c*b[2]
    a[1]   = a1_tmp
    a[2]   = a2_tmp

    @assert bulge != 0.0 "bulge was not properly initialized."
    bulge
end

function cancel_bulge!(a::AbstractVector{Float64}, b::AbstractVector{Float64},
    c::Float64, s::Float64, bulge::Float64)

    #    @assert abs(bulge) >= eps(Float64) "bulge cannot be zero before cancellation."
    @assert size(a)[1] == 2     "input is not a 2x2 block due to a."
    @assert size(b)[1] == 2     "input is not a 2x2 block due to b."

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
    c::Float64, s::Float64, bulge::Float64)

#    @assert abs(bulge) >= 1e-15 "bulge cannot be 0 before movement."

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

function do_bulge_chasing!(a :: AbstractVector, b :: AbstractVector,
    evec_row ::AbstractVector, p :: Int64, q :: Int64)
    
    @assert size(a)[1] - size(b)[1] == 1

    shift = wilkinson_shift(a[q], a[q-1], b[q-1])
    x = a[p] - shift
    z = b[p]
    c, s = givens_rotation(x, z)
#    apply_givens_to_evec_row!(evec_row, c, s, p)
    bulge = make_bulge!(view(a, p:p+1), view(b, p:p+1), c, s)
    x = b[p]
    z = bulge
    println("p ", p)
    println("q ", q)
    if q - p == 2
        println("2x2 eigensolver should be called now.")
        evals, evecs = eigen!(SymTridiagonal(view(a,p:q-1), view(b,p:p)))
        a[p:q-1] = evals
        b[p] = 0.0
        # TODO: apply evecs to evec row
        return
    end

    for i = p:q-3
        c, s = givens_rotation(x, z)
#        apply_givens_to_evec_row!(evec_row, c, s, i) # TODO: CHECK

        bulge = move_bulge!(view(a, i:i+1), view(b, i:i+2), c, s, bulge)

        if abs(b[i]) <= sqrt(eps(Float64))*(abs(a[i]) + abs(a[i+1]))
            display(b)
            println("TRIGGER DEFLATION!")
            b[i] = 0.0

            p_large = p
            q_large = i
            p_small = i+1 #TODO check
            q_small = q

            do_bulge_chasing!(a, b, evec_row, p_large, q_large) # big box.
            do_bulge_chasing!(a, b, evec_row, p_small, q_small) # small box.
        end

        x = b[i+1]
        z = bulge
    end

    c, s = givens_rotation(x, z)
#    apply_givens_to_evec_row!(evec_row, c, s, q)

    cancel_bulge!(view(a, q-1:q), view(b, q-2:q-1), c, s, bulge)
end

function qr_tridiag!(a :: AbstractVector, b :: AbstractVector, max_iter=1000,
    tol=1e-8)
    n = size(a)[1]
    evec_row = zeros(n)
    evec_row[1] = 1.0

    for iter = 1:max_iter
        if iter == max_iter
            println("hit max_iteration")
            break
        end
#        if iter % 100 == 0
#            if norm(b, Inf) < tol
#                println("stopped at iteration ", iter)
#                break
#            end
#        end
        do_bulge_chasing!(a, b, evec_row, 1, n)
    end
    evec_row
end
