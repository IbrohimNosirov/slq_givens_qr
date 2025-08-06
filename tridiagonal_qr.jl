using LinearAlgebra

# TODO: use @views to access a and b.
# use Givens for Givens rotation
# make a T_block and givens_mtrx to carry around for in-place operations.

function givens(x :: Float64, z :: Float64)
    if z == 0.0
        c = sign(x)
        s = 0.0
        r = abs(x)
    elseif x == 0.0
        c = 0.0
        s = -sign(z)
        r = abs(z)
    elseif abs(x) > abs(z)
        tau = z / x
        u = sign(x) * sqrt(1 + tau * tau)
        c = 1 / u
        s = -c * tau
        r = x * u
    else
        tau = x / z
        u = sign(z) * sqrt(1 + tau*tau)
        s = -1 / u
        c = tau / u
        r = z * u
    end

    c, s, r
end

function wilkinson_shift(a :: Vector{Float64}, b :: Vector{Float64})
    d = (a[end-1] - a[end])/2
    if abs(d) < 1e-14 && abs(b[end]) < 1e-14
        return a[end]
    end
    denominator = d + sign(d)*sqrt(d*d + b[end]*b[end])
    if abs(denominator) < 1e-14
        return a[end]
    end
    shift = a[end] - (b[end]*b[end])/denominator

    shift
end

function make_bulge!(a :: Vector{Float64}, b :: Vector{Float64},
                    givens_mtrx :: Matrix{Float64})
    T_block  = diagm(0 => a[1:3], -1 => b[1:2], 1 => b[1:2])
    givens_block = diagm(0 => ones(3))
    givens_block[1:2, 1:2] = givens_mtrx
    result = givens_block' * T_block * givens_block
    a[1:3] = diag(result)
    b[1:2] = diag(result, -1)

    result[3,1]
end

function cancel_bulge!(a :: Vector{Float64}, b :: Vector{Float64},
                      bulge :: Float64, givens_mtrx :: Matrix{Float64})
    T_block = diagm(0 => a[end-2:end], -1 => b[end-1:end], 1 => b[end-1:end])
    T_block[3,1], T_block[1,3] = bulge, bulge
    givens_block = diagm(0 => ones(3))
    givens_block[2:end, 2:end] = givens_mtrx
    result = givens_block' * T_block * givens_block
    a[end-2:end] = diag(result)
    b[end-1:end] = diag(result, -1)

    result[3, 1]
end

function move_bulge!(a :: Vector{Float64}, b :: Vector{Float64},
    bulge :: Float64, j :: Int64, givens_mtrx :: Matrix{Float64})
    @assert j >= 1 "index too small."
    @assert j < size(b)[1] "index too large."
    T_block = diagm(0 => a[j:j+3], 1 => b[j:j+2], -1 => b[j:j+2])
    T_block[3, 1], T_block[1, 3] = bulge, bulge
    givens_block = diagm(0 => ones(4))
    givens_block[2:end-1, 2:end-1] = givens_mtrx
    result = givens_block' * T_block * givens_block
    a[j:j+3] = diag(result)
    b[j:j+2] = diag(result, -1)

    return result[4, 2]
end

function apply_givens_to_evec_row!(evec_row :: Vector{Float64}, c :: Float64,
    s :: Float64, i :: Int64)
    tau1 = evec_row[i]
    tau2 = evec_row[i+1]
    evec_row[i]   = c*tau1 - s*tau2
    evec_row[i+1] = s*tau1 + c*tau2
end

function do_bulge_chasing!(a :: Vector{Float64}, b :: Vector{Float64},
    evec_row ::Vector{Float64})
    #=a givens rotation moves makes a bulge in the first iteration and cancels
    it in the last iteration. As such iter 1, n are 3x3 operations. everything
    in between are 4x4 operations.=#
    @assert size(a)[1] - size(b)[1] == 1

    shift = wilkinson_shift(a, b)
    x = a[1] - shift
    z = b[1]
    c, s, _ = givens(x, z)
    apply_givens_to_evec_row!(evec_row, c, s, 1)
    givens_mtrx = [[c, -s] [s, c]]
    bulge = make_bulge!(a, b, givens_mtrx)
    x = b[1]
    z = bulge

    for i = 2:size(b)[1]-1
        c, s, _ = givens(x, z)
        apply_givens_to_evec_row!(evec_row, c, s, i)
        givens_mtrx = [[c, -s] [s, c]]
        bulge = move_bulge!(a, b, bulge, i-1, givens_mtrx)
        x = b[i]
        z = bulge
    end
    c, s, _ = givens(x, z)
    apply_givens_to_evec_row!(evec_row, c, s, size(b)[1]-1)
    givens_mtrx = [[c, -s] [s, c]]

    bulge = cancel_bulge!(a, b, bulge, givens_mtrx)
    @assert abs(bulge) < 1e-15
end

function qr_tridiag!(a :: Vector{Float64}, b :: Vector{Float64}, max_iter=1000,
    tol=1e-8)
    n = size(a)[1]
    evec_row = zeros(n)
    evec_row[1] = 1.0

    for iter = 1:max_iter
        if iter == max_iter
#            println("hit max_iteration")
            break
        end
        if norm(b, Inf) < tol
#            println("stopped at iteration ", iter)
            break
        end
        do_bulge_chasing!(a, b, evec_row)
    end
    a, evec_row
end
