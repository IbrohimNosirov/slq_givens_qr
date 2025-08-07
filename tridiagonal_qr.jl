using LinearAlgebra
#=
TODO:
- Diagonal doesn't allocate vs diagm, which does.
- what is the relationship between tol and norm(b)?
-check for type stability. Most of the time is spent in compilation.
DONE:
-make a T_block and givens_mtrx to carry around for in-place operations.
-use @views to access a and b -> ends up costing more in garbage collection
=#

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

function make_bulge!(a :: AbstractVector, b :: AbstractVector,
                     c :: Float64, s :: Float64)
    T_block  = diagm(0 => a[1:3], -1 => b[1:2], 1 => b[1:2])
    T_block[3,1], T_block[1,3] = 0.0, 0.0
    T_block = G * T_block * G'
    a[1:3] = diag(T_block)
    b[1:2] = diag(T_block, -1)

    T_block[3,1]
end

function cancel_bulge!(T_block :: AbstractMatrix, a :: AbstractVector,
    b :: AbstractVector, bulge :: Float64, G :: LinearAlgebra.Givens{Float64})
    T_block = diagm(0 => a[end-2:end], -1 => b[end-1:end], 1 => b[end-1:end])
    T_block[3,1], T_block[1,3] = bulge, bulge
    T_block = G * T_block * G'
    a[end-2:end] = diag(T_block)
    b[end-1:end] = diag(T_block, -1)

    T_block[3, 1]
end

function move_bulge!(T_block :: AbstractMatrix, a :: AbstractVector,
    b :: AbstractVector, bulge :: Float64, j :: Int64,
    G :: LinearAlgebra.Givens{Float64})
    @assert j >= 1 "index too small."
    @assert j < size(b)[1] "index too large."
    T_block = diagm(0 => a[j:j+3], 1 => b[j:j+2], -1 => b[j:j+2])
    T_block[3, 1], T_block[1, 3] = bulge, bulge
    T_block[4, 2], T_block[2, 4] = 0.0, 0.0 
    T_block = G * T_block * G'
    a[j:j+3] = diag(T_block)
    b[j:j+2] = diag(T_block, -1)

    T_block[4, 2]
end

function apply_givens_to_evec_row!(evec_row :: AbstractVector, c :: Float64,
    s :: Float64, i :: Int64)
    tau1 = evec_row[i]
    tau2 = evec_row[i+1]
    evec_row[i]   = c*tau1 - s*tau2
    evec_row[i+1] = s*tau1 + c*tau2
end

function do_bulge_chasing!(T_block :: AbstractMatrix, a :: AbstractVector,
    b :: AbstractVector, evec_row ::AbstractVector)
    #=a givens rotation moves makes a bulge in the first iteration and cancels
    it in the last iteration. As such iter 1, n are 3x3 operations. everything
    in between are 4x4 operations.=#
    @assert size(a)[1] - size(b)[1] == 1

    shift = wilkinson_shift(a[end], a[end-1], b[end])
    n = size(b)[1]
    x = a[1] - shift
    z = b[1]
    G, _ = givens(x, z, 1, 2)
    #apply_givens_to_evec_row!(evec_row, c, s, 1)
    bulge = make_bulge!(view(T_block, 1:3, 1:3), a, b, G)
    x = b[1]
    z = bulge

    for i = 2:n-1
        G, _ = givens(x, z, 2, 3)
    #    apply_givens_to_evec_row!(evec_row, c, s, i)
        bulge = move_bulge!(T_block, a, b, bulge, i-1, G)
        x = b[i]
        z = bulge
    end
    G, _ = givens(x, z, 2, 3)
    #apply_givens_to_evec_row!(evec_row, c, s, n-1)
    #givens_mtrx = [[c, -s] [s, c]]

    bulge = cancel_bulge!(view(T_block, 2:4, 2:4), a, b, bulge, G)
    @assert abs(bulge) < 1e-15
end

function qr_tridiag!(a :: AbstractVector, b :: AbstractVector, max_iter=1000,
    tol=1e-8)
    n = size(a)[1]
    evec_row = zeros(n)
    evec_row[1] = 1.0
    T_block = zeros(4,4)

    for iter = 1:2000
#        if iter == max_iter
#            println("hit max_iteration")
#            break
#        end
#        if norm(b, Inf) < tol
#            println("stopped at iteration ", iter)
#            break
#        end
        do_bulge_chasing!(T_block, a, b, evec_row)
    end
end
