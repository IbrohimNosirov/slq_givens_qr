using LinearAlgebra
using Plots

function kronecker_quasirand_vec(N, start=0)
    d = 1
    φ = 1.0 + 1.0/d
    for k = 1:10
        gφ = φ^(d + 1) - φ - 1
        dgφ= (d + 1)*φ^d - 1
        φ -= gφ/dgφ
    end
    αs = [mod(1.0/φ^j, 1.0) for j = 1:d]
    # Compute the quasi-random sequence.
    z = zeros(N)
    for j = 1:N
        z[j] = mod(0.5 + (start+j)*αs[d], 1.0)
    end

    z
end

# TODO: don't form H as a dense matrix.

# This gives me a matrix of Householder reflectors inside of A. 
function tridiag_reduce!(A :: AbstractMatrix)
    n = size(A, 1)
    τ = view(A, 1:n, n)
    for k = 1:n-2
        x = view(A, k+1:n, k)
        τk = LinearAlgebra.reflector!(x)
        LinearAlgebra.reflectorApply!(x, τk, view(A, k+1:n, k+1:n))
        LinearAlgebra.reflectorApply!(x, τk, view(A, k+1:n, k+1:n)')
        τ[k] = τk
    end
end

function tridiag_params!(A, alpha, beta)
    n = size(A, 1)
    for j = 1:n-1
        alpha[j] = A[j,j]
        beta[j] = A[j+1,j]
    end
    alpha[n] = A[n,n]
    alpha, beta
end

tridiag_params(A) = tridiag_params!(A, zeros(size(A,1)), zeros(size(A,1)-1))
get_tridiag(A) = SymTridiagonal(tridiag_params(A))

function mtrx_make(evals :: AbstractVector)
    n = size(evals)[1]
    u = randn(n)
    H = I - 2u*u'./(u'*u) # any unitary transformation.
    A = H * diagm(evals) * H'
    for i = 1:10
        u = randn(n)
        H = I - 2u*u'./(u'*u) # any unitary transformation.
        @assert cond(H) ≈ 1.0
        A = H * A * H'
    end

    A
end

function tridiag_mtrx_make(evals :: AbstractVector)
    A = mtrx_make(evals)
    tridiag_reduce!(A)
    a = diag(A)
    b = diag(A, -1)

    @assert evals ≈ eigen!(SymTridiagonal(a, b)).values
    a, b
end

# TODO: write a function with n number of eigenvalues, m number of well
# separated evals, exponential decay on the rest, and k condition number.
# to impove condition number, just add the order of magnitude to the whole
# spectrum.

function sep_evals_make(n :: Int64)
    @assert n < 10 "to be well-separated, there shouldn't be too many of them."
    evals = collect(range(1,n)) ./ n
    evals
end

function exponential_decay_make(n :: Int64)
    @assert n > 3 "there should be many evals to really get exponential decay."
    evals = exp.(-collect(range(1,n))) .+ 2*sqrt(eps(Float64))
    sort!(evals)
    evals
end

function spectrum_make(n :: Int64, sep_evals_num :: Int64)
    sep_evals = sep_evals_make(sep_evals_num)
    clustered_evals = exponential_decay_make(n - sep_evals_num)
    evals = [clustered_evals; sep_evals]
end

## Main
#n = 100
#sep_evals_num = 2
#sep_evals = sep_evals_make(sep_evals_num)
#clustered_evals = clustered_evals_make(n - sep_evals_num)
#evals = [clustered_evals; sep_evals]
#println("evals ", evals)
#println("condition number ", maximum(evals)/minimum(evals))
#println(evals[1] - evals[2])
#plt = plot(range(1,n), reshape(evals, length(evals), 1), seriestype=scatter)
#display(plt)

#let
#n = 6
#evals = collect(range(1, n))
#a, b = tridiag_mtrx_make(evals)
#end
