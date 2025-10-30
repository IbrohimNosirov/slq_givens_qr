using LinearAlgebra
using Plots

function tridiag_reduce!(A :: AbstractMatrix)
    n = size(A)[1]
    for k = 1:n-2
        x = view(A, k+1:n, k)
        τ = LinearAlgebra.reflector!(x)
        LinearAlgebra.reflectorApply!(x, τ, view(A, k+1:n, k+1:n))
        LinearAlgebra.reflectorApply!(x, τ, view(A, k+1:n, k+1:n)')
    end
end

function mtrx_make(evals :: AbstractVector)
    n = size(evals)[1]
    u = ones(n,1)
    H = I - 2u*u'./(u'*u) # any unitary transformation.
    A = H * diagm(evals) * H'

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

let
n = 6
evals = collect(range(1, n))
a, b = tridiag_mtrx_make(evals)
end
