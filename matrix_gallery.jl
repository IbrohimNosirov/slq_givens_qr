using LinearAlgebra
using Plots
using Random

function tridiag_reduce!(A::AbstractMatrix)
    n = size(A)[1]
    for k = 1:n-2
        x = view(A, k+1:n, k)
        τk = LinearAlgebra.reflector!(x)
        LinearAlgebra.reflectorApply!(x, τk, view(A, k+1:n, k+1:n))
        LinearAlgebra.reflectorApply!(x, τk, view(A, k+1:n, k+1:n)')
    end
end

#function gappy_matrix_make(n::Int64, lo::Float64, hi::Float64, num_gaps::Int64)
#    evals_remain = n
#    evals = []
#    subinterval_len = (hi - lo) / (num_gaps+1)
#    sub_lo = lo
#    sub_hi = 0
#    for i=1:num_gaps
#        sub_hi = sub_lo + sub_interval_len
#        if i == num_gaps
#            sub_interval = collect(range(sub_lo, sub_hi, evals_remain)
#        else
#            sub_interval = collect(range(sub_lo, sub_hi,
#                            Integer(round(n / num_gaps))))
#        end
#        evals
#    end
#end

function clustered_matrix(n::Int64, lo::Float64, hi::Float64, eps)
    @assert n > 2 "there must be at least 3 clusters."

    range_width = hi - lo
    cluster1_center = lo + range_width / 4
    cluster2_center = lo + range_width / 2
    cluster3_center = hi - range_width / 4


    n1 = Integer(round(n / 3))
    n2 = n1
    n3 = n - n1 - n2

    cluster1 = cluster1_center .+ rand(n1)*eps
    cluster2 = cluster1_center .+ rand(n2)*eps
    cluster3 = cluster1_center .+ rand(n3)*eps

    evals = [cluster1; cluster2; cluster3]
    @assert n ≈ length(evals)
    plot(evals, seriestype=:scatter)
    
    #evals
end

function even_matrix(n::Int64)
    evals = collect(range(1, 10, n))
    Q = qr!(randn(n,n)).Q
    A = Q' * diagm(evals) * Q
    tridiag_reduce!(A)
    A
end
