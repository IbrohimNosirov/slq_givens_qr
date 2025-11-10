using LinearAlgebra
using Test
using Plots
using Profile
using StatProfilerHTML

include("tridiagonal_qr.jl")
include("matrix_gallery.jl")

@testset "Givens rotation test " begin
    # 1 3
    # 2 4
    A = [1. 2.;
         3. 4.]
    display(A)
    c, s = givens_rotation(1., 3.)
    givens_mtrx = [c -s;
                   s  c]
    A = givens_mtrx * A
    @test A[2,1] ≈ 0
end

@testset "Wilkinson shift test " begin
    a1 = 1.0
    a2 = 2.0
    b  = 3.0
    @test wilkinson_shift(a1, a2, b) ≈ -1.5413812651491101
end

let
    n = 1000
    a = 2 * ones(n)
    b = -1 * ones(n-1)
    evals_stemr, evecs_stemr = eigen!(SymTridiagonal(a, b))
    p = sortperm(evals_stemr)
    evals_stemr = evals_stemr[p]
    evec_stemr = evecs_stemr[p] 

    evals_mine = 2 * ones(n)
    b = -1 * ones(n-1)
    evec_row = @profilehtml qr_tridiag!(evals_mine, b)
    p = sortperm(evals_mine)
    evals_mine = evals_mine[p]
    evec_row = evec_row[p] 

    evec_err = norm(abs.(evec_row) - abs.(evecs_stemr[1,:]), Inf)
    println("max evec error ", evec_err)

    eval_err = maximum(abs.(evals_mine .- evals_stemr)./abs.(evals_stemr))
    println("max eval error ", eval_err)
    #y = abs.(a .- evals_stemr)./abs.(evals_stemr)
    #x = range(1, n, n)
    #plot(x, abs.(y), yaxis=:log, seriestype=:scatter)
end

#let
#    a = collect(range(1, 100, 100)) .+ 100.0
#    b = ones(99)
#    evals_stemr, evecs_stemr = @time eigen!(SymTridiagonal(a, b))
#    a = collect(range(1, 100, 100)) .+ 100.0
#    b = ones(99)
#    evec_row = @time qr_tridiag!(a, b)
#    println("evec_row error ", (evecs_stemr[1,:] - evec_row))
#    println("stemr evals ", evals_stemr)
#    println("evals ", a)
#    y = (sort!(a) .- sort!(evals_stemr))./norm(evals_stemr, Inf)
#    x = range(0, 100, 100)
#    plot(x, abs.(y), yaxis=:log, seriestype=:scatter)
#end

#let
#    x = Integer.(round.(collect(10 .^ range(1, 2, length=10))))
#    y1 = zeros(10)
#    y2 = zeros(10)
#    time_eigen(a,b) = @timed eigen!(SymTridiagonal(a, b))
#    time_qr_tridiag(a,b) = @timed qr_tridiag!(a, b)
#
#    for i = 1:10
#        n = x[i]
#        a = 2 * ones(n)
#        b = -1 * ones(n-1)
#        y1[i] = time_eigen(a,b)[2]
#        a = 2 * ones(n)
#        b = -1 * ones(n-1)
#        y2[i] = time_qr_tridiag(a,b)[2]
#    end
#
#    println("y1 ", y1)
#    println("y2 ", y2)
#    plot(x,  y1, xaxis=:log, yaxis=:log)
#    plot!(x, y2, xaxis=:log, yaxis=:log)
#end
#
#let
#    n = 10
#    evals = collect(range(1, 100, n))
#    tridiag_mtrx_make(evals)
#end

#@testset "qr_tridiag! tests" begin
#n = 100
#sep_evals_num = 2
#sep_evals = sep_evals_make(sep_evals_num)
#clustered_evals = clustered_evals_make(n - sep_evals_num)
#evals = [clustered_evals; sep_evals]
#end
