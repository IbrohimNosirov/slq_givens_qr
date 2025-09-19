using LinearAlgebra
using Test
using Plots

include("tridiagonal_qr.jl")
include("matrix_gallery.jl")

@testset "Givens rotation test " begin
    # 1 3
    # 2 4
    A = [[1., 2.] [3., 4.]]
    c, s = givens_rotation(1., 2.)
    givens_mtrx = [[c, -s] [s, c]]
    A = givens_mtrx' * A
    @test A[2,1] ≈ 0
end

@testset "Wilkinson shift test " begin
    a1 = 1.0
    a2 = 2.0
    b  = 3.0
    @test wilkinson_shift(a1, a2, b) ≈ -1.5413812651491101
end

@testset "make_bulge!() test " begin
    a = ones(10)
    a[10] = 0.0
    b = ones(9) * 0.4
    c, s = givens_rotation(a[1], b[1])
    bulge = 0.0
    println("type of view ", view(b,1:2))
    bulge = make_bulge!(view(a,1:2), view(b,1:2), c, s)
    @test bulge ≈ 0.1485562705416415
end

#@testset "move_bulge!() test " begin
#    a = ones(10)
#    a[10] = 0
#    b = ones(9)
#    c, s = givens_rotation(a[1], b[1])
#    bulge = 3.0
#    bulge = move_bulge!(a, b, c, s, bulge)
#    bulge ≈ 0.7071067811865475
#end

let
    n = 10
    a_cold_start = 2 * ones(n)
    b_cold_start = -1 * ones(n-1)
    evals_stemr, evecs_stemr = @time eigen!(SymTridiagonal(a_cold_start,
                                    b_cold_start))
    a_cold_start = 2 * ones(n)
    b_cold_start = -1 * ones(n-1)
    evec_row = @time qr_tridiag!(a_cold_start, b_cold_start)
    sort!(a_cold_start)
    println("evec_row mine", evec_row)
    println("evec_row stemr", evecs_stemr[1,:])
    println("stemr evals ", evals_stemr)
    println("evals ", a_cold_start)
    y = (a_cold_start .- evals_stemr)./abs.(evals_stemr)
    x = range(1, n, n)
    plot(x, abs.(y), yaxis=:log, seriestype=:scatter)
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
#    x = Integer.(round.(collect(10 .^ range(1, 4, length=10))))
#    y1 = zeros(10)
#    y2 = zeros(10)
#    time_eigen(a,b) = @timed eigen!(SymTridiagonal(a, b))
#    time_qr_tridiag(a,b) = @timed qr_tridiag!(a, b)
#
#    for i = 1:10
#        a = collect(range(1, x[i]*10, x[i]))
#        b = ones(x[i] - 1)
#        y1[i] = time_eigen(a,b)[2]
#        a = collect(range(1, x[i]*10, x[i]))
#        b = ones(x[i] - 1)
#        y2[i] = time_qr_tridiag(a,b)[2]
#    end
#
#    println("y1 ", y1)
#    println("y2 ", y2)
#    plot(x,  y1, xaxis=:log, yaxis=:log)
#    plot!(x, y2, xaxis=:log, yaxis=:log)
#end

#let
#    n = 100
#    lo = 1.0
#    hi = 1000.0
#    eps = 10.0
#    even_matrix(n)
#end
