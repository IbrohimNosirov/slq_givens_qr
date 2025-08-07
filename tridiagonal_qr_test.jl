using LinearAlgebra
using Test
include("tridiagonal_qr.jl")

#@testset "Givens rotation test " begin
#    # 1 3
#    # 2 4
#    A = [[1., 2.] [3., 4.]]
#    G, _ = givens(1., 2., 1, 2)
#    A = G * A
#    @test A[2,1] ≈ 0
#end

#let
#    a = ones(10)
#    a[10] = 0
#    b = ones(9)
#    println("wilkinson shift ", wilkinson_shift(a,b))
#end

#let
#    a = ones(10)
#    a[10] = 0
#    b = ones(9)
#    c, s, r = givens(a[1], b[1])
#    givens_mtrx = [[c, -s] [s, c]]
#    println("make bulge ", make_bulge!(a, b, givens_mtrx))
#end

#let
#    a = ones(10)
#    a[10] = 0
#    b = ones(9)
#    c, s, r = givens(a[1], b[1])
#    givens_mtrx = [[c, -s] [s, c]]
#    bulge = 3.0
#    println("cancel bulge ", cancel_bulge!(a, b, bulge, givens_mtrx))
#end

#let
#    a = ones(10)
#    a[10] = 0
#    b = ones(9)
#    c, s, r = givens(a[1], b[1])
#    givens_mtrx = [[c, -s] [s, c]]
#    bulge = 3.0
#    println("move bulge ", move_bulge!(a, b, bulge, 7, givens_mtrx))
#end

let
    a_cold_start = collect(range(1, 100, 10))
    b_cold_start = ones(9)
    qr_tridiag!(a_cold_start,b_cold_start)
    evals = collect(range(1, 1000, 1000))
    b = ones(999)
    evec_row = @time qr_tridiag!(evals, b)
    a = collect(range(1, 1000, 1000))
    b = ones(999)
    evals_stemr, evec_row_stemr = @time eigen!(SymTridiagonal(a,b))
    println("evals_givens error ", norm(evals - evals_stemr, Inf))
    println("evec relative error ", norm(evec_row - evec_row_stemr[1,:])/norm(evec_row_stemr[1,:]))
#    println("evals_stemr ", evals_stemr)
#    println("evec_row_stemr ", evec_row_stemr[1,:])
end
