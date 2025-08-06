using LinearAlgebra
using Test
include("tridiagonal_qr.jl")


#@testset "Givens rotation test " begin
#    # 1 3
#    # 2 4
#    A = [[1., 2.] [3., 4.]]
#    c, s, r = givens(1., 2.)
#    givens_mtrx = [[c, -s] [s, c]]
#    A = givens_mtrx' * A
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
    a = collect(range(1, 100, 10))
    b = ones(9)
    evals_givens, evec_row_givens = @time qr_tridiag!(a,b)
    a = collect(range(1, 100, 10))
    b = ones(9)
    evals_stemr, evec_row_stemr = @time eigen!(SymTridiagonal(a,b))
    println("evals_givens ", evals_givens)
    println("evec_row_givens ", evec_row_givens)
    println("evals_stemr ", evals_stemr)
    println("evec_row_stemr ", evec_row_stemr[1,:])
end
