using LinearAlgebra
using Test
using Plots
include("tridiagonal_qr.jl")

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

#TODO gets tested implicitly by the assert inside the function.

@testset "move_bulge!() test " begin
    a = ones(10)
    a[10] = 0
    b = ones(9)
    c, s = givens_rotation(a[1], b[1])
    bulge = 3.0
    bulge = move_bulge!(a, b, c, s, bulge)
    bulge ≈ 0.7071067811865475
end

let
    a_cold_start = collect(range(1, 100, 100))
    b_cold_start = ones(99)
#    evals_stemr, evec_row_stemr = @time eigen!(SymTridiagonal(a_cold_start,
#                                    b_cold_start))
#    println("evals_stemr ", evals_stemr)
    println("evec row", qr_tridiag!(a_cold_start, b_cold_start))
#    evals = collect(range(1, 100, 100))
#    b = ones(99)
#    evec_row = @time qr_tridiag!(evals, b)
#    a = collect(range(1, 100, 100))
#    b = ones(99)
#    evals_stemr, evec_row_stemr = @time eigen!(SymTridiagonal(a,b))
#    display((evals .- evals_stemr)./(evals_stemr))
#    println("evec relative error ", norm(evec_row - evec_row_stemr[1,:])/norm(evec_row_stemr[1,:]))
#    println("evals_mine ", evals)
#    println("evals_stemr ", evals_stemr)
#    println("evec_row_stemr ", evec_row_stemr[1,:])
#logplot( ((evals - evals_stemr) ./ (abs.(evals_stemr))) )
end
