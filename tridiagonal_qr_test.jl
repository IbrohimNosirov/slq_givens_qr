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

let
    a1 = 1.0
    a2 = 2.0
    b  = 3.0
    println("wilkinson shift ", wilkinson_shift(a1, a2, b))
end

let
    a = ones(10)
    a[10] = 0
    b = ones(9) * 0.4
    c, s = givens_rotation(a[1], b[1])
    bulge = 0.0
    bulge = make_bulge!(view(a,1:2), view(b,1:2), c, s, bulge)
    println("bulge ", bulge)
end

# too hard to test on its own
#let
#    a = ones(10)
#    a[10] = 0
#    b = ones(9)
#    c, s = givens_rotation(a[1], b[1])
#    bulge = 3.0
#    println("cancel bulge ", cancel_bulge!(view(a,1:2), view(b,1:2), c, s,
#    bulge))
#end

let
    a = ones(10)
    a[10] = 0
    b = ones(9)
    c, s = givens_rotation(a[1], b[1])
    bulge = 3.0
    bulge = move_bulge!(a, b, c, s, bulge)
    println("bulge ", bulge)
end

let
    a_cold_start = collect(range(1, 100, 10))
    b_cold_start = ones(9)
    qr_tridiag!(a_cold_start,b_cold_start)
#    evals = collect(range(1, 100, 100))
#    b = ones(99)
#    evec_row = @time qr_tridiag!(evals, b)
#    a = collect(range(1, 100, 100))
#    b = ones(99)
#    evals_stemr, evec_row_stemr = @time eigen!(SymTridiagonal(a,b))
#    println("evals_givens error ", norm(evals - evals_stemr, Inf))
#    println("evec relative error ", norm(evec_row - evec_row_stemr[1,:])/norm(evec_row_stemr[1,:]))
#    println("evals_mine ", evals)
#    println("evals_stemr ", evals_stemr)
#    println("evec_row_stemr ", evec_row_stemr[1,:])
#plot( ((evals - evals_stemr) ./ (abs.(evals_stemr)))[4:end] )
end
