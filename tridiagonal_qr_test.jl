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
    evals = sort(randn(n))
    a, b = tridiag_mtrx_make(evals)
    evals_lapack, evecs_lapack = eigen!(SymTridiagonal(a, b))
    evecs_lapack = evecs_lapack[1,:]
    p = sortperm(evals_lapack)
    evals_lapack = evals_lapack[p]
    evecs_lapack = evecs_lapack[p]

    evals_mine, b = tridiag_mtrx_make(evals)
#    evecs_mine = @profilehtml qr_tridiag!(evals_mine, b)
    evecs_mine = qr_tridiag!(evals_mine, b)
    p = sortperm(evals_mine)
    evals_mine = evals_mine[p]
    evecs_mine = evecs_mine[p]

    evec_err = norm(abs.(evecs_mine) - abs.(evecs_lapack), Inf)
    println("max evec error ", evec_err)

    evals_lapack_err = maximum(abs.(evals .- evals_lapack)./abs.(evals))
    println("max lapack eval error ", evals_lapack_err)

    evals_mine_err = maximum(abs.(evals .- evals_mine)./abs.(evals))
    println("max mine eval errror ", evals_mine_err)
    #y = abs.(a .- evals_stemr)./abs.(evals_stemr)
    #x = range(1, n, n)
    #plot(x, abs.(y), yaxis=:log, seriestype=:scatter)
end
