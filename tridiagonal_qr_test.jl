using LinearAlgebra
using Profile
using OptimalTransport
using Distributions
using DataStructures
using StatProfilerHTML

include("tridiagonal_qr.jl")
include("matrix_gallery.jl")
#include("simple_qr.jl")

#let
#evals_count = 100
#evals = zeros(evals_count)
#make_functional_decay!(evals, Interval(0,1), gaussian)
#println(evals)
#gr()
#scatter(range(1,evals_count), evals)
#end
#
#let
#evals_count = 100
#evals = zeros(evals_count)
#make_functional_decay!(evals, Interval(0,1), matern_1_2)
#println(evals)
#gr()
#scatter(range(1,evals_count), evals)
#end
#
#let
#evals_count = 100
#evals = zeros(evals_count)
#make_functional_decay!(evals, Interval(0,1), matern_3_2)
#println(evals)
#gr()
#scatter(range(1,evals_count), evals)
#end
#
#let
#evals_count = 100
#evals = zeros(evals_count)
#make_functional_decay!(evals, Interval(0,1), matern_5_2)
#println(evals)
#gr()
#scatter(range(1,evals_count), evals)
#end
#
#let
#evals_count = 100
#evals = zeros(evals_count)
#make_functional_decay!(evals, Interval(0,1), matern_7_2)
#println(evals)
#gr()
#scatter(range(1,evals_count), evals)
#end
#
#let
#evals_count = 100
#evals = zeros(evals_count)
#make_functional_decay!(evals, Interval(0,1), matern_9_2)
#println(evals)
#gr()
#scatter(range(1,evals_count), evals)
#end
#
#let
#evals_count = 100
#evals = zeros(evals_count)
#make_functional_decay!(evals, Interval(0,1), matern_11_2)
#println(evals)
#gr()
#scatter(range(1,evals_count), evals)
#end
#
#let
#evals_count = 100
#evals = zeros(evals_count)
#make_functional_decay!(evals, Interval(0,1), matern_13_2)
#println(evals)
#gr()
#scatter(range(1,evals_count), evals)
#end
#
#let
## 100 eigenvalues in 1 cluster, each with radius 1e-7
#evals_count = 100
#epsilon = 1e-7
#evals = zeros(evals_count)
#make_cluster!(evals, Interval(0,1), epsilon)
#gr()
#scatter(range(1,evals_count), evals)
#end

# tridiagonal QR tests
let
        evals_count = 7
        evals = zeros(evals_count)
        make_functional_decay!(evals, Interval(0,1), matern_1_2)

        evals_mine, subdiagonal = make_tridiag_matrix(evals)
        evecs_mine = zeros(evals_count)
        qr_tridiag!(evals_mine, subdiagonal, evecs_mine, 1)

        evals_count = 10
        evals = zeros(evals_count)
        make_functional_decay!(evals, Interval(0,1), matern_1_2)

        evals_mine, subdiagonal = make_tridiag_matrix(evals)
        evecs_mine = zeros(evals_count)
        println("started eigensolve")
        @time qr_tridiag!(evals_mine, subdiagonal, evecs_mine, 1)

        evals = zeros(evals_count)
        make_functional_decay!(evals, Interval(0,1), matern_1_2)

        evals_mine, subdiagonal = make_tridiag_matrix(evals)
        evecs_mine = zeros(evals_count)
        evecs_mine = @profilehtml qr_tridiag!(evals_mine, subdiagonal, evecs_mine, 1)

        diagonal, subdiagonal = make_tridiag_matrix(evals)
        println("started LAPACK solver ")
        evals_lapack, evecs_lapack = @time eigen!(SymTridiagonal(diagonal, subdiagonal))
        evecs_lapack = evecs_lapack[1,:]

        evec_err_lapack = sum(evecs_lapack .* evecs_lapack)
        evec_err_mine = sum(evecs_mine .* evecs_mine)
        println("mine QQ^T ", evec_err_mine)
        println("lapack QQ^T ", evec_err_lapack)
        
        evals_lapack_err = maximum(abs.(evals .- evals_lapack) ./ abs.(evals))
        println("max lapack eval error ", evals_lapack_err)

        evals_mine_err = maximum(abs.(evals .- evals_mine) ./ abs.(evals))
        println("max mine eval error ", evals_mine_err)
end
