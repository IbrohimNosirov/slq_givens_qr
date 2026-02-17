using LinearAlgebra
using Plots

function kronecker_quasirand_vec(N, start=0)
  d = 1
  φ = 1.0 + 1.0/d
  for k = 1:10
    gφ = φ^(d + 1) - φ - 1
    dgφ= (d + 1)*φ^d - 1
    φ -= gφ/dgφ
  end
  αs = [mod(1.0/φ^j, 1.0) for j = 1:d]
  # Compute the quasi-random sequence.
  z = zeros(N)
  for j = 1:N
    z[j] = mod(0.5 + (start+j)*αs[d], 1.0)
  end

  z
end

# TODO: don't form H as a dense matrix.

# This gives me a matrix of Householder reflectors inside of A. 
function reduce_tridiag!(A :: AbstractMatrix)
  n = size(A, 1)
  τ = view(A, 1:n, n)
  for k = 1:n-2
    x = view(A, k+1:n, k)
    τk = LinearAlgebra.reflector!(x)
    LinearAlgebra.reflectorApply!(x, τk, view(A, k+1:n, k+1:n))
    LinearAlgebra.reflectorApply!(x, τk, view(A, k+1:n, k+1:n)')
    τ[k] = τk
  end
end

function tridiag_params!(A, alpha, beta)
  n = size(A, 1)
  for j = 1:n-1
    alpha[j] = A[j,j]
    beta[j] = A[j+1,j]
  end
  alpha[n] = A[n,n]
  alpha, beta
end

tridiag_params(A) = tridiag_params!(A, zeros(size(A,1)), zeros(size(A,1)-1))
get_tridiag(A) = SymTridiagonal(tridiag_params(A))

function make_matrix(evals :: AbstractVector)
  n = size(evals, 1)
  u = randn(n)
  H = I - 2u*u'./(u'*u) # any unitary transformation.
  A = H * diagm(evals) * H'
  for i = 1:10
    u = randn(n)
    H = I - 2u*u'./(u'*u) # any unitary transformation.
    @assert cond(H) ≈ 1.0
    A = H * A * H'
  end

  A
end

function make_tridiag_matrix(evals :: AbstractVector)
  A = make_matrix(evals)
  reduce_tridiag!(A)
  a = diag(A)
  b = diag(A, -1)

  @assert evals ≈ eigen(SymTridiagonal(a, b)).values
  a, b
end

struct Interval
  start  :: Float64
  finish :: Float64 
end

function Interval(start :: Float64, finish :: Float64)
  @assert start  > 0
  @assert finish > 0 
  @assert finish - start >= 0

  Interval(start, finish)
end

function make_functional_decay!(evals :: AbstractVector, interval :: Interval, fun :: Function)
  evals_count = size(evals, 1)
  @assert evals_count > 3

  interval_range = interval.finish - interval.start
  evals .= collect(range(1, evals_count)) ./ interval_range .+ interval.start
  evals .= fun.(evals) .+ 2*sqrt(eps(Float64))*rand(evals_count)

  sort!(evals)
end

function make_cluster!(evals :: AbstractVector, interval :: Interval, epsilon :: Float64)
  evals_count = size(evals, 1)
  @assert evals_count > 0
  @assert epsilon > 1e-8
  
  seed = evals_count * 42
  # there are two different kinds of partition: by index (i) and by range (e).
  interval_range = interval.finish - interval.start
  evals .= epsilon .* kronecker_quasirand_vec(evals_count) ./ interval_range .+ interval.start

  sort!(evals)
end

# gaussian kernel
gaussian(r) = exp(-r^2)

# Matérn ν = 1/2 (C^0)
matern_1_2(r, l=1.0) = exp(-r/l)

# Matérn ν = 3/2 (C^2)
matern_3_2(r, l=1.0) = (1 + sqrt(3)*r/l) * exp(-sqrt(3)*r/l)

# Matérn ν = 5/2 (C^4)
matern_5_2(r, l=1.0) = (1 + sqrt(5)*r/l + 5*r^2/(3*l^2)) * exp(-sqrt(5)*r/l)

# Matérn ν = 7/2 (C^6)
matern_7_2(r, l=1.0) = (1 + sqrt(7)*r/l + 14*r^2/(5*l^2) + 7*sqrt(7)*r^3/(15*l^3)) * exp(-sqrt(7)*r/l)

# Matérn ν = 9/2 (C^8)
matern_9_2(r, l=1.0) = (1 + 3*r/l + 27*r^2/(7*l^2) + 18*r^3/(7*l^3) + 27*r^4/(35*l^4)) * exp(-3*r/l)

# Matérn ν = 11/2 (C^10)
matern_11_2(r, l=1.0) = (1 + sqrt(11)*r/l + 55*r^2/(9*l^2) + 55*sqrt(11)*r^3/(27*l^3) + 1375*r^4/(567*l^4) + 275*sqrt(11)*r^5/(1701*l^5)) * exp(-sqrt(11)*r/l)

# Matérn ν = 13/2 (C^12)
matern_13_2(r, l=1.0) = (1 + sqrt(13)*r/l + 26*r^2/(3*l^2) + 13*sqrt(13)*r^3/(9*l^3) + 169*r^4/(54*l^4) + 169*sqrt(13)*r^5/(486*l^5) + 2197*r^6/(4374*l^6)) * exp(-sqrt(13)*r/l)

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

#let
## 100 eigenvalues in 1 cluster, each with radius 1e-7
#evals_count = 100
#epsilon = 1e-7
#evals = zeros(evals_count)
#make_cluster!(evals, Interval(0,1), epsilon)
#gr()
#scatter(range(1,evals_count), evals)
#end


