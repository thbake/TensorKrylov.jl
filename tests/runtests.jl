using TensorKrylov
using Test
using Random

Random.seed(12345)

@testset "TensorKrylov tests" begin

	#@testset "Tensor structures test" begin
	#	include("tensor_structures_tests.jl")
	#end

	#@testset "Orthogonality tests" begin
	#	include("orthonormal_bases_tests.jl")
	#end
    #
end

@testset "Tensor Krylov subspace method tests" begin
    include("tensor_krylov_method.jl")
end
