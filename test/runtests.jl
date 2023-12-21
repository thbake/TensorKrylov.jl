using TensorKrylov
using Test
using Random

Random.seed!(12345)


@testset "TensorKrylov tests" begin 

    #@testset "Preprocessing tests" begin
    #    include("preprocessing.jl")
    #end

    #@testset "Decomposition tests" begin
    #   include("decompositions.jl")
    #end

    #@testset "Eigenvalue tests" begin
    #    include("eigenvalues.jl")
    #end

    #@testset "Kronecker product structures" begin
    #    include("tensor_struct.jl")
    #end

    #@testset "Utils tests" begin
    #    include("utils.jl")
    #end

    @testset "Tensor Krylov subspace method tests" begin
        include("tensor_krylov_method.jl")
    end

end
