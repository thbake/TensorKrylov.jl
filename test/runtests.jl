module TestSets

    using TensorKrylov
    using Test
    using Random

    include("test_utils.jl")

    Random.seed!(12345)

    @testset "TensorKrylov tests" begin 

        @testset "Low rank approximation tests" begin
            include("approximation.jl")
        end

        @testset "Decomposition tests" begin
           include("decompositions.jl")
        end

        @testset "Eigenvalue tests" begin
            include("eigenvalues.jl")
        end

        @testset "Kronecker product structures" begin
            include("tensor_struct.jl")
        end

        #@testset "Utils tests" begin
        #    include("utils.jl")
        #end

        @testset "Tensor Krylov subspace method tests" begin
            include("tensor_krylov_method.jl")
        end

    end

end
