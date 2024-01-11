using TensorKrylov, Test
using Kronecker, TensorToolbox, LinearAlgebra

@testset "Computation of matrix exponentials" begin

    n = 200

    A = SymTridiagonal(2ones(n), -ones(n-1))

    γ = rand()
    exact_exp  = exp(γ .* Matrix(A))
    approx_exp = exponentiate(A, γ)

    @test exact_exp ≈ approx_exp

end

@testset "Monotonic decrease of residual and error in A-norm" begin

    d    = 2
    n    = 100
    Tₖ   = assemble_matrix(n,  Laplace{Float64})
    A    = KroneckerMatrix{Float64, SymInstance}([Tₖ for _ in 1:d], Laplace{Float64})
    b    = [ rand(n) for _ in 1:d ]
    nmax = 99

    tensor_krylov_exact(A, b, nmax, TensorLanczos{Float64})

end

@testset "Symmetric example" begin

    BLAS.set_num_threads(30)

    d   = 10
    n   = 200

    A               = KronMat{Float64, SymInstance}(d, n, Laplace{Float64})
    b               = random_rhs(d, n)
    spd_system      = TensorizedSystem{Float64, SymInstance}(A, b)
    nmax            = 199
    convergencedata = solve_tensorized_system(spd_system, nmax, TensorLanczosReorth{Float64})
    #display(convergencedata)

end

@testset "Nonsymmetric example" begin

    BLAS.set_num_threads(30)

    d    = 5
    n    = 200

    A               = KronMat{Float64, NonSymInstance}(d, n, ConvDiff{Float64})
    b               = random_rhs(d, n)
    nonsym_system   = TensorizedSystem{Float64, NonSymInstance}(A, b)
    nmax            = 199
    #convergencedata = solve_tensorized_system(nonsym_system, nmax, TensorArnoldi{Float64})
    #display(convergencedata)
    
end
