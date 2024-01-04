using TensorKrylov, Test
using Kronecker, TensorToolbox, LinearAlgebra

#@testset "Computation of matrix exponentials" begin
#
#    n = 200
#
#    A = SymTridiagonal(2ones(n), -ones(n-1))
#
#    γ = rand()
#    exact_exp  = exp(γ .* Matrix(A))
#    approx_exp = exponentiate(A, γ)
#
#    @test exact_exp ≈ approx_exp
#
#end
#
#@testset "Monotonic decrease of residual and error in A-norm" begin
#
#    d    = 2
#    n    = 100
#    Tₖ   = assemble_matrix(n, TensorLanczos{Float64})
#    A    = KroneckerMatrix{Float64}([Tₖ for _ in 1:d])
#    b    = [ rand(n) for _ in 1:d ]
#    nmax = 99
#
#    tensor_krylov_exact(A, b, nmax, TensorLanczos{Float64})
#
#end
#
#@testset "Symmetric example" begin
#
#    BLAS.set_num_threads(30)
#
#    d   = 10
#    n   = 200
#
#    spd_system      = TensorizedSystem{Float64}(n, d, TensorLanczosReorth{Float64})
#    nmax            = 199
#    #convergencedata = solve_tensorized_system(spd_system, nmax)
#    #display(convergencedata)
#
#end

@testset "Nonsymmetric example" begin

    BLAS.set_num_threads(30)

    d    = 5
    n    = 200

    c = 10
    h  = inv(n + 1)
    L  =     sparse( inv(h^2) .* SymTridiagonal( 2ones(n), -ones(n - 1)) )
    Aₛ = L + sparse( (c * inv(4 * h)) .* diagm(-1 => ones(n-1), 0 => 3ones(n), 1 => -5ones(n - 1), 2 => ones(n - 2)) )
    A = KronMat{Float64}([ Aₛ for _  in 1:d ])
    bs = rand(n)
    b = [ bs for _ in 1:d ]
    normalize!(b)
    nmax            = 199
    convergencedata = ConvergenceData{Float64}(nmax)
    tensor_krylov!(convergencedata, A, b, 1e-9, nmax, TensorArnoldi{Float64} )
    nonsym_system   = TensorizedSystem{Float64}(n, d, TensorArnoldi{Float64})
    #convergencedata = solve_tensorized_system(nonsym_system, nmax)
    println(convergencedata.relative_residual_norm)
    display(convergencedata)
    
end
