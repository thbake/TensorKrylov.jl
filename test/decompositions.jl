using TensorKrylov
using LinearAlgebra

@testset "Arnoldi and Lanczos steps" begin

    n = 1000
    k = 500
    h = inv(n + 1)

    A = inv(h^2) * SymTridiagonal( 2ones(n), -1ones(n - 1) )
    b = rand(n)

    arnoldi = arnoldi_algorithm(A, b, k)
    lanczos = lanczos_algorithm(A, b, k)

    @test isorthonormal(arnoldi, k)
    @test isorthonormal(lanczos, k-1)

end

@testset "Multiple Arnoldi and Lanczos decomposition steps" begin

    d  = 5
    n = 200

    Aₛ = assemble_matrix(n, TensorLanczos{Float64})
    A  = KroneckerMatrix{Float64}([Aₛ for _ in 1:d])
    b  = [ rand(n) for _ in 1:d ]

    k = 50

    # Initialize Arnoldi and Lanczos decompositions
    t_arnoldi = TensorArnoldi{Float64}(A)
    t_lanczos = TensorLanczos{Float64}(A)

    initial_orthonormalization!(t_arnoldi, b, Arnoldi{Float64})
    initial_orthonormalization!(t_lanczos, b, Lanczos{Float64})

    for j in 2:k

        orthonormal_basis!(t_arnoldi, j)
        orthonormal_basis!(t_lanczos, j)

    end

    # Test for positive definiteness
    for s in 1:d

        @test isposdef(t_lanczos.H[s][1:k, 1:k])

    end

    @test isorthonormal(t_arnoldi, k)
    @test isorthonormal(t_lanczos, k)

end
