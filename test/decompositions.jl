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

    Aₛ = assemble_matrix(n, Laplace)
    A  = KronMat{SymInstance}([Aₛ for _ in 1:d])
    b  = [ rand(n) for _ in 1:d ]

    k = 50

    # Initialize Arnoldi and Lanczos decompositions
    t_arnoldi = TensorArnoldi(A)
    t_lanczos = TensorLanczos(A)

    orthonormalize!(t_arnoldi, b)
    orthonormalize!(t_lanczos, b)

    for j in 2:k

        orthonormalize!(t_arnoldi, j)
        orthonormalize!(t_lanczos, j)

    end

    # Test for positive definiteness
    for s in 1:d

        @test isposdef(t_lanczos.H[s][1:k, 1:k])

    end

    @test isorthonormal(t_arnoldi, k)
    @test isorthonormal(t_lanczos, k)

end
