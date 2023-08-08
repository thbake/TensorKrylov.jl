using TensorKrylov
using LinearAlgebra

@testset "Arnoldi and Lanczos steps" begin

    n = 1000

    h = inv(n + 1)

    A = inv(h^2) * Tridiagonal( -1ones(n - 1), 2ones(n), -1ones(n - 1) )
    
    b = rand(n)

    k = 500

    # Initialize Arnoldi decomposition
    
    arnoldi = Arnoldi{Float64}(rand(n, n), zeros(n, k + 1), zeros(k + 1, k), b)
    lanczos = Lanczos{Float64}(A, zeros(n, k), zeros(k, k), b)

    for j in 1:k

        orthonormal_basis_vector!(arnoldi, j)

    end

    for j in 2:k-1

        orthonormal_basis_vector!(lanczos, j)

    end

    test_arnoldi = zeros(k, k)
    test_lanczos = zeros(k, k)

    arnoldi_basis = @view(arnoldi.V[:, 1:k])
    lanczos_basis = @view(lanczos.V[:, 1:k])

    LinearAlgebra.mul!(test_arnoldi, transpose(arnoldi_basis), arnoldi_basis)

    LinearAlgebra.mul!(test_lanczos, transpose(lanczos_basis), lanczos_basis)

    @test test_arnoldi[1:k, 1:k] ≈ I(k)
    @test test_lanczos[1:k, 1:k] ≈ I(k)

end

@testset "Multiple Arnoldi and Lanczos decomposition steps" begin

    d = 10

    nₛ = 1000

    h = inv(nₛ + 1)

    Aₛ= inv(h^2) * Tridiagonal( -1ones(nₛ - 1) , 2ones(nₛ), -1ones(nₛ - 1) )

    A = KroneckerMatrix{Float64}([Aₛ for _ in 1:d])

    b = [ rand(nₛ) for _ in 1:d ]

    k = 500

    # Initialize Arnoldi and Lanczos decompositions
    t_arnoldi = TensorArnoldi{Float64}(A)
    t_lanczos = TensorLanczos{Float64}(A)

    for j in 1:k

        orthonormal_basis!(t_arnoldi, b, j, Arnoldi)

    end

    for j in 2:k

        orthonormal_basis!(t_lanczos, b, j, Lanczos)

    end

    arnoldi_test = zeros(k, k)
    lanczos_test = zeros(k, k)

    Iₖ = I(k)

    for s in 1:d

        #mul!( arnoldi_test, t_arnoldi.V[s][:, 1:k]', t_arnoldi.V[s][:, 1:k] ) 
        mul!( lanczos_test, t_lanczos.V[s][:, 1:k]', t_lanczos.V[s][:, 1:k] )


        #@test arnoldi_test ≈ Iₖ 
        @test lanczos_test ≈ Iₖ

    end


end
