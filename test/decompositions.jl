using TensorKrylov: Arnoldi, arnoldi_step!
using LinearAlgebra

@testset "Arnoldi and Lanczos steps" begin

    n = 1000

    h = inv(n + 1)

    A = inv(h^2) * Tridiagonal( -1ones(n - 1), 2ones(n), -1ones(n - 1) )
    
    b = rand(n)

    k = 500

    # Initialize Arnoldi decomposition
    
    #arnoldi = Arnoldi{T}(A, b, j)
    lanczos = Lanczos{Float64}(A, zeros(n, k), zeros(k, k), b)

    for j in 2:k-1

        lanczos_step!(lanczos, j)

    end

    C = zeros(k, k)

    LinearAlgebra.mul!(C, transpose(@view(lanczos.V[:, 1:k])), @view(lanczos.V[:, 1:k]))

    @test C[1:k, 1:k] ≈ I(k)
end

@testset "Multiple Arnoldi and Lanczos decomposition steps" begin

    d = 3

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

        multiple_arnoldi!(t_arnoldi, b, j)

    end

    for j in 2:k

        multiple_lanczos!(t_lanczos, b, j)

    end

    arnoldi_test = zeros(k, k)
    lanczos_test = zeros(k, k)

    Iₖ = I(k)

    for s in 1:d

        mul!( arnoldi_test, t_arnoldi.V[s][:, 1:k]', t_arnoldi.V[s][:, 1:k] ) 
        mul!( lanczos_test, t_lanczos.V[s][:, 1:k]', t_lanczos.V[s][:, 1:k] )


        @test arnoldi_test ≈ Iₖ 
        @test lanczos_test ≈ Iₖ

    end


end
