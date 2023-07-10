using TensorKrylov: Arnoldi, arnoldi_step!
using LinearAlgebra

@testset "Arnoldi decomposition steps" begin

    d = 5

    nₛ = 1000

    h = inv(nₛ + 1)

    Aₛ= inv(h^2) * Tridiagonal( -1ones(nₛ - 1) , 2ones(nₛ), -1ones(nₛ - 1) )

    A = KroneckerMatrix{Float64}([Aₛ'Aₛ for _ in 1:d])

    b = [ rand(nₛ) for _ in 1:d ]

    # Initialize Arnoldi decomposition
    arnoldi = Arnoldi{Float64}(A, b)

    k = 200

    for j in 1:k

        arnoldi_step!(arnoldi, j)

    end

    C = zeros(k, k)

    for s in 1:d

        mul!(C, arnoldi.V[s][:, 1:k]', arnoldi.V[s][:, 1:k] ) 

        @test C ≈ I(k) 

    end


end
