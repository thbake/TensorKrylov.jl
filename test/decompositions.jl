using TensorKrylov: Arnoldi, arnoldi_step!
using LinearAlgebra

#@testset "Arnoldi steps" begin
#
#    n = 1000
#
#    h = inv(n + 1)
#
#    A = inv(h^2) * Tridiagonal( -1ones(n - 1), 2ones(n), -1ones(n - 1) )
#    
#    b = rand(n)
#
#    # Initialize Arnoldi decomposition
#    
#    arnoldi = Arnoldi{T}(A, b, j)
#end

@testset "Multiple Arnoldi decomposition steps" begin

    d = 100

    nₛ = 1000

    h = inv(nₛ + 1)

    Aₛ= inv(h^2) * Tridiagonal( -1ones(nₛ - 1) , 2ones(nₛ), -1ones(nₛ - 1) )

    A = KroneckerMatrix{Float64}([Aₛ for _ in 1:d])

    b = [ rand(nₛ) for _ in 1:d ]

    # Initialize Arnoldi decompositions
    t_arnoldi = TensorArnoldi{Float64}(A)

    k = 500

    for j in 1:k

        multiple_arnoldi!(t_arnoldi, b, j)

    end

    C = zeros(k, k)

    for s in 1:d

        mul!(C, t_arnoldi.V[s][:, 1:k]', t_arnoldi.V[s][:, 1:k] ) 

        @test C ≈ I(k) 

    end


end
