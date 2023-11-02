using TensorKrylov, Test, LinearAlgebra, SparseArrays

@testset "Sums of Kronecker products" begin

    d = 4
    n = 50
    Aᵢ= [ sparse(Tridiagonal(-ones(n - 1), 2ones(n), -ones(n - 1))) for _ in 1:d ]
    A = explicit_kroneckersum(Aᵢ)
    In = I(n)
    As = Aᵢ[1]

    A_test = kron(As, In, In, In) + kron(In, As, In, In) + kron(In, In, As, In) + kron(In, In, In, As) 

    @test A_test == A

end
