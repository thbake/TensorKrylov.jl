using TensorKrylov: extract_coefficients, kronprodnorm
using LinearAlgebra, Kronecker

@testset "Stenger coefficients" begin


    d = 5

    nₛ = 1000

    h = inv(nₛ + 1)

    Aₛ= inv(h^2) * Tridiagonal( -1ones(nₛ - 1) , 2ones(nₛ), -1ones(nₛ - 1) )

    A = KroneckerMatrix{Float64}([Aₛ'Aₛ for _ in 1:d])

    A_explicit = kroneckersum( A.𝖳... )


    # Compute smallest eigenvalue and condition number analitically
    λ_min = (2 / h^2) * (1 - cos( π / (nₛ + 1)))
    λ_max = (2 / h^2) * (1 - cos( (nₛ * π) / (nₛ + 1)))

    κ = 4 * (nₛ + 1)^2 * inv(π^2 * d)

    @info κ

    τ = 1e-14

    b = [ rand(nₛ) for _ in 1:d ]

    b_norm = kronprodnorm(b)

    ω, α, rank = extract_coefficients(τ, κ, λ_min, b_norm)

end
