using TensorKrylov: Arnoldi, Lanczos
using TensorKrylov: qr_hessenberg

using LinearAlgebra, SparseArrays

@testset "QR-algorithm for upper Hessenberg matrices" begin

    τ = 1e-5

    n_small = 10

    k_small = 5

    A_small = sparse(Tridiagonal(-ones(n_small - 1), 2ones(n_small), -ones(n_small - 1)))

    eigvalssmall = qr_hessenberg(A_small, τ, 8)

    @info "Eigenvalues of small matrix " eigvalssmall

    sorted_eigenvalues = sort(Vector(eigvalssmall))
    @test sorted_eigenvalues ≈ eigvals(Matrix(A_small))

    n_large = 200

    k_large = 100

    A_large = sparse(Tridiagonal(-ones(n_large - 1), 2ones(n_large), -ones(n_large - 1)))

    eigvalslarge = qr_hessenberg(A_large, τ, 100)

    λ_max = maximum(eigvalslarge)
    λ_min = minimum(eigvalslarge)

end
