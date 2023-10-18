using TensorKrylov: CharacteristicPolynomials
using TensorKrylov: extreme_tensorized_eigenvalues
using LinearAlgebra

@testset "Bisection method for symmetric tridiagonal eigenvalue problems" begin

    n = 50
    k = 5

    A = Tridiagonal(-ones(n - 1), 2ones(n), -ones(n - 1))
    v = inv(sqrt(n)) .* ones(n)

    lanczos = Lanczos{Float64}(A, zeros(n, k + 1), zeros(k + 1, k + 1), v)

    initialize_decomp!(lanczos.V, v)
    orthonormal_basis_vector!(lanczos, 1)

    # Initialize sequence of characteristic polynomials
    char_polynomials = [ [1], [ lanczos.H[1, 1], -1.0] ]


    for j in 2:k 

        orthonormal_basis_vector!(lanczos, j)
        γⱼ = lanczos.H[j, j]
        βⱼ = lanczos.H[j, j - 1]
        next_coefficients!(char_polynomials, j, γⱼ, βⱼ)

    end

    # Test for correct generation of polynomials by evaluating them at μ = 2
    test_values = [1.0, -1.96, -0.04166666666666724, 1.9565217391304344, 0.04545454545454549, -1.9523809523809523]

    μ1 = 2.0
    μ2 = 1.0
    μ3 = 0.25

    for j in 1:length(char_polynomials)

        polynomial = char_polynomials[j]

        @test @evalpoly(μ1, polynomial...) ≈ test_values[j]

    end

    # Test for correct counting of sign change
    @test  sign_changes(μ1, char_polynomials) == 3
    @test  sign_changes(μ2, char_polynomials) == 2
    @test  sign_changes(μ3, char_polynomials) == 1

    γ = diag(lanczos.H, 0)[1:k]
    β = diag(lanczos.H, 1)[1:k-1]

    # Initial interval
    y, z = initial_interval( γ, β )

    approximations = [ bisection(y, z, k, i, char_polynomials) for i in k : -1 : 1 ]

    exact = eigvals(Matrix(lanczos.H[1:k, 1:k]))

    @test approximations ≈ exact

    # Again, test for correct generation of characteristic polynomials by evaluating
    # last one at the eigenvalues of the Lanczos tridiagonal matrix

    last_polynomial = char_polynomials[end]

    for j in 1:length(exact)

        @test @evalpoly(exact[j], last_polynomial...) < 1e-14

    end
end

@testset "Extreme eigenvalues of system with symmetric tridiagonal coefficient matrices" begin

    d = 1
    n = 100
    k = 50

    h = inv(n + 1)

    #Aₛ = inv(h^2) * Tridiagonal( -1ones(n - 1), 2ones(n), -1ones(n - 1) )
    Aₛ =  Tridiagonal( -1ones(n - 1), 2ones(n), -1ones(n - 1) )

    
    bs = rand(n)

    A = KroneckerMatrix{Float64}([Aₛ for _ in 1:d])
    lanczos = Lanczos{Float64}(Aₛ, zeros(n, k), zeros(k, k), bs)

    for j in 1:k-1

        orthonormal_basis_vector!(lanczos, j)

        @test isposdef(lanczos.H[1:j, 1:j])

    end

    b = [bs for _ in 1:d]
    b̃ = [ zeros( size(b[s]) )  for s in eachindex(b) ]

    λ_max = 0.0
    λ_min = 0.0

    t_lanczos = TensorLanczos{Float64}(A)
    char_poly = CharacteristicPolynomials{Float64}(d, t_lanczos.H[1, 1])

    initial_orthonormalization!(t_lanczos, b, Lanczos)

    for j = 2:k

        orthonormal_basis!(t_lanczos, j)
        H_minors = principal_minors(t_lanczos.H, j)

        λ_min, λ_max = extreme_tensorized_eigenvalues(H_minors, char_poly, j)
        
    end

    Tₖ = t_lanczos.H[1][1:k, 1:k]

    γ = diag(Tₖ, 0)[1:k]
    β = diag(Tₖ, 1)[1:k-1]

    # Initial interval
    y, z = initial_interval( γ, β )

    lanczos_basis = t_lanczos.V[1][:, 1:k]

    #display(lanczos_basis' * lanczos_basis)

    # Get first sequence of characteristic polynomials
    first_char_poly_seq = char_poly.coefficients[1]

    a = bisection(y, z, k, k, first_char_poly_seq)
    b = bisection(y, z, k, 1, first_char_poly_seq)

    @info "Small eigenvalues: " a, b 

    approximate_eigenvalues = [ a, b ]
    exact_eigenvalues = eigvals(Matrix(Tₖ))

    @info @evalpoly(a, first_char_poly_seq[end]...)

    exact_extreme         = [(d * exact_eigenvalues[1]), (d * exact_eigenvalues[end])]
    approximation_extreme = [λ_min, λ_max]

    @info "Exact eigenvalues: " exact_extreme
    @info "Approximated eigenvalues: " approximation_extreme

    @info "Relative error: " abs(exact_extreme[end] - approximation_extreme[end]) / exact_extreme[end]

    @info "Exact condition number of Tₖ: " cond(Matrix(Tₖ)[1:k, 1:k])

    @info "Condition number of Tₖ given by exact extreme eigenvalues: " exact_eigenvalues[end] / exact_eigenvalues[1]
    @info "Estimated condition number of Tₖ: " abs(λ_max/ λ_min)

    R = qr_algorithm(Matrix(Tₖ), 1e-5, 100)
    display(R[1])
    display(R[end])



end
