# Structs
# -------

using TensorKrylov
using LinearAlgebra
using Debugger

#@testset "Bisection method for symmetric tridiagonal eigenvalue problems" begin
#
#    n = 50
#    k = 5
#
#    A = Tridiagonal(-ones(n - 1), 2ones(n), -ones(n - 1))
#    v = inv(sqrt(n)) .* ones(n)
#
#    lanczos = Lanczos{Float64}(A, zeros(n, k + 1), zeros(k + 1, k + 1), v)
#
#    for j in 1:k 
#
#        orthonormal_basis_vector!(lanczos, j)
#
#    end
#
#    # Initialize sequence of characteristic polynomials
#    char_polynomials = [ [1], [ lanczos.H[1, 1], -1.0] ]
#
#    for j in 2:k 
#
#        γⱼ = lanczos.H[j, j]
#        βⱼ = lanczos.H[j, j - 1]
#        next_coefficients!(char_polynomials, j, γⱼ, βⱼ)
#
#    end
#
#    # Test for correct generation of polynomials by evaluating them at μ = 2
#    test_values = [1.0, -1.96, -0.04166666666666724, 1.9565217391304344, 0.04545454545454549, -1.9523809523809523]
#
#    μ1 = 2.0
#    μ2 = 1.0
#    μ3 = 0.25
#
#    for j in 1:length(char_polynomials)
#
#        polynomial = char_polynomials[j]
#
#        @test @evalpoly(μ1, polynomial...) ≈ test_values[j]
#
#    end
#
#    γ = diag(lanczos.H, 0)[1:k]
#    β = diag(lanczos.H, 1)[1:k-1]
#
#    # Initial interval
#    y, z = initial_interval( γ, β )
#
#    
#
#    # Test for correct counting of sign change
#    @test  sign_changes(μ1, char_polynomials) == 3
#    @test  sign_changes(μ2, char_polynomials) == 2
#    @test  sign_changes(μ3, char_polynomials) == 1
#    
#
#    approximations = [ bisection(y, z, k, i, char_polynomials) for i in k : -1 : 1 ]
#
#    exact = eigvals(Matrix(lanczos.H[1:k, 1:k]))
#
#    @test approximations ≈ exact
#end

@testset "Extreme eigenvalues of system with symmetric tridiagonal coefficient matrices" begin


    d = 1

    n = 5

    k = 3

    h = inv(n + 1)

    Aₛ = inv(h^2) * Tridiagonal(-ones(n - 1), 2ones(n), ones(n - 1))

    A = KroneckerMatrix{Float64}([ Aₛ for _ in 1:d ])

    b = [ rand(n) for _ in 1:d ]

    # Allocate memory for right-hand side b̃
    b̃ = [ zeros( size(b[s]) )  for s in eachindex(b) ]

    lanczos = Lanczos{Float64}(Aₛ, zeros(n, k), zeros(k, k), 2ones(n))

    for j in 1:k-1

        orthonormal_basis_vector!(lanczos, j)

        #@info "Tₖ after k-th orthonormalization step k =" j
        #show(IOContext(stdout, :compact => false, :limit => false), "text/plain", lanczos.H[1:j, 1:j])
        #show(IOContext(stdout, :compact => false, :limit => false), "text/plain", lanczos.V[1:n, 1:j])

        @test isposdef(lanczos.H[1:j, 1:j])

    end


    #char_poly = CharacteristicPolynomials{Float64}(d, t_lanczos.H[1, 1])


    λ_min = 0.0
    λ_max = 0.0

    #t_lanczos = TensorLanczos{Float64}(A)

    #initial_orthonormalization!(t_lanczos, b, Lanczos)

    #for j = 2:k

    #    orthonormal_basis!(t_lanczos, j)

    #    #H_minors = principal_minors(t_lanczos.H, j)
    #    #b_minors = principal_minors(b̃, j)

    #    #λ_min, λ_max = extreme_tensorized_eigenvalues(H_minors, char_poly, j)

    #end


    #exact_eigenvalues = eigvals(Matrix(t_lanczos.H[1][1:k, 1:k]))

    #lanczos_test = zeros(k, k)

    #Iₖ = I(k)

    #for s in 1:d

    #    mul!( lanczos_test, t_lanczos.V[s][:, 1:k]', t_lanczos.V[s][:, 1:k] ) 

    #    @test lanczos_test ≈ Iₖ

    #end

    #@info "First exact eigenvalues" exact_eigenvalues[1], exact_eigenvalues[end]

    #@info "Test for positive definiteness: " isposdef(Matrix(t_lanczos.H[1][1:k - 1, 1:k- 1]))

    #exact_extreme = [d * exact_eigenvalues[1], d * exact_eigenvalues[end]]

    #@info "Exact extreme: " exact_extreme

    #@info "Approximated extreme eigenvalues: " λ_min, λ_max

    #approximation_extreme = [λ_min, λ_max]

    #@test exact_extreme ≈ approximation_extreme

end



#@testset "QR-algorithm for upper Hessenberg matrices" begin
#
#    τ = 1e-5
#
#    n_small = 10
#
#    k_small = 5
#
#    A_small = sparse(Tridiagonal(-ones(n_small - 1), 2ones(n_small), -ones(n_small - 1)))
#
#    eigvalssmall = qr_hessenberg(A_small, τ, 8)
#
#    @info "Eigenvalues of small matrix " eigvalssmall
#
#    sorted_eigenvalues = sort(Vector(eigvalssmall))
#    @test sorted_eigenvalues ≈ eigvals(Matrix(A_small))
#
#    n_large = 200
#
#    k_large = 100
#
#    A_large = sparse(Tridiagonal(-ones(n_large - 1), 2ones(n_large), -ones(n_large - 1)))
#
#    eigvalslarge = qr_hessenberg(A_large, τ, 100)
#
#    λ_max = maximum(eigvalslarge)
#    λ_min = minimum(eigvalslarge)
#
#end
