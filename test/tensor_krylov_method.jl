using TensorKrylov, Test
using Kronecker, TensorToolbox, LinearAlgebra, BenchmarkTools, SparseArrays
using TensorKrylov: compute_dataframe, exponential_sum_parameters, exp, exponentiate, normalize!

@testset "Computation of matrix exponentials" begin

    n = 200

    A = SymTridiagonal(2ones(n), -ones(n-1))

    γ = rand()
    exact_exp  = exp(γ .* Matrix(A))
    approx_exp = exponentiate(A, γ)

    @test exact_exp ≈ approx_exp

end

@testset "Monotonic decrease of residual and error in A-norm" begin

    function solvecompressed(H::KroneckerMatrix{T}, b::Vector{<:AbstractVector{T}}) where T<:AbstractFloat

        H_expanded = Symmetric(kroneckersum(H.𝖳...))
        b_expanded = kron(b...)
        y          = H_expanded\b_expanded

        return y

    end

    function exactresidualnorm(A::KroneckerMatrix{T}, b::Vector{<:AbstractVector{T}}, xₖ::AbstractVector{T}) where T<:AbstractFloat
        A_expanded = kroneckersum(A.𝖳...)
        b_expanded = kron(b...)
        tmp        = zeros(size(A_expanded, 1))

        mul!(tmp, A_expanded, xₖ)

        rₖ = b_expanded - tmp

        return sqrt(dot(rₖ, rₖ)) * inv(LinearAlgebra.norm(b_expanded))

    end

    function Anormerror(A::AbstractMatrix{T}, x::AbstractVector{T}, xₖ::AbstractVector{T}) where T<: AbstractFloat

        tmp = zeros(size(x)) 
        diff = x - xₖ

        mul!(tmp, A, diff)

        return sqrt(dot(diff, diff))

    end

    function tensor_krylov_exact(A::KronMat{T}, b::KronProd{T}, nmax::Int, t_orthonormalization::Type{<:TensorDecomposition}, tol = 1e-9) where T <: AbstractFloat

        xₖ = Vector{T}(undef, nentries(A))

        A_expanded = kroneckersum(A.𝖳...)
        b_expanded = kron(b...)

        x = Symmetric(A_expanded)\b_expanded

        tensor_decomp      = t_orthonormalization(A)
        orthonormalization = tensor_decomp.orthonormalization

        initial_orthonormalization!(tensor_decomp, b, orthonormalization)

        n = size(A[1], 1)

        b̃         = initialize_compressed_rhs(b, tensor_decomp.V)
        coefficients_dir = compute_package_directory()
        coeffs_df        = compute_dataframe()
        
        for k = 2:nmax

            orthonormal_basis!(tensor_decomp, k)

            H_minors = principal_minors(tensor_decomp.H, k)
            V_minors = principal_minors(tensor_decomp.V, n, k)
            b_minors = principal_minors(b̃, k)

            # Update compressed right-hand side b̃ = Vᵀb
            columns = kth_columns(tensor_decomp.V, k)

            update_rhs!(b_minors, columns, b, k)

            b̃_norm       = kronprodnorm(b_minors)
            λ_min, λ_max = analytic_eigenvalues(d, k)
            κ            = abs(λ_max / λ_min)
            t            = compute_rank(coefficients_dir, κ, tol)
            α, ω         = exponential_sum_parameters(coefficients_dir, t, κ)

            y  = solvecompressed(H_minors, b_minors)
            yₜ = solve_compressed_system(H_minors, b_minors, ω, α, t, λ_min)

            @info "Relative error of solving compressed system: " norm(y - kroneckervectorize(yₜ)) * inv(norm(y))

            mul!(xₖ, kron(V_minors.𝖳...), y)

            r_normexact = exactresidualnorm(A, b, xₖ)

            println(r_normexact)

            error = Anormerror(A_expanded, x, xₖ)

            #@info "Error x - xₖ" error

            #residual_norm(H_minors, y)

        end

    end

    d    = 2
    n    = 100
    h    = inv(n + 1)
    Tₖ   = sparse(inv(h^2) .* SymTridiagonal(2ones(n), -ones(n - 1)))
    A    = KroneckerMatrix{Float64}([Tₖ for _ in 1:d])
    b    = [ rand(n) for _ in 1:d ]
    nmax = 90

    #tensor_krylov_exact(A, b, nmax, TensorLanczos{Float64})

end

@testset "Symmetric example" begin

    BLAS.set_num_threads(30)

    d    = 10
    nₛ   = 200
    nmax = 199
    h    = inv(nₛ + 1)
    Aₛ   = sparse(inv(h^2) .* SymTridiagonal(2ones(nₛ), -1ones(nₛ - 1)))
    A    = KroneckerMatrix{Float64}([Aₛ for _ in 1:d])
    b    = [ rand(nₛ) for _ in 1:d ]

    normalize!(b)
    
    convergencedata = ConvergenceData{Float64}(nmax)
    tensor_krylov!(convergencedata, A, b, 1e-9, nmax, TensorLanczos{Float64}, SilentMode)

end

@testset "Nonsymmetric example" begin

    d    = 10
    n    = 200
    nmax = 199
    h    = inv(n + 1)
    c    = 10

    L  =     sparse( inv(h^2) .* SymTridiagonal( 2ones(n), -ones(n - 1)) )
    Aₛ = L + sparse( (c * inv(4 * h)) .* diagm(-1 => ones(n-1), 0 => 3ones(n), 1 => -5ones(n - 1), 2 => ones(n - 2)) )

    A = KroneckerMatrix{Float64}([Aₛ for _ in 1:d])

    b = [ rand(n) for _ in 1:d ]

    normalize!(b)
    
    BLAS.set_num_threads(30)
    
    #x = tensor_krylov!(A, b, 1e-9, nmax, TensorArnoldi{Float64}, SilentMode)

end
