export tensor_krylov!, solve_compressed_system

using ExponentialUtilities: exponential!, expv


#function matrix_exponential_vector!(y::ktensor, A::KronMat{T}, b::KronProd{T}, γ::T, k::Int) where T<:AbstractFloat
#
#    tmp1 = Matrix(copy(A[1]))
#    tmp  = SymTridiagonal(tmp1)
#    expA = exponentiate(tmp, γ)
#    
#    julia_exp = exp(γ * tmp1)  
#
#    error = LinearAlgebra.norm(julia_exp - expA) / LinearAlgebra.norm(julia_exp)
#    println(error)
#
#    for s in 1:length(A)
#
#        y.fmat[s][:, k] = expA * b[s]
#
#    end
#
#end

function solve_compressed_system(
        H::KronMat{T}, 
        b::Vector{<:AbstractVector{T}}, 
        ω::Array{T},
        α::Array{T},
        λ::T,
    ) where T <: AbstractFloat

    # Since we are considering a canonical decomposition the tensor rank of yₜ
    # is equal to 

    k = dimensions(H)

    t = length(α)

    λ_inv = inv(λ)
    yₜ    = ktensor(λ_inv .* ω, [ ones(k[s], t) for s in 1:length(H)] )


    for k = 1:t

        γ = -α[k] * λ_inv

        matrix_exponential_vector!(yₜ, H, b, γ, k)

    end

    return yₜ
end




# SPD case no convergence data
function tensor_krylov!(A::KronMat{T}, b::KronProd{T}, tol::T, nmax::Int, t_orthonormalization::Type{TensorLanczos{T}}) where T <: AbstractFloat

    println(BLAS.get_num_threads())

    d      = length(A)
    𝔎      = Vector{Int}(undef, d) # Initialize multiindex 𝔎
    b_norm = kronprodnorm(b)
    x      = nothing # Declare approximate solution

    tensor_decomp = t_orthonormalization(A)

    # Initialize list of characteristic polynomials of Jacobi matrices Tₖ
    char_poly          = CharacteristicPolynomials{T}(d, tensor_decomp.H[1, 1])
    orthonormalization = tensor_decomp.orthonormalization
    initial_orthonormalization!(tensor_decomp, b, orthonormalization)


    b̃                = initialize_compressed_rhs(b, tensor_decomp.V) 
    coefficients_dir = compute_coefficients_directory()
    coefficients_df  = compute_dataframe(coefficients_dir)

    n = dimensions(A)[1]

    for k = 2:nmax

        # Compute orthonormal basis and Hessenberg factor of each Krylov subspace 𝓚ₖ(Aₛ, bₛ) 
        orthonormal_basis!(tensor_decomp, k)

        H_minors, V_minors, b_minors = compute_minors(tensor_decomp, b̃, n, k)
        columns                      = kth_columns(tensor_decomp.V, k)

        # Update compressed right-hand side b̃ = Vᵀb
        update_rhs!(b_minors, columns, b, k)

        b̃_norm       = kronprodnorm(b_minors)
        λ_min, λ_max = analytic_eigenvalues(d, k) # Taking eigenvalues of principal minors of A not of H
        κ            = abs(λ_max / λ_min)

        @info "Condition: " κ

        ω, α, rank = optimal_coefficients(coefficients_dir, coefficients_df, tol, κ, λ_min, b̃_norm)
        
        @info "Chosen tensor rank: " rank


        # Approximate solution of compressed system
        y  = solve_compressed_system(H_minors, b_minors, ω, α, λ_min)
        𝔎 .= k 

        subdiagentries = [ tensor_decomp.H[s][k + 1, k] for s in 1:d ]
        r_norm         = residual_norm(H_minors, y, 𝔎, subdiagentries, b_minors) # Compute residual norm
        rel_res_norm   = (r_norm / b_norm)


        @info "Iteration: " k "relative residual norm:" rel_res_norm

        if rel_res_norm < tol

            x        = ktensor( ones(rank), [ zeros(size(A[s], 1), rank) for s in 1:d ])
            x_minors = principal_minors(x, k)

            basis_tensor_mul!(x_minors, V_minors, y)

            println("Convergence")

            return x

        end

    end

    println("No convergence")

end

# SPD case convergence data
function tensor_krylov!(convergencedata::ConvergenceData{T}, A::KronMat{T}, b::KronProd{T}, tol::T, nmax::Int, t_orthonormalization::Type{<:TensorDecomposition}) where T <: AbstractFloat

    d      = length(A)
    𝔎      = Vector{Int}(undef, d) # Initialize multiindex 𝔎
    b_norm = kronprodnorm(b)
    x      = nothing # Declare approximate solution

    tensor_decomp = t_orthonormalization(A)

    # Initialize list of characteristic polynomials of Jacobi matrices Tₖ
    char_poly          = CharacteristicPolynomials{T}(d, tensor_decomp.H[1, 1])
    orthonormalization = tensor_decomp.orthonormalization
    initial_orthonormalization!(tensor_decomp, b, orthonormalization)


    b̃                = initialize_compressed_rhs(b, tensor_decomp.V) 
    coefficients_dir = compute_coefficients_directory()
    coefficients_df  = compute_dataframe(coefficients_dir)

    n = dimensions(A)[1]

    for k = 2:nmax

        # Compute orthonormal basis and Hessenberg factor of each Krylov subspace 𝓚ₖ(Aₛ, bₛ) 
        orthonormal_basis!(tensor_decomp, k)

        H_minors, V_minors, b_minors = compute_minors(tensor_decomp, b̃, n, k)
        columns                      = kth_columns(tensor_decomp.V, k)

        # Update compressed right-hand side b̃ = Vᵀb
        update_rhs!(b_minors, columns, b, k)

        b̃_norm       = kronprodnorm(b_minors)
        λ_min, λ_max = analytic_eigenvalues(d, k) # Taking eigenvalues of principal minors of A and not of H
        κ            = abs(λ_max / λ_min)

        @info "Condition: " κ

        ω, α, rank = optimal_coefficients(coefficients_dir, coefficients_df, tol, κ, λ_min, b̃_norm)
        
        @info "Chosen tensor rank: " rank


        # Approximate solution of compressed system
        y  = solve_compressed_system(H_minors, b_minors, ω, α, λ_min)
        𝔎 .= k 

        subdiagentries = [ tensor_decomp.H[s][k + 1, k] for s in 1:d ]
        r_norm         = residual_norm(H_minors, y, 𝔎, subdiagentries, b_minors) # Compute residual norm
        rel_res_norm   = (r_norm / b_norm)

        convergencedata.relative_residualnorm[k] = rel_res_norm

        @info "Iteration: " k "relative residual norm:" rel_res_norm

        if rel_res_norm < tol

            x        = ktensor( ones(rank), [ zeros(size(A[s], 1), rank) for s in 1:d ])
            x_minors = principal_minors(x, k)

            basis_tensor_mul!(x_minors, V_minors, y)

            println("Convergence")

            return x

        end

    end

    println("No convergence")

end

# Non-symmetric case, no convergence data
function tensor_krylov!(A::KronMat{T}, b::KronProd{T}, tol::T, nmax::Int, t_orthonormalization::Type{TensorArnoldi{T}}) where T <: AbstractFloat

    d      = length(A)
    𝔎      = Vector{Int}(undef, d) # Initialize multiindex 𝔎
    b_norm = kronprodnorm(b)
    x      = nothing # Declare approximate solution

    tensor_decomp = t_orthonormalization(A)

    # Initialize list of characteristic polynomials of Jacobi matrices Tₖ
    orthonormalization = tensor_decomp.orthonormalization
    initial_orthonormalization!(tensor_decomp, b, orthonormalization)

    b̃                = initialize_compressed_rhs(b, tensor_decomp.V) 

    n = dimensions(A)[1]


    for k = 2:nmax

        # Compute orthonormal basis and Hessenberg factor of each Krylov subspace 𝓚ₖ(Aₛ, bₛ) 
        orthonormal_basis!(tensor_decomp, k)

        H_minors, V_minors, b_minors = compute_minors(tensor_decomp, b̃, n, k)
        columns                      = kth_columns(tensor_decomp.V, k)

        # Update compressed right-hand side b̃ = Vᵀb
        update_rhs!(b_minors, columns, b, k)


        rank, λ_min = get_nonsymmetric_rank(H_minors[1], b̃, tol)
        α, ω = nonsymmetric_coefficients(rank)
        
        @info "Chosen tensor rank: " rank

        # Approximate solution of compressed system
        y  = solve_compressed_system(H_minors, b_minors, ω, α, λ_min)
        𝔎 .= k 

        subdiagentries = [ tensor_decomp.H[s][k + 1, k] for s in 1:d ]
        r_norm         = residual_norm(H_minors, y, 𝔎, subdiagentries, b_minors) # Compute residual norm
        rel_res_norm   = (r_norm / b_norm)


        @info "Iteration: " k "relative residual norm:" rel_res_norm

        if rel_res_norm < tol

            x        = ktensor( ones(rank), [ zeros(size(A[s], 1), rank) for s in 1:d ])
            x_minors = principal_minors(x, k)

            basis_tensor_mul!(x_minors, V_minors, y)

            println("Convergence")

            return x

        end

    end

    println("No convergence")

end
