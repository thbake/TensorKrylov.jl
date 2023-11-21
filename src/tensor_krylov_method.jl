export tensor_krylov!, update_rhs!, initialize_compressed_rhs, basis_tensor_mul!, solve_compressed_system

using ExponentialUtilities: exponential!, expv

function matrix_exponential_vector!(y::ktensor, A::KronMat{T}, b::KronProd{T}, γ::T, k::Int) where T<:AbstractFloat

    for s = 1:length(A)

        tmp = Matrix(copy(A[s]))

        #y.fmat[s][:, k] = expv(γ, tmp, b[s]) # Update kth column
        y.fmat[s][:, k] =  exp(γ * tmp) * b[s] # Update kth column

    end

end

function solve_compressed_system(
        H::KronMat{T}, 
        b::Vector{<:AbstractVector{T}}, 
        ω::Array{T},
        α::Array{T},
        t::Int,
        λ::T,
    ) where T <: AbstractFloat

    # Since we are considering a canonical decomposition the tensor rank of yₜ
    # is equal to 

    k = dimensions(H)

    λ_inv = inv(λ)
    yₜ    = ktensor(λ_inv .* ω, [ ones(k[s], t) for s in 1:length(H)] )


    for k = 1:t

        γ = -α[k] * λ_inv

        matrix_exponential_vector!(yₜ, H, b, γ, k)

    end

    return yₜ
end



function initialize_compressed_rhs(b::KronProd{T}, V::KronMat{T}) where T<:AbstractFloat

        b̃        = [ zeros( size(b[s]) )  for s in eachindex(b) ]
        b_minors = principal_minors(b̃, 1)
        columns  = kth_columns(V, 1)
        update_rhs!(b_minors, columns, b, 1)

        return b̃
end

function update_rhs!(b̃::KronProd{T}, V::KronProd{T}, b::KronProd{T}, k::Int) where T<:AbstractFloat
    # b̃ = Vᵀb = ⨂ Vₛᵀ ⋅ ⨂ bₛ = ⨂ Vₛᵀbₛ
    
    for s = 1:length(b̃)

        # Update one entry of each component of b̃ by performing a single inner product 
        b̃[s][k] = dot(V[s], b[s])

    end

end

function basis_tensor_mul!(x::ktensor, V::KronMat{T}, y::ktensor) where T<:AbstractFloat

    x.lambda = copy(y.lambda)

    for s in eachindex(V)

        LinearAlgebra.mul!(x.fmat[s], V[s], y.fmat[s])

    end

end

# SPD case no convergence data
function tensor_krylov!(A::KronMat{T}, b::KronProd{T}, tol::T, nmax::Int, t_orthonormalization::Type{<:TensorDecomposition}) where T <: AbstractFloat

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

        H_minors = principal_minors(tensor_decomp.H, k)
        V_minors = principal_minors(tensor_decomp.V, n, k)
        b_minors = principal_minors(b̃, k)
        columns  = kth_columns(tensor_decomp.V, k)

        # Update compressed right-hand side b̃ = Vᵀb
        update_rhs!(b_minors, columns, b, k)

        b̃_norm       = kronprodnorm(b_minors)
        λ_min, λ_max = analytic_eigenvalues(d, k)
        κ            = abs(λ_max / λ_min)

        @info "Condition: " κ

        ω, α, rank = optimal_coefficients(coefficients_dir, coefficients_df, tol, κ, λ_min, b̃_norm)
        
        @info "Chosen tensor rank: " rank


        # Approximate solution of compressed system
        y  = solve_compressed_system(H_minors, b_minors, ω, α, rank, λ_min)
        𝔎 .= k 

        subdiagentries = [ tensor_decomp.H[s][k + 1, k] for s in 1:d ]
        r_norm         = residual_norm(H_minors, y, 𝔎, subdiagentries, b_minors) # Compute residual norm
        rel_res_norm   = (r_norm / b_norm)


        @info "Iteration: " k "relative residual norm:" rel_res_norm

        if rel_res_norm < tol

            x = ktensor( ones(rank), [ zeros(size(A[s], 1), rank) for s in 1:d ])

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

        H_minors = principal_minors(tensor_decomp.H, k)
        V_minors = principal_minors(tensor_decomp.V, n, k)
        b_minors = principal_minors(b̃, k)
        columns  = kth_columns(tensor_decomp.V, k)

        # Update compressed right-hand side b̃ = Vᵀb
        update_rhs!(b_minors, columns, b, k)

        b̃_norm       = kronprodnorm(b_minors)
        λ_min, λ_max = analytic_eigenvalues(d, k)
        κ            = abs(λ_max / λ_min)

        @info "Condition: " κ

        ω, α, rank = optimal_coefficients(coefficients_dir, coefficients_df, tol, κ, λ_min, b̃_norm)
        
        @info "Chosen tensor rank: " rank


        # Approximate solution of compressed system
        y  = solve_compressed_system(H_minors, b_minors, ω, α, rank, λ_min)
        𝔎 .= k 

        subdiagentries = [ tensor_decomp.H[s][k + 1, k] for s in 1:d ]
        r_norm         = residual_norm(H_minors, y, 𝔎, subdiagentries, b_minors) # Compute residual norm
        rel_res_norm   = (r_norm / b_norm)

        convergencedata.relative_residualnorm[k] = rel_res_norm

        @info "Iteration: " k "relative residual norm:" rel_res_norm

        if rel_res_norm < tol

            x = ktensor( ones(rank), [ zeros(size(A[s], 1), rank) for s in 1:d ])

            x_minors = principal_minors(x, k)

            basis_tensor_mul!(x_minors, V_minors, y)

            println("Convergence")

            return x

        end

    end

    println("No convergence")

end
