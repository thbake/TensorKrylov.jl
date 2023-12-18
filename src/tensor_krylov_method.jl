export tensor_krylov!, solve_compressed_system

using ExponentialUtilities: exponential!, expv
using Logging

function enable_logger(convergencelogger::ConsoleLogger, κ::T, rank::Int, k::Int, rel_res_norm::T) where T<:AbstractFloat

    with_logger(convergencelogger) do

        @debug "Condition: " κ
        @debug "Chosen tensor rank: " rank
        @debug "Iteration: " k "relative residual norm:" rel_res_norm

    end

end

function solve_compressed_system(H::KronMat{T}, b::KronProd{T}, ω::Array{T}, α::Array{T}, λ::T) where T <: AbstractFloat

    # Since we are considering a canonical decomposition the tensor rank of yₜ
    # is equal to 

    k     = dimensions(H)
    t     = length(α)
    λ_inv = inv(λ)
    yₜ    = ktensor(λ_inv .* ω, [ ones(k[s], t) for s in 1:length(H)] )

    for k = 1:t

        γ = -α[k] * λ_inv

        matrix_exponential_vector!(yₜ, H, b, γ, k)

    end

    return yₜ
end




# SPD case no convergence data
function tensor_krylov!(
    A::KronMat{T}, b::KronProd{T},
    tol::T,
    nmax::Int,
    t_orthonormalization::Type{TensorLanczos{T}},
    debug::Bool = false) where T <: AbstractFloat

    d      = length(A)
    𝔎      = Vector{Int}(undef, d) # Initialize multiindex 𝔎
    b_norm = kronprodnorm(b)
    x      = nothing # Declare approximate solution

    tensor_decomp = t_orthonormalization(A)

    # Initialize list of characteristic polynomials of Jacobi matrices Tₖ
    #char_poly          = CharacteristicPolynomials{T}(d, tensor_decomp.H[1, 1])
    orthonormalization = tensor_decomp.orthonormalization
    initial_orthonormalization!(tensor_decomp, b, orthonormalization)


    b̃                = initialize_compressed_rhs(b, tensor_decomp.V) 
    coefficients_dir = compute_package_directory()
    coefficients_df  = compute_dataframe(coefficients_dir)

    n = dimensions(A)[1]

    convergencelogger = AbstractLogger

    if debug == true

        convergencelogger = ConsoleLogger(stdout, Logging.Debug)

    end

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

        @debug "Condition number: " κ

        rank = compute_rank(coefficients_df, κ, tol)
        α, ω = exponential_sum_parameters(coefficients_dir, rank, κ)
        
        @debug "Chosen tensor rank: " rank


        # Approximate solution of compressed system
        y  = solve_compressed_system(H_minors, b_minors, ω, α, λ_min)
        𝔎 .= k 

        subdiagentries = [ tensor_decomp.H[s][k + 1, k] for s in 1:d ]

        r_norm = Inf

        try 

            r_norm = residual_norm(H_minors, y, 𝔎, subdiagentries, b_minors) # Compute residual norm

        catch e

            if isa(e, CompressedNormBreakdown{T})

                println("Early termination at k = " * string(k) * " due to compressed norm breakdown")

                return

            end

        end

        rel_res_norm   = (r_norm / b_norm)

        if debug

            enable_logger(convergencelogger, κ, rank, k, rel_res_norm)

        end


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
function tensor_krylov!(
    convergence_data::ConvergenceData{T},
    A::KronMat{T},
    b::KronProd{T},
    tol::T,
    nmax::Int,
    t_orthonormalization::Type{<:TensorDecomposition},
    debug::Bool = false) where T <: AbstractFloat

    d      = length(A)
    𝔎      = Vector{Int}(undef, d) # Initialize multiindex 𝔎
    b_norm = kronprodnorm(b)
    x      = nothing # Declare approximate solution

    tensor_decomp = t_orthonormalization(A)

    # Initialize list of characteristic polynomials of Jacobi matrices Tₖ
    #char_poly          = CharacteristicPolynomials{T}(d, tensor_decomp.H[1, 1])
    orthonormalization = tensor_decomp.orthonormalization
    initial_orthonormalization!(tensor_decomp, b, orthonormalization)


    b̃                = initialize_compressed_rhs(b, tensor_decomp.V) 
    coefficients_dir = compute_package_directory()
    coefficients_df  = compute_dataframe(coefficients_dir)

    n = dimensions(A)[1]

    convergencelogger = AbstractLogger

    if debug == true

        convergencelogger = ConsoleLogger(stdout, Logging.Debug)

    end

    for k = 2:nmax

        # Compute orthonormal basis and Hessenberg factor of each Krylov subspace 𝓚ₖ(Aₛ, bₛ) 
        orthonormal_basis!(tensor_decomp, k)

        H_minors, V_minors, b_minors = compute_minors(tensor_decomp, b̃, n, k)
        columns                      = kth_columns(tensor_decomp.V, k)

        # Update compressed right-hand side b̃ = Vᵀb
        update_rhs!(b_minors, columns, b, k)

        λ_min, λ_max = analytic_eigenvalues(d, k) # Taking eigenvalues of principal minors of A and not of H
        κ            = abs(λ_max / λ_min)

        rank = compute_rank(coefficients_df, κ, tol)
        α, ω = exponential_sum_parameters(coefficients_dir, rank, κ)
        
        # Approximate solution of compressed system
        y  = solve_compressed_system(H_minors, b_minors, ω, α, λ_min)
        𝔎 .= k 

        subdiagentries = [ tensor_decomp.H[s][k + 1, k] for s in 1:d ]

        r_norm = Inf

        try 

            r_norm = residual_norm(H_minors, y, 𝔎, subdiagentries, b_minors) # Compute residual norm

        catch e

            if isa(e, CompressedNormBreakdown{T})

                println("Early termination due to compressed norm breakdown")

                resize!(convergence_data, k - 1) # Update number of iterations and associated data

                return

            end

        end

        r_norm         = residual_norm!(convergence_data, H_minors, y, 𝔎, subdiagentries, b_minors) # Compute residual norm
        rel_res_norm   = (r_norm / b_norm)

        if debug

            enable_logger(convergencelogger, κ, rank, k, rel_res_norm)

        end

        convergence_data.relative_residual_norm[k]  = rel_res_norm

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
function tensor_krylov!(
    A::KronMat{T}, b::KronProd{T},
    tol::T,
    nmax::Int,
    t_orthonormalization::Type{TensorArnoldi{T}},
    debug::Bool = false) where T <: AbstractFloat

    d      = length(A)
    𝔎      = Vector{Int}(undef, d) # Initialize multiindex 𝔎
    b_norm = kronprodnorm(b)
    x      = nothing # Declare approximate solution

    tensor_decomp = t_orthonormalization(A)

    # Initialize list of characteristic polynomials of Jacobi matrices Tₖ
    orthonormalization = tensor_decomp.orthonormalization
    initial_orthonormalization!(tensor_decomp, b, orthonormalization)

    b̃ = initialize_compressed_rhs(b, tensor_decomp.V) 

    n = dimensions(A)[1]

    #λ_min = minimum(abs.(eigvals(Matrix(A[1])))) * d
    convergencelogger = AbstractLogger

    if debug == true

        convergencelogger = ConsoleLogger(stdout, Logging.Debug)

    end

    for k = 2:nmax

        # Compute orthonormal basis and Hessenberg factor of each Krylov subspace 𝓚ₖ(Aₛ, bₛ) 
        orthonormal_basis!(tensor_decomp, k)

        H_minors, V_minors, b_minors = compute_minors(tensor_decomp, b̃, n, k)
        columns                      = kth_columns(tensor_decomp.V, k)
        A_minors = principal_minors(A, k)

        @info "Check singularity" cond(Matrix(H_minors[1]))

        # Update compressed right-hand side b̃ = Vᵀb
        update_rhs!(b_minors, columns, b, k)

        λ_min = minimum(abs.(eigvals(Matrix(A_minors[1])))) * d
        rank = compute_rank(λ_min, b, tol)
        α, ω = exponential_sum_parameters(rank)
        
        # Approximate solution of compressed system
        y  = solve_compressed_system(H_minors, b_minors, ω, α, λ_min)
        𝔎 .= k 

        subdiagentries = [ tensor_decomp.H[s][k + 1, k] for s in 1:d ]
        r_norm         = residual_norm(H_minors, y, 𝔎, subdiagentries, b_minors) # Compute residual norm
        rel_res_norm   = (r_norm / b_norm)

        if debug

            enable_logger(convergencelogger, λ_min, rank, k, rel_res_norm)

        end


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
function tensor_krylov!(
    convergence_data::ConvergenceData{T},
    A::KronMat{T}, b::KronProd{T},
    tol::T,
    nmax::Int,
    t_orthonormalization::Type{TensorArnoldi{T}},
    debug::Bool = false) where T <: AbstractFloat

    d      = length(A)
    𝔎      = Vector{Int}(undef, d) # Initialize multiindex 𝔎
    b_norm = kronprodnorm(b)
    x      = nothing # Declare approximate solution

    tensor_decomp = t_orthonormalization(A)

    # Initialize list of characteristic polynomials of Jacobi matrices Tₖ
    orthonormalization = tensor_decomp.orthonormalization
    initial_orthonormalization!(tensor_decomp, b, orthonormalization)

    b̃ = initialize_compressed_rhs(b, tensor_decomp.V) 

    n = dimensions(A)[1]

    #λ_min = minimum(abs.(eigvals(Matrix(A[1])))) * d
    convergencelogger = AbstractLogger

    if debug == true

        convergencelogger = ConsoleLogger(stdout, Logging.Debug)

    end

    for k = 2:nmax

        # Compute orthonormal basis and Hessenberg factor of each Krylov subspace 𝓚ₖ(Aₛ, bₛ) 
        orthonormal_basis!(tensor_decomp, k)

        H_minors, V_minors, b_minors = compute_minors(tensor_decomp, b̃, n, k)
        columns                      = kth_columns(tensor_decomp.V, k)
        A_minors = principal_minors(A, k)

        # Update compressed right-hand side b̃ = Vᵀb
        update_rhs!(b_minors, columns, b, k)

        λ_min = minimum(abs.(eigvals(Matrix(A_minors[1])))) * d
        rank = compute_rank(λ_min, b, tol)
        α, ω = exponential_sum_parameters(rank)
        
        # Approximate solution of compressed system
        y  = solve_compressed_system(H_minors, b_minors, ω, α, λ_min)
        𝔎 .= k 

        subdiagentries = [ tensor_decomp.H[s][k + 1, k] for s in 1:d ]
        r_norm         = residual_norm(H_minors, y, 𝔎, subdiagentries, b_minors) # Compute residual norm
        rel_res_norm   = (r_norm / b_norm)

        if debug

            enable_logger(convergencelogger, λ_min, rank, k, rel_res_norm)

        end

        convergence_data.relative_residual_norm[k]  = rel_res_norm

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

function tensor_krylov!(
    convergence_data::ConvergenceData{T},
    A::KronMat{T}, b::KronProd{T},
    tol::T,
    nmax::Int,
    t_orthonormalization::Type{<:TensorDecomposition{T}},
    debug::Bool = false) where T <: AbstractFloat

    d      = length(A)
    𝔎      = Vector{Int}(undef, d) # Initialize multiindex 𝔎
    b_norm = kronprodnorm(b)
    x      = nothing # Declare approximate solution

    tensor_decomp = t_orthonormalization(A)

    orthonormalization = tensor_decomp.orthonormalization
    initial_orthonormalization!(tensor_decomp, b, orthonormalization)

    b̃ = initialize_compressed_rhs(b, tensor_decomp.V) 

    n = dimensions(A)[1]

    #λ_min = minimum(abs.(eigvals(Matrix(A[1])))) * d
    convergencelogger = AbstractLogger

    if debug == true

        convergencelogger = ConsoleLogger(stdout, Logging.Debug)

    end

    for k = 2:nmax

        # Compute orthonormal basis and Hessenberg factor of each Krylov subspace 𝓚ₖ(Aₛ, bₛ) 
        orthonormal_basis!(tensor_decomp, k)

        H_minors, V_minors, b_minors = compute_minors(tensor_decomp, b̃, n, k)
        columns                      = kth_columns(tensor_decomp.V, k)
        A_minors = principal_minors(A, k)

        # Update compressed right-hand side b̃ = Vᵀb
        update_rhs!(b_minors, columns, b, k)

        spectral_data = spectral_data(d, k, t_orthonormalization)
        rank = compute_rank(λ_min, b, tol)
        α, ω = exponential_sum_parameters(rank)
        
        # Approximate solution of compressed system
        y  = solve_compressed_system(H_minors, b_minors, ω, α, spectral_data)
        𝔎 .= k 

        subdiagentries = [ tensor_decomp.H[s][k + 1, k] for s in 1:d ]
        r_norm         = residual_norm(H_minors, y, 𝔎, subdiagentries, b_minors) # Compute residual norm
        rel_res_norm   = (r_norm / b_norm)

        if debug

            enable_logger(convergencelogger, λ_min, rank, k, rel_res_norm)

        end

        convergence_data.relative_residual_norm[k]  = rel_res_norm

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
