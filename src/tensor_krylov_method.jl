export tensor_krylov!, solve_compressed_system

using ExponentialUtilities: exponential!, expv
using Logging

struct DebugConfig 

    debug_mode::Bool
    orthogonality_checks::Bool

end

function enable_logger(
    convergencelogger::ConsoleLogger,
    spectraldata::SpectralData{T},
    rank::Int,
    k::Int,
    rel_res_norm::T,
    ::Type{TensorLanczos{T}}) where T<:AbstractFloat

    with_logger(convergencelogger) do

        @debug "Condition: " spectraldata.κ
        @debug "Chosen tensor rank: " rank
        @debug "Iteration: " k "relative residual norm:" rel_res_norm

    end

end

function enable_logger(
    convergencelogger::ConsoleLogger,
    spectraldata::SpectralData{T},
    rank::Int,
    k::Int,
    rel_res_norm::T,
    ::Type{TensorArnoldi{T}}) where T<:AbstractFloat

    with_logger(convergencelogger) do

        @debug "Smallest eigenvalue: " spectraldata.λ_min
        @debug "Chosen tensor rank: " rank
        @debug "Iteration: " k "relative residual norm:" rel_res_norm

    end

end

function solve_compressed_system(
    H::KronMat{T},
    b::KronProd{T},
    approxdata::ApproximationData{T},
    λ_min::T) where T <: AbstractFloat

    # Since we are considering a canonical decomposition the tensor rank of yₜ
    # is equal to 

    k     = dimensions(H)
    t     = length(approxdata.ω)
    λ_inv = inv(λ_min)
    yₜ    = ktensor(λ_inv .* approxdata.ω, [ ones(k[s], t) for s in 1:length(H)] )

    for k = 1:t

        γ = -approxdata.α[k] * λ_inv

        matrix_exponential_vector!(yₜ, H, b, γ, k)

    end

    return yₜ
end

function tensor_krylov!(
    convergence_data::ConvergenceData{T},
    A::KronMat{T}, b::KronProd{T},
    tol::T,
    nmax::Int,
    orthonormalization_type::Type{<:TensorDecomposition{T}},
    debug::Bool = false) where T <: AbstractFloat

    d      = length(A)
    n      = dimensions(A)[1]
    𝔎      = Vector{Int}(undef, d) # Initialize multiindex 𝔎
    b_norm = kronprodnorm(b)
    x      = nothing # Declare approximate solution

    tensor_decomp      = orthonormalization_type(A)
    orthonormalization = tensor_decomp.orthonormalization

    initial_orthonormalization!(tensor_decomp, b, orthonormalization)

    b̃ = initialize_compressed_rhs(b, tensor_decomp.V) 


    convergencelogger = AbstractLogger

    if debug == true

        convergencelogger = ConsoleLogger(stdout, Logging.Debug)

    end

    spectraldata = SpectralData{T}()
    approxdata   = ApproximationData{T}(tol, orthonormalization_type)

    for k = 2:nmax

        # Compute orthonormal basis and Hessenberg factor of each Krylov subspace 𝓚ₖ(Aₛ, bₛ) 
        orthonormal_basis!(tensor_decomp, k)

        H_minors, V_minors, b_minors = compute_minors(tensor_decomp, b̃, n, k)
        columns                      = kth_columns(tensor_decomp.V, k)

        # Update compressed right-hand side b̃ = Vᵀb
        update_rhs!(b_minors, columns, b, k)

        update_data!(spectraldata, d, k, orthonormalization_type)

        update_data!(approxdata, spectraldata, orthonormalization_type)
        
        y  = solve_compressed_system(H_minors, b_minors, approxdata, spectraldata.λ_min) # Approximate solution of compressed system
        𝔎 .= k 

        subdiagentries = [ tensor_decomp.H[s][k + 1, k] for s in 1:d ]
        r_norm         = residual_norm(H_minors, y, 𝔎, subdiagentries, b_minors) # Compute residual norm
        rel_res_norm   = (r_norm / b_norm)

        if debug

            enable_logger(convergencelogger, spectraldata, approxdata.rank, k, rel_res_norm, orthonormalization_type)

        end

        convergence_data.relative_residual_norm[k]  = rel_res_norm

        if rel_res_norm < tol

            x        = ktensor( ones(approxdata.rank), [ zeros(size(A[s], 1), approxdata.rank) for s in 1:d ])
            x_minors = principal_minors(x, k)

            basis_tensor_mul!(x_minors, V_minors, y)

            println("Convergence")

            return x

        end

    end

    println("No convergence")

end
