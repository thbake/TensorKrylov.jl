export tensor_krylov!, solve_compressed_system
export DebugMode, SilentMode

using ExponentialUtilities: exponential!, expv
using Logging

abstract type Mode       end
struct DebugMode  <:Mode end
struct SilentMode <:Mode end

function log_convergence_data(
    debuglogger::ConsoleLogger,
    convergencedata::ConvergenceData{T},
    k::Int,
    ::Type{LanczosUnion{T}}) where T

    with_logger(debuglogger) do
        @debug "Condition: " convergencedata.spectraldata[k].κ
        @debug "Iteration: " k "relative residual norm:" convergencedata.relative_residual_norm[k]

    end

end

function log_convergence_data(
    debuglogger::ConsoleLogger,
    convergencedata::ConvergenceData{T},
    k::Int,
    ::Type{TensorArnoldi{T}}) where T

    with_logger(debuglogger) do

        @debug "Smallest eigenvalue: " convergencedata.spectraldata[k].λ_min
        @debug "Iteration: " k "relative residual norm:" convergencedata.relative_residual_norm[k]

    end

end

function process_log(::ConvergenceData{T}, k::Int, ::Type{SilentMode}, ::Type{<:TensorDecomposition{T}}) where T

    return

end

function process_log(convergencedata::ConvergenceData{T}, k::Int, ::Type{DebugMode}, orthonormalization_type::Type{<:TensorDecomposition{T}}) where T

    debuglogger = ConsoleLogger(stdout, Logging.Debug)

    log_convergence_data(debuglogger, convergencedata, k, orthonormalization_type) 

end

function solve_compressed_system(
    H         ::KronMat{T},
    b         ::KronProd{T},
    approxdata::ApproximationData{T},
    λ_min     ::T) where T 

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

distancetosingularity(H::KronMat{T}) where T = cond(first(H))

function tensor_krylov!(
    convergence_data       ::ConvergenceData{T},
    A                      ::KronMat{T},
    b                      ::KronProd{T},
    tol                    ::T,
    nmax                   ::Int,
    orthonormalization_type::Type{<:TensorDecomposition{T}},
    mode                   ::Type{<:Mode} = SilentMode) where T 

    d      = length(A)
    n      = dimensions(A)[1]
    𝔎      = Vector{Int}(undef, d) # Initialize multiindex 𝔎
    b_norm = kronprodnorm(b)
    x      = nothing # Declare approximate solution

    tensor_decomp = orthonormalization_type(A)

    initial_orthonormalization!(tensor_decomp, b, tensor_decomp.orthonormalization)

    b̃ = initialize_compressed_rhs(b, tensor_decomp.V) 
    
    #spectraldata = SpectralData{T}(d, n, nmax, orthonormalization_type)
    approxdata   = ApproximationData{T}(tol, orthonormalization_type)
    r_comp       = Inf
    r_norm       = Inf

    for k = 2:nmax

        # Compute orthonormal basis and Hessenberg factor of each Krylov subspace 𝓚ₖ(Aₛ, bₛ) 
        orthonormal_basis!(tensor_decomp, k)

        H_minors, V_minors, b_minors = compute_minors(tensor_decomp, b̃, n, k)
        columns                      = kth_columns(tensor_decomp.V, k)

        update_rhs!(b_minors, columns, b, k) # b̃ = Vᵀb
        update_data!(convergence_data.spectraldata, d, n, k,      orthonormalization_type)
        update_data!(approxdata,   convergence_data.spectraldata, orthonormalization_type)
        #spectraldata.κ = distancetosingularity(principal_minors(tensor_decomp.H, k + 1, k))

        y  = solve_compressed_system(H_minors, b_minors, approxdata, convergence_data.spectraldata.λ_min[k]) # Hy = b̃ 
        𝔎 .= k 

        subdiagentries = [ tensor_decomp.H[s][k + 1, k] for s in 1:d ]

        try

            r_comp, r_norm = residual_norm!(H_minors, y, 𝔎, subdiagentries, b_minors) # Compute residual norm

        catch e 

            if isa(e, CompressedNormBreakdown{T})

                println("Early termination at k = " * string(k) * " due to compressed norm breakdown")
                convergence_data.niterations = k - 1

                return

            end

        end
        rel_res_norm   = (r_norm / b_norm)

        convergence_data.relative_residual_norm[k]  = rel_res_norm
        convergence_data.projected_residual_norm[k] = r_comp
        convergence_data.orthogonality_data[k]      = orthogonality_loss(first(V_minors), k)


        process_log(convergence_data, k, mode, orthonormalization_type)

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
