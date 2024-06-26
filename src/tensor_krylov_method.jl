export tensorkrylov!, solve_compressed_system, solve_exactly

using ExponentialUtilities: exponential!, expv
using Logging

abstract type Mode       end
struct DebugMode  <:Mode end
struct SilentMode <:Mode end

function solve_compressed_system(
    H,         
    b,
    approxdata::ApproximationData{T, U},
    λ_min     ::T,
    class::Type{<:MatrixGallery}) where {T, U<:Instance}

    # Since we are considering a canonical decomposition the tensor rank of yₜ
    # is equal to 

    k     = dimensions(H)
    t     = length(approxdata.ω)
    λ_inv = inv(λ_min)
    yₜ    = KruskalTensor{T}(λ_inv .* approxdata.ω, [ ones(k[s], t) for s in 1:length(H)] )

    @inbounds for k = 1:t

        γ = -approxdata.α[k] * λ_inv

        matrix_exponential_vector!(yₜ, H, b, γ, k, class)

    end

    return yₜ
end

function tensorkrylov!(
    convergence_data       ::ConvergenceData{T},
    A                      ::KronMat{matT, U},
    b                      ::KronProd{T},
    tol                    ::T,
    nmax                   ::Int,
    orthonormalization_type::Type{<:TensorDecomposition},
    mode                   ::Type{<:Mode} = SilentMode) where {matT, T, U<:Instance}

    d      = length(A)
    n      = dimensions(A)[1]
    𝔎      = Vector{Int}(undef, d) # Initialize multiindex 𝔎
    b_norm = kronprodnorm(b)
    x      = KruskalTensor{T}() # Declare approximate solution

    tensor_decomp = orthonormalization_type(A)

    orthonormalize!(tensor_decomp, b)

    b̃ = initialize_compressed_rhs(b, tensor_decomp.V) 
    
    spectraldata = SpectralData{matT, T, U}(A, nmax)
    approxdata   = ApproximationData{T, U}(tol)

    r_comp       = Inf
    r_norm       = Inf

    @inbounds for k = 2:nmax

        # Compute orthonormal basis and Hessenberg factor of each Krylov subspace 𝓚ₖ(Aₛ, bₛ) 
        orthonormalize!(tensor_decomp, k)

        H_minors, V_minors, b_minors = compute_minors(tensor_decomp, b̃, n, k)
        columns                      = kth_columns(tensor_decomp.V, k)

        update_rhs!(b_minors, columns, b, k) # b̃ = Vᵀb
        update_data!(spectraldata, d, A.matrixclass())
        update_data!(approxdata, spectraldata)


        y  = solve_compressed_system(H_minors, b_minors, approxdata, spectraldata.λ_min[k], A.matrixclass) # Hy = b̃ 
        𝔎 .= k 

        subdiagentries = [ tensor_decomp.H[s][k + 1, k] for s in 1:d ]

        try

            r_comp, r_norm = residualnorm!(H_minors, y, 𝔎, subdiagentries, b_minors, b_norm) # Compute residual norm

        catch e 

            if isa(e, CompressedNormBreakdown{T})

                println("Early termination at k = " * string(k) * " due to compressed norm breakdown")
                convergence_data.niterations = k - 1

                resize!(convergence_data, convergence_data.niterations)

                return

            end

        end
        rel_res_norm   = (r_norm / b_norm)

        convergence_data.relative_residual_norm[k]  = rel_res_norm
        convergence_data.projected_residual_norm[k] = r_comp
        convergence_data.orthogonality_data[k]      = orthogonality_loss(first(V_minors), k)


        #process_log(convergence_data, k, mode, orthonormalization_type)

        if rel_res_norm < tol

            x = KruskalTensor{T}( ones(approxdata.rank), [ zeros(size(A[s], 1), approxdata.rank) for s in 1:d ])

            basis_tensor_mul!(x, V_minors, y)

            println("Convergence")

            return x

        end

    end

    println("No convergence")


end
