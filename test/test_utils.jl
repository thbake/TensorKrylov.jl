using LinearAlgebra, Kronecker,  TensorToolbox, TensorKrylov, SparseArrays
using TensorKrylov: compressed_residual, compute_lower_outer!, 
                    compute_lower_triangles!, cp_tensor_coefficients,
                    explicit_kroneckersum, maskprod, matrix_vector, MVnorm, 
                    squared_tensor_entries, tensorinnerprod 
using TensorKrylov: compute_minors, exponential_sum_parameters!, exponentiate,  
                    initialize_compressed_rhs, normalize!, update_data!, 
                    update_rhs!

using TensorKrylov: ApproximationData, SpectralData

# test/preprocessing.jl functions
# test/decompositions.jl functions

# test/eigenvalues.jl functions
function getextreme(v::Vector{T}) where T

    first  = minimum(v)
    second = maximum(v)

    return first, second
end

# test/utils.jl functions
function initialize_matrix_products(M, x)

    d      = length(M)
    t      = length(x.lambda)
    Λ      = LowerTriangular( zeros(t, t) )
    lowerX = [ zeros(t, t) for _ in 1:d ]
    Z      = matrix_vector(M, x)
    
    compute_lower_outer!(Λ, x.lambda)
    compute_lower_triangles!(lowerX, x)

    return Λ, lowerX, Z

end


function tensorsquarednorm(x::ktensor)

    d = length(x.fmat)
    t = length(x.lambda)

    value = 0.0 

    lowerX = [ zeros(t, t) for _ in 1:d ]
    
    compute_lower_triangles!(lowerX, x)

    X = Symmetric.(lowerX, :L)

    for j in 1:t, i in 1:t

        value += maskprod(X, i, j)

    end

    return value

end

# test/tensor_krylov_method.jl functions

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

    tmp  = zeros(size(x)) 
    diff = x - xₖ

    mul!(tmp, A, diff)

    return sqrt(dot(diff, diff))

end

function monotonic_decrease(errors::Vector{T}) where T 

    tmp = copy(errors[2:end])

    return all(tmp .< @view errors[1: end - 1])

end

function tensor_krylov_exact(A::KronMat{T}, b::KronProd{T}, nmax::Int, orthonormalization_type::Type{<:TensorDecomposition{T}}, tol = 1e-9) where T <: AbstractFloat

    d  = length(A)
    xₖ = Vector{T}(undef, nentries(A))

    A_expanded = kroneckersum(A.𝖳...)
    b_expanded = kron(b...)

    x = Symmetric(A_expanded)\b_expanded

    tensor_decomp      = orthonormalization_type(A)
    orthonormalization = tensor_decomp.orthonormalization

    initial_orthonormalization!(tensor_decomp, b, orthonormalization)

    n = size(A[1], 1)
    b̃ = initialize_compressed_rhs(b, tensor_decomp.V)
    
    spectraldata = SpectralData{T}(d, n, nmax, orthonormalization_type)
    approxdata   = ApproximationData{T}(tol, orthonormalization_type)
    r_norm       = Inf
    errors       = zeros(nmax)

    H_minors, V_minors, b_minors = compute_minors(tensor_decomp, b̃, n, 1)

    y = solvecompressed(H_minors, b_minors)
    
    mul!(xₖ, kron(V_minors.𝖳...), y)

    errors[1] = Anormerror(A_expanded, x, xₖ)

    for k = 2:nmax

        orthonormal_basis!(tensor_decomp, k)

        H_minors, V_minors, b_minors = compute_minors(tensor_decomp, b̃, n, k)
        columns                      = kth_columns(tensor_decomp.V, k) 
        update_rhs!(b_minors, columns, b, k) # Update compressed right-hand side b̃ = Vᵀb
        update_data!(spectraldata, d, n, k, orthonormalization_type)
        update_data!(approxdata, spectraldata, orthonormalization_type)

        y  = solvecompressed(H_minors, b_minors)
        yₜ = solve_compressed_system(H_minors, b_minors, approxdata, spectraldata.λ_min[k])

        #@info "Relative error of solving compressed system: " norm(y - kroneckervectorize(yₜ)) * inv(norm(y))

        mul!(xₖ, kron(V_minors.𝖳...), y)

        r_norm     = exactresidualnorm(A, b, xₖ)
        errors[k]  = Anormerror(A_expanded, x, xₖ)

    end

    @test monotonic_decrease(errors)

end
