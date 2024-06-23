using LinearAlgebra, Kronecker, TensorKrylov, SparseArrays
using TensorKrylov: compressed_residual, compute_lower_outer!, 
                    compute_lower_triangles!, cp_tensor_coefficients,
                    maskprod, matrix_exponential_vector!, matrix_vector, MVnorm, 
                    squared_tensor_entries, tensorinnerprod 
using TensorKrylov: compute_minors, exponential_sum_parameters!,
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
function test_update_rhs(b̃::KronProd, V, b::KronProd, k::Int)

    d        = length(b)
    b_minors = [ zeros(k)    for _ in 1:d ]

    for j in 1:k

        b_minors = principal_minors(b̃, j)
        columns = kth_columns(V, j)
        update_rhs!(b_minors, columns, b, j)

    end

    exact  = zeros(k^2)
    approx = kron(b_minors...)


    mul!(exact,  kron( adjoint(V)...), kron(b...))

    return exact ≈ approx

end

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

function exact_solution(M::KronMat, x::KruskalTensor{T}) where T

    M_kroneckersum = kroneckersum(M.M...)
    x_explicit     = kroneckervectorize(x)
    
    return M_kroneckersum * x_explicit

end

function error_MVnorm(
    x       ::KruskalTensor{T},
    Λ       ::AbstractMatrix{T},
    lowerX  ::Vector{<:AbstractMatrix{T}},
    Z       ::Vector{<:AbstractMatrix{T}},
    solution::Vector{T}) where T
    
    exact_efficient_MVnorm = MVnorm(x, Λ, lowerX, Z) # Compute ||Mx||²
    exactMVnorm            = dot(solution, solution)
    relative_error         = (exact_efficient_MVnorm - exactMVnorm) / exactMVnorm 

    return relative_error

end

function error_tensorinnerprod(
    Z::Vector{Matrix{T}},
    x::KruskalTensor{T},
    b::KronProd{T},
    solution::Vector{T}) where T

    b_norm          = kronprodnorm(b)
    b_explicit      = kron(b...)
    approxinnerprod = tensorinnerprod(Z, x, b_norm)
    exactinnerprod  = dot(solution, b_explicit) # Compute <Mx, b>₂
    relative_error  = abs(exactinnerprod - approxinnerprod) / exactinnerprod 

    return relative_error

end

function error_compressed_residualnorm(
    b       ::KronProd{T},
    solution::Vector{T},
    Λ       ::AbstractMatrix{T},
    lowerX  ::Vector{Matrix{T}},
    M       ::KronMat,
    x       ::KruskalTensor{T}) where {T}

    # Explicit compressed residual norm
    b_norm               = kronprodnorm(b)
    exp_comp_res_norm    = norm(kron(b...) - solution)^2
    approx_comp_res_norm = compressed_residual(lowerX, Λ, M, x, b, b_norm)
    relative_error       = (exp_comp_res_norm - approx_comp_res_norm) * inv(exp_comp_res_norm)

    return relative_error

end



function tensorsquarednorm(x::KruskalTensor{T}) where T

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

function solvecompressed(H::KronMat, b::Vector{<:AbstractVector{T}}) where {T}

    H_expanded = Symmetric(kroneckersum(H.𝖳...))
    b_expanded = kron(b...)
    y          = H_expanded\b_expanded

    return y

end

function exactresidualnorm(A::KronMat, b::Vector{<:AbstractVector{T}}, xₖ::AbstractVector{T}) where {T}

    A_expanded = kroneckersum(A.M...)
    b_expanded = kron(b...)
    tmp        = zeros(size(A_expanded, 1))

    mul!(tmp, A_expanded, xₖ)

    rₖ = b_expanded - tmp

    return sqrt(dot(rₖ, rₖ)) * inv(LinearAlgebra.norm(b_expanded))

end

function Anormerror(A::AbstractMatrix{T}, x::AbstractVector{T}, xₖ::AbstractVector{T}) where T

    tmp  = zeros(size(x)) 
    diff = x - xₖ

    mul!(tmp, A, diff)

    return sqrt(dot(diff, diff))

end

function monotonic_decrease(errors::Vector{T}) where T 

    tmp = copy(errors[2:end])

    return all(tmp .< @view errors[1: end - 1])

end

function tensor_krylov_exact(A::KronMat, b::KronProd{T}, nmax::Int, orthonormalization_type::Type{<:TensorDecomposition}, tol = 1e-9) where {T}

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
    
    spectraldata = SpectralData{T,U}(nmax)
    approxdata   = ApproximationData{T, U}(tol)
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
        update_data!(spectraldata, d, A.matrixclass())
        update_data!(approxdata, spectraldata)

        y  = solvecompressed(H_minors, b_minors)
        #yₜ = solve_compressed_system(H_minors, b_minors, approxdata, spectraldata.λ_min[k])

        #@info "Relative error of solving compressed system: " norm(y - kroneckervectorize(yₜ)) * inv(norm(y))

        mul!(xₖ, kron(V_minors.𝖳...), y)

        r_norm     = exactresidualnorm(A, b, xₖ)
        errors[k]  = Anormerror(A_expanded, x, xₖ)

    end

    @test monotonic_decrease(errors)

end
