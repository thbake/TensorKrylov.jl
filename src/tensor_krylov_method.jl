export tensor_krylov, update_rhs!, KronMat, KronProd

using ExponentialUtilities: exponential!, expv

# Aliases
const KronProd{T}      = Vector{<:AbstractVector{T}} 
const KronMat{T}       = KroneckerMatrix{T}
const LowerTriangle{T} = LowerTriangular{T, <:AbstractMatrix{T}} 
const FMatrices{T}     = Vector{<:AbstractMatrix{T}} 


function exponentiate(A::AbstractMatrix{T}) where T <: AbstractFloat

    # Compute eigenvalues. Recall that the considered matrix is the Jacobi
    # matrix resulting from the Hermitian Lanczos algorithm. Therefore, the 
    # matrix is not necessarily Toepliz. However, we know that such a matrix
    # is of the form
    #   T = UᵀAU, where the eigenvalues of T are also eigenvalues of U. The 
    # question is then, which ones? The first k?


end

function get_subdiagonal_entries(A::KronMat, k::Int) 

    entries = [A[s][k + 1, k] for s in 1:length(A)]

    return entries

end

function matrix_exponential_vector!(
        y::ktensor,
        A::KronMat{T},
        b::KronProd{T},
        γ::T,
        k::Int) where T<:AbstractFloat

    for s = 1:length(A)

        #y.fmat[s][:, k] = LinearAlgebra.BLAS.gemv('N' , exp(- γ .*  A[s]), b[s])

        tmp = Matrix(copy(A[s]))

        #y.fmat[s][:, k] = γ .* exponential!(tmp) * b[s]
        y.fmat[s][:, k] = expv(γ, tmp, b[s])

    end

end

function innerprod_kronsum_tensor!(
        yX::FMatrices{T},
        yAx::FMatrices{T},
        Ax::FMatrices{T},
        x::ktensor,
        y::KronProd{T}) where T <: AbstractFloat

    # Computes <Ax, y>₂, where A is a matrix (Kronecker sum) and y is a Kruskal tensor.
    mul!(yX, y, x)    
    mul!(yAx, y, Ax)  

    mask = trues(length(Ax))

    @assert length(Ax) == ndims(x)

    Ax_y = 0.0

    for s = 1:length(Ax)

        mask[s] = false

        yX_mask  = yX[mask]
        yAx_mask = yAx[.!mask]
        
        for i = 1:ncomponents(x)

            # Scale here with lambda
            Ax_y += x.lambda[i] * maskprod(yX_mask, i) * maskprod(yAx_mask, i)

        end

        mask[s] = true

    end

    return Ax_y

end

function solve_compressed_system(
        H::KronMat{T}, 
        b::Vector{<:AbstractVector{T}}, 
        ω::Array{T},
        α::Array{T},
        t::Int,
        λ::T,
    ) where T <: AbstractFloat

    reciprocal = inv(λ)

    # Since we are considering a canonical decomposition the tensor rank of yₜ
    # is equal to 

    k = dimensions(H)
    
    yₜ = ktensor(reciprocal .* ω, [ ones(k[s], t) for s in 1:length(H)] )

    for k = 1:t

        γ = -α[k] * reciprocal

        matrix_exponential_vector!(yₜ, H, b, γ, k)

    end

    return yₜ
end

function compute_lower_outer!(L::AbstractMatrix{T}, γ::Array{T}) where T <: AbstractFloat

    # Lower triangular matrix representing the outer product of a vector with itself

    t = size(L, 1)

    for j = 1:t, i = j:t

        L[i, j] = γ[i] * γ[j]

    end

end

function compute_coefficients(Λ::LowerTriangle{T}, δ::Array{T}) where T <: AbstractFloat

    t = length(δ)

    Δ = ones(t, t)

    # Lower triangle of outer product
    compute_lower_outer!(Δ, δ) # ∈ ℝᵗᵗ

    Γ = Δ .* Λ

    return Γ

end

function matrix_vector(
        A::KronMat{T},
        x::ktensor)::AbstractVector where T<:AbstractFloat

    # Compute the matrix vector products 
    #   
    #   z⁽ˢ⁾ᵢ = Aₛ⋅ x⁽ˢ⁾ᵢ for s = 1,…,d, i = 1,…,t
    #
    # This is equivalent as computing the product Z⁽ˢ⁾ = Aₛ⋅X⁽ˢ⁾, where X⁽ˢ⁾
    # are the factor matrices of the CP-tensor x.

    length(A) == ndims(x) || throw(DimensionMismatch("Kronecker matrix and vector (Kruskal tensor) have different number of components"))

    orders = [ size(A[s], 1) for s in 1:length(A) ]

    rank   = ncomponents(x)

    # Return vector of matrices as described above
    Z = [ zeros(orders[s], rank) for s in eachindex(A) ]

    @info A[1]

    for s = 1:length(A)

        LinearAlgebra.mul!(Z[s], A[s], x.fmat[s])

    end

    return Z

end

function skipindex(index::Int, range::UnitRange{Int})

    return Iterators.filter(r -> r != index, range)

end

function maskprod(A::FMatrices{T}, i::Int, j::Int) where T <: AbstractFloat

    # Compute product of entries (i,j) of the matrices contained in A.

    return prod(getindex.(A, i, j)) 

end


function maskprod(x::FMatrices{T}, i::Int) where T <: AbstractFloat

    return prod(getindex.(x, i)) 

end

function efficient_matrix_vector_norm(
        x::ktensor,
        Λ::AbstractMatrix{T},
        X_inner::FMatrices{T},
        Z::FMatrices{T}) where T <: AbstractFloat

    # Compute the squared 2-norm ||Ax||², where A ∈ ℝᴺ×ᴺ is a Kronecker sum and
    # x ∈ ℝᴺ is given as a Kruskal tensor of rank t.
    #
    # X_inner holds the inner products 
    #
    #   xᵢ⁽ˢ⁾ᵀxⱼ⁽ˢ⁾ for s = 1,…,d, i,j = 1,…,t
    #
    # And Z contains the matrices that represent the matrix vector products
    # 
    #   z⁽ˢ⁾ᵢ = Aₛ⋅ x⁽ˢ⁾ᵢ for s = 1,…,d, i = 1,…,t
    #
    # A is not passed explicitly, as the precomputed inner products are given.

    d      = ndims(x)
    rank   = ncomponents(x)

    # The following contain inner products of the form 
    #
    #   zᵢ⁽ˢ⁾ᵀzⱼ⁽ˢ⁾ for s = 1,…,d, i,j = 1,…,t,
    # 
    # and 
    #
    #   zᵢ⁽ˢ⁾ᵀxⱼ⁽ˢ⁾ for s = 1,…,d, i,j = 1,…,t,
    #
    # respcetively

    Z_inner = [ zeros(rank, rank) for _ in 1:d ]
    ZX      = [ zeros(rank, rank) for _ in 1:d ]

    compute_lower_triangles!(Z_inner, Z)

    #@info Z_inner

    for s in 1:d

        BLAS.gemm!('T', 'N', 1.0, Z[s], x.fmat[s], 1.0, ZX[s]) 

    end

    result = 0.0

    mask_s = trues(d)
    mask_r = trues(d)

    # We can separate the large sum 
    #
    #   ΣₛΣᵣΣᵢΣⱼ xᵢ⁽¹⁾ᵀxⱼ⁽¹⁾ ⋯ zᵢ⁽ˢ⁾ᵀxⱼ⁽ˢ⁾ ⋯ xᵢ⁽ʳ⁾ᵀzⱼ⁽ʳ⁾ ⋯ xᵢ⁽ᵈ⁾ᵀxⱼ⁽ᵈ⁾
    #
    # into the cases 
    #
    #   (1) s  = r, i  = j,
    #   (2) s  = r, i != j,
    #   (3) s != r, i  = j,
    #   (4) s != r, i != j
    #
    # and simplify the calculation using the fact that some inner products 
    # appear twice (only access lower triangle of matrices) and that the norm
    # of the columns of the factor matrices are one.

    for s in 1:d

        for j = 1:rank # case (1)

            result += Λ[j, j] * Z_inner[s][j, j]

            mask_s[s] = false

            tmp = 0.0

            for i = skipindex(j, j:rank) # case (2)

                tmp += Λ[i, j] * maskprod(X_inner[mask_s], i, j) * maskprod(Z_inner[.!mask_s], i, j)

            end

            result += 2 * tmp

        end

        ZX_masked = ZX[.!mask_s]

        for r = skipindex(s, 1:d) # case (3)

            mask_r[r] = false

            mask_sr = mask_s .&& mask_r

            X_masked  = X_inner[mask_sr]
            XZ_masked =      ZX[.!mask_r]

            for i = 1:rank

                result += Λ[i, i] * ZX[s][i, i] * ZX[r][i, i]

                tmp = 0.0

                for j = skipindex(i, 1:rank) # case (4)

                    tmp += Λ[j, i] * maskprod(X_masked, i, j) *  maskprod(ZX_masked, i, j) * maskprod(XZ_masked, j, i)

                end

                result += 2 * tmp

            end

            mask_r[r] = true
        end

        mask_s[s] = true

    end

    return result

end


function compressed_residual(H::KronMat{T}, y::ktensor, b::KronProd{T}) where T<:AbstractFloat

    # This variant expands the matrices/tensors

    N = nentries(H)

    H_expanded = sparse(kroneckersum(H.𝖳...))
    y_expanded = reshape(full(y), N)
    b_expanded = kronecker(b...)

    @assert issparse(H_expanded)

    comp_res = (H_expanded * y_expanded) - b_expanded
    
    @info dot(comp_res, comp_res)
    return dot(comp_res, comp_res)

end

function squared_tensor_entries(Y_masked::FMatrices{T}, Γ::AbstractMatrix{T}) where T <: AbstractFloat

    # Compute Σ |y_𝔏|² with formula in paper, when y is given in CP format:
    #
    #   Σ |y_𝔏|² = ||Σᵢ eₖₛᵀ yᵢ⁽ˢ⁾ ⨂ ⱼ≠ ₛ yᵢ⁽ʲ⁾||², 
    #
    # where δ represents the vector holding kₛ-th entry of each column of the 
    # s-th factor matrix of y.
    
    t = size(Y_masked, 1)

    value = 0.0

    for k = 1:t

        value += Γ[k, k] 

        for i = skipindex(k, k:t)

            value += 2 * Γ[i, k] * maskprod(Y_masked, i, k) # Symmetry

        end
    end

    return value 
end

function compute_lower_triangles!(LowerTriangles::FMatrices{T}, x::ktensor) where T<:AbstractFloat

    for s = 1:length(LowerTriangles)

        BLAS.syrk!('L', 'T', 1.0, x.fmat[s], 1.0, LowerTriangles[s])

    end

end

function compute_lower_triangles!(LowerTriangles::FMatrices{T}, x::FMatrices{T}) where T<:AbstractFloat

    for s = 1:length(LowerTriangles)

        BLAS.syrk!('L', 'T', 1.0, x[s], 1.0, LowerTriangles[s])

    end

end

function residual_norm(
        H::KronMat{T},
        y::ktensor,
        𝔎::Vector{Int},
        subdiagonal_entries::Vector{T},
        b::KronProd{T}) where T<:AbstractFloat
    
    # Compute squared norm of the residual according to Lemma 3.4 of paper.
    
    # Σ |hˢₖ₊₁ₖ|² * Σ |y\_𝔏|² + ||ℋy - b̃||²
    
    # Get entries at indices (kₛ+1, kₛ) for each dimension with pair of 
    # multiindices 𝔎+1, 𝔎

    d = length(H) # Number of dimensions

    t = ncomponents(y) # Tensor rank

    # Allocate memory for (lower triangular) matrices representing inner products
    Ly = [ zeros(t, t) for _ in 1:d ]

    compute_lower_triangles!(Ly, y)

    # Allocate memory for (lower triangular) matrix representing outer product
    # of coefficients.
    
    Λ = LowerTriangular(zeros(t, t))

    compute_lower_outer!(Λ, y.lambda)

    # Make matrices lower triangular
    Ly = map(LowerTriangular, Ly)

    res_norm = 0.0

    mask = trues(d)

    for s = 1:d

        Γ = compute_coefficients(Λ, y.fmat[s][𝔎[s], :]) # Symmetric matrix 

        mask[s] = false

        y² = squared_tensor_entries(Ly[.!mask], Γ)

        res_norm += abs( subdiagonal_entries[s] )^2 * y²

        mask[s] = true

    end

    # Compute squared compressed residual norm
    #r_compressed = compressed_residual(Ly, Λ, H, y, b)
    r_compressed = compressed_residual(H, y, b)

    #@info r_compressed
    
    return sqrt(res_norm + r_compressed)

end

function update_rhs!(b̃::KronProd{T}, V::KronProd{T}, b::KronProd{T}) where T<:AbstractFloat

    # b̃ = Vᵀb = ⨂ Vₛᵀ ⋅ ⨂ bₛ = ⨂ Vₛᵀbₛ
    
    for s = 1:length(b̃)

        # Update one entry of each component of b̃ by performing a single inner product 
        b̃[s][end] = dot( V[s] , b[s] )
     
    end

end

function basis_tensor_mul!(x::ktensor, V::KronMat{T}, y::ktensor) where T<:AbstractFloat

    x.lambda = copy(y.lambda)

    for s in eachindex(V)

        LinearAlgebra.mul!(x.fmat[s], V[s], y.fmat[s])

    end

end

function initialize(
        A::KronMat{T},
        b::KronProd{T},
        orthonormalization::Type{<:TensorDecomposition}) where T <: AbstractFloat


    # Initialize the d Arnoldi decompositions of Aₛ
    tensor_decomposition = orthonormalization{T}(A)

    # Initialize multiindex 𝔎
    𝔎 = Vector{Int}(undef, d)

    # Allocate memory for right-hand side b̃
    b̃ = [ zeros( size(b[s]) )  for s in eachindex(b) ]

    orthonormal_basis!(tensor_decomposition, b, 1, orthonormalization)

end


function tensor_krylov(
        A::KronMat{T},
        b::KronProd{T},
        tol::T,
        nmax::Int,
        orthonormalization::Type{<:TensorDecomposition}) where T <: AbstractFloat

	# Initilialize implicit tensorized Krylov subspace basis and upper Hessenberg 
    d = length(A)

    # Initialize the d Arnoldi decompositions of Aₛ
    tensor_decomposition = orthonormalization{T}(A)

    # Initialize multiindex 𝔎
    𝔎 = Vector{Int}(undef, d)

    # Allocate memory for right-hand side b̃
    b̃ = [ zeros( size(b[s]) )  for s in eachindex(b) ]

    #@info "b̃ after update" b̃

    # Allocate memory for approximate solution
    #x = ktensor( ones(1), [ zeros(size(A[s], 1), 1) for s in 1:d ] )
    x = nothing

    coefficients_df = compute_dataframe()

    orthonormal_basis!(tensor_decomposition, b, 1, orthonormalization)

    H_minors = principal_minors(tensor_decomposition.H, 1)
    V_minors = principal_minors(tensor_decomposition.V, 1)
    b_minors = principal_minors(b̃, 1)

    y = inv( sum(H_minors[1, 1]) ) * prod( [ b_minors[s][1] for s in 1:d ] )

    x = prod(V_minors[1, 1]) * y

    for k = 2:nmax

        # Compute orthonormal basis and Hessenberg factor of each Krylov subspace 𝓚ₖ(Aₛ, bₛ) 
        orthonormal_basis!(tensor_decomposition, b, k, orthonormalization)

        H_minors = principal_minors(tensor_decomposition.H, k)
        V_minors = principal_minors(tensor_decomposition.V, k)
        b_minors = principal_minors(b̃, k)

        columns = kth_columns(tensor_decomposition.V, k)

        # Update compressed right-hand side b̃ = Vᵀb
        update_rhs!(b_minors, columns, b)

        b_norm = kronprodnorm(b_minors)

        λ_min, λ_max = projected_kronecker_eigenvalues(H_minors)

        κ = λ_max / λ_min

        ω, α, rank = optimal_coefficients(coefficients_df, tol, κ, λ_min, b_norm)

        # Approximate solution of compressed system
        y = solve_compressed_system(H_minors, b_minors, ω, α, rank, λ_min)

        𝔎 .= k 

        subdiagonal_entries = [ tensor_decomposition.H[s][k + 1, k] for s in 1:d ]

        @info subdiagonal_entries

        # Compute residual norm
        r_norm = residual_norm(H_minors, y, 𝔎, subdiagonal_entries, b_minors)

        rel_res_norm = (r_norm / kronprodnorm(b_minors))

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
