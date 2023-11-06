export KronMat, KronProd, LowerTriangle, FMatrices

# Aliases
const KronProd{T}      = Vector{<:AbstractVector{T}} 
const KronMat{T}       = KroneckerMatrix{T}
const LowerTriangle{T} = LowerTriangular{T, <:AbstractMatrix{T}} 
const FMatrices{T}     = Vector{<:AbstractMatrix{T}} 

function compute_lower_outer!(L::AbstractMatrix{T}, γ::Array{T}) where T <: AbstractFloat

    # Lower triangular matrix representing the outer product of a vector with itself

    t = size(L, 1)

    for j = 1:t, i = j:t

        L[i, j] = γ[i] * γ[j]

    end

end

function compute_coefficients(Λ::LowerTriangle{T}, δ::Array{T}) where T <: AbstractFloat

    # Given a collection of lower triangular matrices containing all values of 
    # λ⁽ˢ⁾corresponding to each factor matrix in the CP-decomposition of the 
    # tensor y, and an array δ containing the k-th entry of a column of said 
    # factor matrices, compute the product of both (see section 3.3. bottom).

    t = length(δ)

    Δ = ones(t, t)

    # Lower triangle of outer product
    compute_lower_outer!(Δ, δ) # ∈ ℝᵗᵗ

    Γ = Δ .* Λ

    return Γ

end

function skipindex(index::Int, range::UnitRange{Int})

    return Iterators.filter(r -> r != index, range)

end

function maskprod(A::FMatrices{T}, i::Int, j::Int) where T <: AbstractFloat

    # Compute product of entries (i,j) of the matrices contained in A.

    return prod(getindex.(A, i, j)) 

end


function maskprod(x::FMatrices{T}, i::Int) where T <: AbstractFloat

    return prod(getindex.(x, 1, i)) 

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

function squared_tensor_entries(Y_masked::FMatrices{T}, Γ::AbstractMatrix{T}) where T <: AbstractFloat

    # Compute Σ |y_𝔏|² with formula in paper, when y is given in CP format:
    #
    #   Σ |y_𝔏|² = ||Σᵢ eₖₛᵀ yᵢ⁽ˢ⁾ ⨂ ⱼ≠ ₛ yᵢ⁽ʲ⁾||², 
    #
    # where δ represents the vector holding kₛ-th entry of each column of the 
    # s-th factor matrix of y.
    #
    # We use the symmetry of the inner products and only require to iterate in
    # the correct way:
    #
    # 2 ⋅Σₖ₌₁ Σᵢ₌ₖ₊₁ Γ[i, k] ⋅ Πⱼ≠ ₛ<yᵢ⁽ʲ⁾,yₖ⁽ʲ⁾> + Σᵢ₌₁ Πⱼ≠ ₛ||yᵢ⁽ʲ⁾||²
    
    t = size(Γ, 1)

    value = 0.0

    for k = 1:t, i = 1:t

        value += Γ[i, k] * maskprod(Y_masked, i, k)

    end

    return value 
end


function matrix_vector(A::KronMat{T}, x::ktensor)::AbstractVector where T<:AbstractFloat

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

    for s = 1:length(A)

        LinearAlgebra.mul!(Z[s], A[s], x.fmat[s])

    end

    return Z

end

function MVnorm(x::ktensor, Λ::AbstractMatrix{T}, lowerX::FMatrices{T}, Z::FMatrices{T}) where T<:AbstractFloat

    Λ_complete = Symmetric(Λ, :L)
    X          = Symmetric.(lowerX, :L)
    Z_inner    = [ Z[s]'Z[s] for s in 1:length(Z) ]
    XZ         = [ x.fmat[s]'Z[s] for s in 1:ndims(x) ]

    d    = length(lowerX)
    rank = length(x.lambda)

    mask_s = falses(d)
    mask_r = falses(d)

    MVnorm = 0.0

    for j = 1:rank, i = 1:rank

        for s in 1:d

            mask_s[s] = true

            for r in 1:d

                mask_r[r] = true

                MVnorm += Λ_complete[i, j] * maskprod( X[.!(mask_s .|| mask_r)], i, j ) *  maskprod(XZ[mask_s .⊻ mask_r], i, j) * maskprod(Z_inner[mask_s .&& mask_r], i, j)

                mask_r[r] = false
            end

            mask_s[s] = false

        end

    end

    @assert MVnorm > 0.0

    return MVnorm
    
end

function tensorinnerprod(Ax::FMatrices{T}, x::ktensor, y::KronProd{T}) where T<:AbstractFloat

    d   = ndims(x)
    yX  = [ transpose(y[s]) * x.fmat[s] for s in 1:d ]
    yAx = [ transpose(y[s]) * Ax[s] for s in 1:d ]

    Ax_y = 0.0

    mask = falses(d)
    
    for s in 1:d

        for i in 1:ncomponents(x)

            mask[s] = true

            Ax_y += maskprod(yAx[mask], i) * maskprod(yX[.!mask], i)

            mask[s] = false

        end

    end

    @assert Ax_y > 0.0

    return Ax_y

end

function efficientMVnorm(x::ktensor, Λ::AbstractMatrix{T}, X_inner::FMatrices{T}, Z::FMatrices{T}) where T <: AbstractFloat

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
            XZ_masked =     ZX[.!mask_r]

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

    d = ndims(y)
    Z = matrix_vector(H, y)

    vec     = zeros(prod(size(y))) # Here I run out of memory
    indices = collect(1:d)

    for i in 1:ncomponents(y)

        mask = trues(d)

        for s in 1:d

            tmp = [ zeros(size(y)[s]) for s in 1:d ]

            mask[s] = false

            tmp[s]             = @view(Z[s][:, i])
            columns            = kth_columns(y.fmat[mask], i)
            tmp[indices[mask]] = columns

            vec += kron(tmp...)

            mask[s] = true

        end

    end

    comp_res      = vec - kron(b...)
    comp_res_norm = dot(comp_res, comp_res)

    return comp_res_norm

end

function compressed_residual(
        Ly::FMatrices{T},
        Λ::AbstractMatrix{T},
        H::KronMat{T},
        y::ktensor,
        b::KronProd{T}) where T <:AbstractFloat

    # We know that 
    
    #   ||Hy - b||² = ||Hy||² -2⋅bᵀ(Hy) + ||b||² 

    # For this we evaluate all z⁽ˢ⁾ᵢ=  Z⁽ˢ⁾[:, i] = Hₛy⁽ˢ⁾ᵢ ∈ ℝᵏₛ for i = 1,…,t
    Z = matrix_vector(H, y)

    Ly = Symmetric.(Ly, :L)

    # First we compute ||Hy||²
    Hy_norm = MVnorm(y, Symmetric(Λ, :L), Ly, Z)

    # Now we proceed with <Hy, b>₂
    Hy_b = tensorinnerprod(Z, y, b)

    # Finally we compute the squared 2-norm of b
    b_norm = kronproddot(b)

    comp_res = Hy_norm - 2* Hy_b + b_norm

    @assert comp_res > 0.0

    return comp_res
    
end

#function compressed_residual(H::KronMat{T}, y::ktensor, b::KronProd{T}) where T<:AbstractFloat
#
#    # This variant expands the matrices/tensors
#
#    N = nentries(H)
#
#    H_expanded = sparse(Matrix(kroneckersum(H.𝖳...)))
#    y_expanded = reshape(full(y), N)
#    b_expanded = kronecker(b...)
#
#    x = zeros(N)
#
#    @assert issparse(H_expanded)
#
#    #mul!(x, H_expanded, y_expanded)
#
#    comp_res = (H_expanded * y_expanded) - b_expanded
#    comp_res = x - b_expanded
#    
#    @info "Compressed residual" dot(comp_res, comp_res)
#    return dot(comp_res, comp_res)
#
#end

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
    Ly = Symmetric.(Ly, :L)

    res_norm = 0.0

    mask = trues(d)

    for s = 1:d

        Γ = Symmetric(compute_coefficients(Λ, y.fmat[s][𝔎[s], :]), :L) # Symmetric matrix 

        mask[s] = false

        y² = squared_tensor_entries(Ly[mask], Γ)

        res_norm += abs( subdiagonal_entries[s] )^2 * y²

        mask[s] = true

    end

    # Compute squared compressed residual norm
    r_compressed = compressed_residual(Ly, Λ, H, y, b)
    
    #r_compressed = compressed_residual(H, y, b)
    @info r_compressed
    @info res_norm

    return sqrt(res_norm + r_compressed)

end
