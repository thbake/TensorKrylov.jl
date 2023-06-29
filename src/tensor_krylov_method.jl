# Aliases
const KronProd{T}      = Vector{Vector{T}} 
const KronMat{T}       = KroneckerMatrix{T}
const LowerTriangle{T} = LowerTriangular{T, Matrix{T}} 
const FMatrices{T}     = Vector{AbstractMatrix{T}} 


function matrix_exponential_vector!(
        factors::AbstractVector,
        A::KronMat{T},
        b::KronProd{T},
        γ::T) where T<:AbstractFloat

    for s = 1:length(A)

        factors[s] = LinearAlgebra.BLAS.gemv('N' , exp(- γ * A[s]), b[s] )

    end

end

function innerprod_kronsum_tensor!(
        yX::Vector{Matrix{T}},
        yAx::Vector{Matrix{T}},
        Ax::Vector{Matrix{T}},
        x::ktensor,
        y::Vector{Vector{T}}) where T <: AbstractFloat

    # Computes <Ax, y>₂, where A is a matrix (Kronecker sum) and y is a Kruskal tensor.
    mul!(yX, y, x)    
    mul!(yAx, y, Ax)  

    mask = trues(length(Ax))

    @assert length(Ax) == ndims(x)

    Ax_y = 0.0

    for s = 1:length(Ax)

        mask[s] = false
        
        for i = 1:ncomponents(x)

            # Scale here with lambda
            Ax_y += x.lambda[i] * maskprod(yX[mask], i) * maskprod(yAx[.!mask], i)

        end

        mask[s] = true

    end

    return Ax_y

end

function solve_compressed_system(
        H::KronMat{T}, 
        b::Vector{Vector{T}}, 
        ω::Array{T},
        α::Array{T},
        t::Int,
        j::Int
    ) where T <: AbstractFloat

    λ = min_eigenvalue(H) # This might be different depending on the system

    reciprocal = inv(λ)

    # Since we are considering a canonical decomposition the tensor rank of yₜ
    # is equal to 
    
    yₜ = ktensor(reciprocal .* ω, [ ones(t,t) for _ in 1:length(H)] )
    
    for j = 1:t

        γ = -α[j] * reciprocal

        matrix_exponential_vector!(yₜ.fmat, H, b, γ)

    end

    return yₜ
end

function compute_lower_outer!(L::LowerTriangle{T}, γ::Array{T}) where T <: AbstractFloat

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
        A::KroneckerMatrix{T},
        x::ktensor)::AbstractVector where T<:AbstractFloat

    # Compute the matrix vector products 
    #   
    #   z⁽ˢ⁾ᵢ = Aₛ⋅ x⁽ˢ⁾ᵢ for s = 1,…,d, i = 1,…,t
    #
    # This is equivalent as computing the product Z⁽ˢ⁾ = Aₛ⋅X⁽ˢ⁾, where X⁽ˢ⁾
    # are the factor matrices of the CP-tensor x.

    orders = dimensions(A)
    rank   = ncomponents(x)

    # Return vector of matrices as described above
    Z = [ zeros(orders[s], rank) for s in eachindex(A) ]

    for s = 1:length(A)

        LinearAlgebra.mul!(Z[s], A[s], x.fmat[s])

    end

    return Z

end

function skipindex(index::Int, range::UnitRange{Int})

    return Iterators.filter(r -> r != index, range)

end

function maskprod(A::Vector{Matrix{T}}, i::Int, j::Int) where T <: AbstractFloat

    # Compute product of entries (i,j) of the matrices contained in A.

    return prod(getindex.(A, i, j)) 

end

function maskprod(A::Vector{LowerTriangle{T}}, i::Int, j::Int) where T <: AbstractFloat

    # Compute product of entries (i,j) of the matrices contained in A.

    return prod(getindex.(A, i, j)) 

end

function maskprod(x::Vector{Matrix{T}}, i::Int) where T <: AbstractFloat

    return prod(getindex.(x, i)) 

end

function efficient_matrix_vector_norm(
        x::ktensor,
        Λ::AbstractMatrix{T},
        X_inner::Vector{LowerTriangle{T}},
        Z::Vector{Matrix{T}}) where T <: AbstractFloat

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

        for r = skipindex(s, 1:d) # case (3)

            mask_r[r] = false

            for i = 1:rank

                result += Λ[i, i] * ZX[s][i, i] * ZX[r][i, i]

                for j = skipindex(i, 1:rank) # case (4)

                    mask_sr = mask_s .&& mask_r

                    result += Λ[j, i] * 2 * maskprod(X_inner[mask_sr], i, j) *  maskprod(ZX[.!mask_s], i, j) * maskprod(ZX[.!mask_r], j, i)

                end

            end

            mask_r[r] = true
        end

        mask_s[s] = true

    end

    return result

end


function compressed_residual(
        Ly::Vector{LowerTriangle{T}},
        Λ::LowerTriangle{T},
        H::KroneckerMatrix{T},
        y::ktensor,
        b) where T <:AbstractFloat

    # TODO: 
    #
    # We know that 
    #
    # ||Hy - b||² = ||Hy||² -2⋅bᵀ(Hy) + ||b||² 
    
    d = length(H)
    t = ncomponents(y)

    # For this we evaluate all Z⁽ˢ⁾[:, i] = Hₛy⁽ˢ⁾ᵢ ∈ ℝⁿₛ for i = 1,…,t
    Z = matrix_vector(H, y)

    # First we compute ||Hy||²
    Hy_norm = efficient_matrix_vector_norm(y, Symmetric(Λ, :L), Ly, Z)

    # Now we compute <Hy, b>₂
    bY = [ zeros(1, t) for _ in 1:d ] # bₛᵀyᵢ⁽ˢ⁾
    bZ = [ zeros(1, t) for _ in 1:d ] # bₛᵀzᵢ⁽ˢ⁾, where zᵢ⁽ˢ⁾ = Hₛ⋅yᵢ⁽ˢ⁾

    Hy_b = innerprod_kronsum_tensor!(bY, bZ, Z, y, b)

    # Finally we compute the squared 2-norm of b
    b_norm = prod( dot(b[s], b[s]) for s in 1:d )

    return Hy_norm - 2 * Hy_b + b_norm
    
end

function squared_tensor_entries(
        Y_masked::Vector{LowerTriangle{T}},
        Γ::LowerTriangle{T}) where T <: AbstractFloat

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

    function compute_lower_triangles!(LowerTriangles::Vector{Matrix{T}}, x::Vector{Matrix{T}}) where T<:AbstractFloat

    for s = 1:length(LowerTriangles)

        BLAS.syrk!('L', 'T', 1.0, x[s], 1.0, LowerTriangles[s])

    end

end

function residual_norm(H::KronMat{T}, y::ktensor, 𝔎::Vector{Int}, b) where T<:AbstractFloat
    
    # Compute norm of the residual according to Lemma 3.4 of paper.
    
    # Σ |hˢₖ₊₁ₖ|² * Σ |y\_𝔏|² + ||ℋy - b̃||²
    
    # Get entries at indices (kₛ+1, kₛ) for each dimension with pair of 
    # multiindices 𝔎+1, 𝔎

    d = length(H) # Number of dimensions

    t = ncomponents(y) # Tensor rank

    # Allocate memory for (lower triangular) matrices representing inner products
    Ly = [ zeros(t, t) for _ in 1:d ]

    compute_lower_triangles!(Ly, y.fmat)

    # Allocate memory for (lower triangular) matrix representing outer product
    # of coefficients.
    
    Λ = LowerTriangular(zeros(t, t))

    Λ = compute_lower_outer!(Λ, y.lambda)

    # Make matrices lower triangular
    Ly = map(LowerTriangular, Ly)

    res_norm = 0.0

    mask = trues(d)

    for s = 1:d

        Γ = compute_coefficients(Λ, y.fmat[s][𝔎[s], :]) # Symmetric matrix 

        mask[s] = false

        y² = squared_tensor_entries(Ly[mask], Γ)

        res_norm += abs( H[𝔎[s] + 1, 𝔎[s]] )^2 * y²

        mask[s] = true

    end

    # Compute squared compressed residual norm
    r_compressed = compressed_residual(Ly, Λ, H, y, b)

    return res_norm + r_compressed

end

function update_rhs!(b̃::KronProd{T}, V::KronMat{T}, b::KronProd{T}, k::Int) where T<:AbstractFloat

    # b̃ = Vᵀb = ⨂ Vₛᵀ ⋅ ⨂ bₛ = ⨂ Vₛᵀbₛ
    
    for s = 1:length(b̃)

        # Update one entry of each component of b̃ by performing a single inner product 
        mul!( b̃[s][k], transpose( @view(V[s][:, k]) ), b[s] )
     
    end

end

function basis_tensor_mul!(x::ktensor, V::KronMat{T}, y::ktensor) where T<:AbstractFloat

    x.lambda = copy(y.lambda)

    for s in eachindex(V)

        mul!(x.fmat[s], V[s], y.fmat[s])

    end

end

function tensor_krylov(A::KronMat{T}, b::KronProd{T}, tol::T, nmax::Int, ω, α, rank) where T <: AbstractFloat

	# Initilialize implicit tensorized Krylov subspace basis and upper Hessenberg 
    d = length(A)

    # Initialize the d Arnoldi decompositions of Aₛ
    decomps = Arnoldis(A, b)

    # Initialize multiindex 𝔎
    𝔎 = Vector{Int}(undef, d)

    # Allocate memory for right-hand side b̃
    b̃ = [ zeros( size(b[s]) )  for s in eachindex(b) ]

    # Allocate memory for approximate solution
    x = ktensor( ones(rank), zeros(d, rank))

    for j = 1:nmax

        # Compute orthonormal basis and Hessenberg factor of each Krylov subspace 𝓚ₖ(Aₛ, bₛ) 
        arnoldi_step!(decomps, j)

        update_rhs!(b̃, decomps.V, b, j)

        y = solve_compressed_system(decomps.H, b̃, ω, α, rank, j)

        𝔎 .= j 

        r_norm = residual_norm(decomps.H, y, 𝔎, b̃)

        if r_norm < tol

            basis_tensor_mul!(x, decomps.V, y)

            return x

        end

    end

end
