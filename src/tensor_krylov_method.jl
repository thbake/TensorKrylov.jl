function mul!(
        result::Vector{Array{T}},
        KP::Vector{Array{T}},
        x::ktensor) where T <: AbstractFloat

    nᵢ = ndims(x)

   for s = 1:nᵢ

       result[s] = transpose(KP[s]) * (x.lambda[s] * x.fmat[s])

   end

end

function mul!(
        result::Vector{Array{T}},
        KP::Vector{Array{T}},
        X::Vector{Matrix{T}}) where T <: AbstractFloat

    nᵢ = length(result)

   for s = 1:nᵢ

       result[s] = transpose(KP[s]) * X.fmat[s]

   end

end

function matrix_exponential_vector!(
        factors::AbstractVector,
        A::KroneckerMatrix{T},
        b::Vector{Array{T}}, γ::T) where T<:AbstractFloat

    for s = 1:length(A)

        factors[s] = LinearAlgebra.BLAS.gemv('N' , exp(- γ * A[s]), b[s] )

    end

end

#function solve_compressed_system(
#        H::KroneckerMatrix{T}, 
#        b::Vector{Array{T}}, 
#        ω::Array{T},
#        α::Array{T},
#        t::Int) where T <: AbstractFloat
#
#    λ = smallest_eigenvalue(H) # This might be different depending on the system
#
#    reciprocal = inv(λ)
#
#    # Since we are considering a canonical decomposition the tensor rank of yₜ
#    # is equal to 
#    
#    yₜ = ktensor(reciprocal .* ω, [ ones(t,t) for _ in 1:length(H)] )
#    
#    for j = 1:t
#
#        γ = -α[j] * reciprocal
#
#        matrix_exponential_vector!(yₜ.fmat, H, b, γ)
#
#    end
#
#    return yₜ
#end

function hessenberg_subdiagonals(H::AbstractMatrix, 𝔎::Vector{Int})

    # Extract subdiagonal entries (kₛ₊₁, kₛ) of matrix H⁽ˢ⁾ of ℋ     

    d = length(𝔎)

    entries = Array{Float64}(undef, d)

    for s = 1:d

        entries[s] = H[𝔎[s] + 1, 𝔎[s]]

    end

    return entries

end

function compute_lower_triangle!(
        LowerTriangle::LowerTriangular{T, Matrix{T}},
        A::Matrix{T},
        B::Matrix{T},
        γ::Array{T},
        k::Int) where T <: AbstractFloat

    t = length(γ)

    for j = 1:t-k, i = j+k:t

        LowerTriangle[i, j] = (γ[i]*γ[j])dot(@view(A[:, j]), @view(B[:, i]))
        
    end

end

function compute_lower_triangle!(
        LowerTriangle::LowerTriangular{T, Matrix{T}},
        A::Matrix{T},
        k::Int) where T <: AbstractFloat

    t = size(A, 2)

    for j = 1:t-k, i = j+k:t

        LowerTriangle[i, j] = dot(@view(A[:, j]), @view(A[:, i]))
        
    end

end

function compute_lower_triangle!(
        LowerTriangle::LowerTriangular{T, Matrix{T}},
        γ::Array{T}) where T <: AbstractFloat

    t = size(LowerTriangle, 1)

    for j = 1:t, i = j:t

        LowerTriangle[i, j] = γ[i] * γ[j]

    end

end

function compute_coefficients(
        LowerTriangle::LowerTriangular{T, Matrix{T}},
        δ::Vector{T}) where T <: AbstractFloat

    # Compute Σ |y_𝔏|² with formula in paper, when y is given in CP format:
    #
    # Σ |y_𝔏|² = ||Σᵢ eₖₛᵀ yᵢ⁽ˢ⁾ ⨂ ⱼ≠ ₛ yᵢ⁽ʲ⁾||².
    
    # Get the kₛ-th entry of each column of the s-th factor matrix of y.
    t = length(δ)

    Δ = ones(t, t)

    compute_lower_triangle!(Δ, δ) # ∈ ℝᵗᵗ

    Γ = Δ .* LowerTriangle

    return Γ

end

function innerproducts!(
        LowerTriangles::Vector{LowerTriangular{T, Matrix{T}}},
        factormatrix,
        k::Int) where T <: AbstractFloat

    for s in eachindex(LowerTriangles)

        compute_lower_triangle!(LowerTriangles[s], factormatrix[s], k)
    end
    
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

function efficient_matrix_vector_norm(
        A::KroneckerMatrix{T},
        x::ktensor,
        X_inner::Vector{Matrix{T}},
        Z::Vector{Matrix{T}}) where T <: AbstractFloat

    orders = dimensions(A)
    d      = ndims(x)
    rank   = ncomponents(x)

    Z_inner = [ zeros(rank, rank) for s in 1:d ]
    ZX      = [ zeros(rank, rank) for s in 1:d ]

    for s in 1:d

        BLAS.syrk!('L', 'T', 1.0, Z[s], 1.0,  Z_inner[s])
        BLAS.gemm!('T', 'N', 1.0, Z[s], x.fmat[s], 1.0, ZX[s])

    end

    result = 0.0

    mask_s = trues(d)
    mask_r = trues(d)

    for s in 1:d

        for j = 1:rank

            result += x.lambda[j]^2 * Z_inner[s][j, j]

            mask_s[s] = false

            for i = skipindex(j, j:rank)

                result += 2 * x.lambda[j] * x.lambda[i] * prod(getindex.(X_inner[mask_s], i, j)) * Z_inner[.!mask_s][1][i, j]

            end

        end

        for r = skipindex(s, 1:d)

            mask_r[r] = false

            for i = 1:rank

                result += x.lambda[i]^2 * ZX[s][i, i]^2 

                for j = skipindex(i, 1:rank)

                    result += x.lambda[i] * x.lambda[j] * prod(getindex.(ZX[mask_r], i, j)) * ZX[.!mask_r][1][j, i]

                end

            end

            mask_r[r] = true
        end

        mask_s[s] = true
    end

    return result

end

# Last try of squared norm
function lastnorm(A::KroneckerMatrix{T}, x::ktensor) where T<:AbstractFloat

    orders = dimensions(A)
    d      = ndims(x)
    rank   = ncomponents(x)

    # Return vector of matrices as described above
    Z = matrix_vector(A, x)

    X_inner = [ ones(rank, rank) for s in 1:d ]
    Z_inner = [ ones(rank, rank) for s in 1:d ]
    ZX      = [ ones(rank, rank) for s in 1:d ]
    

    for s = 1:length(A)

        LinearAlgebra.mul!(X_inner[s], transpose(x.fmat[s]), x.fmat[s])
        LinearAlgebra.mul!(Z_inner[s], transpose(Z[s]), Z[s])
        LinearAlgebra.mul!(ZX[s], transpose(Z[s]), x.fmat[s])

    end

    my_norm = 0.0

    mask_s = trues(d)
    mask_r = trues(d)

    for s in 1:d

        for i = 1:rank

            my_norm += x.lambda[i]^2 * Z_inner[s][i, i]

            mask_s[s] = false

            for j = skipindex(i, 1:rank)

                my_norm += x.lambda[i] * x.lambda[j] * prod(getindex.(X_inner[mask_s], i, j)) * Z_inner[.!mask_s][1][i, j]

            end

        end

        for r = skipindex(s, 1:d)

            mask_r[r] = false

            for i = 1:rank

                my_norm += x.lambda[i]^2 * ZX[s][i, i]^2 

                for j = skipindex(i, 1:rank)

                    #my_norm += x.lambda[i] * x.lambda[j] * prod(getindex.(ZX[mask_r], i, j)) * XZ[.!mask_r][1][i, j]
                    my_norm += x.lambda[i] * x.lambda[j] * prod(getindex.(ZX[mask_r], i, j)) * ZX[.!mask_r][1][j, i]

                end

            end

            mask_r[r] = true
        end

        mask_s[s] = true
    end

    return my_norm

end

function compressed_residual(
        Ly::Vector{LowerTriangular{T, Matrix{T}}},
        H::KroneckerMatrix{T},
        y::ktensor,
        b::Vector{Array{T}}) where T <:AbstractFloat

    # TODO: 
    #
    # We know that 
    #
    # ||Hy - b||² = ||Hy||² -2⋅bᵀ(Hy) + ||b||² 
    
    d = length(H)
    t = ncomponents(y)

    # For this we evaluate all B⁽ˢ⁾[:, i] = Hₛy⁽ˢ⁾ᵢ ∈ ℝⁿₛ for i = 1,…,t
    B = matrix_vector(H, y)

    # First we compute ||Hy||²
    Hy_norm = squared_matrix_vector(Ly, B, H, y)

    # Now we compute <Hy, b>₂

    Hy_b = 0.0

    bY = [ zeros(t) for _ in 1:d ]
    bZ = [ zeros(t) for _ in 1:d ]


    mul!(bY, b, y)
    mul!(bZ, b, B)
    


    for s = 1:d, r = skipindex(s, 1:d), i = 1:t

        Hy_b += bY[r][i] * bY[s][i]

    end

    # Finally we compute the 2-norm of b
    b_norm = norm(b)

    return Hy_norm - 2 * Hy_b + b_norm
    
    
end

    
function residual_norm(H::KroneckerMatrix, y::ktensor, 𝔎::Vector{Int}, b)
    
    # Compute norm of the residual according to Lemma 3.4 of paper.
    
    # Σ |hˢₖ₊₁ₖ|² * Σ |y\_𝔏|² + ||ℋy - b̃||²
    
    # Get entries at indices (kₛ+1, kₛ) for each dimension with pair of 
    # multiindices 𝔎+1, 𝔎

    # Number of dimensions
    d = size(H)

    # Tensor rank
    t = ncomponents(y)

    # Extract subdiagonal entries of upper Hesseberg matrices
    h² = map(abs, hessenberg_subdiagonals(H, 𝔎)).^2

    # Allocate memory for (lower triangular) matrices representing inner products
    Ly = [ zeros(t, t) for _ in 1:d ]

    for s = 1:d

        BLAS.syrk!('L', 'T', 1.0, y.fmat[s], 1.0, Ly[s])

    end

    res_norm = 0.0

    for s = 1:d

        y² = 0.0

        C = compute_coefficients(Ly[s], y.fmat[s][𝔎[s], :])

        for k = 1:t, i = k:t

            product = 1.0

            for j = skipindex(s, 1:d)

                product *= Ly[j][i, k]

            end

            y² += C[i, k] * product

        end

        # Here I'm counting the diagonal twice... Need to account for that.
        y² *= 2.0

        res_norm += h²[s] * y²

    end


    # Compute squared compressed residual norm
    rₕ = compressed_residual(Ly, H, y, b)

    return res_norm + rₕ

end

function tensor_krylov(A::KroneckerMatrix, b::AbstractVector, tol) 

	# Initilialize implicit tensorized Krylov subspace basis and upper Hessenberg 
	# matrix of compressed linear system
	decompositions = [Arnoldi(A[s], size(A[s], 1)) for s = 1:length(A)]

	# Compute basis and Hessenberg factor of each Krylov subspace 𝓚ₖ(Aₛ, bₛ) 
	for s = 1:length(A)
		
		arnoldi_modified!(A[s], b[s], 100, decompositions[s])
		
	end


    #H = KroneckerMatrix(decompositions)

    
    #y = solve_compressed_system()

	return decompositions
end
